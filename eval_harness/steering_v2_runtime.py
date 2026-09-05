from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import Config
from .steering_agent import _read_jsonl, summarize_agent_results
from .steering_repr import _atomic_json, _git_commit, _load_local_model, _sha256_file
from .steering_runtime import (
    _aggregate_intervention,
    _reference_map,
    _run_case_with_steering,
)
from .steering_v2_diagnostics import WRONG_INDICES, compute_v2_intervention
from .steering_v2_repr import ROLE_TO_INDEX


class V2SoftPairwiseSteerer:
    """Bounded, smoothed soft-pairwise intervention at GPT-OSS MLP input."""

    def __init__(
        self,
        layers,
        layer: int,
        rho_max: float,
        representation: dict,
        calibration: dict,
    ) -> None:
        if layer not in representation["probe_weight"]:
            raise ValueError(f"v2 representation has no layer {layer}")
        if layer not in calibration["layers"]:
            raise ValueError(f"v2 calibration has no layer {layer}")
        if not 0 < rho_max <= 0.1:
            raise ValueError("rho_max must be in (0, 0.1]")
        self.layer = layer
        self.rho_max = float(rho_max)
        self.window = int(calibration["window"])
        self.route_temperature = float(calibration["route_temperature"])
        self.gamma = float(calibration["gamma"])
        self.direction_floor = float(calibration["direction_floor"])
        self.positions: torch.Tensor | None = None
        self.segment_ranges: list[tuple[int, int]] = []
        self.call_stats: list[dict] = []
        layer_calibration = calibration["layers"][layer]
        self._cpu_state = {
            "weight": representation["probe_weight"][layer].float(),
            "bias": representation["probe_bias"][layer].float(),
            "center": layer_calibration["margin_center"].float(),
            "scale": layer_calibration["margin_scale"].float(),
            "beta": float(layer_calibration["joint_threshold"]),
            "unit": representation["unit_pair_direction"][layer][
                list(WRONG_INDICES), ROLE_TO_INDEX["tool"]
            ].float(),
        }
        self._device_state: dict = {}
        self.handle = layers[layer].mlp.register_forward_pre_hook(self._hook)

    def prepare(self, positions: list[int]) -> None:
        self.positions = (
            torch.tensor(positions, dtype=torch.long) if positions else None
        )
        self.segment_ranges = []
        if not positions:
            return
        start = 0
        for index in range(1, len(positions)):
            if positions[index] != positions[index - 1] + 1:
                self.segment_ranges.append((start, index))
                start = index
        self.segment_ranges.append((start, len(positions)))

    def reset_case(self) -> None:
        self.positions = None
        self.segment_ranges = []
        self.call_stats = []

    def _state(self, device: torch.device) -> dict:
        key = str(device)
        if self._device_state.get("device") == key:
            return self._device_state
        self._device_state = {
            name: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in self._cpu_state.items()
        }
        self._device_state["device"] = key
        return self._device_state

    def _hook(self, _module, inputs):
        if self.positions is None or not len(self.positions):
            return None
        hidden = inputs[0]
        if hidden.shape[0] != 1:
            raise ValueError("v2 steering requires batch size one")
        if hidden.shape[1] <= int(self.positions.max()):
            return None
        state = self._state(hidden.device)
        positions = self.positions.to(hidden.device)
        values = hidden[0, positions].float()
        delta = torch.zeros_like(values)
        stats = defaultdict(float)
        stats["token_applications"] = len(values)
        for start, end in self.segment_ranges:
            observed = compute_v2_intervention(
                values[start:end],
                state["weight"],
                state["bias"],
                state["center"],
                state["scale"],
                state["beta"],
                state["unit"],
                window=self.window,
                route_temperature=self.route_temperature,
                gamma=self.gamma,
                rho_max=self.rho_max,
                direction_floor=self.direction_floor,
            )
            delta[start:end] = observed["delta"]
            ratios = observed["delta_norm"] / observed["residual_norm"].clamp_min(1e-12)
            stats["steered_token_applications"] += int(observed["gate"].sum())
            stats["intervention_norm_sum"] += float(observed["delta_norm"].sum())
            stats["intervention_norm_max"] = max(
                stats["intervention_norm_max"], float(observed["delta_norm"].max())
            )
            stats["relative_norm_sum"] += float(ratios.sum())
            stats["relative_norm_max"] = max(
                stats["relative_norm_max"], float(ratios.max())
            )
            stats["joint_score_sum"] += float(observed["score"].sum())
            stats["joint_score_max"] = max(
                stats["joint_score_max"], float(observed["score"].max())
            )
            stats["tool_probability_change_sum"] += float(
                (observed["post_p_tool"] - observed["p_tool"]).sum()
            )
            stats["dominant_margin_change_sum"] += float(
                (observed["post_dominant_margin"] - observed["dominant_margin"]).sum()
            )
        if stats["relative_norm_max"] > self.rho_max + 1e-5:
            raise AssertionError("v2 relative intervention cap was violated")
        self.call_stats.append(dict(stats))
        changed = hidden.clone()
        changed[0, positions] = (values + delta).to(hidden.dtype)
        return (changed, *inputs[1:])

    def summary(self) -> dict:
        totals = defaultdict(float)
        for call in self.call_stats:
            for key, value in call.items():
                if key.endswith("_max"):
                    totals[key] = max(totals[key], value)
                else:
                    totals[key] += value
        applications = int(totals["token_applications"])
        steered = int(totals["steered_token_applications"])
        denominator = max(1, applications)
        return {
            "method": "soft-pairwise-v2",
            "hook": "GPT-OSS MLP input / all_pre_mlp_hidden_states",
            "layer": self.layer,
            "rho_max": self.rho_max,
            "window": self.window,
            "prefill_calls_with_tool_tokens": len(self.call_stats),
            "token_applications": applications,
            "steered_token_applications": steered,
            "steered_fraction": steered / denominator,
            "mean_intervention_norm": totals["intervention_norm_sum"] / denominator,
            "max_intervention_norm": totals["intervention_norm_max"],
            "mean_relative_intervention_norm": totals["relative_norm_sum"]
            / denominator,
            "max_relative_intervention_norm": totals["relative_norm_max"],
            "mean_joint_score": totals["joint_score_sum"] / denominator,
            "max_joint_score": totals["joint_score_max"],
            "mean_tool_probability_change": totals["tool_probability_change_sum"]
            / denominator,
            "mean_dominant_wrong_margin_change": totals["dominant_margin_change_sum"]
            / denominator,
        }

    def close(self) -> None:
        self.handle.remove()


def run_v2_steering(
    repo: Path,
    config_path: Path,
    run_name: str,
    rho_max: float = 0.005,
    resume: bool = False,
) -> Path:
    manifest_path = repo / "eval_data/steering_attack_layer_manifest.jsonl"
    all_cases = _read_jsonl(manifest_path)
    reference_by_id = _reference_map(all_cases)
    cases = [case for case in all_cases if case["variant"] != "clean"]
    config = Config.load(config_path)
    representation_path = (
        repo
        / "research_outputs/phase3_steering_v2/representation_analysis/steering_state.pt"
    )
    calibration_path = (
        repo / "research_outputs/phase3_steering_v2/calibration/calibration_state.pt"
    )
    representation = torch.load(
        representation_path, map_location="cpu", weights_only=False
    )
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    layer = calibration["selected_layer"]
    if layer is None:
        raise RuntimeError("v2 diagnostics did not authorize agent evaluation")
    output = repo / "research_outputs/phase3_steering_v2/runs" / run_name
    result_path = output / "results.jsonl"
    metadata_path = output / "run.json"
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "layer",
        "manifest_path": str(manifest_path.relative_to(repo)),
        "manifest_sha256": _sha256_file(manifest_path),
        "selected_case_ids": [int(case["case_id"]) for case in cases],
        "attack_only": True,
        "config_fingerprint": config.fingerprint(),
        "method": "soft-pairwise-v2",
        "layer": layer,
        "rho_max": rho_max,
        "window": calibration["window"],
        "joint_quantile": calibration["joint_quantile"],
        "route_temperature": calibration["route_temperature"],
        "gamma": calibration["gamma"],
        "direction_floor": calibration["direction_floor"],
        "hook": calibration["hook"],
        "scope": "all true Tool content tokens during every prompt prefill",
        "representation_state_sha256": _sha256_file(representation_path),
        "calibration_state_sha256": _sha256_file(calibration_path),
        "git_commit": _git_commit(repo),
    }
    if output.exists():
        if not resume:
            raise FileExistsError(f"{output} exists; pass --resume")
        previous = json.loads(metadata_path.read_text())
        invariant = (
            "manifest_sha256",
            "selected_case_ids",
            "config_fingerprint",
            "method",
            "layer",
            "rho_max",
            "window",
            "joint_quantile",
            "representation_state_sha256",
            "calibration_state_sha256",
        )
        mismatch = [key for key in invariant if previous[key] != metadata[key]]
        if mismatch:
            raise ValueError(f"v2 steering resume mismatch: {mismatch}")
        metadata = previous
    else:
        output.mkdir(parents=True)
        _atomic_json(metadata_path, metadata)
    rows = _read_jsonl(result_path) if result_path.exists() else []
    completed = {int(row["case_id"]) for row in rows}
    pending = [case for case in cases if int(case["case_id"]) not in completed]
    _atomic_json(
        output / "run_state.json",
        {
            "status": "running" if pending else "complete",
            "completed_case_ids": sorted(completed),
            "remaining_case_ids": [int(case["case_id"]) for case in pending],
        },
    )
    if pending:
        tokenizer, model = _load_local_model(config)
        steerer = V2SoftPairwiseSteerer(
            model.model.layers,
            layer,
            rho_max,
            representation,
            calibration,
        )
        try:
            with result_path.open("a") as handle:
                for case in pending:
                    case_id = int(case["case_id"])
                    result = _run_case_with_steering(
                        case,
                        reference_by_id[case_id],
                        model,
                        tokenizer,
                        config,
                        steerer,
                    )
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(result)
                    completed.add(case_id)
                    remaining = [
                        int(item["case_id"])
                        for item in cases
                        if int(item["case_id"]) not in completed
                    ]
                    _atomic_json(
                        output / "run_state.json",
                        {
                            "status": "running" if remaining else "complete",
                            "completed_case_ids": sorted(completed),
                            "remaining_case_ids": remaining,
                        },
                    )
                    print(
                        f"v2 case {case_id}: {result['output_class']} "
                        f"task={result['legitimate_task_success']} "
                        f"steered={result['intervention']['steered_fraction']:.4f} "
                        f"in {result['elapsed_seconds']:.3f}s",
                        flush=True,
                    )
        finally:
            steerer.close()
    summary = summarize_agent_results(rows)
    summary["intervention"] = _aggregate_intervention(rows)
    applications = sum(row["intervention"]["token_applications"] for row in rows)
    summary["v2_intervention"] = {
        "mean_relative_intervention_norm": sum(
            row["intervention"]["mean_relative_intervention_norm"]
            * row["intervention"]["token_applications"]
            for row in rows
        )
        / max(1, applications),
        "max_relative_intervention_norm": max(
            row["intervention"]["max_relative_intervention_norm"] for row in rows
        ),
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "run_state.json",
        {
            "status": "complete",
            "completed_case_ids": sorted(completed),
            "remaining_case_ids": [],
        },
    )
    metadata["status"] = "complete"
    metadata["result_count"] = len(rows)
    metadata["results_sha256"] = _sha256_file(result_path)
    metadata["elapsed_seconds"] = round(
        sum(float(row["elapsed_seconds"]) for row in rows), 3
    )
    _atomic_json(metadata_path, metadata)
    return output
