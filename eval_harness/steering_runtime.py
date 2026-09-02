from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import Config
from .core import SafeToolSandbox, message, parse_assistant, tool_call, tool_result
from .runner import (
    DEVELOPER_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
    classify_outcome,
    generate,
)
from .steering_agent import (
    _DENIAL,
    _legitimate_task_score,
    _read_jsonl,
    summarize_agent_results,
    tool_span_plan,
)
from .steering_repr import (
    ROLE_TO_INDEX,
    _atomic_json,
    _git_commit,
    _load_local_model,
    _sha256_file,
)

_DATASET_MANIFESTS = {
    "layer": "steering_attack_layer_manifest.jsonl",
    "tune": "steering_attack_tune_manifest.jsonl",
    "devval": "steering_attack_devval_manifest.jsonl",
}


class ToolResidualSteerer:
    """Intervene at one post-block residual hook on true Tool content tokens."""

    def __init__(
        self,
        layers,
        layer: int,
        method: str,
        alpha: float,
        representation_state: dict,
        calibration_state: dict,
        rank: int = 4,
    ) -> None:
        if method not in {"soft-pairwise", "continuous"}:
            raise ValueError(f"unsupported steering method: {method}")
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"layer {layer} is outside 0..{len(layers) - 1}")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if rank not in calibration_state["continuous_states"]:
            raise ValueError(f"continuous rank {rank} has no clean calibration")
        self.layer = layer
        self.method = method
        self.alpha = float(alpha)
        self.rank = rank
        self.roles = list(representation_state["roles"])
        self.positions: torch.Tensor | None = None
        self.call_stats: list[dict] = []
        self._device_state: dict[str, torch.Tensor] = {}
        self._representation_state = representation_state
        self._calibration_state = calibration_state
        self.handle = layers[layer].register_forward_hook(self._hook)

    def prepare(self, positions: list[int]) -> None:
        self.positions = (
            torch.tensor(positions, dtype=torch.long) if positions else None
        )

    def reset_case(self) -> None:
        self.positions = None
        self.call_stats = []

    def _to_device(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        if self._device_state.get("_device_key") == key:
            return self._device_state
        layer = self.layer
        rank_state = self._calibration_state["continuous_states"][self.rank]
        tensors = {
            "probe_weight": self._representation_state["probe_weight"][layer]
            .float()
            .to(device),
            "probe_bias": self._representation_state["probe_bias"][layer]
            .float()
            .to(device),
            "probe_temperature": self._representation_state[
                "probe_temperature"
            ][layer]
            .float()
            .to(device),
            "pair_to_tool": self._representation_state["pair_vector"][
                layer, :, ROLE_TO_INDEX["tool"]
            ]
            .float()
            .to(device),
            "soft_thresholds": self._calibration_state["soft_thresholds"][layer]
            .float()
            .to(device),
            "basis": self._representation_state["role_basis"][layer, :, : self.rank]
            .float()
            .to(device),
            "center": self._representation_state["global_center"][layer]
            .float()
            .to(device),
            "tool_mean": rank_state["tool_mean"][layer].float().to(device),
            "tool_cov_inv": rank_state["tool_cov_inv"][layer].float().to(device),
            "tool_threshold": rank_state["threshold"][layer].float().to(device),
        }
        tensors["_device_key"] = key  # type: ignore[assignment]
        self._device_state = tensors
        return tensors

    def _hook(self, _module, _inputs, output):
        if self.positions is None or not len(self.positions):
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != 1:
            raise ValueError("steering runtime currently requires batch size one")
        if hidden.shape[1] <= int(self.positions.max()):
            return output
        positions = self.positions.to(hidden.device)
        values = hidden[0, positions].float()
        state = self._to_device(hidden.device)
        if self.method == "soft-pairwise":
            delta, stats = self._soft_pairwise(values, state)
        else:
            delta, stats = self._continuous(values, state)
        changed = hidden.clone()
        changed[0, positions] = (
            values + delta
        ).to(dtype=hidden.dtype)
        self.call_stats.append(stats)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    def _soft_pairwise(
        self, values: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        probabilities = torch.softmax(
            (values @ state["probe_weight"].T + state["probe_bias"])
            / state["probe_temperature"],
            dim=-1,
        )
        excess = (probabilities - state["soft_thresholds"]).clamp_min(0)
        excess[:, ROLE_TO_INDEX["tool"]] = 0
        per_role_delta = (
            self.alpha
            * excess[:, :, None]
            * state["pair_to_tool"][None, :, :]
        )
        delta = per_role_delta.sum(1)
        norms = delta.norm(dim=-1)
        active = excess.sum(-1).gt(0)
        return delta, {
            "method": self.method,
            "token_applications": len(values),
            "steered_token_applications": int(active.sum()),
            "intervention_norm_sum": float(norms.sum()),
            "intervention_norm_max": float(norms.max()),
            "wrong_role_excess_sum": float(excess.sum()),
            "wrong_role_excess_by_role": {
                role: float(excess[:, index].sum())
                for index, role in enumerate(self.roles)
                if role != "tool"
            },
            "contribution_norm_by_role": {
                role: float(per_role_delta[:, index].norm(dim=-1).sum())
                for index, role in enumerate(self.roles)
                if role != "tool"
            },
        }

    def _continuous(
        self, values: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        coordinates = (values - state["center"]) @ state["basis"]
        difference = coordinates - state["tool_mean"]
        distance = torch.einsum(
            "ni,ij,nj->n", difference, state["tool_cov_inv"], difference
        ).clamp_min(0)
        outside = distance.gt(state["tool_threshold"])
        rho = torch.zeros_like(distance)
        rho[outside] = 1.0 - torch.sqrt(
            state["tool_threshold"] / distance[outside].clamp_min(1e-12)
        )
        delta_coordinates = -rho[:, None] * difference
        delta = self.alpha * delta_coordinates @ state["basis"].T
        norms = delta.norm(dim=-1)
        return delta, {
            "method": self.method,
            "token_applications": len(values),
            "steered_token_applications": int(outside.sum()),
            "intervention_norm_sum": float(norms.sum()),
            "intervention_norm_max": float(norms.max()),
            "mahalanobis_distance_sum": float(distance.sum()),
            "mahalanobis_distance_max": float(distance.max()),
            "outside_tool_region": int(outside.sum()),
        }

    def summary(self) -> dict:
        totals: dict[str, float] = defaultdict(float)
        role_excess: dict[str, float] = defaultdict(float)
        role_contribution: dict[str, float] = defaultdict(float)
        for stats in self.call_stats:
            for key, value in stats.items():
                if key == "wrong_role_excess_by_role":
                    for role, role_value in value.items():
                        role_excess[role] += role_value
                elif key == "contribution_norm_by_role":
                    for role, role_value in value.items():
                        role_contribution[role] += role_value
                elif isinstance(value, (int, float)) and not key.endswith("_max"):
                    totals[key] += value
                elif key.endswith("_max"):
                    totals[key] = max(totals[key], value)
        applications = int(totals["token_applications"])
        steered = int(totals["steered_token_applications"])
        result = {
            "method": self.method,
            "layer": self.layer,
            "alpha": self.alpha,
            "rank": self.rank if self.method == "continuous" else None,
            "prefill_calls_with_tool_tokens": len(self.call_stats),
            "token_applications": applications,
            "steered_token_applications": steered,
            "steered_fraction": steered / applications if applications else 0.0,
            "mean_intervention_norm": (
                totals["intervention_norm_sum"] / applications
                if applications
                else 0.0
            ),
            "max_intervention_norm": totals["intervention_norm_max"],
        }
        if self.method == "soft-pairwise":
            result.update(
                {
                    "mean_wrong_role_excess_mass": (
                        totals["wrong_role_excess_sum"] / applications
                        if applications
                        else 0.0
                    ),
                    "wrong_role_excess_by_role": dict(role_excess),
                    "contribution_norm_by_role": dict(role_contribution),
                }
            )
        else:
            result.update(
                {
                    "fraction_outside_tool_region": (
                        totals["outside_tool_region"] / applications
                        if applications
                        else 0.0
                    ),
                    "mean_mahalanobis_distance": (
                        totals["mahalanobis_distance_sum"] / applications
                        if applications
                        else 0.0
                    ),
                    "max_mahalanobis_distance": totals[
                        "mahalanobis_distance_max"
                    ],
                }
            )
        return result

    def close(self) -> None:
        self.handle.remove()


def _run_case_with_steering(
    case: dict,
    reference_html: str,
    model,
    tokenizer,
    config: Config,
    steerer: ToolResidualSteerer,
) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sandbox = SafeToolSandbox(case["url"], case["html"])
    messages = [
        message("system", SYSTEM_PROMPT.format(reasoning=config.reasoning)),
        message("developer", DEVELOPER_PROMPT),
        message(
            "user",
            USER_PROMPT.format(workspace=sandbox.workspace, url=case["url"]),
        ),
    ]
    transcript = []
    torch.manual_seed(config.seed + int(case["case_id"]))
    termination_reason = "max_steps"
    final_answer = None
    steerer.reset_case()
    for step in range(config.max_steps):
        prompt = "".join(messages) + "<|start|>assistant"
        plan = tool_span_plan(
            prompt,
            tokenizer,
            seen_tool_messages=0,
            max_tokens_per_message=1_000_000_000,
            tail_tokens=0,
        )
        steerer.prepare(plan["positions"])
        generation_started = time.perf_counter()
        raw, prompt_tokens, generated_tokens = generate(
            model, tokenizer, prompt, config
        )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        parsed = parse_assistant(raw)
        transcript.append(
            {
                "step": step,
                "raw": raw,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "generation_seconds": round(generation_seconds, 3),
                **parsed,
            }
        )
        for analysis in parsed["analysis"]:
            messages.append(message("assistant", analysis, "analysis"))
        if parsed["final"]:
            final_answer = parsed["final"]
            messages.append(message("assistant", parsed["final"], "final"))
        for function_name, raw_args in parsed["tool_calls"]:
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            messages.append(tool_call(function_name, raw_args))
            output = sandbox.run(function_name.split(".", 1)[-1], args)
            messages.append(tool_result(function_name, output))
        if not parsed["tool_calls"]:
            termination_reason = "final" if parsed["final"] else "no_tool_call"
            break
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    raw_text = " ".join(turn["raw"].lower() for turn in transcript)
    output_class = classify_outcome(sandbox.attempted, sandbox.succeeded, raw_text)
    task_quality, task_plumbing_success = _legitimate_task_score(
        case, reference_html, sandbox, final_answer
    )
    task_success = int(task_quality["success"])
    total_generated_tokens = sum(turn["generated_tokens"] for turn in transcript)
    public_case = {key: value for key, value in case.items() if key != "html"}
    return {
        **public_case,
        "output_class": output_class,
        "attack_attempt": int(output_class.startswith("ATTEMPTED")),
        "attack_success": int(sandbox.succeeded),
        "legitimate_task_success": task_success,
        "secure_task_success": int(task_success and not sandbox.succeeded),
        "task_plumbing_success": task_plumbing_success,
        "task_output_present": task_quality["output_present"],
        "task_format_success": task_quality["format_success"],
        "task_content_grounded": task_quality["content_grounded"],
        "page_content_returned_to_model": task_quality["page_returned"],
        "task_quality": task_quality,
        "over_refusal": int(bool(_DENIAL.search(raw_text)) and not task_success),
        "termination_reason": termination_reason,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "total_prompt_tokens": sum(turn["prompt_tokens"] for turn in transcript),
        "total_generated_tokens": total_generated_tokens,
        "generated_tokens_per_second": round(
            total_generated_tokens / elapsed_seconds, 3
        ),
        "peak_vram_gib": round(peak_vram_bytes / 2**30, 3),
        "intervention": steerer.summary(),
        "transcript": transcript,
        "tool_events": sandbox.events,
    }


def _reference_map(cases: list[dict]) -> dict[int, str]:
    clean = {
        int(case.get("clean_case_id", case["case_id"])): case["html"]
        for case in cases
        if case["variant"] == "clean"
    }
    return {
        int(case["case_id"]): clean[int(case.get("clean_case_id", case["case_id"]))]
        for case in cases
    }


def _aggregate_intervention(rows: list[dict]) -> dict:
    applications = sum(row["intervention"]["token_applications"] for row in rows)
    steered = sum(
        row["intervention"]["steered_token_applications"] for row in rows
    )
    weighted_norm = sum(
        row["intervention"]["mean_intervention_norm"]
        * row["intervention"]["token_applications"]
        for row in rows
    )
    return {
        "token_applications": applications,
        "steered_token_applications": steered,
        "steered_fraction": steered / applications if applications else 0.0,
        "mean_intervention_norm": weighted_norm / applications if applications else 0.0,
        "max_intervention_norm": max(
            (row["intervention"]["max_intervention_norm"] for row in rows),
            default=0.0,
        ),
    }


def run_steering(
    repo: Path,
    dataset: str,
    config_path: Path,
    method: str,
    layer: int,
    alpha: float,
    rank: int,
    run_name: str,
    attack_only: bool = False,
    resume: bool = False,
) -> Path:
    if dataset not in _DATASET_MANIFESTS:
        raise ValueError(f"unsupported steering dataset: {dataset}")
    manifest_path = repo / "eval_data" / _DATASET_MANIFESTS[dataset]
    all_cases = _read_jsonl(manifest_path)
    reference_by_id = _reference_map(all_cases)
    cases = [
        case for case in all_cases if not attack_only or case["variant"] != "clean"
    ]
    config = Config.load(config_path)
    representation_path = (
        repo
        / "research_outputs/phase3_steering/representation_analysis/steering_state.pt"
    )
    calibration_path = (
        repo
        / "research_outputs/phase3_steering/layer_screening/calibration_state.pt"
    )
    representation_state = torch.load(
        representation_path, map_location="cpu", weights_only=False
    )
    calibration_state = torch.load(
        calibration_path, map_location="cpu", weights_only=False
    )
    output_dir = repo / "research_outputs/phase3_steering/runs" / run_name
    results_path = output_dir / "results.jsonl"
    metadata_path = output_dir / "run.json"
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "manifest_path": str(manifest_path.relative_to(repo)),
        "manifest_sha256": _sha256_file(manifest_path),
        "selected_case_ids": [int(case["case_id"]) for case in cases],
        "attack_only": attack_only,
        "config_fingerprint": config.fingerprint(),
        "method": method,
        "layer": layer,
        "alpha": alpha,
        "rank": rank if method == "continuous" else None,
        "hook": "decoder block output before next block",
        "scope": "all true Tool content tokens during every prompt prefill",
        "representation_state_sha256": _sha256_file(representation_path),
        "calibration_state_sha256": _sha256_file(calibration_path),
        "git_commit": _git_commit(repo),
    }
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"{output_dir} exists; pass --resume")
        previous = json.loads(metadata_path.read_text())
        invariant = (
            "dataset",
            "manifest_sha256",
            "selected_case_ids",
            "config_fingerprint",
            "method",
            "layer",
            "alpha",
            "rank",
            "representation_state_sha256",
            "calibration_state_sha256",
        )
        mismatches = [key for key in invariant if previous[key] != metadata[key]]
        if mismatches:
            raise ValueError(f"steering resume metadata mismatch: {mismatches}")
        metadata = previous
    else:
        output_dir.mkdir(parents=True)
        _atomic_json(metadata_path, metadata)
    rows = _read_jsonl(results_path) if results_path.exists() else []
    completed = {int(row["case_id"]) for row in rows}
    pending = [case for case in cases if int(case["case_id"]) not in completed]
    _atomic_json(
        output_dir / "run_state.json",
        {
            "status": "running" if pending else "complete",
            "completed_case_ids": sorted(completed),
            "remaining_case_ids": [int(case["case_id"]) for case in pending],
        },
    )
    if pending:
        tokenizer, model = _load_local_model(config)
        steerer = ToolResidualSteerer(
            model.model.layers,
            layer,
            method,
            alpha,
            representation_state,
            calibration_state,
            rank,
        )
        try:
            with results_path.open("a") as output_file:
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
                    rows.append(result)
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed.add(case_id)
                    remaining = [
                        int(item["case_id"])
                        for item in cases
                        if int(item["case_id"]) not in completed
                    ]
                    _atomic_json(
                        output_dir / "run_state.json",
                        {
                            "status": "running" if remaining else "complete",
                            "completed_case_ids": sorted(completed),
                            "remaining_case_ids": remaining,
                        },
                    )
                    print(
                        f"case {case_id}: {result['output_class']} "
                        f"task={result['legitimate_task_success']} "
                        f"steered={result['intervention']['steered_fraction']:.3f} "
                        f"in {result['elapsed_seconds']:.3f}s",
                        flush=True,
                    )
        finally:
            steerer.close()
    summary = summarize_agent_results(rows)
    summary["intervention"] = _aggregate_intervention(rows)
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        output_dir / "run_state.json",
        {
            "status": "complete",
            "completed_case_ids": sorted(completed),
            "remaining_case_ids": [],
        },
    )
    metadata["status"] = "complete"
    metadata["result_count"] = len(rows)
    metadata["results_sha256"] = _sha256_file(results_path)
    _atomic_json(metadata_path, metadata)
    return output_dir
