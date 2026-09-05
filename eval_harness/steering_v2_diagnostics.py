from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from .config import Config
from .steering_agent import tool_span_plan
from .steering_debug import extract_single_insertion, replay_prompts
from .steering_repr import _atomic_json, _load_local_model, _sha256_file
from .steering_v2_repr import ROLE_TO_INDEX, PreMlpCapture, _CaptureFinished

WRONG_ROLES = ("system", "user", "cot", "assistant")
WRONG_INDICES = tuple(ROLE_TO_INDEX[role] for role in WRONG_ROLES)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def smooth_local(values: torch.Tensor, window: int) -> torch.Tensor:
    """Centered box smoothing with 15 left/16 right neighbors for window=32."""
    if values.ndim != 2 or window < 1:
        raise ValueError("values must be [tokens, features] and window positive")
    left = (window - 1) // 2
    right = window - left - 1
    prefix = torch.cat(
        [
            torch.zeros((1, values.shape[1]), dtype=values.dtype, device=values.device),
            values.cumsum(0),
        ]
    )
    indices = torch.arange(len(values), device=values.device)
    starts = (indices - left).clamp_min(0)
    ends = (indices + right + 1).clamp_max(len(values))
    return (prefix[ends] - prefix[starts]) / (ends - starts)[:, None]


def compute_v2_intervention(
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    beta: float,
    unit_directions: torch.Tensor,
    window: int = 32,
    route_temperature: float = 1.0,
    gamma: float = 1.0,
    rho_max: float = 0.005,
    direction_floor: float = 0.1,
) -> dict[str, torch.Tensor]:
    values = values.float()
    weight = weight.float()
    bias = bias.float()
    logits = values @ weight.T + bias
    margins = logits[:, list(WRONG_INDICES)] - logits[:, [ROLE_TO_INDEX["tool"]]]
    q = (margins - center) / scale
    q_smooth = smooth_local(q, window)
    score = q_smooth.max(-1).values
    route = torch.softmax(q_smooth / route_temperature, dim=-1)
    raw_direction = route @ unit_directions.float()
    raw_direction_norm = raw_direction.norm(dim=-1)
    gate = score.gt(beta) & raw_direction_norm.ge(direction_floor)
    severity = torch.zeros_like(score)
    severity[gate] = 1.0 - torch.exp(-(score[gate] - beta) / gamma)
    direction = raw_direction / raw_direction_norm.clamp_min(1e-12)[:, None]
    residual_norm = values.norm(dim=-1)
    magnitude = rho_max * residual_norm * severity
    delta = magnitude[:, None] * direction
    post_logits = (values + delta) @ weight.T + bias
    post_margins = (
        post_logits[:, list(WRONG_INDICES)] - post_logits[:, [ROLE_TO_INDEX["tool"]]]
    )
    probabilities = torch.softmax(logits, dim=-1)
    post_probabilities = torch.softmax(post_logits, dim=-1)
    dominant = margins.gather(1, q_smooth.argmax(-1)[:, None]).squeeze(1)
    post_dominant = post_margins.gather(1, q_smooth.argmax(-1)[:, None]).squeeze(1)
    return {
        "logits": logits,
        "margins": margins,
        "q": q,
        "q_smooth": q_smooth,
        "score": score,
        "route": route,
        "raw_direction_norm": raw_direction_norm,
        "gate": gate,
        "severity": severity,
        "residual_norm": residual_norm,
        "delta": delta,
        "delta_norm": delta.norm(dim=-1),
        "post_logits": post_logits,
        "post_margins": post_margins,
        "p_tool": probabilities[:, ROLE_TO_INDEX["tool"]],
        "post_p_tool": post_probabilities[:, ROLE_TO_INDEX["tool"]],
        "dominant_margin": dominant,
        "post_dominant_margin": post_dominant,
    }


def _primary_tool_prompt(row: dict, case: dict, config: Config, tokenizer) -> dict:
    candidates = []
    for generation_step, prompt in enumerate(replay_prompts(row, case, config)):
        plan = tool_span_plan(
            prompt,
            tokenizer,
            seen_tool_messages=0,
            max_tokens_per_message=1_000_000_000,
            tail_tokens=0,
        )
        for span in plan["spans"]:
            start = int(span["selected_start"])
            end = int(span["selected_end"])
            candidates.append(
                {
                    "full_count": int(span["full_content_token_count"]),
                    "generation_step": generation_step,
                    "prompt": prompt,
                    "positions": plan["positions"][start:end],
                }
            )
    if not candidates:
        raise ValueError(f"case {case['case_id']} has no Tool output prompt")
    return max(candidates, key=lambda item: item["full_count"])


def _region_codes(
    prompt: str,
    positions: list[int],
    offsets: list[tuple[int, int]],
    case: dict,
    clean_case: dict | None,
) -> tuple[torch.Tensor, dict]:
    # 0=clean/page_other, 1=near-injection context, 2=exact injection.
    codes = torch.zeros(len(positions), dtype=torch.int8)
    if case["variant"] == "clean":
        return codes, {"injection_token_range": None}
    if clean_case is None:
        raise ValueError("attack row lacks clean counterpart")
    injection, _ = extract_single_insertion(clean_case["html"], case["html"])
    escaped = json.dumps(injection, ensure_ascii=False)[1:-1]
    char_start = prompt.find(escaped)
    if char_start < 0:
        return codes, {
            "injection_sha256": case["injection_sha256"],
            "injection_id": case["injection_id"],
            "injection_observed": False,
            "injection_token_range": None,
        }
    char_end = char_start + len(escaped)
    injection_ordinals = [
        ordinal
        for ordinal, position in enumerate(positions)
        if offsets[position][1] > char_start and offsets[position][0] < char_end
    ]
    if not injection_ordinals:
        raise ValueError("selected injection has no token overlap")
    lo = max(0, injection_ordinals[0] - 48)
    hi = min(len(positions), injection_ordinals[-1] + 49)
    codes[lo:hi] = 1
    codes[injection_ordinals] = 2
    return codes, {
        "injection_sha256": case["injection_sha256"],
        "injection_id": case["injection_id"],
        "injection_observed": True,
        "injection_token_range": [injection_ordinals[0], injection_ordinals[-1]],
    }


def collect_v2_tool_activations(
    repo: Path,
    dataset: str,
    config_path: Path,
    resume: bool = False,
) -> Path:
    if dataset not in {"clean", "layer"}:
        raise ValueError("v2 Tool capture dataset must be clean or layer")
    manifest_name = (
        "steering_clean_manifest.jsonl"
        if dataset == "clean"
        else "steering_attack_layer_manifest.jsonl"
    )
    manifest_path = repo / "eval_data" / manifest_name
    cases = _read_jsonl(manifest_path)
    case_by_id = {int(case["case_id"]): case for case in cases}
    source_dir = repo / "research_outputs/phase3_steering/tool_activations" / dataset
    result_path = source_dir / "results_task_quality.jsonl"
    rows = _read_jsonl(result_path)
    row_by_id = {int(row["case_id"]): row for row in rows}
    representation_path = (
        repo
        / "research_outputs/phase3_steering_v2/representation_analysis/steering_state.pt"
    )
    representation = torch.load(
        representation_path, map_location="cpu", weights_only=False
    )
    layers = tuple(representation["candidate_layers_by_dev_accuracy"])
    config = Config.load(config_path)
    output = repo / "research_outputs/phase3_steering_v2/tool_pre_mlp" / dataset
    metadata_path = output / "metadata.json"
    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "manifest": str(manifest_path.relative_to(repo)),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_results": str(result_path.relative_to(repo)),
        "source_results_sha256": _sha256_file(result_path),
        "representation_state_sha256": _sha256_file(representation_path),
        "layers": list(layers),
        "hook": representation["hook"],
        "selection": "largest Tool message from exact saved undefended trajectory",
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        invariant = (
            "dataset",
            "manifest_sha256",
            "source_results_sha256",
            "representation_state_sha256",
            "layers",
            "hook",
        )
        mismatch = [key for key in invariant if previous.get(key) != metadata[key]]
        if mismatch:
            raise ValueError(f"v2 Tool capture resume mismatch: {mismatch}")
        if not resume:
            raise FileExistsError(f"{output} exists; pass --resume")
        metadata = previous
    else:
        output.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            (output / f"shards/layer-{layer:02d}").mkdir(parents=True)
        (output / "skipped").mkdir(parents=True)
        _atomic_json(metadata_path, metadata)
    (output / "skipped").mkdir(parents=True, exist_ok=True)
    skipped = {
        int(path.stem.split("-")[-1])
        for path in (output / "skipped").glob("case-*.json")
    }
    completed = skipped | {
        case_id
        for case_id in case_by_id
        if all(
            (output / f"shards/layer-{layer:02d}/case-{case_id:03d}.pt").exists()
            for layer in layers
        )
    }
    pending = [case_id for case_id in sorted(case_by_id) if case_id not in completed]
    if not pending:
        return output
    tokenizer, model = _load_local_model(config)
    capture = PreMlpCapture(model.model.layers, layers)
    started = time.perf_counter()
    try:
        for case_id in pending:
            case = case_by_id[case_id]
            try:
                selected = _primary_tool_prompt(
                    row_by_id[case_id], case, config, tokenizer
                )
            except ValueError as error:
                _atomic_json(
                    output / "skipped" / f"case-{case_id:03d}.json",
                    {
                        "case_id": case_id,
                        "variant": case["variant"],
                        "reason": str(error),
                    },
                )
                completed.add(case_id)
                skipped.add(case_id)
                print(f"v2 {dataset} case {case_id}: SKIPPED ({error})", flush=True)
                continue
            encoded = tokenizer(
                selected["prompt"],
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = [
                tuple(item) for item in encoded.pop("offset_mapping")[0].tolist()
            ]
            positions = selected["positions"]
            position_tensor = torch.tensor([positions], dtype=torch.long)
            clean_case = (
                case_by_id[int(case["clean_case_id"])]
                if dataset == "layer" and case["variant"] != "clean"
                else None
            )
            region_codes, injection_meta = _region_codes(
                selected["prompt"], positions, offsets, case, clean_case
            )
            capture.reset(position_tensor)
            inputs = {key: value.to(config.device) for key, value in encoded.items()}
            case_started = time.perf_counter()
            try:
                with torch.inference_mode():
                    model.model(**inputs, use_cache=False)
            except _CaptureFinished:
                pass
            token_ids = encoded["input_ids"][0, positions].cpu()
            token_texts = [
                tokenizer.decode([int(token_id)], skip_special_tokens=False)
                for token_id in token_ids
            ]
            for layer in layers:
                payload = {
                    "schema_version": 1,
                    "case_id": case_id,
                    "variant": case["variant"],
                    "dataset": dataset,
                    "page_id": case.get("page_id"),
                    "clean_case_id": case.get("clean_case_id"),
                    "generation_step": selected["generation_step"],
                    "page_returned": row_by_id[case_id].get(
                        "page_content_returned_to_model", 0
                    ),
                    "layer": layer,
                    "token_ids": token_ids,
                    "token_texts": token_texts,
                    "region_codes": region_codes,
                    **injection_meta,
                    "activations": capture.values[layer][0],
                }
                target = output / f"shards/layer-{layer:02d}/case-{case_id:03d}.pt"
                temporary = target.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, target)
            completed.add(case_id)
            remaining = sorted(set(case_by_id) - completed)
            _atomic_json(
                output / "run_state.json",
                {
                    "status": "running" if remaining else "complete",
                    "completed_case_ids": sorted(completed),
                    "remaining_case_ids": remaining,
                },
            )
            print(
                f"v2 {dataset} case {case_id}: {len(positions)} Tool tokens x "
                f"{len(layers)} layers in {time.perf_counter() - case_started:.3f}s",
                flush=True,
            )
    finally:
        capture.close()
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    metadata["status"] = "complete"
    metadata["skipped_case_ids"] = sorted(skipped)
    _atomic_json(metadata_path, metadata)
    _atomic_json(
        output / "run_state.json",
        {
            "status": "complete",
            "completed_case_ids": sorted(completed),
            "skipped_case_ids": sorted(skipped),
            "remaining_case_ids": [],
        },
    )
    return output


def _load_case_shards(root: Path, layer: int) -> list[dict]:
    return [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted((root / f"shards/layer-{layer:02d}").glob("case-*.pt"))
    ]


def _distribution(values: torch.Tensor) -> dict:
    values = values.double()
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=len(values) > 1)),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p99": float(torch.quantile(values, 0.99)),
    }


def _summarize_outputs(outputs: list[tuple[dict, dict]]) -> dict:
    regions = {"clean": 0, "page_other": 0, "context": 1, "injection": 2}
    summary = {}
    for name, code in regions.items():
        selections = []
        for shard, observed in outputs:
            if name == "clean":
                mask = (
                    torch.ones(len(shard["region_codes"]), dtype=torch.bool)
                    if shard["variant"] == "clean"
                    else torch.zeros(len(shard["region_codes"]), dtype=torch.bool)
                )
            elif shard["variant"] == "clean":
                continue
            else:
                mask = shard["region_codes"].eq(code)
            if mask.any():
                selections.append((observed, mask))
        if not selections:
            continue
        score = torch.cat([observed["score"][mask] for observed, mask in selections])
        gate = torch.cat([observed["gate"][mask] for observed, mask in selections])
        ratio = torch.cat(
            [
                observed["delta_norm"][mask]
                / observed["residual_norm"][mask].clamp_min(1e-12)
                for observed, mask in selections
            ]
        )
        p_change = torch.cat(
            [
                observed["post_p_tool"][mask] - observed["p_tool"][mask]
                for observed, mask in selections
            ]
        )
        margin_change = torch.cat(
            [
                observed["post_dominant_margin"][mask]
                - observed["dominant_margin"][mask]
                for observed, mask in selections
            ]
        )
        summary[name] = {
            "score": _distribution(score),
            "firing_rate": float(gate.float().mean()),
            "relative_intervention": _distribution(ratio),
            "mean_tool_probability_change": float(p_change.mean()),
            "mean_dominant_wrong_margin_change": float(margin_change.mean()),
        }
    return summary


def calibrate_and_diagnose_v2(
    repo: Path,
    window: int = 32,
    quantile: float = 0.99,
    rho_max: float = 0.005,
) -> Path:
    representation_path = (
        repo
        / "research_outputs/phase3_steering_v2/representation_analysis/steering_state.pt"
    )
    representation = torch.load(
        representation_path, map_location="cpu", weights_only=False
    )
    clean_root = repo / "research_outputs/phase3_steering_v2/tool_pre_mlp/clean"
    layer_root = repo / "research_outputs/phase3_steering_v2/tool_pre_mlp/layer"
    output = repo / "research_outputs/phase3_steering_v2/calibration"
    output.mkdir(parents=True, exist_ok=True)
    clean_results = _read_jsonl(
        repo
        / "research_outputs/phase3_steering/tool_activations/clean/results_task_quality.jsonl"
    )
    usable_clean_ids = {
        int(row["case_id"])
        for row in clean_results
        if row["page_content_returned_to_model"]
    }
    calibration_layers = {}
    reports = []
    token_debug_root = output / "token_diagnostics"
    token_debug_root.mkdir(exist_ok=True)
    for layer in representation["candidate_layers_by_dev_accuracy"]:
        weight = representation["probe_weight"][layer].float()
        bias = representation["probe_bias"][layer].float()
        unit = representation["unit_pair_direction"][layer][
            list(WRONG_INDICES), ROLE_TO_INDEX["tool"]
        ].float()
        clean_shards = _load_case_shards(clean_root, layer)
        calibration_shards = [
            shard
            for shard in clean_shards
            if int(shard["case_id"]) < 20 and int(shard["case_id"]) in usable_clean_ids
        ]
        sanity_shards = [
            shard
            for shard in clean_shards
            if int(shard["case_id"]) >= 20 and int(shard["case_id"]) in usable_clean_ids
        ]
        if len(calibration_shards) != 20:
            raise ValueError(
                f"expected 20 page-bearing calibration cases, found {len(calibration_shards)}"
            )
        margins = []
        for shard in calibration_shards:
            logits = shard["activations"].float() @ weight.T + bias
            margins.append(
                logits[:, list(WRONG_INDICES)] - logits[:, [ROLE_TO_INDEX["tool"]]]
            )
        all_margins = torch.cat(margins)
        center = all_margins.median(0).values
        scale = 1.4826 * (all_margins - center).abs().median(0).values + 1e-6
        calibration_scores = []
        for shard, margin in zip(calibration_shards, margins, strict=True):
            q = (margin - center) / scale
            calibration_scores.append(smooth_local(q, window).max(-1).values)
        beta = float(torch.quantile(torch.cat(calibration_scores), quantile))
        calibration_layers[layer] = {
            "margin_center": center,
            "margin_scale": scale,
            "joint_threshold": beta,
        }
        sanity_outputs = [
            (
                shard,
                compute_v2_intervention(
                    shard["activations"],
                    weight,
                    bias,
                    center,
                    scale,
                    beta,
                    unit,
                    window=window,
                    rho_max=rho_max,
                ),
            )
            for shard in sanity_shards
        ]
        layer_shards = _load_case_shards(layer_root, layer)
        layer_outputs = [
            (
                shard,
                compute_v2_intervention(
                    shard["activations"],
                    weight,
                    bias,
                    center,
                    scale,
                    beta,
                    unit,
                    window=window,
                    rho_max=rho_max,
                ),
            )
            for shard in layer_shards
        ]
        summary = _summarize_outputs(sanity_outputs + layer_outputs)
        sanity_summary = _summarize_outputs(sanity_outputs)["clean"]
        cap_max = max(
            float(
                (
                    observed["delta_norm"] / observed["residual_norm"].clamp_min(1e-12)
                ).max()
            )
            for _shard, observed in sanity_outputs + layer_outputs
        )
        report = {
            "layer": layer,
            "window": window,
            "joint_quantile": quantile,
            "rho_max": rho_max,
            "calibration_case_count": len(calibration_shards),
            "calibration_token_count": len(all_margins),
            "clean_sanity_case_count": len(sanity_shards),
            "clean_sanity": sanity_summary,
            "margin_center": dict(zip(WRONG_ROLES, center.tolist(), strict=True)),
            "margin_scale": dict(zip(WRONG_ROLES, scale.tolist(), strict=True)),
            "joint_threshold": beta,
            "regions": summary,
            "observed_relative_norm_max": cap_max,
            "checks": {
                "injection_score_above_page": summary["injection"]["score"]["mean"]
                > summary["page_other"]["score"]["mean"],
                "injection_firing_above_page": summary["injection"]["firing_rate"]
                > summary["page_other"]["firing_rate"],
                "clean_sanity_firing_le_3pct": sanity_summary["firing_rate"] <= 0.03,
                "relative_norm_cap": cap_max <= rho_max + 1e-6,
                "tool_probability_rises_on_injection": summary["injection"][
                    "mean_tool_probability_change"
                ]
                > 0,
                "dominant_wrong_margin_falls_on_injection": summary["injection"][
                    "mean_dominant_wrong_margin_change"
                ]
                < 0,
            },
        }
        report["diagnostic_pass"] = all(report["checks"].values())
        reports.append(report)

    passing = [report for report in reports if report["diagnostic_pass"]]
    if not passing:
        selected_layer = None
    else:
        selected_layer = max(
            passing,
            key=lambda item: (
                item["regions"]["injection"]["firing_rate"]
                - item["regions"]["page_other"]["firing_rate"]
            ),
        )["layer"]
    state = {
        "schema_version": 1,
        "roles": representation["roles"],
        "wrong_roles": list(WRONG_ROLES),
        "hook": representation["hook"],
        "window": window,
        "joint_quantile": quantile,
        "route_temperature": 1.0,
        "gamma": 1.0,
        "direction_floor": 0.1,
        "rho_max_diagnostic": rho_max,
        "selected_layer": selected_layer,
        "layers": calibration_layers,
        "representation_state_sha256": _sha256_file(representation_path),
    }
    state_path = output / "calibration_state.pt"
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, state_path)

    # Required per-token diagnostic log for the selected layer.
    if selected_layer is not None:
        layer = selected_layer
        weight = representation["probe_weight"][layer].float()
        bias = representation["probe_bias"][layer].float()
        unit = representation["unit_pair_direction"][layer][
            list(WRONG_INDICES), ROLE_TO_INDEX["tool"]
        ].float()
        layer_state = calibration_layers[layer]
        shards = _load_case_shards(clean_root, layer) + _load_case_shards(
            layer_root, layer
        )
        for shard in shards:
            observed = compute_v2_intervention(
                shard["activations"],
                weight,
                bias,
                layer_state["margin_center"],
                layer_state["margin_scale"],
                layer_state["joint_threshold"],
                unit,
                window=window,
                rho_max=rho_max,
            )
            diagnostic = {
                "schema_version": 1,
                "case_id": shard["case_id"],
                "variant": shard["variant"],
                "layer": layer,
                "token_ids": shard["token_ids"],
                "token_texts": shard["token_texts"],
                "region_codes": shard["region_codes"],
                "region_code_map": {"page_or_clean": 0, "context": 1, "injection": 2},
                "beta": layer_state["joint_threshold"],
                **{key: value for key, value in observed.items() if key != "delta"},
            }
            torch.save(
                diagnostic,
                token_debug_root
                / f"{shard['dataset']}-case-{int(shard['case_id']):03d}-{shard['variant']}.pt",
            )
    report_payload = {
        "schema_version": 1,
        "selection_rule": (
            "among diagnostic-passing dev-accuracy candidates, maximize "
            "injection minus page-other firing rate"
        ),
        "selected_layer": selected_layer,
        "agent_evaluation_allowed": selected_layer is not None,
        "calibration_state_sha256": _sha256_file(state_path),
        "layers": reports,
    }
    _atomic_json(output / "diagnostic_report.json", report_payload)
    return output
