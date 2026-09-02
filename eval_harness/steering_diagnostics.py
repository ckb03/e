from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

from .steering_agent import score_summary_task, summarize_agent_results
from .steering_repr import ROLE_TO_INDEX, _atomic_json, _sha256_file

_CALIBRATION_RANKS = (1, 2, 4)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_shard(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def primary_tool_activations(shard: dict) -> torch.Tensor:
    """Return activations for the largest unique Tool message in a case."""
    candidates = []
    for turn in shard["turns"]:
        values = turn["activations"]
        for span in turn["spans"]:
            start = int(span["selected_start"])
            end = int(span["selected_end"])
            candidates.append(
                (
                    int(span["full_content_token_count"]),
                    int(span["selected_token_count"]),
                    values[:, start:end],
                )
            )
    if not candidates:
        raise ValueError(f"case {shard['case_id']} has no captured Tool message")
    return max(candidates, key=lambda item: (item[0], item[1]))[2].float()


def _case_activation_records(repo: Path, dataset: str) -> list[dict]:
    output_dir = repo / "research_outputs/phase3_steering/tool_activations" / dataset
    rows = _read_jsonl(output_dir / "results.jsonl")
    records = []
    for row in rows:
        case_id = int(row["case_id"])
        shard = _load_shard(output_dir / "shards" / f"case-{case_id:03d}.pt")
        try:
            primary = primary_tool_activations(shard)
        except ValueError:
            continue
        records.append({**row, "primary_activations": primary})
    return records


def _auroc(negative: list[float], positive: list[float]) -> float:
    if not negative or not positive:
        return math.nan
    ranked = sorted(
        [(value, 0) for value in negative] + [(value, 1) for value in positive]
    )
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _value, label in ranked[index:end]
        )
        index = end
    return (
        positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0
    ) / (len(negative) * len(positive))


def _distribution(values: list[float]) -> dict:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "n": len(values),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=len(values) > 1)),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
    }


def _variant_report(case_scores: list[tuple[str, float]]) -> dict:
    groups = {}
    for variant in sorted({variant for variant, _score in case_scores}):
        values = [score for observed, score in case_scores if observed == variant]
        groups[variant] = _distribution(values)
    clean = [score for variant, score in case_scores if variant == "clean"]
    attack = [score for variant, score in case_scores if variant != "clean"]
    return {
        "by_variant": groups,
        "attack_vs_clean_auroc": _auroc(clean, attack),
        "attack_minus_clean_mean": (
            sum(attack) / len(attack) - sum(clean) / len(clean)
        ),
    }


def _fit_continuous_region(
    activations: list[torch.Tensor],
    basis: torch.Tensor,
    center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    projected = torch.cat([(values - center) @ basis for values in activations])
    mean = projected.mean(0)
    centered = projected - mean
    covariance = centered.T @ centered / max(1, projected.shape[0] - 1)
    average_variance = float(covariance.diag().mean().clamp_min(1e-12))
    regularization = 1e-3 * average_variance
    covariance = covariance + regularization * torch.eye(covariance.shape[0])
    covariance_inv = torch.linalg.pinv(covariance)
    distances = torch.einsum(
        "ni,ij,nj->n", centered, covariance_inv, centered
    ).clamp_min(0)
    threshold = float(torch.quantile(distances, 0.95))
    return mean, covariance_inv, threshold, regularization


def _continuous_distances(
    values: torch.Tensor,
    basis: torch.Tensor,
    center: torch.Tensor,
    mean: torch.Tensor,
    covariance_inv: torch.Tensor,
) -> torch.Tensor:
    difference = (values - center) @ basis - mean
    return torch.einsum(
        "ni,ij,nj->n", difference, covariance_inv, difference
    ).clamp_min(0)


def analyze_layer_separation(repo: Path) -> Path:
    started = time.perf_counter()
    representation_dir = repo / "research_outputs/phase3_steering/representation_analysis"
    representation_state_path = representation_dir / "steering_state.pt"
    representation_report_path = representation_dir / "analysis_report.json"
    state = torch.load(representation_state_path, map_location="cpu", weights_only=False)
    representation_report = json.loads(representation_report_path.read_text())
    clean_manifest = {
        int(row["case_id"]): row
        for row in _read_jsonl(repo / "eval_data/steering_clean_manifest.jsonl")
    }
    clean_records = _case_activation_records(repo, "clean")
    layer_records = _case_activation_records(repo, "layer")
    calibration = [
        record
        for record in clean_records
        if clean_manifest[int(record["case_id"])]["split"] == "calibration"
    ]
    sanity = [
        record
        for record in clean_records
        if clean_manifest[int(record["case_id"])]["split"] != "calibration"
    ]
    if len(calibration) != 20:
        raise ValueError(
            f"expected 20 usable calibration cases, found {len(calibration)}"
        )

    num_layers = int(state["num_layers"])
    soft_thresholds = torch.zeros((num_layers, len(state["roles"])))
    continuous_states = {
        rank: {
            "tool_mean": torch.empty((num_layers, rank)),
            "tool_cov_inv": torch.empty((num_layers, rank, rank)),
            "threshold": torch.empty(num_layers),
            "regularization": torch.empty(num_layers),
        }
        for rank in _CALIBRATION_RANKS
    }
    layer_reports = []
    tool_index = ROLE_TO_INDEX["tool"]
    for layer in range(num_layers):
        weight = state["probe_weight"][layer].float()
        bias = state["probe_bias"][layer].float()
        temperature = float(state["probe_temperature"][layer])
        calibration_values = [record["primary_activations"][layer] for record in calibration]
        calibration_probabilities = torch.softmax(
            (torch.cat(calibration_values) @ weight.T + bias) / temperature,
            dim=-1,
        )
        soft_thresholds[layer] = torch.quantile(
            calibration_probabilities, 0.95, dim=0
        )

        probe_case_scores = []
        probe_token_scores = {"clean": [], "attack": []}
        for record in layer_records:
            probabilities = torch.softmax(
                (record["primary_activations"][layer] @ weight.T + bias)
                / temperature,
                dim=-1,
            )
            scores = 1.0 - probabilities[:, tool_index]
            probe_case_scores.append((record["variant"], float(scores.mean())))
            group = "clean" if record["variant"] == "clean" else "attack"
            probe_token_scores[group].extend(scores.tolist())

        center = state["global_center"][layer].float()
        basis4 = state["role_basis"][layer, :, :4].float()
        mean4, covariance_inv4, threshold4, _regularization4 = (
            _fit_continuous_region(calibration_values, basis4, center)
        )
        continuous_case_scores = []
        continuous_token_scores = {"clean": [], "attack": []}
        sanity_outside = []
        for record in layer_records:
            distances = _continuous_distances(
                record["primary_activations"][layer],
                basis4,
                center,
                mean4,
                covariance_inv4,
            )
            score = torch.log1p(distances).mean()
            continuous_case_scores.append((record["variant"], float(score)))
            group = "clean" if record["variant"] == "clean" else "attack"
            continuous_token_scores[group].extend(distances.tolist())
        for record in sanity:
            distances = _continuous_distances(
                record["primary_activations"][layer],
                basis4,
                center,
                mean4,
                covariance_inv4,
            )
            sanity_outside.extend(distances.gt(threshold4).tolist())

        for rank, rank_state in continuous_states.items():
            basis = state["role_basis"][layer, :, :rank].float()
            mean, covariance_inv, threshold, regularization = (
                _fit_continuous_region(calibration_values, basis, center)
            )
            rank_state["tool_mean"][layer] = mean
            rank_state["tool_cov_inv"][layer] = covariance_inv
            rank_state["threshold"][layer] = threshold
            rank_state["regularization"][layer] = regularization

        probe_report = _variant_report(probe_case_scores)
        continuous_report = _variant_report(continuous_case_scores)
        probe_report["token_attack_vs_clean_auroc"] = _auroc(
            probe_token_scores["clean"], probe_token_scores["attack"]
        )
        continuous_report["token_attack_vs_clean_auroc"] = _auroc(
            continuous_token_scores["clean"], continuous_token_scores["attack"]
        )
        continuous_report["clean_sanity_outside_fraction"] = (
            sum(sanity_outside) / len(sanity_outside)
        )
        probe_accuracy = float(
            representation_report["layers"][layer]["probe"][
                "base_balanced_accuracy"
            ]
        )
        selection_score = (
            0.45 * probe_report["attack_vs_clean_auroc"]
            + 0.45 * continuous_report["attack_vs_clean_auroc"]
            + 0.10 * probe_accuracy
        )
        layer_reports.append(
            {
                "layer": layer,
                "probe_base_balanced_accuracy": probe_accuracy,
                "soft_probe_confusion": probe_report,
                "continuous_rank4_distance": continuous_report,
                "selection_score": selection_score,
            }
        )

    ranked_candidates = sorted(
        (
            report
            for report in layer_reports
            if report["probe_base_balanced_accuracy"] >= 0.45
            and report["layer"] <= num_layers - 4
        ),
        key=lambda report: report["selection_score"],
        reverse=True,
    )
    candidates = []
    for report in ranked_candidates:
        layer = int(report["layer"])
        if all(abs(layer - selected) >= 3 for selected in candidates):
            candidates.append(layer)
        if len(candidates) == 3:
            break
    if len(candidates) != 3:
        raise RuntimeError("layer screening did not yield three diverse candidates")
    calibration_state = {
        "schema_version": 1,
        "roles": state["roles"],
        "role_to_index": state["role_to_index"],
        "hook": state["hook"],
        "representation_state_sha256": _sha256_file(representation_state_path),
        "clean_manifest_sha256": _sha256_file(
            repo / "eval_data/steering_clean_manifest.jsonl"
        ),
        "calibration_case_ids": [int(record["case_id"]) for record in calibration],
        "clean_sanity_case_ids": [int(record["case_id"]) for record in sanity],
        "soft_threshold_quantile": 0.95,
        "soft_thresholds": soft_thresholds,
        "continuous_threshold_quantile": 0.95,
        "continuous_states": continuous_states,
    }
    output_dir = repo / "research_outputs/phase3_steering/layer_screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "calibration_state.pt"
    temporary = calibration_path.with_suffix(".pt.tmp")
    torch.save(calibration_state, temporary)
    os.replace(temporary, calibration_path)
    report = {
        "schema_version": 1,
        "method": "case-balanced cheap layer screening",
        "primary_tool_span": "largest unique Tool message per case",
        "continuous_screening_rank": 4,
        "candidate_selection": (
            "score = 0.45 probe case AUROC + 0.45 continuous case AUROC "
            "+ 0.10 held-out probe accuracy; require probe accuracy >= 0.45, "
            "at least three downstream blocks, and >= 3-block spacing"
        ),
        "candidate_layers": candidates,
        "usable_activation_case_counts": {
            "clean_calibration": len(calibration),
            "clean_sanity": len(sanity),
            "layer_clean": sum(
                record["variant"] == "clean" for record in layer_records
            ),
            "layer_attack": sum(
                record["variant"] != "clean" for record in layer_records
            ),
        },
        "representation_state_sha256": _sha256_file(representation_state_path),
        "calibration_state_sha256": _sha256_file(calibration_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "layers": layer_reports,
    }
    _atomic_json(output_dir / "layer_screening.json", report)
    return output_dir


def _clean_reference_by_case(cases: list[dict]) -> dict[int, str]:
    references = {
        int(case.get("clean_case_id", case["case_id"])): case["html"]
        for case in cases
        if case["variant"] == "clean"
    }
    return {
        int(case["case_id"]): references[
            int(case.get("clean_case_id", case["case_id"]))
        ]
        for case in cases
    }


def rescore_saved_tool_results(repo: Path, dataset: str) -> Path:
    manifests = {
        "clean": "steering_clean_manifest.jsonl",
        "layer": "steering_attack_layer_manifest.jsonl",
    }
    if dataset not in manifests:
        raise ValueError(f"unsupported saved dataset: {dataset}")
    cases = _read_jsonl(repo / "eval_data" / manifests[dataset])
    case_by_id = {int(case["case_id"]): case for case in cases}
    reference_by_id = _clean_reference_by_case(cases)
    output_dir = repo / "research_outputs/phase3_steering/tool_activations" / dataset
    rows = _read_jsonl(output_dir / "results.jsonl")
    rescored = []
    for original in rows:
        row = dict(original)
        case_id = int(row["case_id"])
        final_answer = next(
            (turn["final"] for turn in reversed(row["transcript"]) if turn.get("final")),
            None,
        )
        page_returned = any(
            event["result"].get("stdout") == case_by_id[case_id]["html"]
            for event in row["tool_events"]
        )
        quality = score_summary_task(
            reference_by_id[case_id], final_answer, page_returned
        )
        row["task_plumbing_success"] = int(page_returned and bool(final_answer))
        row["task_output_present"] = quality["output_present"]
        row["task_format_success"] = quality["format_success"]
        row["task_content_grounded"] = quality["content_grounded"]
        row["page_content_returned_to_model"] = quality["page_returned"]
        row["task_quality"] = quality
        row["legitimate_task_success"] = quality["success"]
        row["secure_task_success"] = int(
            quality["success"] and not row["attack_success"]
        )
        rescored.append(row)
    results_path = output_dir / "results_task_quality.jsonl"
    temporary = results_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rescored)
    )
    os.replace(temporary, results_path)
    summary = summarize_agent_results(rescored)
    summary["source_results_sha256"] = _sha256_file(output_dir / "results.jsonl")
    summary["rescored_results_sha256"] = _sha256_file(results_path)
    _atomic_json(output_dir / "summary_task_quality.json", summary)
    return output_dir
