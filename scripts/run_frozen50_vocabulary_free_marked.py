#!/usr/bin/env python3
"""Run loose vocabulary-free live detection plus marked strong CoT on frozen-50."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from eval_harness.config import Config
from eval_harness.defense import OpenAICompatibleJudge
from eval_harness.defense_replay import _summary, run_defended_case
from eval_harness.live_detector import LiveRoleProbeDetector
from eval_harness.steering_repr import _load_local_model
from scripts.run_frozen_heldout50 import (
    _annotate_judge_ground_truth,
    _atomic_json,
    _injection_templates_by_hash,
    _load_completed,
    _read_jsonl,
    _sha256,
    _wait_for_judge,
    prepare_selection,
)

STAGE = "vocabulary_free_k2_cot_k4_user_strong_cot_marked"
COT_CAP = 2
USER_CAP = 4


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _progress(root: Path, done: int, total: int, case_id: int | None) -> None:
    width = 30
    filled = width * done // total
    bar = "#" * filled + "." * (width - filled)
    suffix = "complete" if case_id is None else f"case={case_id}"
    print(f"[{bar}] {done:2d}/{total} {suffix}", flush=True)
    _atomic_json(
        root / "status.json",
        {
            "status": "complete" if done == total else "running",
            "stage": STAGE,
            "completed": done,
            "total": total,
            "current_case_id": case_id,
        },
    )


def run(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[1]
    output_root = (
        args.output_root if args.output_root.is_absolute() else repo / args.output_root
    )
    source_root = (
        args.source_root if args.source_root.is_absolute() else repo / args.source_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    selected, clean_references = prepare_selection(repo, source_root)
    selected_ids = [int(row["case_id"]) for row in selected]
    selected_by_id = {int(row["case_id"]): row for row in selected}
    baseline_path = source_root / "baseline/results.jsonl"
    baselines = {int(row["case_id"]): row for row in _read_jsonl(baseline_path)}
    if set(baselines) != set(selected_ids):
        raise ValueError("the matched frozen-50 baseline is incomplete")

    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = Config.load(config_path)
    results_path = output_root / "results.jsonl"
    rows = _load_completed(results_path, selected_ids)
    completed = {int(row["case_id"]) for row in rows}
    probe_state = (
        repo
        / "research_outputs/phase3_steering_v2/representation_analysis/steering_state.pt"
    )
    metadata = {
        "schema_version": 1,
        "stage": STAGE,
        "case_ids": selected_ids,
        "config": asdict(config),
        "config_fingerprint": config.fingerprint(),
        "selection_manifest": str(source_root / "selection_manifest.jsonl"),
        "selection_manifest_sha256": _sha256(source_root / "selection_manifest.jsonl"),
        "matched_baseline": str(baseline_path),
        "matched_baseline_sha256": _sha256(baseline_path),
        "selector_mode": "vocabulary_free",
        "channel_caps": {"cot8": COT_CAP, "user18": USER_CAP},
        "selector_audit": (
            "all sustained seeds, all merged regions with structural features, "
            "cumulative filter-stage counts, all passing candidates, selected spans, "
            "five-role logits, local contrasts, and offline recall labels"
        ),
        "judge_url": args.judge_url,
        "judge_model": args.judge_model,
        "confirmed_text_mode": "marked",
        "correction": "strong state-preserving open analysis CoT",
    }
    metadata_path = output_root / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("refusing to resume because metadata changed")
    if not metadata_path.exists():
        _atomic_json(metadata_path, metadata)
    if len(completed) == len(selected_ids):
        _progress(output_root, len(selected_ids), len(selected_ids), None)
        return output_root

    _atomic_json(
        output_root / "status.json",
        {
            "status": "loading_target",
            "stage": STAGE,
            "completed": len(completed),
            "total": len(selected_ids),
            "current_case_id": None,
        },
    )
    _wait_for_judge(args.judge_url)
    tokenizer, model = _load_local_model(config)
    detector = LiveRoleProbeDetector(
        model,
        tokenizer,
        probe_state,
        diagnostic_root=output_root / "probe_diagnostics",
        selector_mode="vocabulary_free",
        vocabulary_free_cot_cap=COT_CAP,
        vocabulary_free_user_cap=USER_CAP,
    )
    judge = OpenAICompatibleJudge(args.judge_url, args.judge_model)
    templates = _injection_templates_by_hash(repo)
    _progress(output_root, len(completed), len(selected_ids), -1)
    for case_id in selected_ids:
        if case_id in completed:
            continue
        case = selected_by_id[case_id]
        injection = templates[str(case["injection_sha256"])]
        detector.set_case_context(STAGE, case_id, injection)
        result = run_defended_case(
            case=case,
            reference_html=clean_references[str(case["source_url"])],
            baseline=baselines[case_id],
            model=model,
            tokenizer=tokenizer,
            config=config,
            judge=judge,
            repo=repo,
            candidates={},
            confirmed_text_mode="marked",
            candidate_detector=detector,
        )
        _annotate_judge_ground_truth(result, injection)
        if not result["pre_intervention_turn_matches_baseline"]:
            raise ValueError(f"case {case_id}: pre-intervention turn changed")
        if result["defense_processed_tool_results"] != len(result["tool_events"]):
            raise ValueError(f"case {case_id}: unprocessed Tool result")
        rows.append(result)
        _append(results_path, result)
        completed.add(case_id)
        _progress(output_root, len(completed), len(selected_ids), case_id)
    _atomic_json(output_root / "summary.json", _summary(rows))
    _progress(output_root, len(selected_ids), len(selected_ids), None)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gpt-oss-20b.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "research_outputs/prompt_injection_defense/frozen_heldout50_vocabulary_free_k2_k4_marked"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("research_outputs/prompt_injection_defense/frozen_heldout50"),
    )
    parser.add_argument("--judge-url", default="http://127.0.0.1:8017")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.8-27B-FP8")
    args = parser.parse_args()
    try:
        output = run(args)
    except Exception as error:
        root = (
            args.output_root
            if args.output_root.is_absolute()
            else Path(__file__).resolve().parents[1] / args.output_root
        )
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            root / "status.json",
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
