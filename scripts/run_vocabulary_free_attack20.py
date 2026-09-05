#!/usr/bin/env python3
"""Run the vocabulary-free live selector with strong CoT correction on dev-20."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from eval_harness.config import Config
from eval_harness.defense import OpenAICompatibleJudge
from eval_harness.defense_replay import _summary, run_defended_case
from eval_harness.defense_smoke import DEFAULT_RESULTS, _load_jsonl
from eval_harness.live_detector import LiveRoleProbeDetector
from eval_harness.steering_repr import _load_local_model
from scripts.run_frozen_heldout50 import (
    _annotate_judge_ground_truth,
    _atomic_json,
    _injection_templates_by_hash,
)

CASE_IDS = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29)
STAGE = "vocabulary_free_strong_cot_marked"


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _progress(root: Path, done: int, case_id: int | None) -> None:
    width = 30
    filled = width * done // len(CASE_IDS)
    bar = "#" * filled + "." * (width - filled)
    suffix = "complete" if case_id is None else f"case={case_id}"
    print(f"[{bar}] {done:2d}/{len(CASE_IDS)} {suffix}", flush=True)
    _atomic_json(
        root / "status.json",
        {
            "status": "complete" if done == len(CASE_IDS) else "running",
            "stage": STAGE,
            "completed": done,
            "total": len(CASE_IDS),
            "current_case_id": case_id,
        },
    )


def run(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[1]
    root = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = Config.load(config_path)
    cases = {
        int(row["case_id"]): row
        for row in _load_jsonl(repo / "eval_data/steering_attack_layer_manifest.jsonl")
    }
    baselines = {
        int(row["case_id"]): row for row in _load_jsonl(repo / DEFAULT_RESULTS)
    }
    templates = _injection_templates_by_hash(repo)
    results_path = root / "results.jsonl"
    rows = _load_jsonl(results_path) if results_path.exists() else []
    completed = {int(row["case_id"]) for row in rows}
    if len(completed) != len(rows) or not completed <= set(CASE_IDS):
        raise ValueError("invalid or duplicate case IDs in resumable results")

    probe_state = repo / "research_outputs/phase3_steering_v2/representation_analysis/steering_state.pt"
    metadata = {
        "schema_version": 1,
        "stage": STAGE,
        "case_ids": list(CASE_IDS),
        "config": asdict(config),
        "config_fingerprint": config.fingerprint(),
        "selector_mode": "vocabulary_free",
        "selector": (
            "CoT8 contrast >0.50 for >=8 tokens, reserve top 1; User18 contrast "
            ">0.13 for >=8, vocabulary-free prose structure, rank by word_count * "
            "(1-markup_fraction), reserve top 2"
        ),
        "judge_url": args.judge_url,
        "judge_model": args.judge_model,
        "confirmed_text_mode": "marked",
        "correction": "strong state-preserving open analysis CoT",
    }
    metadata_path = root / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("refusing to resume because metadata changed")
    if not metadata_path.exists():
        _atomic_json(metadata_path, metadata)

    tokenizer, model = _load_local_model(config)
    detector = LiveRoleProbeDetector(
        model,
        tokenizer,
        probe_state,
        diagnostic_root=root / "probe_diagnostics",
        selector_mode="vocabulary_free",
    )
    judge = OpenAICompatibleJudge(args.judge_url, args.judge_model)
    _progress(root, len(completed), None if len(completed) == len(CASE_IDS) else -1)
    for case_id in CASE_IDS:
        if case_id in completed:
            continue
        case = cases[case_id]
        baseline = baselines[case_id]
        if case["html_sha256"] != baseline["html_sha256"]:
            raise ValueError(f"case {case_id}: baseline HTML mismatch")
        injection = templates[str(case["injection_sha256"])]
        detector.set_case_context(STAGE, case_id, injection)
        result = run_defended_case(
            case=case,
            reference_html=cases[int(case["clean_case_id"])]["html"],
            baseline=baseline,
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
        _progress(root, len(completed), case_id)
    _atomic_json(root / "summary.json", _summary(rows))
    _progress(root, len(CASE_IDS), None)
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gpt-oss-20b.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research_outputs/prompt_injection_defense/vocabulary_free_strong_cot_attack20"),
    )
    parser.add_argument("--judge-url", default="http://127.0.0.1:8017")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.8-27B-FP8")
    args = parser.parse_args()
    try:
        output = run(args)
    except Exception as error:
        root = args.output_root
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            root / "status.json",
            {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
