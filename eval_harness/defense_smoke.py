from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from .core import message
from .defense import (
    JUDGE_SYSTEM_PROMPT,
    CandidateSpan,
    Judge,
    OpenAICompatibleJudge,
    post_tool_defense,
)
from .runner import USER_PROMPT

DEFAULT_RESULTS = Path(
    "research_outputs/phase3_steering/tool_activations/layer/results_task_quality.jsonl"
)
DEFAULT_CANDIDATES = Path(
    "research_outputs/prompt_injection_detector_study/plateau_candidates.csv"
)
DEFAULT_DIAGNOSTICS = Path(
    "research_outputs/phase3_steering_v2/calibration/token_diagnostics"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _candidate_map(
    path: Path, cap: int
) -> dict[tuple[str, int, str], list[CandidateSpan]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        if row["passes_broad_filter"] != "True":
            continue
        key = (row["dataset"], int(row["case_id"]), row["variant"])
        output.setdefault(key, []).append(
            CandidateSpan(
                start_token=int(row["start"]),
                stop_token=int(row["stop"]),
                score=float(row["rank_score"]),
                channel=row["channels"],
            )
        )
    for key in output:
        output[key] = sorted(output[key], key=lambda item: item.score, reverse=True)[
            :cap
        ]
    return output


def _tool_event(row: dict[str, Any], serialized: str) -> dict[str, Any]:
    for event in row["tool_events"]:
        if json.dumps(event["result"], ensure_ascii=False) == serialized:
            return event
    raise ValueError(f"case {row['case_id']}: no Tool event matches probe tokenization")


def run_smoke(
    repo: Path,
    case_ids: list[int],
    judge: Judge,
    output_dir: Path,
    cap: int = 3,
) -> Path:
    rows = {row["case_id"]: row for row in _load_jsonl(repo / DEFAULT_RESULTS)}
    candidates = _candidate_map(repo / DEFAULT_CANDIDATES, cap)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for case_id in case_ids:
        row = rows[case_id]
        variant = row["variant"]
        diagnostic_path = (
            repo / DEFAULT_DIAGNOSTICS / f"layer-case-{case_id:03d}-{variant}.pt"
        )
        diagnostic = torch.load(diagnostic_path, map_location="cpu", weights_only=False)
        token_texts = list(diagnostic["token_texts"])
        serialized = "".join(token_texts)
        event = _tool_event(row, serialized)
        selected = candidates.get(("layer", case_id, variant), [])
        user_request = USER_PROMPT.format(
            workspace="/workspace/eval-sandbox", url=row["url"]
        )
        tool_call = f"functions.{event['tool']}(command={event['command']!r})"
        decision = post_tool_defense(
            user_request=user_request,
            tool_call=tool_call,
            tool_purpose=(
                "The page was fetched because the user asked for an end-to-end "
                "webpage summary."
            ),
            serialized_tool_result=serialized,
            token_texts=token_texts,
            candidates=selected,
            judge=judge,
        )
        injection = diagnostic["region_codes"].eq(2)
        true_injection_retrieved = any(
            bool(injection[item.start_token : item.stop_token].any())
            for item in selected
        )
        attempts = []
        for attempt in decision.attempts:
            attempts.append(
                {
                    "candidate_span": {
                        "start_token": attempt.candidate.candidate.start_token,
                        "stop_token": attempt.candidate.candidate.stop_token,
                        "score": attempt.candidate.candidate.score,
                        "channel": attempt.candidate.candidate.channel,
                    },
                    "expanded_char_span": [
                        attempt.candidate.char_start,
                        attempt.candidate.char_end,
                    ],
                    "expanded_candidate_text": attempt.candidate.candidate_text,
                    "previous_text": attempt.candidate.previous_text,
                    "next_text": attempt.candidate.next_text,
                    "judge_prompt": attempt.prompt,
                    "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
                    "judge_messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": attempt.prompt,
                        },
                    ],
                    "raw_verdict": attempt.raw_verdict,
                    "raw_verdicts": list(attempt.raw_verdicts),
                    "verdict": attempt.verdict,
                }
            )
        output = {
            "case_id": case_id,
            "variant": variant,
            "source_url": row["source_url"],
            "candidate_spans": [
                {
                    "start_token": item.start_token,
                    "stop_token": item.stop_token,
                    "score": item.score,
                    "channel": item.channel,
                }
                for item in selected
            ],
            "true_injection_retrieved": true_injection_retrieved,
            "judge_attempts": attempts,
            "confirmed_injection": decision.confirmed_injection is not None,
            "corrective_cot": decision.corrective_cot,
            "forced_analysis_message": (
                message("assistant", decision.corrective_cot, "analysis")
                if decision.corrective_cot
                else None
            ),
            "skipped_reason": decision.skipped_reason,
            "baseline_attack_attempt": bool(row["attack_attempt"]),
            "baseline_attack_success": bool(row["attack_success"]),
            "baseline_original_task_completed": bool(row["legitimate_task_success"]),
        }
        outputs.append(output)
        print(
            f"case {case_id}: candidates={len(selected)} "
            f"verdicts={[item['verdict'] for item in attempts]} "
            f"correction={decision.corrective_cot is not None}",
            flush=True,
        )

    result_path = output_dir / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "case_ids": case_ids,
        "judge_model": getattr(judge, "model", "unknown"),
        "judge_url": getattr(judge, "url", "unknown"),
        "temperature": 0,
        "candidate_cap": cap,
        "cases": len(outputs),
        "confirmed_injections": sum(row["confirmed_injection"] for row in outputs),
        "judge_attempts": sum(len(row["judge_attempts"]) for row in outputs),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the V1 post-Tool defense smoke test"
    )
    parser.add_argument("--case-ids", default="13,12")
    parser.add_argument(
        "--judge-backend",
        choices=("local", "openai"),
        default="local",
    )
    parser.add_argument("--judge-url", default="http://127.0.0.1:8000")
    parser.add_argument("--judge-model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir
    if not output.is_absolute():
        output = repo / output
    if args.judge_backend == "local":
        from .local_judge import GptOssTransformersJudge

        model_name = args.judge_model or "openai/gpt-oss-20b"
        judge: Judge = GptOssTransformersJudge(model=model_name)
    else:
        model_name = args.judge_model or "Qwen/Qwen3.8-27B-FP8"
        judge = OpenAICompatibleJudge(args.judge_url, model_name)
    completed = run_smoke(
        repo,
        [int(value) for value in args.case_ids.split(",")],
        judge,
        output,
        cap=args.cap,
    )
    print(f"Artifacts: {completed}")


if __name__ == "__main__":
    main()
