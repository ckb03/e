#!/usr/bin/env python3
"""Run cached Qwen3.8-27B-FP8 directly on saved defense-judge prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)
from transformers.integrations.finegrained_fp8 import FP8Experts

from eval_harness.defense import JUDGE_SYSTEM_PROMPT, parse_verdict

DEFAULT_INPUT = Path(
    "research_outputs/prompt_injection_defense/"
    "full_replay_state_preserving_attack20/results.jsonl"
)
DEFAULT_OUTPUT = Path(
    "research_outputs/prompt_injection_defense/"
    "qwen_direct_judge_cases_26_29/results.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def saved_attempts(row: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for intervention in row["interventions"]:
        for attempt_index, attempt in enumerate(intervention["decision"]["attempts"]):
            attempts.append(
                {
                    "tool_event_index": intervention["tool_event_index"],
                    "attempt_index": attempt_index,
                    "candidate": attempt["candidate"]["candidate"],
                    "candidate_text": attempt["candidate"]["candidate_text"],
                    "prompt": attempt["prompt"],
                    "previous_judge_verdict": attempt["verdict"],
                }
            )
    return attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-ids", default="26,29")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    wanted = {int(value) for value in args.case_ids.split(",")}
    rows = [row for row in load_jsonl(args.input) if row["case_id"] in wanted]
    if {row["case_id"] for row in rows} != wanted:
        raise ValueError("not all requested case IDs were found")

    # Transformers 5.16.1 assumes every FP8 model has an MoE implementation.
    # Qwen3.8-27B is dense, so its implementation key is None. Treat that as
    # having no tensor-parallel MoE rewrites rather than calling .get on None.
    FP8Experts._impl_tp_layer_overrides.setdefault(None, {})

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    skip_modules = config.quantization_config["modules_to_not_convert"]
    # The checkpoint skip list contains nonexistent dense-MLP modules named
    # `mlp.gate`. Transformers uses an unanchored regex for this list, which
    # accidentally matches `mlp.gate_proj` and discards its FP8 scale tensor.
    # Remove only the exact nonexistent gate entries, retaining all real skips.
    removed_gate_skips = [name for name in skip_modules if name.endswith(".mlp.gate")]
    config.quantization_config["modules_to_not_convert"] = [
        name for name in skip_modules if not name.endswith(".mlp.gate")
    ]
    config.quantization_config["dequantize"] = True
    print(f"removed {len(removed_gate_skips)} colliding mlp.gate skip entries", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        local_files_only=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    load_seconds = time.perf_counter() - load_started

    outputs: list[dict[str, Any]] = []
    for row in rows:
        for attempt in saved_attempts(row):
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": attempt["prompt"]},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            raw = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            try:
                verdict = parse_verdict(raw)
            except ValueError:
                verdict = "INVALID"
            record = {
                "case_id": row["case_id"],
                "variant": row["variant"],
                **attempt,
                "judge_model": args.model,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "load_seconds": round(load_seconds, 3),
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0,
                "use_cache": True,
                "dequantized_to_bf16": True,
                "raw_verdict": raw,
                "verdict": verdict,
                "generation_seconds": round(generation_seconds, 3),
            }
            outputs.append(record)
            print(
                f"case {row['case_id']} attempt {attempt['attempt_index']}: "
                f"old={attempt['previous_judge_verdict']} qwen={verdict} "
                f"raw={raw!r}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
        encoding="utf-8",
    )
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
