from __future__ import annotations

import torch

from eval_harness.core import tool_result
from eval_harness.steering_agent import (
    ToolResidualCapture,
    summarize_agent_results,
    tool_span_plan,
)


class CharacterOffsetTokenizer:
    def __call__(self, text: str, **_: object) -> dict:
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_tool_span_plan_samples_only_new_message_content_and_tail() -> None:
    first = tool_result("functions.bash", {"stdout": "first", "exit_code": 0})
    second = tool_result(
        "functions.bash",
        {"stdout": "abcdefghijklmnopqrstuvwxyz", "exit_code": 0},
    )
    prompt = first + second + "<|start|>assistant"

    plan = tool_span_plan(
        prompt,
        CharacterOffsetTokenizer(),
        seen_tool_messages=1,
        max_tokens_per_message=10,
        tail_tokens=3,
    )

    assert plan["total_tool_messages"] == 2
    assert len(plan["spans"]) == 1
    assert len(plan["positions"]) <= 10
    span = plan["spans"][0]
    assert span["tool_message_index"] == 1
    assert span["selected_token_count"] == len(plan["positions"])
    assert span["tail_selected_count"] == 3
    assert all(prompt[position] not in "<>|" for position in plan["positions"][-3:])


def test_tool_capture_ignores_decode_calls_after_prefill() -> None:
    layer = torch.nn.Identity()
    capture = ToolResidualCapture([layer])
    capture.prepare([1, 3])
    prefill = torch.arange(15).reshape(1, 5, 3).float()
    try:
        layer(prefill)
        expected = capture.stacked().clone()
        layer(torch.full((1, 1, 3), 999.0))
        observed = capture.stacked()
    finally:
        capture.close()

    assert torch.equal(observed, expected)
    assert torch.equal(observed[0].float(), prefill[0, [1, 3]])


def _row(variant: str, attack: int, task: int) -> dict:
    return {
        "variant": variant,
        "attack_success": attack,
        "attack_attempt": attack,
        "legitimate_task_success": task,
        "secure_task_success": int(task and not attack),
        "over_refusal": 0,
        "elapsed_seconds": 1.0,
        "captured_tool_tokens": 10,
        "output_class": "ATTEMPTED_SUCCESSFUL" if attack else "UNNOTICED",
    }


def test_agent_summary_reports_stsr_uncertainty_and_role_spoof_gap() -> None:
    rows = [
        _row("base-injection", 0, 1),
        _row("base-injection", 1, 0),
        _row("cot-forgery-injection", 1, 0),
        _row("cot-forgery-injection", 1, 1),
    ]

    summary = summarize_agent_results(rows)

    assert summary["by_variant"]["base-injection"]["attack_success"]["rate"] == 0.5
    assert summary["by_variant"]["base-injection"]["secure_task_success"]["rate"] == 0.5
    assert summary["role_spoof_gap"]["difference"] == 0.5
    assert len(summary["overall"]["attack_success"]["wilson_95"]) == 2
