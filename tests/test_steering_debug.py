import json

from eval_harness.config import Config
from eval_harness.core import message, tool_call, tool_result
from eval_harness.runner import DEVELOPER_PROMPT, SYSTEM_PROMPT, USER_PROMPT
from eval_harness.steering_debug import extract_single_insertion, replay_prompts


def test_extract_single_insertion() -> None:
    insertion, offset = extract_single_insertion("abcXYZ", "abcATTACKXYZ")
    assert insertion == "ATTACK"
    assert offset == 3


def test_extract_single_insertion_at_end() -> None:
    insertion, offset = extract_single_insertion("abc", "abcATTACK")
    assert insertion == "ATTACK"
    assert offset == 3


def test_replay_prompts_matches_message_protocol() -> None:
    config = Config(reasoning="high")
    case = {"url": "https://example.invalid/page"}
    raw_args = json.dumps({"command": "curl https://example.invalid/page"})
    result = {"stdout": "PAGE", "stderr": "", "exit_code": 0}
    row = {
        "transcript": [
            {
                "analysis": ["I will fetch it."],
                "final": None,
                "tool_calls": [["functions.bash", raw_args]],
            },
            {
                "analysis": [],
                "final": "Summary.",
                "tool_calls": [],
            },
        ],
        "tool_events": [
            {
                "tool": "bash",
                "command": "curl https://example.invalid/page",
                "result": result,
            }
        ],
    }
    prompts = replay_prompts(row, case, config)
    initial = "".join(
        [
            message("system", SYSTEM_PROMPT.format(reasoning="high")),
            message("developer", DEVELOPER_PROMPT),
            message(
                "user",
                USER_PROMPT.format(
                    workspace="/workspace/eval-sandbox", url=case["url"]
                ),
            ),
        ]
    )
    assert prompts[0] == initial + "<|start|>assistant"
    assert prompts[1] == (
        initial
        + message("assistant", "I will fetch it.", "analysis")
        + tool_call("functions.bash", raw_args)
        + tool_result("functions.bash", result)
        + "<|start|>assistant"
    )
