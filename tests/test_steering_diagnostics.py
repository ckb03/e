from __future__ import annotations

import torch

from eval_harness.steering_diagnostics import primary_tool_activations


def test_primary_tool_activations_selects_largest_message() -> None:
    first = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    second = torch.arange(2 * 4 * 3).reshape(2, 4, 3) + 100
    shard = {
        "case_id": 7,
        "turns": [
            {
                "activations": first,
                "spans": [
                    {
                        "full_content_token_count": 5,
                        "selected_token_count": 2,
                        "selected_start": 0,
                        "selected_end": 2,
                    },
                    {
                        "full_content_token_count": 200,
                        "selected_token_count": 3,
                        "selected_start": 2,
                        "selected_end": 5,
                    },
                ],
            },
            {
                "activations": second,
                "spans": [
                    {
                        "full_content_token_count": 20,
                        "selected_token_count": 4,
                        "selected_start": 0,
                        "selected_end": 4,
                    }
                ],
            },
        ],
    }

    observed = primary_tool_activations(shard)

    assert torch.equal(observed, first[:, 2:5].float())
