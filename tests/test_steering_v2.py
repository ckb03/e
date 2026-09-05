from __future__ import annotations

import torch

from eval_harness.steering_v2_diagnostics import smooth_local
from eval_harness.steering_v2_runtime import V2SoftPairwiseSteerer


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Identity()

    def forward(self, values):
        return self.mlp(values)


def _states() -> tuple[dict, dict]:
    pair = torch.zeros((5, 5, 2))
    pair[0, 4] = torch.tensor([-1.0, 0.0])
    representation = {
        "probe_weight": {0: torch.tensor([[1.0, 0.0]] + [[0.0, 0.0]] * 4)},
        "probe_bias": {0: torch.zeros(5)},
        "unit_pair_direction": {0: pair},
    }
    calibration = {
        "window": 1,
        "route_temperature": 1.0,
        "gamma": 1.0,
        "direction_floor": 0.1,
        "layers": {
            0: {
                "margin_center": torch.zeros(4),
                "margin_scale": torch.ones(4),
                "joint_threshold": 0.0,
            }
        },
    }
    return representation, calibration


def test_centered_smoothing_has_correct_edge_denominators() -> None:
    values = torch.arange(1, 6).float()[:, None]
    observed = smooth_local(values, window=3).squeeze(1)
    assert torch.allclose(observed, torch.tensor([1.5, 2.0, 3.0, 4.0, 4.5]))


def test_v2_pre_mlp_hook_changes_only_selected_token_and_honors_cap() -> None:
    block = _Block()
    representation, calibration = _states()
    steerer = V2SoftPairwiseSteerer([block], 0, 0.01, representation, calibration)
    steerer.prepare([1])
    values = torch.tensor([[[1.0, 0.0], [10.0, 0.0], [2.0, 0.0]]])
    try:
        observed = block(values)
        summary = steerer.summary()
    finally:
        steerer.close()
    assert torch.equal(observed[0, 0], values[0, 0])
    assert torch.equal(observed[0, 2], values[0, 2])
    assert observed[0, 1, 0] < values[0, 1, 0]
    assert summary["steered_fraction"] == 1.0
    assert summary["max_relative_intervention_norm"] <= 0.01 + 1e-6


def test_v2_smoothing_segments_do_not_cross_tool_messages() -> None:
    block = _Block()
    representation, calibration = _states()
    calibration["window"] = 3
    steerer = V2SoftPairwiseSteerer([block], 0, 0.01, representation, calibration)
    steerer.prepare([1, 2, 7, 8])
    assert steerer.segment_ranges == [(0, 2), (2, 4)]
    steerer.close()
