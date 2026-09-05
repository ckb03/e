from __future__ import annotations

import torch

from eval_harness.steering_runtime import ToolResidualSteerer


def _states() -> tuple[dict, dict]:
    roles = ["system", "user", "cot", "assistant", "tool"]
    pair = torch.zeros((1, 5, 5, 2))
    pair[0, 0, 4] = torch.tensor([2.0, 0.0])
    representation = {
        "roles": roles,
        "probe_weight": torch.zeros((1, 5, 2)),
        "probe_bias": torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]]),
        "probe_temperature": torch.ones(1),
        "pair_vector": pair,
        "role_basis": torch.eye(2).reshape(1, 2, 2),
        "global_center": torch.zeros((1, 2)),
    }
    calibration = {
        "soft_thresholds": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]),
        "continuous_states": {
            1: {
                "tool_mean": torch.zeros((1, 1)),
                "tool_cov_inv": torch.ones((1, 1, 1)),
                "threshold": torch.ones(1),
            }
        },
    }
    return representation, calibration


def test_soft_pairwise_hook_changes_only_selected_tool_position() -> None:
    layer = torch.nn.Identity()
    representation, calibration = _states()
    steerer = ToolResidualSteerer(
        [layer], 0, "soft-pairwise", 1.0, representation, calibration, rank=1
    )
    steerer.prepare([1])
    values = torch.zeros((1, 3, 2))
    try:
        observed = layer(values)
        summary = steerer.summary()
    finally:
        steerer.close()

    assert observed[0, 1, 0] > 1.9
    assert torch.equal(observed[0, 0], values[0, 0])
    assert summary["steered_fraction"] == 1.0
    assert summary["wrong_role_excess_by_role"]["system"] > 0.99


def test_continuous_hook_moves_outlier_to_calibrated_boundary() -> None:
    layer = torch.nn.Identity()
    representation, calibration = _states()
    steerer = ToolResidualSteerer(
        [layer], 0, "continuous", 1.0, representation, calibration, rank=1
    )
    steerer.prepare([0])
    values = torch.tensor([[[2.0, 7.0], [0.0, 0.0]]])
    try:
        observed = layer(values)
        summary = steerer.summary()
    finally:
        steerer.close()

    assert torch.allclose(observed[0, 0], torch.tensor([1.0, 7.0]))
    assert torch.equal(observed[0, 1], values[0, 1])
    assert summary["fraction_outside_tool_region"] == 1.0
    assert summary["mean_mahalanobis_distance"] == 4.0
