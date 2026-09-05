from __future__ import annotations

import torch

from eval_harness.continuous_diagnostics import (
    projected_continuous_intervention,
    trailing_mean,
)
from eval_harness.continuous_geometry import deterministic_positions
from eval_harness.continuous_runtime import ContinuousRoleRegionSteerer


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Identity()

    def forward(self, values):
        return self.mlp(values)


def _states() -> tuple[dict, dict]:
    geometry = {
        "rank": 1,
        "basis": {0: torch.tensor([[1.0], [0.0]])},
        "representation_center": {0: torch.zeros(2)},
    }
    calibration = {
        "window": 1,
        "layers": {
            0: {
                "tool_mean": torch.zeros(1),
                "inverse_covariance": torch.ones((1, 1)),
                "tau_token": 1.0,
                "beta": 0.0,
            }
        },
    }
    return geometry, calibration


def test_deterministic_positions_are_even_and_unique() -> None:
    assert deterministic_positions(5, 10).tolist() == [0, 1, 2, 3, 4]
    observed = deterministic_positions(10, 4)
    assert observed.tolist() == [0, 3, 6, 9]
    assert len(observed.unique()) == 4


def test_trailing_mean_uses_short_prefix_denominators() -> None:
    values = torch.arange(1, 6).float()
    assert torch.allclose(
        trailing_mean(values, 3), torch.tensor([1.0, 1.5, 2.0, 3.0, 4.0])
    )


def test_uncapped_continuous_correction_reaches_radial_boundary() -> None:
    observed = projected_continuous_intervention(
        torch.tensor([[10.0]]),
        torch.tensor([10.0]),
        torch.zeros(1),
        torch.ones((1, 1)),
        tau_token=1.0,
        beta=0.0,
        window=1,
        rho_max=1.0,
    )
    assert observed["correction_gate"].item()
    assert not observed["cap_activated"].item()
    assert torch.allclose(observed["d2_after"], torch.tensor([1.0]), atol=1e-5)


def test_continuous_correction_honors_relative_cap() -> None:
    observed = projected_continuous_intervention(
        torch.tensor([[10.0]]),
        torch.tensor([10.0]),
        torch.zeros(1),
        torch.ones((1, 1)),
        tau_token=1.0,
        beta=0.0,
        window=1,
        rho_max=0.01,
    )
    assert observed["cap_activated"].item()
    assert observed["delta_over_h"].item() <= 0.01 + 1e-6
    assert observed["d2_after"].item() < observed["d2_before"].item()


def test_continuous_runtime_does_not_smooth_across_tool_spans() -> None:
    block = _Block()
    geometry, calibration = _states()
    steerer = ContinuousRoleRegionSteerer([block], 0, 0.01, geometry, calibration)
    steerer.prepare([1, 2, 7, 8])
    try:
        assert steerer.segment_ranges == [(0, 2), (2, 4)]
    finally:
        steerer.close()
