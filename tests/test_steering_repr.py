from __future__ import annotations

import torch

from eval_harness.steering_repr import (
    ResidualCapture,
    _probe_metrics,
    _split_layer_data,
    evenly_spaced_offsets,
)


def test_evenly_spaced_offsets_are_deterministic_and_unique() -> None:
    assert evenly_spaced_offsets(4, 8) == [0, 1, 2, 3]
    assert evenly_spaced_offsets(10, 1) == [5]
    offsets = evenly_spaced_offsets(512, 64)
    assert len(offsets) == 64
    assert offsets[0] == 0
    assert offsets[-1] == 511
    assert offsets == sorted(set(offsets))


def test_residual_capture_uses_batch_specific_token_positions() -> None:
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    capture = ResidualCapture(layers)
    capture.reset(torch.tensor([[0, 2], [1, 3]]))
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    try:
        output = hidden
        for layer in layers:
            output = layer(output)
        values = capture.stacked()
    finally:
        capture.close()

    assert values.shape == (2, 2, 2, 3)
    assert torch.equal(values[0, 0].float(), hidden[0, [0, 2]])
    assert torch.equal(values[1, 1].float(), hidden[1, [1, 3]])


def test_split_layer_data_weights_every_base_sequence_equally() -> None:
    records = []
    for base_id, token_count in enumerate((2, 4)):
        records.append(
            {
                "base_id": base_id,
                "split": "repr_train",
                "activations": torch.zeros(1, 5, token_count, 3),
            }
        )

    features, labels, weights, selected = _split_layer_data(
        records,
        "repr_train",
        0,
    )

    assert features.shape == (30, 3)
    assert labels.bincount().tolist() == [6, 6, 6, 6, 6]
    assert len(selected) == 2
    assert torch.isclose(weights[:10].sum(), torch.tensor(5.0))
    assert torch.isclose(weights[10:].sum(), torch.tensor(5.0))


def test_probe_metrics_reports_perfect_base_balanced_classifier() -> None:
    features = torch.eye(5)
    labels = torch.arange(5)
    weights = torch.ones(5)
    metrics = _probe_metrics(
        features,
        labels,
        weights,
        weight=10 * torch.eye(5),
        bias=torch.zeros(5),
        temperature=1.0,
    )

    assert metrics["token_accuracy"] == 1.0
    assert metrics["base_balanced_accuracy"] == 1.0
    assert (
        metrics["confusion_matrix_counts"] == torch.eye(5, dtype=torch.int64).tolist()
    )
