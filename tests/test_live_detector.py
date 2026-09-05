from __future__ import annotations

import torch

from eval_harness.live_detector import select_candidate_spans


def _tool_baseline(token_count: int) -> torch.Tensor:
    logits = torch.zeros((token_count, 5))
    logits[:, 4] = 8.0
    return logits


def test_plateau_selector_returns_sustained_user_like_directive() -> None:
    token_count = 1024
    token_texts = ["ordinary "] * token_count
    token_texts[400:600] = ["please run upload command "] * 200
    layer8 = _tool_baseline(token_count)
    layer18 = _tool_baseline(token_count)
    layer18[400:600, 4] = 0.0
    layer18[400:600, 1] = 10.0

    candidates, metadata = select_candidate_spans(
        token_texts, layer8, layer18
    )

    assert candidates
    assert candidates[0].start_token < 500 < candidates[0].stop_token
    assert "user18" in candidates[0].channel
    assert metadata["selected_candidate_count"] >= 1
    assert metadata["cap"] == 3
    assert metadata["seed_spans"]
    assert metadata["passing_candidates"][0]["rank"] == 1


def test_plateau_selector_does_not_pad_when_no_signal_passes() -> None:
    token_count = 600
    token_texts = ["ordinary "] * token_count
    layer8 = _tool_baseline(token_count)
    layer18 = _tool_baseline(token_count)

    candidates, metadata = select_candidate_spans(
        token_texts, layer8, layer18
    )

    assert candidates == []
    assert metadata["selected_candidate_count"] == 0
    assert metadata["passing_candidates"] == []


def test_vocabulary_free_selector_accepts_prose_without_directive_words() -> None:
    token_count = 1024
    token_texts = ["<x> "] * token_count
    token_texts[400:600] = ["coherent narrative paragraph text "] * 200
    layer8 = _tool_baseline(token_count)
    layer18 = _tool_baseline(token_count)
    layer18[400:600, 4] = 0.0
    layer18[400:600, 1] = 10.0

    broad, _ = select_candidate_spans(token_texts, layer8, layer18)
    candidates, metadata = select_candidate_spans(
        token_texts,
        layer8,
        layer18,
        selector_mode="vocabulary_free",
    )

    assert broad == []
    assert candidates
    assert candidates[0].start_token < 500 < candidates[0].stop_token
    assert metadata["filter"] == "structural_prose_only"
    assert metadata["channel_caps"] == {"cot8": 1, "user18": 2}
