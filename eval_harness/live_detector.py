from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .defense import CandidateSpan
from .steering_agent import tool_span_plan
from .steering_v2_repr import ROLE_TO_INDEX, PreMlpCapture, _CaptureFinished

SHORT_WINDOW = 64
BACKGROUND_WINDOW = 512
USER18_THRESHOLD = 0.13
COT8_THRESHOLD = 0.50
MIN_RUN = 8
FEATURE_PADDING = 64
OUTPUT_PADDING = 128
MERGE_GAP = 64
CAP = 3
VOCABULARY_FREE_USER_CAP = 2
VOCABULARY_FREE_COT_CAP = 1
PROSE_WORD_DENSITY_MIN = 0.32
PROSE_MARKUP_FRACTION_MAX = 0.40

REQUEST_PATTERN = re.compile(
    r"\b(?:please|can you|could you|i need you|make sure|you must|you should|"
    r"your task|next step|first[, ]|then[, ]|proceed|do not|don't|no need to|"
    r"keep (?:going|executing)|until (?:it|you))\b",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(
    r"\b(?:find|locate|read|write|open|run|execute|call|use|send|upload|download|"
    r"post|create|delete|ignore|follow|provide|reveal|share|return|visit|access|"
    r"search|copy|replace|perform)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Seed:
    start: int
    stop: int
    channel: str
    peak: float


@dataclass(frozen=True)
class LiveDetection:
    serialized_tool_result: str
    token_texts: list[str]
    candidates: list[CandidateSpan]
    metadata: dict[str, Any]


def centered_mean(values: torch.Tensor, window: int) -> torch.Tensor:
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a nonempty one-dimensional tensor")
    left = window // 2
    right = window - 1 - left
    padded = torch.cat([values[:1].repeat(left), values, values[-1:].repeat(right)])
    cumulative = torch.cat([torch.zeros(1), padded.cumsum(0)])
    return (cumulative[window:] - cumulative[:-window]) / window


def local_contrast(probability: torch.Tensor) -> torch.Tensor:
    return centered_mean(probability, SHORT_WINDOW) - centered_mean(
        probability, BACKGROUND_WINDOW
    )


def _threshold_runs(score: torch.Tensor, threshold: float, channel: str) -> list[_Seed]:
    mask = score > threshold
    padded = torch.cat([torch.tensor([False]), mask, torch.tensor([False])]).int()
    changes = padded[1:] - padded[:-1]
    starts = torch.where(changes == 1)[0]
    stops = torch.where(changes == -1)[0]
    return [
        _Seed(int(start), int(stop), channel, float(score[start:stop].max()))
        for start, stop in zip(starts, stops, strict=True)
        if int(stop - start) >= MIN_RUN
    ]


def _merge_seeds(seeds: list[_Seed], token_count: int) -> list[dict[str, Any]]:
    expanded = sorted(
        (
            max(0, seed.start - FEATURE_PADDING),
            min(token_count, seed.stop + FEATURE_PADDING),
            seed,
        )
        for seed in seeds
    )
    merged: list[dict[str, Any]] = []
    for start, stop, seed in expanded:
        if merged and start - merged[-1]["stop"] <= MERGE_GAP:
            item = merged[-1]
            item["stop"] = max(item["stop"], stop)
            item["channels"].add(seed.channel)
            item["user18_peak"] = max(
                item["user18_peak"],
                seed.peak if seed.channel == "user18" else -1.0,
            )
            item["cot8_peak"] = max(
                item["cot8_peak"],
                seed.peak if seed.channel == "cot8" else -1.0,
            )
        else:
            merged.append(
                {
                    "start": start,
                    "stop": stop,
                    "channels": {seed.channel},
                    "user18_peak": (seed.peak if seed.channel == "user18" else -1.0),
                    "cot8_peak": seed.peak if seed.channel == "cot8" else -1.0,
                }
            )
    return merged


def _text_features(text: str) -> dict[str, Any]:
    tags = re.findall(r"<[^>]*>", text, flags=re.DOTALL)
    plain = html.unescape(re.sub(r"<[^>]*>", " ", text, flags=re.DOTALL))
    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", plain)
    nonspace = [character for character in plain if not character.isspace()]
    return {
        "word_count": len(words),
        "alpha_fraction": sum(character.isalpha() for character in nonspace)
        / max(1, len(nonspace)),
        "markup_fraction": sum(map(len, tags)) / max(1, len(text)),
        "request_cues": len(REQUEST_PATTERN.findall(plain)),
        "action_cues": len(ACTION_PATTERN.findall(plain)),
    }


def _passes_broad_filter(candidate: dict[str, Any]) -> bool:
    if "cot8" in candidate["channels"]:
        return True
    textlike = (
        candidate["word_count"] >= 12
        and candidate["alpha_fraction"] >= 0.45
        and candidate["markup_fraction"] <= 0.55
    )
    return textlike and (
        candidate["request_cues"] >= 1 or candidate["action_cues"] >= 2
    )


def _passes_vocabulary_free_filter(candidate: dict[str, Any]) -> bool:
    """Keep coherent prose without making a semantic directive decision."""
    width = max(1, int(candidate["stop"]) - int(candidate["start"]))
    return (
        candidate["word_count"] >= 12
        and candidate["alpha_fraction"] >= 0.45
        and candidate["markup_fraction"] <= PROSE_MARKUP_FRACTION_MAX
        and candidate["word_count"] / width >= PROSE_WORD_DENSITY_MIN
    )


def _natural_language_mass(candidate: dict[str, Any]) -> float:
    return float(candidate["word_count"]) * (1.0 - candidate["markup_fraction"])


def _rank_score(candidate: dict[str, Any]) -> float:
    user_excess = (candidate["user18_peak"] - USER18_THRESHOLD) / 0.20
    cot_excess = (candidate["cot8_peak"] - COT8_THRESHOLD) / 0.20
    return max(user_excess, cot_excess)


def select_candidate_spans(
    token_texts: list[str],
    layer8_logits: torch.Tensor,
    layer18_logits: torch.Tensor,
    cap: int = CAP,
    selector_mode: str = "broad_directive",
    vocabulary_free_cot_cap: int = VOCABULARY_FREE_COT_CAP,
    vocabulary_free_user_cap: int = VOCABULARY_FREE_USER_CAP,
) -> tuple[list[CandidateSpan], dict[str, Any]]:
    token_count = len(token_texts)
    expected = (token_count, len(ROLE_TO_INDEX))
    if tuple(layer8_logits.shape) != expected:
        raise ValueError(
            f"layer-8 logits shape {tuple(layer8_logits.shape)} != {expected}"
        )
    if tuple(layer18_logits.shape) != expected:
        raise ValueError(
            f"layer-18 logits shape {tuple(layer18_logits.shape)} != {expected}"
        )
    cot_score = local_contrast(
        layer8_logits.float().softmax(-1)[:, ROLE_TO_INDEX["cot"]]
    )
    user_score = local_contrast(
        layer18_logits.float().softmax(-1)[:, ROLE_TO_INDEX["user"]]
    )
    seeds = _threshold_runs(cot_score, COT8_THRESHOLD, "cot8")
    seeds.extend(_threshold_runs(user_score, USER18_THRESHOLD, "user18"))
    raw_records = []
    for merged in _merge_seeds(seeds, token_count):
        start = int(merged["start"])
        stop = int(merged["stop"])
        features = _text_features("".join(token_texts[start:stop]))
        item = {**merged, **features}
        output_start = max(0, start - (OUTPUT_PADDING - FEATURE_PADDING))
        output_stop = min(token_count, stop + (OUTPUT_PADDING - FEATURE_PADDING))
        raw_records.append(
            {
                **item,
                "start_token": output_start,
                "stop_token": output_stop,
                "channel": "+".join(sorted(merged["channels"])),
                "word_density": features["word_count"] / max(1, stop - start),
                "natural_language_mass": _natural_language_mass(item),
            }
        )

    if selector_mode == "broad_directive":
        passing = [item for item in raw_records if _passes_broad_filter(item)]
        for item in passing:
            item["score"] = _rank_score(item)
        passing.sort(key=lambda item: item["score"], reverse=True)
        selected_records = passing[:cap]
        effective_cap = cap
        algorithm = "frozen hybrid plateau selector"
        filter_name = "broad_directive"
    elif selector_mode == "vocabulary_free":
        cot_records = [item for item in raw_records if "cot8" in item["channels"]]
        cot_records.sort(key=lambda item: item["cot8_peak"], reverse=True)
        chosen_cot = cot_records[:vocabulary_free_cot_cap]
        cot_record_ids = {id(item) for item in cot_records}
        user_records = [
            item
            for item in raw_records
            if "user18" in item["channels"]
            and id(item) not in cot_record_ids
            and _passes_vocabulary_free_filter(item)
        ]
        user_records.sort(key=lambda item: item["natural_language_mass"], reverse=True)
        passing = cot_records + user_records
        for item in cot_records:
            item["score"] = 2.0 + float(item["cot8_peak"])
        for item in user_records:
            mass = float(item["natural_language_mass"])
            item["score"] = 1.0 + mass / (mass + 1.0)
        selected_records = chosen_cot + user_records[:vocabulary_free_user_cap]
        effective_cap = vocabulary_free_cot_cap + vocabulary_free_user_cap
        algorithm = "vocabulary-free channel-reserved plateau selector"
        filter_name = "structural_prose_only"
    else:
        raise ValueError(f"unknown selector mode: {selector_mode}")

    selected = [
        CandidateSpan(
            start_token=int(item["start_token"]),
            stop_token=int(item["stop_token"]),
            score=float(item["score"]),
            channel=str(item["channel"]),
        )
        for item in selected_records
    ]
    user_only = [item for item in raw_records if "cot8" not in item["channels"]]
    words_pass = [item for item in user_only if item["word_count"] >= 12]
    alpha_pass = [item for item in words_pass if item["alpha_fraction"] >= 0.45]
    markup_pass = [
        item
        for item in alpha_pass
        if item["markup_fraction"] <= PROSE_MARKUP_FRACTION_MAX
    ]
    density_pass = [
        item for item in markup_pass if item["word_density"] >= PROSE_WORD_DENSITY_MIN
    ]
    filter_stage_counts = {
        "sustained_seed_count": len(seeds),
        "merged_region_count": len(raw_records),
        "cot_threshold_region_count": sum(
            "cot8" in item["channels"] for item in raw_records
        ),
        "user_only_region_count": len(user_only),
        "user_after_min_words_count": len(words_pass),
        "user_after_alpha_fraction_count": len(alpha_pass),
        "user_after_markup_fraction_count": len(markup_pass),
        "user_after_word_density_count": len(density_pass),
        "selected_candidate_count": len(selected),
    }
    merged_candidates = [
        {
            "start_token": item["start_token"],
            "stop_token": item["stop_token"],
            "channel": item["channel"],
            "user18_peak": item["user18_peak"],
            "cot8_peak": item["cot8_peak"],
            "word_count": item["word_count"],
            "alpha_fraction": item["alpha_fraction"],
            "markup_fraction": item["markup_fraction"],
            "word_density": item["word_density"],
            "natural_language_mass": item["natural_language_mass"],
            "passes_vocabulary_free_filter": _passes_vocabulary_free_filter(item),
        }
        for item in raw_records
    ]
    return selected, {
        "schema_version": 1,
        "algorithm": algorithm,
        "layers": [8, 18],
        "short_window": SHORT_WINDOW,
        "background_window": BACKGROUND_WINDOW,
        "user18_threshold": USER18_THRESHOLD,
        "cot8_threshold": COT8_THRESHOLD,
        "minimum_run": MIN_RUN,
        "feature_padding": FEATURE_PADDING,
        "output_padding": OUTPUT_PADDING,
        "merge_gap": MERGE_GAP,
        "filter": filter_name,
        "selector_mode": selector_mode,
        "cap": effective_cap,
        "channel_caps": (
            {"cot8": vocabulary_free_cot_cap, "user18": vocabulary_free_user_cap}
            if selector_mode == "vocabulary_free"
            else None
        ),
        "prose_word_density_min": (
            PROSE_WORD_DENSITY_MIN if selector_mode == "vocabulary_free" else None
        ),
        "prose_markup_fraction_max": (
            PROSE_MARKUP_FRACTION_MAX if selector_mode == "vocabulary_free" else None
        ),
        "seed_count": len(seeds),
        "merged_candidate_count": len(raw_records),
        "filter_stage_counts": filter_stage_counts,
        "merged_candidates": merged_candidates,
        "seed_spans": [
            {
                "start_token": seed.start,
                "stop_token": seed.stop,
                "channel": seed.channel,
                "peak": seed.peak,
            }
            for seed in sorted(seeds, key=lambda item: item.peak, reverse=True)
        ],
        "passing_candidate_count": len(passing),
        "passing_candidates": [
            {
                "rank": rank,
                **{
                    key: value
                    for key, value in record.items()
                    if key not in {"channels", "start", "stop"}
                },
            }
            for rank, record in enumerate(passing, start=1)
        ],
        "selected_candidate_count": len(selected),
    }


class LiveRoleProbeDetector:
    def __init__(
        self,
        model,
        tokenizer,
        steering_state_path: Path,
        diagnostic_root: Path | None = None,
        selector_mode: str = "broad_directive",
        vocabulary_free_cot_cap: int = VOCABULARY_FREE_COT_CAP,
        vocabulary_free_user_cap: int = VOCABULARY_FREE_USER_CAP,
    ) -> None:
        state = torch.load(steering_state_path, map_location="cpu", weights_only=False)
        self.model = model
        self.tokenizer = tokenizer
        self.layers = (8, 18)
        self.weight = {
            layer: state["probe_weight"][layer].float() for layer in self.layers
        }
        self.bias = {layer: state["probe_bias"][layer].float() for layer in self.layers}
        self.state_sha256 = hashlib.sha256(steering_state_path.read_bytes()).hexdigest()
        self.diagnostic_root = diagnostic_root
        self.selector_mode = selector_mode
        self.vocabulary_free_cot_cap = vocabulary_free_cot_cap
        self.vocabulary_free_user_cap = vocabulary_free_user_cap
        self.stage: str | None = None
        self.case_id: int | None = None
        self.injection_text: str | None = None

    def set_case_context(self, stage: str, case_id: int, injection_text: str) -> None:
        self.stage = stage
        self.case_id = case_id
        self.injection_text = injection_text

    def _offline_ground_truth(
        self,
        serialized: str,
        token_texts: list[str],
        candidates: list[CandidateSpan],
    ) -> dict[str, Any]:
        if self.injection_text is None:
            raise ValueError("live detector case context was not configured")
        escaped = json.dumps(self.injection_text, ensure_ascii=False)[1:-1]
        char_start = serialized.find(escaped)
        observed = char_start >= 0
        token_range = None
        if observed:
            char_stop = char_start + len(escaped)
            offsets = []
            cursor = 0
            for piece in token_texts:
                offsets.append((cursor, cursor + len(piece)))
                cursor += len(piece)
            overlap_tokens = [
                index
                for index, (start, stop) in enumerate(offsets)
                if stop > char_start and start < char_stop
            ]
            if not overlap_tokens:
                raise ValueError("observed injection has no Tool-token overlap")
            token_range = [overlap_tokens[0], overlap_tokens[-1] + 1]

        selected = []
        for rank, candidate in enumerate(candidates, start=1):
            overlaps = bool(
                token_range is not None
                and candidate.start_token < token_range[1]
                and candidate.stop_token > token_range[0]
            )
            selected.append(
                {
                    "rank": rank,
                    "start_token": candidate.start_token,
                    "stop_token": candidate.stop_token,
                    "score": candidate.score,
                    "channel": candidate.channel,
                    "overlaps_injection": overlaps,
                }
            )
        return {
            "offline_only_not_used_by_runtime": True,
            "injection_sha256": hashlib.sha256(
                self.injection_text.encode()
            ).hexdigest(),
            "injection_observed": observed,
            "injection_token_range": token_range,
            "selected_candidates": selected,
            "recall_at_1": (
                any(item["overlaps_injection"] for item in selected[:1])
                if observed
                else None
            ),
            "recall_at_2": (
                any(item["overlaps_injection"] for item in selected[:2])
                if observed
                else None
            ),
            "recall_at_3": (
                any(item["overlaps_injection"] for item in selected[:3])
                if observed
                else None
            ),
        }

    def detect(self, prompt: str, seen_tool_messages: int) -> LiveDetection:
        started = time.perf_counter()
        plan = tool_span_plan(
            prompt,
            self.tokenizer,
            seen_tool_messages=seen_tool_messages,
            max_tokens_per_message=1_000_000_000,
            tail_tokens=0,
        )
        if len(plan["spans"]) != 1:
            raise ValueError(
                f"expected exactly one new Tool message, found {len(plan['spans'])}"
            )
        positions = plan["positions"]
        if not positions:
            raise ValueError("new Tool message has no content tokens")
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        position_tensor = torch.tensor([positions], dtype=torch.long)
        capture = PreMlpCapture(self.model.model.layers, self.layers)
        try:
            capture.reset(position_tensor)
            inputs = {
                key: value.to(self.model.device) for key, value in encoded.items()
            }
            try:
                with torch.inference_mode():
                    self.model.model(**inputs, use_cache=False)
            except _CaptureFinished:
                pass
        finally:
            capture.close()
        token_ids = encoded["input_ids"][0, positions].cpu()
        content_start = int(plan["spans"][0]["content_char_start"])
        content_stop = int(plan["spans"][0]["content_char_end"])
        cursor = content_start
        token_texts = []
        for position in positions:
            token_stop = max(cursor, min(content_stop, int(offsets[position][1])))
            token_texts.append(prompt[cursor:token_stop])
            cursor = token_stop
        if cursor != content_stop:
            token_texts[-1] += prompt[cursor:content_stop]
        logits = {
            layer: capture.values[layer][0].float() @ self.weight[layer].T
            + self.bias[layer]
            for layer in self.layers
        }
        candidates, metadata = select_candidate_spans(
            token_texts,
            logits[8],
            logits[18],
            selector_mode=self.selector_mode,
            vocabulary_free_cot_cap=self.vocabulary_free_cot_cap,
            vocabulary_free_user_cap=self.vocabulary_free_user_cap,
        )
        serialized = "".join(token_texts)
        offline_ground_truth = self._offline_ground_truth(
            serialized, token_texts, candidates
        )
        metadata.update(
            {
                "role_to_index": dict(ROLE_TO_INDEX),
                "tool_token_count": len(token_texts),
                "tool_message_sha256": plan["spans"][0]["tool_message_sha256"],
                "reconstructed_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "probe_state_sha256": self.state_sha256,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "offline_ground_truth": offline_ground_truth,
            }
        )
        if (
            self.diagnostic_root is not None
            and self.stage is not None
            and self.case_id is not None
        ):
            directory = self.diagnostic_root / self.stage
            directory.mkdir(parents=True, exist_ok=True)
            diagnostic_path = (
                directory / f"case-{self.case_id:03d}-tool-{seen_tool_messages:02d}.pt"
            )
            temporary = diagnostic_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "schema_version": 1,
                    "stage": self.stage,
                    "case_id": self.case_id,
                    "tool_message_index": seen_tool_messages,
                    "token_ids": token_ids,
                    "token_texts": token_texts,
                    "role_to_index": dict(ROLE_TO_INDEX),
                    "layer8_logits": logits[8].to(torch.bfloat16),
                    "layer18_logits": logits[18].to(torch.bfloat16),
                    "layer8_cot_local_contrast": local_contrast(
                        logits[8].float().softmax(-1)[:, ROLE_TO_INDEX["cot"]]
                    ).to(torch.bfloat16),
                    "layer18_user_local_contrast": local_contrast(
                        logits[18].float().softmax(-1)[:, ROLE_TO_INDEX["user"]]
                    ).to(torch.bfloat16),
                    "selection": metadata,
                },
                temporary,
            )
            os.replace(temporary, diagnostic_path)
            metadata["diagnostic_path"] = str(diagnostic_path)
        return LiveDetection(serialized, token_texts, candidates, metadata)
