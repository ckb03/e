from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .steering_data import (
    C4_DATASET,
    DOLMA_DATASET,
    DOLMA_REVISION,
    ROLES,
    _canonical_text,
    _read_jsonl,
    _source_url,
    _write_jsonl,
    render_single_role,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _role_prefix(role: str) -> str:
    return render_single_role(role, "").removesuffix("<|end|>")


def _sample_source(
    tokenizer,
    dataset,
    source: str,
    count: int,
    excluded_urls: set[str],
    excluded_page_texts: list[str],
    seen_hashes: set[str],
) -> list[dict]:
    rows = []
    for example in dataset:
        source_url = _source_url(example)
        if source_url in excluded_urls:
            continue
        token_ids = tokenizer(
            str(example["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        ).input_ids
        if not token_ids:
            continue
        content = tokenizer.decode(token_ids)
        canonical = _canonical_text(content)
        if len(canonical) < 128:
            continue
        content_hash = _sha256_text(canonical)
        if content_hash in seen_hashes:
            continue
        if any(canonical in page_text for page_text in excluded_page_texts):
            continue
        rows.append(
            {
                "source": source,
                "source_url": source_url,
                "content": content,
                "content_sha256": content_hash,
                "content_token_ids": token_ids,
                "content_token_count": len(token_ids),
            }
        )
        seen_hashes.add(content_hash)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"collected {len(rows)} of {count} {source} texts")
    return rows


def _excluded_eval_content(repo: Path) -> tuple[set[str], list[str]]:
    manifest_names = [
        "frozen_manifest-v2.jsonl",
        "steering_clean_manifest.jsonl",
        "steering_attack_layer_manifest.jsonl",
        "steering_attack_tune_manifest.jsonl",
        "steering_attack_devval_manifest.jsonl",
    ]
    rows = []
    for name in manifest_names:
        path = repo / "eval_data" / name
        if path.exists():
            rows.extend(_read_jsonl(path))
    urls = {str(row["source_url"]) for row in rows if row.get("source_url")}
    page_texts = [
        _canonical_text(row["html"])
        for row in rows
        if row.get("html") and row.get("variant") in {None, "clean"}
    ]
    return urls, page_texts


def prepare_v2_representation_data(
    repo: Path,
    seed: int = 20260905,
    force: bool = False,
) -> Path:
    output = repo / "eval_data/steering_repr_v2_manifest.jsonl"
    report_path = repo / "research_outputs/phase3_steering_v2/data_report.json"
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        cache_dir="/workspace/.hf_home/hub",
        local_files_only=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    excluded_urls, excluded_page_texts = _excluded_eval_content(repo)
    cache_dir = str(repo / "data_cache/steering_v2")
    c4 = load_dataset(
        C4_DATASET,
        "en",
        split="validation",
        streaming=True,
        cache_dir=cache_dir,
    ).shuffle(seed=seed, buffer_size=50_000)
    dolma = load_dataset(
        DOLMA_DATASET,
        split="train",
        revision=DOLMA_REVISION,
        streaming=True,
        cache_dir=cache_dir,
    ).shuffle(seed=seed, buffer_size=50_000)
    seen_hashes: set[str] = set()
    rows = _sample_source(
        tokenizer,
        c4,
        "c4",
        62,
        excluded_urls,
        excluded_page_texts,
        seen_hashes,
    )
    rows.extend(
        _sample_source(
            tokenizer,
            dolma,
            "dolma3",
            188,
            excluded_urls,
            excluded_page_texts,
            seen_hashes,
        )
    )
    random.Random(seed).shuffle(rows)
    splits = ["repr_train"] * 175 + ["repr_dev"] * 35 + ["repr_test"] * 40
    for base_id, (row, split) in enumerate(zip(rows, splits, strict=True)):
        row["base_id"] = base_id
        row["split"] = split
        row["role_variants"] = {}
        expected = row["content_token_ids"]
        for role in ROLES:
            prompt = render_single_role(role, row["content"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            prefix_ids = tokenizer(
                _role_prefix(role), add_special_tokens=False
            ).input_ids
            start = len(prefix_ids)
            end = start + len(expected)
            if prompt_ids[start:end] != expected:
                raise AssertionError(
                    f"content-token alignment failed for base {base_id}, role {role}"
                )
            row["role_variants"][role] = {
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "content_token_start": start,
                "content_token_end": end,
            }
    if len({row["content_sha256"] for row in rows}) != 250:
        raise AssertionError("v2 representation contents are not unique")
    if Counter(row["source"] for row in rows) != {"c4": 62, "dolma3": 188}:
        raise AssertionError("v2 source allocation mismatch")
    if Counter(row["split"] for row in rows) != {
        "repr_train": 175,
        "repr_dev": 35,
        "repr_test": 40,
    }:
        raise AssertionError("v2 split allocation mismatch")
    if any(row.get("source_url") in excluded_urls for row in rows):
        raise AssertionError("v2 representation URL overlaps evaluation data")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "seed": seed,
        "manifest": str(output.relative_to(repo)),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "base_count": len(rows),
        "roles": list(ROLES),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "max_content_tokens": max(row["content_token_count"] for row in rows),
        "mean_content_tokens": sum(row["content_token_count"] for row in rows)
        / len(rows),
        "content_tokens_by_split": {
            split: sum(
                row["content_token_count"] for row in rows if row["split"] == split
            )
            for split in ("repr_train", "repr_dev", "repr_test")
        },
        "checks": {
            "same_content_ids_all_roles": True,
            "base_level_split": True,
            "unique_content": True,
            "eval_url_disjoint": True,
            "native_gptoss_wrappers": True,
            "content_only_bounds": True,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return output
