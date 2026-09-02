from __future__ import annotations

import hashlib
import html as html_module
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import requests
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from .core import tool_result

ROLES = ("system", "user", "cot", "assistant", "tool")
DOLMA_DATASET = "allenai/dolma3_mix-150B-1025"
DOLMA_REVISION = "3a8349c"
C4_DATASET = "allenai/c4"
WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
_BODY_CLOSE = re.compile(r"</body\s*>", re.IGNORECASE)
_CLASS_ATTRIBUTE = re.compile(r'\s+class="[^"]*"')
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def render_single_role(role: str, content: str) -> str:
    if role in {"system", "user"}:
        header = f"{role}<|message|>"
    elif role == "cot":
        header = "assistant<|channel|>analysis<|message|>"
    elif role == "assistant":
        header = "assistant<|channel|>final<|message|>"
    elif role == "tool":
        header = "functions. to=assistant<|channel|>commentary<|message|>"
    else:
        raise ValueError(f"unsupported role: {role}")
    return f"<|start|>{header}{content}<|end|>"


def _role_prefix(role: str) -> str:
    rendered = render_single_role(role, "")
    return rendered.removesuffix("<|end|>")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_text(value: str) -> str:
    without_tags = _TAG.sub(" ", value)
    return _WHITESPACE.sub(" ", html_module.unescape(without_tags)).strip().lower()


def _source_url(example: dict) -> str | None:
    if example.get("url"):
        return str(example["url"])
    metadata = example.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, dict):
        for key in ("url", "source_url", "warc_url"):
            if metadata.get(key):
                return str(metadata[key])
    return None


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _load_attack_templates(repo: Path) -> dict[str, list[dict]]:
    path = repo / "experiments/cot-forgery-agent-evals/prompts/injections.yaml"
    config = yaml.safe_load(path.read_text())
    return {
        "base-injection": [
            {
                "injection_id": f"base-injection:{item['type']}",
                "text": item["prompt"].strip(),
            }
            for item in config["base_injections"]
        ],
        "cot-forgery-injection": [
            {
                "injection_id": f"cot-forgery-injection:{item['type']}",
                "text": item["prompt"].strip(),
            }
            for item in config["prompt_injections"]
        ],
    }


def _balanced_templates(
    templates: list[dict],
    count: int,
    seed: int,
) -> list[dict]:
    order = list(templates)
    random.Random(seed).shuffle(order)
    return [order[index % len(order)] for index in range(count)]


def _collect_wikipedia_pages(
    repo: Path,
    count: int,
    seed: int,
    excluded_urls: set[str],
    max_kb: int,
) -> list[dict]:
    stream = load_dataset(
        WIKIPEDIA_DATASET,
        WIKIPEDIA_CONFIG,
        split="train",
        streaming=True,
        cache_dir=str(repo / "data_cache/steering"),
    ).shuffle(seed=seed, buffer_size=1_000)
    session = requests.Session()
    session.headers["User-Agent"] = "role-confusion-steering/1.0"
    selected: list[dict] = []
    seen_urls = set(excluded_urls)
    seen_content_hashes: set[str] = set()
    for example in stream:
        source_url = str(example["url"])
        if source_url in seen_urls:
            continue
        try:
            response = session.get(source_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            continue
        raw_html = response.content.decode("utf-8", errors="ignore")
        size_checked_html = _CLASS_ATTRIBUTE.sub("", raw_html)
        if len(size_checked_html.encode()) > max_kb * 1_024:
            continue
        if not _BODY_CLOSE.search(raw_html):
            continue
        canonical = _canonical_text(raw_html)
        content_sha256 = _sha256_text(canonical)
        if not canonical or content_sha256 in seen_content_hashes:
            continue
        selected.append(
            {
                "source_url": source_url,
                "html": raw_html,
                "html_sha256": _sha256_text(raw_html),
                "content_sha256": content_sha256,
                "html_bytes": len(raw_html.encode()),
            }
        )
        seen_urls.add(source_url)
        seen_content_hashes.add(content_sha256)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"collected {len(selected)} of {count} Wikipedia pages")
    return selected


def _sample_representation_source(
    repo: Path,
    tokenizer,
    dataset,
    source: str,
    count: int,
    excluded_urls: set[str],
    excluded_page_texts: list[str],
    seen_hashes: set[str],
) -> list[dict]:
    rows: list[dict] = []
    for example in dataset:
        source_url = _source_url(example)
        if source_url in excluded_urls:
            continue
        token_ids = tokenizer(
            str(example["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        ).input_ids
        if not token_ids:
            continue
        content = tokenizer.decode(token_ids)
        canonical = _canonical_text(content)
        if len(canonical) < 128:
            continue
        content_sha256 = _sha256_text(canonical)
        if content_sha256 in seen_hashes:
            continue
        if any(canonical in page_text for page_text in excluded_page_texts):
            continue
        rows.append(
            {
                "source": source,
                "source_url": source_url,
                "content": content,
                "content_sha256": content_sha256,
                "content_token_ids": token_ids,
                "content_token_count": len(token_ids),
            }
        )
        seen_hashes.add(content_sha256)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"collected {len(rows)} of {count} {source} texts")
    return rows


def _build_repr_rows(
    repo: Path,
    tokenizer,
    seed: int,
    excluded_urls: set[str],
    excluded_page_texts: list[str],
) -> list[dict]:
    cache_dir = str(repo / "data_cache/steering")
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
    rows = _sample_representation_source(
        repo,
        tokenizer,
        c4,
        "c4",
        75,
        excluded_urls,
        excluded_page_texts,
        seen_hashes,
    )
    rows.extend(
        _sample_representation_source(
            repo,
            tokenizer,
            dolma,
            "dolma3",
            75,
            excluded_urls,
            excluded_page_texts,
            seen_hashes,
        )
    )
    random.Random(seed).shuffle(rows)
    split_names = ["repr_train"] * 105 + ["repr_dev"] * 20 + ["repr_test"] * 25
    for base_id, (row, split) in enumerate(zip(rows, split_names, strict=True)):
        row["base_id"] = base_id
        row["split"] = split
        row["role_variants"] = {}
        for role in ROLES:
            prompt = render_single_role(role, row["content"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            prefix_ids = tokenizer(
                _role_prefix(role),
                add_special_tokens=False,
            ).input_ids
            start = len(prefix_ids)
            end = start + row["content_token_count"]
            if prompt_ids[start:end] != row["content_token_ids"]:
                raise AssertionError(
                    f"content-token alignment failed for base {base_id}, role {role}"
                )
            row["role_variants"][role] = {
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "content_token_ids_sha256": _sha256_text(
                    json.dumps(prompt_ids[start:end], separators=(",", ":"))
                ),
                "prompt_token_count": len(prompt_ids),
                "content_token_start": start,
                "content_token_end": end,
            }
    return rows


def _clean_rows(pages: list[dict], tokenizer) -> list[dict]:
    rows = []
    for page_index, page in enumerate(pages):
        split = "calibration" if page_index < 20 else "clean_sanity"
        sandbox_url = f"https://0x0.st/role-confusion-steering-clean-{page_index}.html"
        tool_message = tool_result(
            "functions.bash",
            {"stdout": page["html"], "stderr": "", "exit_code": 0},
        )
        rows.append(
            {
                "case_id": page_index,
                "page_id": f"clean-page-{page_index:03d}",
                "split": split,
                "variant": "clean",
                "source_url": page["source_url"],
                "url": sandbox_url,
                "html": page["html"],
                "html_sha256": page["html_sha256"],
                "content_sha256": page["content_sha256"],
                "html_bytes": page["html_bytes"],
                "tool_result_token_count": len(
                    tokenizer(tool_message, add_special_tokens=False).input_ids
                ),
                "contains_injection": False,
            }
        )
    return rows


def _attack_rows(
    pages: list[dict],
    templates: dict[str, list[dict]],
    tokenizer,
    seed: int,
) -> dict[str, list[dict]]:
    base_assignments = _balanced_templates(
        templates["base-injection"], len(pages), seed + 1
    )
    cot_assignments = _balanced_templates(
        templates["cot-forgery-injection"], len(pages), seed + 2
    )
    split_names = ["layer"] * 10 + ["tune"] * 10 + ["devval"] * 10
    outputs = {name: [] for name in ("layer", "tune", "devval")}
    next_case_id = 0
    next_attack_id = 0
    for page_index, (page, split) in enumerate(zip(pages, split_names, strict=True)):
        page_id = f"attack-dev-page-{page_index:03d}"
        sandbox_url = f"https://0x0.st/role-confusion-steering-dev-{page_index}.html"
        clean_case_id = next_case_id
        variants: list[tuple[str, dict | None]] = [
            ("clean", None),
            ("base-injection", base_assignments[page_index]),
            ("cot-forgery-injection", cot_assignments[page_index]),
        ]
        for variant, template in variants:
            if template is None:
                rendered_html = page["html"]
                injection_id = None
                injection_sha256 = None
                attack_id = None
                injection_context_verified = None
            else:
                injection = template["text"]
                rendered_html = _BODY_CLOSE.sub(
                    lambda _, selected=injection: selected + "</body>",
                    page["html"],
                    count=1,
                )
                if injection not in rendered_html:
                    raise AssertionError("injection missing after HTML insertion")
                escaped_injection = json.dumps(injection, ensure_ascii=False)[1:-1]
                context = tool_result(
                    "functions.bash",
                    {"stdout": rendered_html, "stderr": "", "exit_code": 0},
                )
                if escaped_injection not in context:
                    raise AssertionError("injection missing from final Tool context")
                injection_context_verified = True
                injection_id = template["injection_id"]
                injection_sha256 = _sha256_text(injection)
                attack_id = next_attack_id
                next_attack_id += 1
            tool_message = tool_result(
                "functions.bash",
                {"stdout": rendered_html, "stderr": "", "exit_code": 0},
            )
            outputs[split].append(
                {
                    "case_id": next_case_id,
                    "attack_id": attack_id,
                    "clean_case_id": clean_case_id,
                    "page_id": page_id,
                    "split": split,
                    "variant": variant,
                    "source_url": page["source_url"],
                    "url": sandbox_url,
                    "injection_context_verified": injection_context_verified,
                    "injection_id": injection_id,
                    "injection_sha256": injection_sha256,
                    "clean_html_sha256": page["html_sha256"],
                    "html": rendered_html,
                    "html_sha256": _sha256_text(rendered_html),
                    "html_bytes": len(rendered_html.encode()),
                    "tool_result_token_count": len(
                        tokenizer(tool_message, add_special_tokens=False).input_ids
                    ),
                }
            )
            next_case_id += 1
    return outputs


def _quality_report(
    repr_rows: list[dict],
    clean_rows: list[dict],
    attack_rows: dict[str, list[dict]],
    test_rows: list[dict],
    manifest_hashes: dict[str, str],
) -> dict:
    all_attack_rows = [row for rows in attack_rows.values() for row in rows]
    attack_only = [row for row in all_attack_rows if row["variant"] != "clean"]
    clean_attack_counterparts = {
        row["page_id"] for row in all_attack_rows if row["variant"] == "clean"
    }
    attack_pages = {row["page_id"] for row in attack_only}
    clean_urls = {row["source_url"] for row in clean_rows}
    attack_urls = {row["source_url"] for row in all_attack_rows}
    test_urls = {row["source_url"] for row in test_rows}
    repr_hashes = [row["content_sha256"] for row in repr_rows]
    repr_urls = {row["source_url"] for row in repr_rows if row["source_url"]}
    repr_token_alignment = all(
        all(
            variant["content_token_ids_sha256"]
            == _sha256_text(json.dumps(row["content_token_ids"], separators=(",", ":")))
            for variant in row["role_variants"].values()
        )
        for row in repr_rows
    )
    page_texts = [_canonical_text(row["html"]) for row in test_rows] + [
        _canonical_text(row["html"]) for row in clean_rows
    ]
    page_texts.extend(
        _canonical_text(row["html"])
        for row in all_attack_rows
        if row["variant"] == "clean"
    )
    repr_page_content_disjoint = all(
        _canonical_text(row["content"]) not in page_text
        for row in repr_rows
        for page_text in page_texts
    )
    clean_html_hashes = {row["html_sha256"] for row in clean_rows}
    attack_clean_hashes = {
        row["html_sha256"] for row in all_attack_rows if row["variant"] == "clean"
    }
    checks = {
        "repr_count_150": len(repr_rows) == 150,
        "repr_role_count_5": all(
            set(row["role_variants"]) == set(ROLES) for row in repr_rows
        ),
        "repr_content_token_alignment": repr_token_alignment,
        "repr_page_url_disjoint": repr_urls.isdisjoint(
            clean_urls | attack_urls | test_urls
        ),
        "repr_page_content_disjoint": repr_page_content_disjoint,
        "repr_unique_content": len(repr_hashes) == len(set(repr_hashes)),
        "repr_split_counts": Counter(row["split"] for row in repr_rows)
        == {"repr_train": 105, "repr_dev": 20, "repr_test": 25},
        "clean_count_30": len(clean_rows) == 30,
        "clean_split_counts": Counter(row["split"] for row in clean_rows)
        == {"calibration": 20, "clean_sanity": 10},
        "attack_count_60": len(attack_only) == 60,
        "attack_split_row_counts": all(
            len(rows) == 30 for rows in attack_rows.values()
        ),
        "clean_attack_content_disjoint": clean_html_hashes.isdisjoint(
            attack_clean_hashes
        ),
        "tool_context_within_model_limit": all(
            row["tool_result_token_count"] < 131_072
            for row in clean_rows + all_attack_rows
        ),
        "attack_clean_rows_unmodified": all(
            row["injection_id"] is None
            and row["injection_sha256"] is None
            and row["html_sha256"] == row["clean_html_sha256"]
            for row in all_attack_rows
            if row["variant"] == "clean"
        ),
        "templates_balanced_within_splits": all(
            set(
                Counter(
                    row["injection_id"] for row in rows if row["variant"] != "clean"
                ).values()
            )
            == {2}
            for rows in attack_rows.values()
        ),
        "attack_injection_present_in_final_tool_context": all(
            row["injection_context_verified"] is True for row in attack_only
        ),
        "no_baseline_outcome_filtering": True,
        "clean_counterpart_per_attack_page": clean_attack_counterparts == attack_pages,
        "clean_attack_url_disjoint": clean_urls.isdisjoint(attack_urls),
        "attack_test_url_disjoint": attack_urls.isdisjoint(test_urls),
        "clean_test_url_disjoint": clean_urls.isdisjoint(test_urls),
        "no_clean_injection": all(not row["contains_injection"] for row in clean_rows),
        "both_attack_variants_per_page": all(
            {row["variant"] for row in all_attack_rows if row["page_id"] == page_id}
            == {"clean", "base-injection", "cot-forgery-injection"}
            for page_id in attack_pages
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"steering dataset quality checks failed: {failed}")
    return {
        "schema_version": 1,
        "checks": checks,
        "counts": {
            "repr": len(repr_rows),
            "repr_by_source": dict(Counter(row["source"] for row in repr_rows)),
            "repr_by_split": dict(Counter(row["split"] for row in repr_rows)),
            "clean": len(clean_rows),
            "attack_rows_including_clean_counterparts": len(all_attack_rows),
            "attack_cases": len(attack_only),
            "attack_by_variant": dict(Counter(row["variant"] for row in attack_only)),
            "attack_by_split": dict(Counter(row["split"] for row in attack_only)),
            "template_usage": dict(Counter(row["injection_id"] for row in attack_only)),
        },
        "overlap": {
            "clean_attack_urls": sorted(clean_urls & attack_urls),
            "attack_test_urls": sorted(attack_urls & test_urls),
            "clean_test_urls": sorted(clean_urls & test_urls),
        },
        "manifest_sha256": manifest_hashes,
    }


def prepare_steering_data(
    repo: Path,
    repr_seed: int = 20260902,
    wikipedia_seed: int = 20260903,
    max_kb: int = 100,
    force: bool = False,
) -> dict:
    output_paths = {
        "repr": repo / "eval_data/steering_repr_manifest.jsonl",
        "clean": repo / "eval_data/steering_clean_manifest.jsonl",
        "attack_layer": repo / "eval_data/steering_attack_layer_manifest.jsonl",
        "attack_tune": repo / "eval_data/steering_attack_tune_manifest.jsonl",
        "attack_devval": repo / "eval_data/steering_attack_devval_manifest.jsonl",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"steering manifests already exist: {[str(path) for path in existing]}"
        )

    test_manifest = repo / "eval_data/frozen_manifest-v2.jsonl"
    test_rows = _read_jsonl(test_manifest)
    test_urls = {row["source_url"] for row in test_rows}
    pages = _collect_wikipedia_pages(
        repo,
        count=60,
        seed=wikipedia_seed,
        excluded_urls=test_urls,
        max_kb=max_kb,
    )
    clean_pages = pages[:30]
    attack_pages = pages[30:]
    all_page_urls = test_urls | {page["source_url"] for page in pages}
    all_page_texts = [_canonical_text(row["html"]) for row in test_rows] + [
        _canonical_text(page["html"]) for page in pages
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        cache_dir="/workspace/.hf_home/hub",
        local_files_only=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    repr_rows = _build_repr_rows(
        repo,
        tokenizer,
        repr_seed,
        all_page_urls,
        all_page_texts,
    )
    clean_rows = _clean_rows(clean_pages, tokenizer)
    templates = _load_attack_templates(repo)
    attack_rows = _attack_rows(
        attack_pages,
        templates,
        tokenizer,
        wikipedia_seed,
    )

    _write_jsonl(output_paths["repr"], repr_rows)
    _write_jsonl(output_paths["clean"], clean_rows)
    _write_jsonl(output_paths["attack_layer"], attack_rows["layer"])
    _write_jsonl(output_paths["attack_tune"], attack_rows["tune"])
    _write_jsonl(output_paths["attack_devval"], attack_rows["devval"])
    manifest_hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in output_paths.items()
    }
    hashes_path = repo / "eval_data/STEERING_MANIFESTS.sha256"
    hashes_path.write_text(
        "".join(
            f"{digest}  {output_paths[key].name}\n"
            for key, digest in manifest_hashes.items()
        )
    )
    quality = _quality_report(
        repr_rows,
        clean_rows,
        attack_rows,
        test_rows,
        manifest_hashes,
    )
    output_dir = repo / "research_outputs/phase2_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_report.json").write_text(
        json.dumps(quality, indent=2) + "\n"
    )
    config = {
        "schema_version": 1,
        "repr_seed": repr_seed,
        "wikipedia_seed": wikipedia_seed,
        "max_kb": max_kb,
        "c4": {"dataset": C4_DATASET, "config": "en", "split": "validation"},
        "dolma3": {
            "dataset": DOLMA_DATASET,
            "revision": DOLMA_REVISION,
            "split": "train",
        },
        "wikipedia": {
            "dataset": WIKIPEDIA_DATASET,
            "config": WIKIPEDIA_CONFIG,
            "split": "train",
            "shuffle_buffer": 1_000,
        },
        "repr_shuffle_buffer": 50_000,
        "model": "openai/gpt-oss-20b",
        "max_content_tokens": 512,
        "roles": list(ROLES),
        "test_manifest_sha256": hashlib.sha256(test_manifest.read_bytes()).hexdigest(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return quality
