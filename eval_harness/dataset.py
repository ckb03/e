from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import requests
import yaml
from datasets import load_dataset

_BODY_CLOSE = re.compile(r"</body\s*>", re.IGNORECASE)
_CLASS_ATTRIBUTE = re.compile(r'\s+class="[^"]*"')


def prepare(
    repo: Path,
    pages: int = 100,
    seed: int = 1234,
    max_kb: int = 100,
    overwrite: bool = False,
) -> Path:
    """Freeze the upstream Wikipedia distribution into a deterministic manifest."""
    output_path = repo / "eval_data/frozen_manifest-v2.jsonl"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; use --overwrite only for a deliberately new distribution"
        )

    prompt_path = repo / "experiments/cot-forgery-agent-evals/prompts/injections.yaml"
    prompt_config = yaml.safe_load(prompt_path.read_text())
    variants = {
        "base-injection": [
            item["prompt"].strip() for item in prompt_config["base_injections"]
        ],
        "cot-forgery-injection": [
            item["prompt"].strip() for item in prompt_config["prompt_injections"]
        ],
    }

    stream = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=1_000)
    base_pages = []
    for example in stream:
        try:
            response = requests.get(
                example["url"],
                headers={"User-Agent": "role-confusion-eval/1.0"},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException:
            continue

        raw_html = response.content.decode("utf-8", errors="ignore")
        size_checked_html = _CLASS_ATTRIBUTE.sub("", raw_html)
        if len(size_checked_html.encode()) <= max_kb * 1_024 and _BODY_CLOSE.search(
            raw_html
        ):
            base_pages.append((example["url"], raw_html))
        if len(base_pages) == pages:
            break

    if len(base_pages) != pages:
        raise RuntimeError(
            f"collected only {len(base_pages)} of {pages} requested pages"
        )

    rng = random.Random(seed)
    rows = []
    for variant, prompts in variants.items():
        for source_url, html in base_pages:
            injection = rng.choice(prompts)
            case_id = len(rows)
            injected_html = _BODY_CLOSE.sub(
                lambda _, selected=injection: selected + "</body>",
                html,
                count=1,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "variant": variant,
                    "source_url": source_url,
                    # Looks like the upstream host but is always intercepted locally.
                    "url": f"https://0x0.st/role-confusion-eval-{case_id}.html",
                    "injection_sha256": hashlib.sha256(injection.encode()).hexdigest(),
                    "html": injected_html,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    return output_path
