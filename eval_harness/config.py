from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    model_id: str = "openai/gpt-oss-20b"
    model_label: str = "gpt-oss-20b"
    seed: int = 1234
    sample_size: int = 200
    max_steps: int = 8
    max_new_tokens: int = 4096
    temperature: float = 1.0
    reasoning: str = "high"
    device: str = "cuda:0"
    dtype: str = "auto"
    attn_implementation: str | None = "kernels-community/vllm-flash-attn3"
    cache_dir: str = "/workspace/.hf_home/hub"
    dataset_manifest: str = "eval_data/frozen_manifest-v2.jsonl"
    output_root: str = "eval_runs"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
