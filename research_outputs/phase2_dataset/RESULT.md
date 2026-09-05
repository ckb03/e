# Phase 2: frozen steering datasets

Status: complete.

The fixed-seed builder produced the exact minimum first-run budget from the dataset brief:

- `D_repr`: 150 base texts, 75 C4 validation and 75 Dolma3, each frozen under the five native GPT-OSS role wrappers. Split by base text into 105 train, 20 dev, and 25 test bases.
- `D_clean`: 30 clean Wikipedia pages, split 20 calibration and 10 sanity.
- `D_attack_dev`: 30 disjoint Wikipedia pages with paired clean, base-injection, and CoT-forgery rows. Each of layer/tune/devval contains 10 pages, 20 attacks, and 10 clean counterparts.
- `D_test`: the existing 200-case manifest remained unchanged at SHA-256 `bbd34526386002629f7b2d253751e627ef722b33175830a67cf78ad19ee1a0d9`.

All 23 frozen quality checks pass. In particular, the representation content tokens align across all five wrappers, source content is unique, the three page datasets have no forbidden URL overlap, each attack has an unmodified clean counterpart, every injection occurs in the final serialized Tool context, all Tool contexts fit under 131,072 tokens, both attack families are balanced, and no case was filtered using baseline outcomes.

Machine-readable evidence:

- `quality_report.json`: counts, overlap sets, all quality booleans, and manifest hashes.
- `config.json`: seeds, source datasets/revision, shuffle buffers, tokenizer/model, and frozen final-test hash.
- `failures.json`: failed setup and verification attempts plus resolutions.
- `../../eval_data/STEERING_MANIFESTS.sha256`: portable checksum registry for all five JSONL manifests.

The environment now pins `zstandard==0.23.0`, which was required to stream the chosen Dolma3 `.zst` shard.
