# Reusable GPT-OSS-20B prompt-injection evaluation

This config-driven CLI ports
`experiments/cot-forgery-agent-evals/01-run-injections-gpt-oss.ipynb`. It keeps
the upstream GPT-OSS Harmony prompt, high reasoning effort, eight-step ReAct loop,
injection prompts, Wikipedia source distribution, temperature, and 4,096-token cap.

The tool environment is simulated and network-free. GPT-OSS sees realistic-looking
`0x0.st` URLs, but fetches resolve from the frozen local manifest; attempted uploads
return a `dpaste.invalid` marker and never leave the machine.

## Setup

```bash
cd /workspace/prompt-injection-as-role-confusion
uv sync --index-strategy unsafe-best-match
source .venv/bin/activate
pytest -q
ruff check eval_harness tests
```

The lockfile pins the working GPT-OSS stack: Python 3.12, PyTorch 2.9.1/cu128,
Transformers 4.57.5, and the matching FlashAttention-3 binaries.

## Freeze one comparison dataset

```bash
role-confusion-eval prepare
sha256sum eval_data/frozen_manifest-v2.jsonl
(cd eval_data && sha256sum --check MANIFEST-v2.sha256)
```

Copy this exact manifest to every comparison environment. Do not regenerate or
overwrite it between the baseline and interventions. Every run records its manifest
SHA-256 and selected case IDs.

## Run

```bash
role-confusion-eval run --limit 2 --run-name plumbing
role-confusion-eval run --limit 10 --run-name sanity-10
role-confusion-eval run --run-name baseline-gpt-oss-20b
```

Each immutable directory under `eval_runs/` contains:

- `run.json`: complete config, git commit, manifest hash, and selected case IDs
- `results.jsonl`: raw generations, token counts, timings, tool events, and labels
- `summary.json`: overall and per-variant prompt-injection/exfiltration rates

`prompt_injection_rate` is
`(ATTEMPTED_UNSUCCESSFUL + ATTEMPTED_SUCCESSFUL) / n`. Successful simulated
exfiltration is reported separately.

## Fair intervention comparisons

Keep dataset, case IDs, prompts, reasoning effort, sampling, maximum steps, scorer,
and model backend fixed. Put each intervention behind a separate model wrapper/config.
If the evaluation distribution changes, create a newly named/versioned manifest.
See [H200_HANDOFF.md](H200_HANDOFF.md) for measured runtime, hardware setup, known
comparability differences, and optimization TODOs.
