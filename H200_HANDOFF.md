# H200 handoff: GPT-OSS-20B prompt-injection evaluation

## Status and bottom line

The repository was cloned from `role-confusion/prompt-injection-as-role-confusion`
and a reusable CLI harness was added under `eval_harness/`.

The final 10-case, upstream-cap sanity run **passed as an infrastructure and
directional sanity check**:

- Model: `openai/gpt-oss-20b`
- Generation cap: 4,096 new tokens (same as the upstream notebook)
- Reasoning effort: `high` (same as the upstream agent notebook)
- Sample: 5 base injections + 5 CoT-forgery injections
- Runtime on one H100 80 GB: 16m 50s
- Base injection: 0/5 attack attempts/successes
- CoT Forgery: 2/5 attack attempts/successes
- Overall prompt-injection rate: 2/10 = 20%
- CoT-Forgery prompt-injection rate: 2/5 = 40%
- Completed without CUDA OOM after installing the correct FA3 stack

Artifacts:

`eval_runs/sanity-fullcap-10-gpt-oss-20b-5e4cf679998f/`

This does **not** constitute a paper reproduction. Ten cases have wide sampling
error, and the paper/project site's `61% -> 10%` result is from the separate
StrongREJECT chat/style-ablation experiment, not this exact agent evaluation.
The relevant agent claim is directional: CoT-like/User-like role cues increase
agent hijacking. The sanity result agrees with that direction.

## Safe URL simulation (fixed in manifest v2)

The first harness version used `benchmark.invalid` URLs, which GPT-OSS correctly recognized as artificial. Manifest v2 now emits realistic-looking deterministic `0x0.st` URLs while intercepting every fetch locally. `dpaste.com/api` uploads also remain simulated and recorded; no benchmark page or synthetic secret leaves the machine.

The tool simulator now models `curl -o`, redirects, subsequent file reads, directory listings, and synthetic exfiltration results. Regenerate `frozen_manifest-v2.jsonl` before the definitive baseline and use its recorded SHA-256 for every intervention comparison.

The primary local labeler is deterministic and tool-event based. The upstream notebook additionally sends transcripts to Gemini 2.5 Pro via OpenRouter for four-way classification. For exact paper compatibility, retain deterministic event labels as the canonical auditable metric and add the upstream classifier as a secondary label.

## Author settings confirmed from the upstream notebook

`experiments/cot-forgery-agent-evals/01-run-injections-gpt-oss.ipynb` uses:

- `model_prefix = 'gptoss-20b'`
- system prompt line `Reasoning: high`
- `max_new_tokens = 1024 * 4`
- `do_sample = True`
- `temperature = 1.0`
- maximum 8 ReAct steps
- 200 sampled rows from 100 base + 100 CoT-forgery HTML pages
- H200 for the paper experiments (per repository README)

Therefore, yes: the author's local agent evaluation uses **high reasoning**.

## Exact dependency lesson

Do not use the newest PyTorch build merely because CUDA works. The first setup
accidentally resolved PyTorch 2.13/cu130. The upstream FA3 hub kernel had no
matching build, forcing eager attention. Eager attention OOMed on a long HTML
page while attempting a 92.6 GiB allocation on the H100.

The working stack is:

- Python 3.12
- PyTorch `2.9.1+cu128`
- CUDA wheel runtime 12.8
- Transformers `4.57.5`
- Triton `3.5.1`
- `kernels==0.11.5`
- `compressed-tensors==0.13.0`
- FlashAttention `2.8.3+cu128torch2.9`
- attention implementation `kernels-community/vllm-flash-attn3`

These are now captured in `pyproject.toml` and `uv.lock`, including the PyTorch
cu128 index and the upstream-compatible prebuilt FlashAttention wheel.

## H200 / Vast image setup

The proposed `vastai/pytorch:cuda-12.8.1-auto` image is a better match than a
CUDA 12.4 image. The exact driver version does not need to equal CUDA 12.8; the
host driver only needs to support it. Do not install or replace the NVIDIA driver
inside the container.

On the new instance:

```bash
vast-capabilities metrics,packages | jq '{gpu:.hardware.gpu, volume:.instance.workspace_is_volume}'
nvidia-smi
cd /workspace/prompt-injection-as-role-confusion
uv sync --index-strategy unsafe-best-match
source .venv/bin/activate
python - <<'PY'
import torch, flash_attn, transformers
print(torch.__version__, torch.version.cuda)
print(flash_attn.__version__, transformers.__version__)
x = torch.ones(1, device="cuda")
print("CUDA OK:", x.item())
PY
```

Expected key output:

```text
2.9.1+cu128 12.8
2.8.3 4.57.5
CUDA OK: 1.0
```

The H200's 140 GB VRAM and roughly 4.0 TB/s HBM bandwidth should eliminate the
capacity concern and outperform the H100, but autoregressive decoding and the
sequential agent loop will remain the main bottlenecks.

## Transfer recommendation

Best option: commit these changes to a private fork/branch and clone that exact
commit on the H200. This preserves `pyproject.toml`, `uv.lock`, configs, harness,
and documentation with an auditable revision.

Do **not** commit:

- `.venv/`
- the 13 GB Hugging Face model cache
- run artifacts containing large transcripts
- real secrets or `.env` files

The frozen manifest is currently ignored by git. For fair cross-method work,
store it in durable versioned storage or deliberately commit it if licensing and
repository-size policy allow (~15.5 MB). Always record and verify its SHA-256.
The completed 10-case sanity run used the superseded v1 manifest, whose hash was:

`4b20c7f340ebee692757a5dc835a8f222766fa9dfb044239029123ff41563343`

Do not use that v1 hash for new intervention comparisons.

Manifest v2 contains the realistic, locally intercepted URLs. Generate it once on the H200, record its hash, and reuse that exact file for the definitive baseline and every intervention.

The locally validated 200-case v2 manifest has SHA-256:

`bbd34526386002629f7b2d253751e627ef722b33175830a67cf78ad19ee1a0d9`

If pushing is inconvenient, another agent can reproduce from upstream plus this
handoff, but cloning a pinned fork is less error-prone and much more auditable.

## Clean experiment organization

Keep these layers separate:

1. **Dataset**: immutable/versioned manifest with page, variant, injection, seed.
2. **Protocol**: Harmony prompt, tools, max steps, sampling, reasoning level.
3. **Model backend**: Transformers today; vLLM/SGLang later.
4. **Intervention**: separate wrapper/config per method; no benchmark mutation.
5. **Scoring**: deterministic event score plus optional upstream LLM classifier.
6. **Artifacts**: config, git commit, manifest hash, selected case IDs, raw
   transcripts, tool events, labels, aggregate summary, timing, package versions.

Never compare intervention runs if manifest hash, case IDs, protocol settings,
or sampling seed differ unless that difference is the explicitly studied factor.

## Performance observations

On H100 80 GB with the correct FA3 stack:

- 10 cases at the full 4,096-token cap: 16m 50s
- measured average: ~101 seconds/case
- projected 200 cases: ~5h 37m, with a practical range of 5-7 hours
- model memory was commonly ~15-23 GB, so memory capacity was not the steady-state
  bottleneck after FA3 was fixed
- long reasoning traces and repeated transcript prefill dominated runtime

The earlier 1,024-token test took 11 minutes for 10 cases, but truncated several
first-turn reasoning traces before a tool call. Its aggregate is not a valid
paper comparison. Do not lower the cap for final runs.

## Optimization TODOs

Optimizations that can preserve benchmark semantics:

- [ ] Add a vLLM or SGLang backend with continuous batching across independent
      agent cases. Pause a sequence at a tool call, execute the simulated tool,
      then requeue it.
- [ ] Batch/prefill cases of similar token length to reduce padding waste.
- [ ] Reuse KV cache across ReAct steps instead of reconstructing and prefilling
      the entire transcript after every tool result.
- [ ] Add resumability: skip case IDs already present in `results.jsonl` and
      write an explicit run-state file.
- [x] Add per-case timings plus prompt and generated token counts.
- [ ] Add peak VRAM and explicit termination reason.
- [ ] Shard the manifest across multiple GPUs/instances and merge by case ID.
- [ ] Benchmark H200 throughput on 10 fixed cases before launching all 200.
- [x] Add automated Harmony parser and safe-tool simulator tests.
- [ ] Expand parser fixtures with more real GPT-OSS output variants.
- [ ] Validate deterministic scorer vs the upstream Gemini classifier on a
      stratified transcript sample.
- [x] Correct simulated URL semantics and realistic curl `-o`/redirect behavior.

Optimizations that change evaluation semantics and should not be used for the
final fair comparison without declaring a new protocol:

- lowering `max_new_tokens`
- changing `Reasoning: high`
- switching temperature/sampling behavior
- truncating or cleaning HTML differently
- changing maximum ReAct steps
- changing injection prompts or page distribution
- speculative decoding without first demonstrating output-distribution parity

## Run sequence on H200

After generating and freezing manifest v2:

```bash
# 1. Exact two-case plumbing check
role-confusion-eval run --limit 2 --run-name h200-plumbing

# 2. Fixed ten-case timing/sanity check
role-confusion-eval run --limit 10 --run-name h200-sanity-10

# Inspect transcripts and labels manually before continuing.

# 3. Definitive baseline
role-confusion-eval run --run-name h200-baseline-gpt-oss-20b
```

Before interventions, copy the baseline's `run.json`, manifest hash, case IDs,
and protocol settings into a comparison registry. Every intervention must reuse
them exactly.
