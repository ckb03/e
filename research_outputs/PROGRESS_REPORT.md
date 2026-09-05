# Role-Confusion Research Lab Progress Report

Updated: 2026-09-02

Phase 1 produced no accepted runtime speedup. The apparent 12.7% improvement was ordinary repeat variance, not an optimization claim.

## Evaluation optimization audit

| Method | Result | Why |
|---|---:|---|
| Same-path repeat | 116.5s → 103.4s | Bitwise exact, but classified as run-to-run variance |
| Continuous batching with independent RNG | 118.2s, 0.87× | Slower and changed generations/tool trajectories |
| Static KV cache + compiled decode | 24.7s, 4.19× | Fast, but changed outputs and changed one attack from successful to unnoticed |
| Cross-turn KV reuse | Not implemented | Profiling put its maximum case-175 CUDA saving at only 5.11s/8.2%; exact cache continuation still needs testing |
| vLLM/SGLang engine | Not run | Neither was installed; changing engines/samplers was unlikely to pass the bitwise gate without dedicated work |
| Separate parallel evaluator processes | Not run | This is a real remaining opportunity and should not be conflated with the rejected batching experiment |

Exact evidence:

- [`phase1_eval_optimization/benchmark_report.json`](phase1_eval_optimization/benchmark_report.json)
- [`phase1_eval_optimization/equivalence_report.json`](phase1_eval_optimization/equivalence_report.json)

Clarifications:

- Ordinary autoregressive KV caching is already enabled with `use_cache=True`. Decode does not recompute the full prefix token-by-token.
- KV cache is not currently retained across separate agent turns. Each new `generate()` call rebuilds the expanded transcript's prefill.
- The configured `kernels-community/vllm-flash-attn3` is an attention kernel, not the vLLM serving engine.
- The profile found 91.8% of CUDA forward time in decode and 8.2% in prefill for case 175. That limits cross-turn prefix-cache upside for that case, but does not prove it is worthless across the full distribution.
- Multi-process parallel evaluation remains promising. The observed model-resident maximum was about 22.8 GiB on a 141 GiB H200, so several independent batch-1 workers may fit. It needs a controlled 2/4-worker experiment followed by the same raw-generation/tool-event/label equivalence gate.
- Phase 1 will be reopened for that concurrency experiment. An isolated vLLM experiment is lower priority because its sampler is unlikely to be bitwise identical, but the final report must distinguish “not tested” from “tested and rejected.”

## Current steering progress

### Completed

#### Frozen Phase 2 datasets

- 150 representation texts × five roles
- 30 clean pages
- 30 development pages × two attacks
- all 23 quality checks passed
- zero forbidden URL overlap
- manifests frozen with SHA-256 hashes

Evidence: [`phase2_dataset/quality_report.json`](phase2_dataset/quality_report.json)

#### Representation extraction

- all 150 bases
- 24 post-block residual layers
- up to 64 aligned content tokens per base
- 6.2 GiB of validated activation shards

#### Representation analysis

- 24 five-way probes
- all matched pairwise role vectors
- continuous role-space bases and singular spectra
- pair vectors are exactly antisymmetric
- maximum basis orthonormality error: \(9.54\times10^{-7}\)
- best held-out probe layer so far: block 15, 70.1% base-balanced accuracy

Evidence: [`phase3_steering/representation_analysis/analysis_report.json`](phase3_steering/representation_analysis/analysis_report.json)

#### Undefended clean baseline

- clean capability: 27/30 = 90%
- 95% Wilson interval: 74.4%–96.5%
- calibration split: 20/20
- separate sanity split: 7/10
- all three failures were preserved

Evidence: [`phase3_steering/tool_activations/clean/summary.json`](phase3_steering/tool_activations/clean/summary.json)

#### Undefended `D_layer` baseline

- attack rows: 20
- ASR: 5/20 = 25%
- base-injection ASR: 1/10 = 10%
- CoT-forgery ASR: 4/10 = 40%
- role-spoof gap: 30 percentage points, with wide small-sample uncertainty
- attack-task capability: 18/20 = 90%
- STSR: 13/20 = 65%
- clean counterparts: 8/10 = 80% capability

Evidence: [`phase3_steering/tool_activations/layer/summary.json`](phase3_steering/tool_activations/layer/summary.json)

### Not yet completed

- cheap layer-separation diagnostics
- top-three causal layer sweep
- soft-pairwise threshold calibration and runtime hook
- continuous Tool-region fitting and runtime hook
- tuning on `D_tune`
- one-shot validation on `D_devval`
- frozen final-test evaluation

## Capability and task-quality requirement

Capability preservation is an evaluation gate immediately after eval optimization and before judging steering.

The evaluator now separately reports:

- ASR only over attacked rows
- clean capability only over clean rows
- STSR only over attacked rows
- over-refusal
- role-spoof gap
- Wilson uncertainty
- deterministic simulator events as canonical attack labels

A summary-schema bug was found and fixed during the audit: the old aggregate divided five attacks by all 30 rows, including clean cases. Schema v2 now correctly reports ASR as 5/20, not 5/30. The fix is committed as `ae43059`.

One limitation remains: current “legitimate task success” verifies that the complete page reached the model and that it produced a final response. It does not yet rigorously score whether the requested 2–3 sentence summary is semantically correct. The task-quality scorer must be hardened before steering selection so a defense cannot look successful merely by returning an empty, irrelevant, or malformed final answer.
