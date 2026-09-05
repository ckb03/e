# Role-confusion unattended research goal

Work only inside `/workspace/role_confusion_research_lab`. Never modify `/workspace/e`.
Read `AGENTS.md`, `H200_HANDOFF.md`, and every file in `research_briefs/` before acting.

## Required order

### Phase 1: exact evaluation acceleration

Optimize the current evaluation before doing steering work. The optimized path must evaluate the same model behavior, not a cheaper substitute.

- Keep the frozen v2 manifest and its SHA-256 unchanged.
- Keep model, Harmony messages, prompts, tool simulator, scorer, case IDs, per-case seeds, temperature 1.0, high reasoning, 4,096-token cap, and eight-step limit unchanged.
- Do not truncate/clean HTML differently, lower reasoning/tokens, change sampling, or alter attack/tool semantics.
- Use fixed representative cases and retain the current sequential Transformers backend as the correctness oracle.
- A speed optimization is accepted for direct use only when raw generations, parsed tool calls/events, labels, and summaries match the oracle for the fixed equivalence suite. If a backend cannot be bitwise-equivalent, label it experimental and do not silently replace the oracle.
- Measure end-to-end case time, prefill/decode time where possible, tokens/s, peak VRAM, and GPU utilization.
- Profile before optimizing. Investigate high-ROI methods such as safe continuous batching with independent per-case RNG, length bucketing, exact cross-step KV reuse, static/compiled execution, prefix caching, and alternative engines. Do not assume any method is valid until equivalence is tested.
- Ensure resumability and flush results after every case.
- Write speed/equivalence results to `research_outputs/phase1_eval_optimization/`.

For every accepted optimization, preserve a production-quality path for later research:

- Keep existing CLI/config behavior compatible; expose new behavior through explicit flags when needed.
- Put accepted changes in focused git commits on a named lab branch; do not mix steering work into those commits.
- Add automated equivalence tests and a one-command benchmark/reproduction script.
- Generate `research_outputs/handoff/accepted_eval.patch` and record accepted commit hashes.
- Generate `research_outputs/handoff/USE_OPTIMIZED_EVAL.md`, machine-readable equivalence/benchmark reports, package/GPU versions, and example configs.
- Ensure the optimized evaluator runs directly here or applies cleanly to another checkout without copying environments, caches, or old runs.

Do not begin Phase 2 until Phase 1 has a tested best configuration and a written result, even if the result is that no faster exact method was found.

### Phase 2: steering dataset preparation

Follow `research_briefs/01_role_confusion_steering_dataset_preparation.md` exactly. Keep D_repr, D_clean, D_attack_dev, and D_test separate; enforce URL/content overlap rules; freeze manifests and hashes; run all quality checks. D_test is never used for selection or tuning.

### Phase 3: the two steering algorithms

Follow `research_briefs/02_role_confusion_steering_algorithms.md` exactly and implement only the two requested first methods:

1. Soft pairwise steering.
2. Continuous role-space steering.

Use the same residual-stream hook for extraction and intervention. Select layers/hyperparameters only on development splits. Evaluate against the unchanged baseline with ASR, clean capability, STSR, role-spoof gap where supported, over-refusal, and compute cost. Use enough matched development cases for directional evidence; do not run the full 200-case D_test merely to spend compute.

## Research discipline

- Keep all environments, caches, code changes, manifests, runs, logs, plots, and reports in this lab directory.
- The shared `/workspace/.hf_home/hub` model cache may be read/reused; do not delete or rewrite it deliberately.
- Never use real secrets or perform real uploads/exfiltration.
- Record configs, commits/diffs, seeds, case IDs, manifest hashes, package versions, GPU state, failures, and negative results.
- Run tests/lint and validate artifact consistency after changes.
- Do not modify `/workspace/e`; leave reviewed changes in this lab for the user to cherry-pick manually.
- Prefer reliable evidence over breadth. Do not claim causal steering from probe movement alone.

## Rate-limit behavior

If Codex reports that the rolling five-hour usage window is exhausted, preserve all state, do not switch to a paid API key or consume a reset credit, and wait for the account window to reset before resuming this same goal. Keep the tmux session alive.

## Definition of done

Produce `research_outputs/FINAL_REPORT.md` with: exact setup; phase-1 equivalence and speed table; accepted/rejected optimizations; dataset hashes/quality checks; soft-pairwise and continuous-steering results; security/utility tradeoffs; failures and limitations; and the highest-ROI next experiments. Ensure every reported number links to a machine-readable artifact. The goal is not complete until the portable optimized-eval handoff passes from a clean environment.
