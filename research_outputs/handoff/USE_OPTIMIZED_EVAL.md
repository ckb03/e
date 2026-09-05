# Using the accepted exact evaluator

## Bottom line

No faster backend passed the fixed bitwise gate. Use the sequential Transformers
backend in the example config; do not use the experimental batching or static
cache results as replacements. The accepted implementation commit is listed in
[accepted_commits.json](accepted_commits.json), and the portable mail-format
patch is [accepted_eval.patch](accepted_eval.patch).

## Apply in another checkout

From a clean checkout at upstream commit
`35c9f787a7356836afe71d6b40dfe0443de1e9a8`:

```bash
git am /path/to/accepted_eval.patch
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check eval_harness tests
```

The patch does not contain environments, caches, model weights, or old run
directories. Verify its checksum with
`sha256sum --check accepted_eval.patch.sha256`.

## Run and resume

```bash
uv run --frozen role-confusion-eval run \
  --config research_outputs/handoff/configs/oracle-gpt-oss-20b.yaml \
  --case-ids 54,175 \
  --run-name fixed-suite

uv run --frozen role-confusion-eval run \
  --config research_outputs/handoff/configs/oracle-gpt-oss-20b.yaml \
  --case-ids 54,175 \
  --run-name fixed-suite \
  --resume
```

The run validates config fingerprint, manifest hash, and ordered case IDs before
resuming. It appends and flushes one JSONL row per completed case and updates
`run_state.json`.

## One-command reproduction

Inside this lab checkout:

```bash
scripts/reproduce_phase1.sh
```

That runs cases 54 and 175 and compares the result against
[oracle_reference_fixed](../phase1_eval_optimization/oracle_reference_fixed).
The command exits nonzero on any raw-generation, parsed-call, simulator-event,
label, termination, metadata, ordering, or semantic-summary mismatch.

## Recorded environment and results

- [Phase 1 result](../phase1_eval_optimization/RESULT.md)
- [Benchmark JSON](../phase1_eval_optimization/benchmark_report.json)
- [Equivalence JSON](../phase1_eval_optimization/equivalence_report.json)
- [Environment JSON](../phase1_eval_optimization/environment.json)
- [Package versions](../phase1_eval_optimization/package_versions.txt)
- [GPU versions](../phase1_eval_optimization/gpu_versions.csv)
- [Fixed suite](fixed_equivalence_suite.json)
- [Example config](configs/oracle-gpt-oss-20b.yaml)

The production path preserves the frozen manifest, Harmony formatting, prompts,
tool simulator, scorer, case seeds, sampling temperature, high reasoning,
4,096-token cap, and eight-step limit.
