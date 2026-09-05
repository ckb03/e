# Isolated role-confusion research lab

- This checkout is the only writable scope for the unattended research goal.
- Never edit `/workspace/e` or files elsewhere under `/workspace`.
- Reuse `/workspace/.hf_home/hub` only as the shared model-weight cache.
- Keep environments, package caches, logs, runs, plots, and reports in this tree.
- Do not use real credentials, external uploads, or unsimulated exfiltration.

## Experiment rules

- Preserve upstream protocol semantics in speed comparisons unless clearly labeled.
- Use fixed manifest hashes, case IDs, seeds, configs, and matched conditions.
- Record failures and negative results, not only successful experiments.
- Evaluate ASR, clean capability, STSR, and role-spoof gap where applicable.
- Treat deterministic simulator events as canonical attack labels.
- Keep clean/attack tasks paired and report uncertainty for small samples.

## Verification

- Run unit tests and lint after code changes.
- Verify every result artifact is parseable and internally consistent.
- Write conclusions and limitations to `research_outputs/FINAL_REPORT.md`.

## Cleanup

The entire lab can be removed later as one directory:

`/workspace/role_confusion_research_lab`
