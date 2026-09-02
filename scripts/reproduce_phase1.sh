#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

run_name="${1:-phase1-reproduction}"
fingerprint="$(uv run --frozen python -c 'from eval_harness.config import Config; print(Config.load("configs/gpt-oss-20b.yaml").fingerprint())')"
candidate="eval_runs/${run_name}-gpt-oss-20b-${fingerprint}"
report="research_outputs/phase1_eval_optimization/${run_name}-equivalence.json"

uv run --frozen role-confusion-eval run \
  --case-ids 54,175 \
  --run-name "$run_name"
uv run --frozen role-confusion-eval compare \
  --reference research_outputs/phase1_eval_optimization/oracle_reference_fixed \
  --candidate "$candidate" \
  --output "$report"

echo "Exact-equivalence report: $report"
