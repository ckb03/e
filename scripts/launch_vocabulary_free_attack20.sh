#!/usr/bin/env bash
set -euo pipefail

cd /workspace/role_confusion_research_lab

output_root="research_outputs/prompt_injection_defense/vocabulary_free_strong_cot_attack20"
mkdir -p "$output_root"

exec 9>"$output_root/launcher.lock"
if ! flock -n 9; then
  echo "The vocabulary-free attack20 launcher already holds the lock."
  exit 1
fi

judge_log="$output_root/qwen_judge.log"
env PYTHONPATH=. .judge-venv/bin/python scripts/serve_qwen_bf16_judge.py \
  --host 127.0.0.1 --port 8017 >"$judge_log" 2>&1 &
judge_pid=$!

cleanup() {
  kill "$judge_pid" 2>/dev/null || true
  wait "$judge_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for attempt in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8017/health >/dev/null; then
    break
  fi
  if ! kill -0 "$judge_pid" 2>/dev/null; then
    echo "Qwen judge exited during startup; see $judge_log"
    exit 1
  fi
  if [[ "$attempt" == "180" ]]; then
    echo "Timed out waiting for Qwen judge; see $judge_log"
    exit 1
  fi
  sleep 1
done

env PYTHONPATH=. .venv/bin/python scripts/run_vocabulary_free_attack20.py \
  --output-root "$output_root" \
  --judge-url http://127.0.0.1:8017
