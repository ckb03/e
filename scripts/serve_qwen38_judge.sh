#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
judge_python="${repo_dir}/.judge-venv/bin/python"
judge_site_packages="${repo_dir}/.judge-venv/lib/python3.12/site-packages"

export HF_HOME="/workspace/.hf_home"
export VLLM_CACHE_ROOT="${repo_dir}/data_cache/vllm"
export FLASHINFER_WORKSPACE_BASE="${repo_dir}/data_cache/flashinfer"
export LD_LIBRARY_PATH="${judge_site_packages}/nvidia/cu13/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${judge_python}" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.45 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --generation-config vllm
