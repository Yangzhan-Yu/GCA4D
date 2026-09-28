#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/Agentic-Spatial-Reasoning/gca-main
cd "$ROOT"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /data2/conda_envs/gca
source scripts/gca_env.sh

required_vars=(
  AGENT_COT_REASONER_MODEL
  AGENT_COT_REASONER_BASE_URL
  AGENT_COT_REASONER_API_KEY
  AGENT_CODE_GENERATOR_MODEL
  AGENT_CODE_GENERATOR_BASE_URL
  AGENT_CODE_GENERATOR_API_KEY
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing environment variable: $name" >&2
    exit 1
  fi
done

mkdir -p logs work_dir/mindcube_10x3_qwen3vl235b

for question_type in rotation around among; do
  echo "=== Running MindCube ${question_type}: 10 samples ==="
  ray stop --force >/dev/null 2>&1 || true

  python -m entrypoints.agent \
    --benchmark mindcube \
    --config config/agent_mindcube.json \
    --question_type "$question_type" \
    --limit 10 \
    --concurrency 1 \
    --resume \
    --work_dir "work_dir/mindcube_10x3_qwen3vl235b/$question_type" \
    2>&1 | tee "logs/mindcube_10x3_qwen3vl235b_${question_type}.log"
done

python scripts/collect_mindcube_10x3.py
