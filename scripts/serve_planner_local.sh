#!/bin/bash
# Serve the Planner locally with vLLM instead of calling a cloud API.
#
# The agent is planner-only (text): every request is a plain-text JSON prompt,
# it never sees images.  So use a TEXT model - a VL model's vision tower would
# only cost VRAM and throughput.
#
# VRAM budget for the candidates:
#   Qwen3-30B-A3B-Instruct      BF16 ~61GB  -> needs --tp 4
#   Qwen3-30B-A3B-Instruct-AWQ  4bit ~18GB  -> --tp 1
#   Qwen3-32B                   BF16 ~64GB  -> needs --tp 4
#   Qwen3-14B                   BF16 ~28GB  -> --tp 2
#   Qwen3-8B                    BF16 ~16GB  -> --tp 1
#
# Usage (BF16 30B on four GPUs):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL=Qwen/Qwen3-30B-A3B-Instruct TP=4 \
#       bash scripts/serve_planner_local.sh
#
# Usage (AWQ 30B on one GPU):
#   CUDA_VISIBLE_DEVICES=6 MODEL=Qwen/Qwen3-30B-A3B-Instruct-AWQ TP=1 \
#       bash scripts/serve_planner_local.sh
set -eu

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen3-30b-a3b}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
LOG="${LOG:-logs/vllm_planner.log}"
# Extra vLLM flags, e.g. EXTRA_ARGS='--quantization gptq_marlin' if the
# checkpoint's quantization is not auto-detected.
EXTRA_ARGS="${EXTRA_ARGS:-}"
# vLLM lives outside the gca env; point this at whichever env has it.
VLLM_PYTHON="${VLLM_PYTHON:-/data2/conda_envs/ICL/bin/python}"

cd "$(dirname "$0")/.."
mkdir -p logs

# Share the project HF cache so a model downloaded once is found here too.
# gca_env.sh sets HF_HUB_OFFLINE=1, which is what we want at serve time: the
# weights are expected to be cached already, and staying offline avoids a
# slow/hanging metadata check against the hub.
if [ -f scripts/gca_env.sh ]; then
    # shellcheck disable=SC1091
    . scripts/gca_env.sh
fi

if [ ! -x "$VLLM_PYTHON" ]; then
    echo "[serve] vLLM python not found: $VLLM_PYTHON" >&2
    echo "[serve] set VLLM_PYTHON=/path/to/env/bin/python" >&2
    exit 1
fi

case "$MODEL" in
    *AWQ*|*awq*|*GPTQ*|*gptq*|*FP8*|*fp8*|*w4a16*)
        echo "[serve] quantized weights detected: expecting a small VRAM footprint"
        ;;
    *)
        echo "[serve] NOTE: '$MODEL' looks like BF16 (~2 bytes/param)."
        echo "[serve]       30B needs --tp 4 on 24GB cards; 32B likewise."
        ;;
esac

echo "[serve] model       : $MODEL"
echo "[serve] served name : $SERVED_NAME"
echo "[serve] port        : $PORT   tp: $TP"
echo "[serve] vis. GPUs   : ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[serve] log         : $LOG"

# --enable-prefix-caching matters a lot: the planner prompt shares a long
# unchanged prefix (tool list, rules, constraint, scene state) across steps.
# shellcheck disable=SC2086
set -- "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --trust-remote-code \
    $EXTRA_ARGS

echo "[serve] cmd         : $*"
nohup "$@" > "$LOG" 2>&1 &

echo "[serve] pid $!  -> waiting for /v1/models ..."
for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[serve] ready: http://127.0.0.1:${PORT}/v1"
        echo
        echo "export AGENT_PLANNER_MODEL='${SERVED_NAME}'"
        echo "export AGENT_PLANNER_BASE_URL='http://127.0.0.1:${PORT}/v1'"
        echo "export AGENT_PLANNER_API_KEY='local'"
        exit 0
    fi
    sleep 5
done

echo "[serve] still not up after 15min - check $LOG" >&2
tail -40 "$LOG" >&2 || true
exit 1
