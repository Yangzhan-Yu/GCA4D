#!/bin/bash
set -ex

export AGENT_COT_REASONER_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_COT_REASONER_API_KEY='bearer'
export AGENT_COT_REASONER_BASE_URL='vllm'
export AGENT_COT_REASONER_PROXY=''

# Planner (text-only) and VLM (vision) can point at different services.
# Leave these unset to reuse AGENT_COT_REASONER_* for both roles.
export AGENT_PLANNER_MODEL="${AGENT_PLANNER_MODEL:-$AGENT_COT_REASONER_MODEL}"
export AGENT_PLANNER_API_KEY="${AGENT_PLANNER_API_KEY:-$AGENT_COT_REASONER_API_KEY}"
export AGENT_PLANNER_BASE_URL="${AGENT_PLANNER_BASE_URL:-$AGENT_COT_REASONER_BASE_URL}"
export AGENT_VLM_MODEL="${AGENT_VLM_MODEL:-$AGENT_COT_REASONER_MODEL}"
export AGENT_VLM_API_KEY="${AGENT_VLM_API_KEY:-$AGENT_COT_REASONER_API_KEY}"
export AGENT_VLM_BASE_URL="${AGENT_VLM_BASE_URL:-$AGENT_COT_REASONER_BASE_URL}"
export AGENT_CODE_GENERATOR_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_CODE_GENERATOR_API_KEY='bearer'
export AGENT_CODE_GENERATOR_BASE_URL='vllm'
export AGENT_CODE_GENERATOR_PROXY=''

BENCHMARK=$1
MODEL_NAME=${AGENT_COT_REASONER_MODEL##*/}

mkdir -p logs/
START_TIME=`date +%Y%m%d-%H:%M:%S`
LOG_FILE=logs/agent_${BENCHMARK}_${MODEL_NAME}_${START_TIME}.log

python -m entrypoints.agent --benchmark $BENCHMARK --concurrency 24 --resume \
    2>&1 | tee -a $LOG_FILE > /dev/null &

sleep 1s
tail -f $LOG_FILE
