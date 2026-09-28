#!/usr/bin/env bash
# Source this file before running GCA:
#   source scripts/gca_env.sh
# For fully offline execution after all assets are present, also export:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export GCA_ROOT=/data3/Agentic-Spatial-Reasoning/gca-main
export GCA_HF_HOME=/data3/Agentic-Spatial-Reasoning/hf_cache
export GCA_HF_HUB="$GCA_HF_HOME/hub"

export HF_HOME="$GCA_HF_HOME"
export HF_HUB_CACHE="$GCA_HF_HUB"
export AGENT_CACHE_DIR="$GCA_HF_HUB"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export U2NET_HOME=/data3/Agentic-Spatial-Reasoning/u2net
export EASYOCR_MODULE_PATH=/data3/Agentic-Spatial-Reasoning/easyocr
export GCA_RUNTIME_CACHE="$GCA_ROOT/.cache"
export XDG_CACHE_HOME="$GCA_RUNTIME_CACHE/xdg"
export MPLCONFIGDIR="$GCA_RUNTIME_CACHE/matplotlib"
export NUMBA_CACHE_DIR="$GCA_RUNTIME_CACHE/numba"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX="$GCA_RUNTIME_CACHE/pycache"

# SAM3 code, checkpoint and interpreter.  SAM3 needs torch>=2.7, while the
# gca env pins torch 2.5, so SAM3 runs in its own environment as a subprocess.
# Weights are intentionally not tracked.
export SAM3_ROOT="$GCA_ROOT/tools/third_party/sam3"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$SAM3_ROOT/checkpoints/sam3.pt}"
export PYTHON_SAM3="${PYTHON_SAM3:-/data2/conda_envs/sam3/bin/python}"
export SAM3_PYTHONPATH="${SAM3_PYTHONPATH:-$SAM3_ROOT}"

mkdir -p "$GCA_ROOT" "$GCA_HF_HUB" "$U2NET_HOME" \
         "$EASYOCR_MODULE_PATH/model" "$GCA_RUNTIME_CACHE" \
         "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PYTHONPYCACHEPREFIX"
