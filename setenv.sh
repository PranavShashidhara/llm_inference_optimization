#!/usr/bin/env bash

echo "⚡ Activating Edge LLM Inference Environment..."

# ----------------------------
# 1. Activate virtual environment
# ----------------------------
source venv/bin/activate

# ----------------------------
# 2. GPU / CUDA environment flags
# ----------------------------
export CUDA_VISIBLE_DEVICES=0

# Optional: improves deterministic behavior (useful for benchmarking)
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Optional: reduce fragmentation (important for KV cache workloads)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ----------------------------
# 3. Project root
# ----------------------------
export PROJECT_ROOT=$(pwd)

# ----------------------------
# 4. Python path (important for modular structure)
# ----------------------------
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# ----------------------------
# 5. Nice terminal indicator
# ----------------------------
export PS1="(llm-opt-env) \u@\h:\w$ "

echo "Environment ready!"
echo "Project root: $PROJECT_ROOT"