#!/usr/bin/env bash
# Start vLLM server serving the local Phi-4 model.
# VRAM allocation: 85% of RTX 4090 (24GB) ≈ 20.4 GB for weights + KV cache.
#
# Usage:
#   bash scripts/start_vllm.sh          # foreground (Ctrl-C to stop)
#   bash scripts/start_vllm.sh &        # background

set -euo pipefail

# Prefix caching buys ~27% throughput (cache hit 54%) but is the prime suspect
# for the CUDA illegal-memory-access that killed the engine 1h40m into M-1:
# vLLM 0.7.2 has known block-manager races in that path. Set NO_PREFIX_CACHE=1
# to trade the throughput back for stability.
if [ "${NO_PREFIX_CACHE:-0}" = "1" ]; then
    PREFIX_CACHE_FLAG=""
    echo "  Prefix caching: DISABLED"
else
    PREFIX_CACHE_FLAG="--enable-prefix-caching"
fi

MODEL_PATH="/mnt/NewSSD/CS_project/Microsoft_Phi4_Local_LM"
SERVED_NAME="phi4"
PORT=8000

echo "Starting vLLM server..."
echo "  Model : $MODEL_PATH"
echo "  Name  : $SERVED_NAME"
echo "  Port  : $PORT"
echo "  VRAM  : 85%"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON:-python3}" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --dtype float16 \
    --seed 42 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --port "$PORT" \
    ${PREFIX_CACHE_FLAG} \
    --guided-decoding-backend outlines \
    --disable-log-requests
