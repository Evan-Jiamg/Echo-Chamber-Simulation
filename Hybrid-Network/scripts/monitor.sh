#!/usr/bin/env bash
# Live sweep progress, refreshed every 10 seconds.
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=$(ls -t "$PROJ"/logs/vllm/run_vllm_*.log 2>/dev/null | head -1)

while true; do
    clear
    echo "============================  $(date '+%H:%M:%S')  ============================"
    echo ""

    # job progress
    OK=$(grep -c "\[OK\]"      "$LOG" 2>/dev/null || echo 0)
    ERR=$(grep -c "\[ERR\]"    "$LOG" 2>/dev/null || echo 0)
    RETRY=$(grep -c "Retrying" "$LOG" 2>/dev/null || echo 0)
    WORKERS=$(pgrep -f run_hybrid | wc -l 2>/dev/null || echo 0)

    echo "[jobs]      done: $OK / 100   failed: $ERR   running: $WORKERS workers"
    echo "[retries]   $RETRY  (0 is normal; a rising count means a retry loop)"
    echo ""

    # most recently finished jobs
    echo "[recent]"
    grep "\[OK\]\|\[ERR\]" "$LOG" 2>/dev/null | tail -8 || echo "  (none yet)"
    echo ""

    # GPU
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu \
          --format=csv,noheader,nounits 2>/dev/null)
    echo "[GPU]       $GPU  (utilisation %, power W, temperature C)"

    # vLLM
    echo ""
    echo "[vLLM throughput]"
    tmux capture-pane -t hcog:vllm -p -S -20 2>/dev/null | grep "throughput" | tail -1 \
        || echo "  (unavailable)"

    echo ""
    echo "=================================================================="
    echo "  Ctrl-C to stop"
    sleep 10
done
