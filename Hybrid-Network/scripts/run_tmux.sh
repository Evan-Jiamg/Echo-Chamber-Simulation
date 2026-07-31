#!/usr/bin/env bash
# Launch vLLM + parallel experiment runner in a tmux session.
#
# Usage:
#   bash scripts/run_tmux.sh
#
# Layout:
#   window 0 "vllm"   — vLLM server (keep alive)
#   window 1 "runner" — waits for server, then runs all experiments

set -euo pipefail

SESSION="hcog"
# Project root from this script's location, so the repository runs
# wherever it is cloned.
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Override for a specific environment:  PYTHON=/path/to/python ./this.sh
PYTHON="${PYTHON:-python3}"
VLLM_URL="http://localhost:8000/v1"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session (window 0: vLLM server)
tmux new-session -d -s "$SESSION" -n "vllm" -x 220 -y 50

tmux send-keys -t "$SESSION:vllm" \
    "cd $PROJ && bash scripts/start_vllm.sh" Enter

# Window 1: wait for vLLM, then run experiments
tmux new-window -t "$SESSION" -n "runner"
tmux send-keys -t "$SESSION:runner" "cd $PROJ" Enter
tmux send-keys -t "$SESSION:runner" "export VLLM_BASE_URL=$VLLM_URL" Enter

# Poll until vLLM is ready, then launch
tmux send-keys -t "$SESSION:runner" \
    "echo 'Waiting for vLLM server...' && \
until curl -sf $VLLM_URL/models > /dev/null 2>&1; do sleep 3; done && \
echo 'vLLM ready — starting experiments' && \
$PYTHON simulation/run_all_parallel.py 2>&1 | tee logs/run_vllm_\$(date +%Y%m%d_%H%M%S).log" \
    Enter

# Focus on runner window when user attaches
tmux select-window -t "$SESSION:runner"

echo ""
echo "tmux session '$SESSION' started in background."
echo ""
echo "  Attach :  tmux attach -t $SESSION"
echo "  Windows:  Ctrl-b 0 = vLLM server"
echo "            Ctrl-b 1 = experiment runner"
echo "  Detach :  Ctrl-b d"
