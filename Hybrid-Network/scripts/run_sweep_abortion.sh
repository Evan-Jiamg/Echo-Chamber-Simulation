#!/bin/bash
# Sweep: N=50, alpha={0,0.25,0.5,0.75,1.0}, scale_free, T=35, seeds=1-10
# Topic: abortion | LLM: phi4 via Ollama (port 11434)

set -euo pipefail

# Project root from this script's location, so the repository runs
# wherever it is cloned.
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Override for a specific environment:  PYTHON=/path/to/python ./this.sh
PYTHON="${PYTHON:-python3}"
LOG="$PROJ/sweep_abortion.log"

STEPS=35
N=50
K=5
NETWORK=scale_free
TOPIC=abortion
BG_LABEL=phi4
MODEL=phi4
SEEDS=(1 2 3 4 5)

mkdir -p "$PROJ/experiments"

echo "========================================" | tee -a "$LOG"
echo "Sweep started: $(date)" | tee -a "$LOG"
echo "Seeds: 1-10 | Alphas: 0 0.25 0.5 0.75 1.0 | N=$N K=$K T=$STEPS" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

cd "$PROJ/simulation"

# Phase 1: alpha=1.0 (pure numeric, no LLM) — run all seeds in parallel
echo "" | tee -a "$LOG"
echo "[Phase 1] alpha=1.0 — all seeds in parallel" | tee -a "$LOG"
for SEED in "${SEEDS[@]}"; do
    echo "  Launching seed=$SEED alpha=1.0" | tee -a "$LOG"
    $PYTHON run_hybrid.py \
        --network_type $NETWORK \
        --K $K \
        --alpha 1.0 \
        --num_agents $N \
        --step_count $STEPS \
        --seed $SEED \
        --topic $TOPIC \
        --gpt_model $MODEL \
        --backgrounds_label $BG_LABEL \
        >> "$LOG" 2>&1 &
done
wait
echo "[Phase 1] Done: $(date)" | tee -a "$LOG"

# Phase 2: LLM alphas — sequential to avoid Ollama contention
echo "" | tee -a "$LOG"
echo "[Phase 2] LLM alphas — sequential" | tee -a "$LOG"
for SEED in "${SEEDS[@]}"; do
    for ALPHA in 0.75 0.5 0.25 0; do
        echo "  seed=$SEED alpha=$ALPHA — $(date)" | tee -a "$LOG"
        $PYTHON run_hybrid.py \
            --network_type $NETWORK \
            --K $K \
            --alpha $ALPHA \
            --num_agents $N \
            --step_count $STEPS \
            --seed $SEED \
            --topic $TOPIC \
            --gpt_model $MODEL \
            --backgrounds_label $BG_LABEL \
            >> "$LOG" 2>&1
        echo "  Done: $(date)" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Sweep finished: $(date)" | tee -a "$LOG"
echo "Results in: $PROJ/experiments/$NETWORK/$TOPIC/" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
