#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export STEPS="${STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export EVAL_EPISODES="${EVAL_EPISODES:-10}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
export USE_RESIDUAL_BASE="${USE_RESIDUAL_BASE:-true}"
export BASE_ALPHA_INIT="${BASE_ALPHA_INIT:-0.1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pi0_neurips_baseline}"
export JOB_NAME="${JOB_NAME:-pi0_residual_expB}"
export OUT_DIR="${OUT_DIR:-/mnt/sda/xzh/lerobot_outputs/${JOB_NAME}_${TS}}"
export LOG_FILE="${LOG_FILE:-$HOME/projects/lerobot/logs/${JOB_NAME}_${TS}.log}"
exec "$HOME/projects/lerobot/run_pi0_residual_pusht_sanity.sh" "$@"
