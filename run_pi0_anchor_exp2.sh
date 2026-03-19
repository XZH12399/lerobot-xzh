#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export STEPS="${STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export EVAL_EPISODES="${EVAL_EPISODES:-10}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
export USE_COARSE_PRIOR="${USE_COARSE_PRIOR:-true}"
export COARSE_DETACH="${COARSE_DETACH:-true}"
export LAMBDA_PRIOR="${LAMBDA_PRIOR:-0.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-pi0_neurips_baseline}"
export JOB_NAME="${JOB_NAME:-pi0_anchor_exp2}"
export OUT_DIR="${OUT_DIR:-/mnt/sda/xzh/lerobot_outputs/${JOB_NAME}_${TS}}"
export LOG_FILE="${LOG_FILE:-$HOME/projects/lerobot/logs/${JOB_NAME}_${TS}.log}"
exec "$HOME/projects/lerobot/run_pi0_anchor_pusht_sanity.sh" "$@"
