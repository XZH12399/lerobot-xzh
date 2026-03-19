#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/sda/xzh/lerobot_outputs}"
GPU="${GPU:-1}"
RUN_MODE="${RUN_MODE:-bg}"   # bg|fg

STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-pi0_minimal_ablation}"

EXP="${EXP:-exp1}"           # exp1|exp2|exp3
TS="$(date +%Y%m%d_%H%M%S)"

case "$EXP" in
  exp1)
    # 最小改动版: coarse prior + detach + lambda=0.1
    JOB_NAME="${JOB_NAME:-pi0_anchor_exp1_minimal_${TS}}"
    OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/pi0_anchor_exp1_${TS}}"
    LOG_FILE="${LOG_FILE:-$PROJECT_DIR/logs/pi0_anchor_exp1_${TS}.log}"
    CUDA_VISIBLE_DEVICES="$GPU" \
    STEPS="$STEPS" \
    SAVE_FREQ="$SAVE_FREQ" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRAIN_EXPERT_ONLY="$TRAIN_EXPERT_ONLY" \
    USE_COARSE_PRIOR="true" \
    COARSE_DETACH="true" \
    LAMBDA_PRIOR="0.1" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    JOB_NAME="$JOB_NAME" \
    OUT_DIR="$OUT_DIR" \
    LOG_FILE="$LOG_FILE" \
    RUN_MODE="$RUN_MODE" \
    bash "$PROJECT_DIR/run_pi0_anchor_pusht_sanity.sh"
    ;;
  exp2)
    # coarse start + no prior loss
    JOB_NAME="${JOB_NAME:-pi0_anchor_exp2_lambda0_${TS}}"
    OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/pi0_anchor_exp2_${TS}}"
    LOG_FILE="${LOG_FILE:-$PROJECT_DIR/logs/pi0_anchor_exp2_${TS}.log}"
    CUDA_VISIBLE_DEVICES="$GPU" \
    STEPS="$STEPS" \
    SAVE_FREQ="$SAVE_FREQ" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRAIN_EXPERT_ONLY="$TRAIN_EXPERT_ONLY" \
    USE_COARSE_PRIOR="true" \
    COARSE_DETACH="true" \
    LAMBDA_PRIOR="0.0" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    JOB_NAME="$JOB_NAME" \
    OUT_DIR="$OUT_DIR" \
    LOG_FILE="$LOG_FILE" \
    RUN_MODE="$RUN_MODE" \
    bash "$PROJECT_DIR/run_pi0_anchor_pusht_sanity.sh"
    ;;
  exp3)
    # 原始 pi0 baseline
    JOB_NAME="${JOB_NAME:-pi0_exp3_baseline_${TS}}"
    OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/pi0_exp3_${TS}}"
    LOG_FILE="${LOG_FILE:-$PROJECT_DIR/logs/pi0_exp3_${TS}.log}"
    CUDA_VISIBLE_DEVICES="$GPU" \
    STEPS="$STEPS" \
    SAVE_FREQ="$SAVE_FREQ" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRAIN_EXPERT_ONLY="$TRAIN_EXPERT_ONLY" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    JOB_NAME="$JOB_NAME" \
    OUT_DIR="$OUT_DIR" \
    LOG_FILE="$LOG_FILE" \
    RUN_MODE="$RUN_MODE" \
    bash "$PROJECT_DIR/run_pi0_pusht_sanity.sh"
    ;;
  *)
    echo "[ERROR] Unknown EXP=$EXP (expected: exp1|exp2|exp3)"
    exit 1
    ;;
esac
