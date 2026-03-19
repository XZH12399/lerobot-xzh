#!/usr/bin/env bash
set -euo pipefail

# Quick ablation for pi0_anchor: train + eval per setting.
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/sda/xzh/lerobot_outputs}"
AB_ROOT_BASE="${AB_ROOT_BASE:-$PROJECT_DIR/outputs/ablation}"

# Keep it quick by default.
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_EVAL_EPISODES="${TRAIN_EVAL_EPISODES:-10}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
FORCE_TASK_TEXT="${FORCE_TASK_TEXT:-Push the T shaped block to the target.}"
GPU_TRAIN="${GPU_TRAIN:-0}"

# space separated sigma:lambda pairs
MATRIX="${MATRIX:-0.03:1.0 0.10:1.0 0.30:0.1}"

TS="$(date +%Y%m%d_%H%M%S)"
AB_ROOT="${AB_ROOT_BASE}/pi0_anchor_quick_${TS}"
mkdir -p "$AB_ROOT" "$PROJECT_DIR/logs"

SUMMARY_CSV="$AB_ROOT/summary.csv"
echo "tag,prior_sigma,lambda_prior,train_run_dir,checkpoint_step,avg_sum_reward,avg_max_reward,pc_success,eval_info" > "$SUMMARY_CSV"

cd "$PROJECT_DIR"

for pair in $MATRIX; do
  sigma="${pair%%:*}"
  lambda="${pair##*:}"
  tag="s${sigma//./p}_l${lambda//./p}"

  run_dir="$OUTPUT_ROOT/pi0_anchor_ablate_${tag}_${TS}"
  train_log="$PROJECT_DIR/logs/pi0_anchor_ablate_${tag}_${TS}.log"

  echo "[ABLA] train $tag sigma=$sigma lambda=$lambda"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" \
  STEPS="$STEPS" \
  SAVE_FREQ="$SAVE_FREQ" \
  EVAL_EPISODES="$TRAIN_EVAL_EPISODES" \
  EVAL_BATCH_SIZE="$TRAIN_EVAL_BATCH_SIZE" \
  BATCH_SIZE="$BATCH_SIZE" \
  TRAIN_EXPERT_ONLY="$TRAIN_EXPERT_ONLY" \
  PRIOR_SIGMA="$sigma" \
  LAMBDA_PRIOR="$lambda" \
  JOB_NAME="pi0_anchor_ablate_${tag}" \
  OUT_DIR="$run_dir" \
  LOG_FILE="$train_log" \
  RUN_MODE="fg" \
  bash "$PROJECT_DIR/run_pi0_anchor_pusht_sanity.sh"

  ckpt_step=$(printf "%06d" "$STEPS")
  eval_out="$AB_ROOT/eval_${tag}_${TS}"
  eval_log="$PROJECT_DIR/logs/eval_pi0_anchor_ablate_${tag}_${TS}.log"

  echo "[ABLA] eval  $tag ckpt=$ckpt_step"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" \
  POLICY_FAMILY="pi0_anchor" \
  TRAIN_RUN_DIR="$run_dir" \
  CHECKPOINT_STEP="$ckpt_step" \
  EVAL_EPISODES="$EVAL_EPISODES" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  FORCE_TASK_TEXT="$FORCE_TASK_TEXT" \
  OUT_DIR="$eval_out" \
  LOG_FILE="$eval_log" \
  RUN_MODE="fg" \
  bash "$PROJECT_DIR/run_pi0_pusht_eval_ckpt.sh"

  eval_info="$eval_out/eval_info.json"
  metrics=$(python3 - <<PY
import json
p = "$eval_info"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)["overall"]
print(f"{d['avg_sum_reward']},{d['avg_max_reward']},{d['pc_success']}")
PY
)

  echo "$tag,$sigma,$lambda,$run_dir,$ckpt_step,$metrics,$eval_info" >> "$SUMMARY_CSV"
  echo "[ABLA] done $tag metrics=$metrics"
done

echo "[ABLA] all done"
echo "[ABLA] summary: $SUMMARY_CSV"
