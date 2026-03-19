#!/usr/bin/env bash
set -euo pipefail

# Eval script for LeRobot pi0/pi0_anchor PushT checkpoints.
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/lerobot}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-/mnt/sda/xzh/lerobot_outputs}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/lerobot/hf_token}"
FORCE_TASK_TEXT="${FORCE_TASK_TEXT:-}"
POLICY_FAMILY="${POLICY_FAMILY:-pi0_anchor}" # pi0 | pi0_anchor

TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-005000}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
JOB_NAME="${JOB_NAME:-pi0_anchor_pusht_ckpt_eval}"

RUN_MODE="${RUN_MODE:-bg}" # bg|fg

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found: $CONDA_SH"
  exit 1
fi

source "$CONDA_SH"
conda activate "$ENV_NAME"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs/eval"

if [[ -z "$TRAIN_RUN_DIR" ]]; then
  TRAIN_RUN_DIR="$(ls -1dt "$TRAIN_OUTPUT_ROOT"/${POLICY_FAMILY}_pusht_* "$PROJECT_DIR"/outputs/${POLICY_FAMILY}_pusht_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$TRAIN_RUN_DIR" || ! -d "$TRAIN_RUN_DIR" ]]; then
  echo "[ERROR] Could not find training run dir for policy family '$POLICY_FAMILY'. Set TRAIN_RUN_DIR explicitly."
  exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$TRAIN_RUN_DIR/checkpoints/$CHECKPOINT_STEP/pretrained_model}"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[ERROR] checkpoint directory not found: $CHECKPOINT_DIR"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs/eval/${POLICY_FAMILY}_ckpt${CHECKPOINT_STEP}_${TS}}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/logs/eval_${POLICY_FAMILY}_ckpt${CHECKPOINT_STEP}_${TS}.log}"

export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ -n "$FORCE_TASK_TEXT" ]]; then
  export LEROBOT_FORCE_TASK_TEXT="$FORCE_TASK_TEXT"
fi

# Load token from file when env var is not provided.
if [[ -z "$HF_TOKEN" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d " \t\r\n" < "$HF_TOKEN_FILE")"
fi

if [[ -n "$HF_TOKEN" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  echo "[INFO] HF token loaded."
else
  echo "[WARN] HF token not set."
  echo "[WARN] Set HF_TOKEN or put token in $HF_TOKEN_FILE"
fi

CMD=(
  lerobot-eval
  --policy.path="$CHECKPOINT_DIR"
  --env.type=pusht
  --eval.batch_size="$EVAL_BATCH_SIZE"
  --eval.n_episodes="$EVAL_EPISODES"
  --policy.device=cuda
  --policy.use_amp=false
  --output_dir="$OUT_DIR"
  --job_name="$JOB_NAME"
)

echo "[INFO] project: $PROJECT_DIR"
echo "[INFO] train output root: $TRAIN_OUTPUT_ROOT"
echo "[INFO] policy family: $POLICY_FAMILY"
echo "[INFO] train run: $TRAIN_RUN_DIR"
echo "[INFO] checkpoint: $CHECKPOINT_DIR"
echo "[INFO] env: $ENV_NAME"
echo "[INFO] HF_ENDPOINT: $HF_ENDPOINT"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "[INFO] TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "[INFO] FORCE_TASK_TEXT: ${FORCE_TASK_TEXT:-<empty>}"
echo "[INFO] output: $OUT_DIR"
echo "[INFO] log: $LOG_FILE"

cd "$PROJECT_DIR"

if [[ "$RUN_MODE" == "fg" || "${1:-}" == "--fg" ]]; then
  echo "[INFO] running eval in foreground"
  exec "${CMD[@]}"
else
  echo "[INFO] running eval in background"
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "[OK] started PID: $PID"
  echo "[OK] tail log: tail -f $LOG_FILE"
  echo "[OK] video dir (after finish): $OUT_DIR/videos"
fi
