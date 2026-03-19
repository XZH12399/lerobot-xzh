#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/lerobot}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-$HOME/.codex_runtime}"

HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"

POLICY_PATH="${POLICY_PATH:-$HOME/.cache/huggingface/hub/models--lerobot--pi05_libero_base/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/sda/xzh/huggingface/lerobot/HuggingFaceVLA/libero}"
ENV_TASK="${ENV_TASK:-libero_10}"

JOB_NAME="${JOB_NAME:-pi05_libero10_smoke}"
RUN_MODE="${RUN_MODE:-bg}"
DRY_RUN="${DRY_RUN:-0}"

STEPS="${STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
EVAL_FREQ="${EVAL_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
LOG_FREQ="${LOG_FREQ:-10}"

POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
POLICY_REPO_ID="${POLICY_REPO_ID:-XZH/pi05-libero10-smoke}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs/pi05_family/${JOB_NAME}_${TS}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/pi05_family}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${JOB_NAME}_${TS}.log}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found: $CONDA_SH"
  exit 1
fi

mkdir -p "$PROJECT_DIR/outputs/pi05_family" "$LOG_DIR"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export HF_ENDPOINT
export MUJOCO_GL
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PYTHONPATH_EXTRA:${PYTHONPATH:-}"

if [[ -z "$HF_TOKEN" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d ' 	
' < "$HF_TOKEN_FILE")"
fi

if [[ -n "$HF_TOKEN" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  echo "[INFO] HF token loaded."
else
  echo "[WARN] HF token not set. Gated model downloads may fail."
fi

CMD=(
  lerobot-train
  --policy.path="$POLICY_PATH"
  --policy.push_to_hub="$PUSH_TO_HUB"
  --dataset.repo_id="$DATASET_REPO_ID"
  --env.type=libero
  --env.task="$ENV_TASK"
  --output_dir="$OUT_DIR"
  --job_name="$JOB_NAME"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --eval.batch_size="$EVAL_BATCH_SIZE"
  --eval.n_episodes="$EVAL_EPISODES"
  --eval_freq="$EVAL_FREQ"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --policy.dtype="$POLICY_DTYPE"
  --policy.device="$POLICY_DEVICE"
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
  --wandb.enable=false
)

if [[ -n "$DATASET_ROOT" ]]; then
  CMD+=(--dataset.root="$DATASET_ROOT")
fi

if [[ "$PUSH_TO_HUB" == "true" ]]; then
  CMD+=(--policy.repo_id="$POLICY_REPO_ID")
fi

echo "[INFO] project: $PROJECT_DIR"
echo "[INFO] env: $ENV_NAME"
echo "[INFO] HF_ENDPOINT: $HF_ENDPOINT"
echo "[INFO] MUJOCO_GL: $MUJOCO_GL"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "[INFO] policy.path: $POLICY_PATH"
echo "[INFO] dataset.repo_id: $DATASET_REPO_ID"
echo "[INFO] dataset.root: $DATASET_ROOT"
echo "[INFO] env.task: $ENV_TASK"
echo "[INFO] output: $OUT_DIR"
echo "[INFO] log: $LOG_FILE"
echo "[INFO] run_mode: $RUN_MODE"
echo "[INFO] command:"
printf ' %q' "${CMD[@]}"
printf '
'

cd "$PROJECT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$RUN_MODE" == "fg" || "${1:-}" == "--fg" ]]; then
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "[OK] started PID: $PID"
  echo "[OK] tail log: tail -f $LOG_FILE"
fi
