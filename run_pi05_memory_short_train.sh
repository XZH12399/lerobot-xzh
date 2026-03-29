#!/usr/bin/env bash
set -euo pipefail

detect_conda_sh() {
  local candidates=(
    "${CONDA_SH:-}"
    "$HOME/.conda/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

default_project_dir() {
  local candidates=(
    "$HOME/XZH/project/lerobot-xzh"
    "$HOME/projects/lerobot-xzh"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$HOME/XZH/project/lerobot-xzh"
}

CONDA_SH="${CONDA_SH:-$(detect_conda_sh || true)}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
PROJECT_DIR="${PROJECT_DIR:-$(default_project_dir)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/pi05_family}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/pi05_family}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-$HOME/.codex_runtime}"

HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"

POLICY_TYPE="${POLICY_TYPE:-pi05_memory}"
POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-lerobot/pi05_libero_base}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/sda/xzh/huggingface/lerobot/HuggingFaceVLA/libero}"

JOB_NAME="${JOB_NAME:-pi05_memory_libero10_shorttrain}"
RUN_MODE="${RUN_MODE:-bg}"
DRY_RUN="${DRY_RUN:-0}"

STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-250}"
LOG_FREQ="${LOG_FREQ:-10}"
EVAL_FREQ="${EVAL_FREQ:-0}"

POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
POLICY_REPO_ID="${POLICY_REPO_ID:-XZH/pi05-memory-libero10-shorttrain}"
USE_HISTORY_MEMORY="${USE_HISTORY_MEMORY:-true}"
MEMORY_SIZE="${MEMORY_SIZE:-16}"
MEMORY_FUSION="${MEMORY_FUSION:-gated}"
RESET_MEMORY_ON_NEW_EPISODE="${RESET_MEMORY_ON_NEW_EPISODE:-true}"
USE_TIME_EMBEDDING_IN_MEMORY="${USE_TIME_EMBEDDING_IN_MEMORY:-true}"
MEMORY_RESIDUAL_SCALE_INIT="${MEMORY_RESIDUAL_SCALE_INIT:-0.0}"
MEMORY_RETRIEVAL_LAYERS="${MEMORY_RETRIEVAL_LAYERS:-2}"
MEMORY_CONSOLIDATE_TYPE="${MEMORY_CONSOLIDATE_TYPE:-fifo}"
MEMORY_UPDATE_FUSED="${MEMORY_UPDATE_FUSED:-false}"
MEMORY_TRAINING_LAYOUT="${MEMORY_TRAINING_LAYOUT:-batch_sorted}"
TRAINING_HISTORY_MODE="${TRAINING_HISTORY_MODE:-batch_fifo}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/${JOB_NAME}_${TS}}"
LOG_FILE="${LOG_FILE:-$LOG_ROOT/${JOB_NAME}_${TS}.log}"

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH_EXTRA:${PYTHONPATH:-}"

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "$HF_TOKEN" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE")"
fi

if [[ -n "$HF_TOKEN" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

CMD=(
  python -m lerobot.scripts.lerobot_train
  --policy.type="$POLICY_TYPE"
  --policy.pretrained_path="$POLICY_PRETRAINED_PATH"
  --dataset.repo_id="$DATASET_REPO_ID"
  --output_dir="$OUT_DIR"
  --job_name="$JOB_NAME"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --eval_freq="$EVAL_FREQ"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --policy.dtype="$POLICY_DTYPE"
  --policy.device="$POLICY_DEVICE"
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
  --policy.push_to_hub="$PUSH_TO_HUB"
  --policy.repo_id="$POLICY_REPO_ID"
  --policy.use_history_memory="$USE_HISTORY_MEMORY"
  --policy.memory_size="$MEMORY_SIZE"
  --policy.memory_fusion="$MEMORY_FUSION"
  --policy.reset_memory_on_new_episode="$RESET_MEMORY_ON_NEW_EPISODE"
  --policy.use_time_embedding_in_memory="$USE_TIME_EMBEDDING_IN_MEMORY"
  --policy.memory_residual_scale_init="$MEMORY_RESIDUAL_SCALE_INIT"
  --policy.memory_retrieval_layers="$MEMORY_RETRIEVAL_LAYERS"
  --policy.memory_consolidate_type="$MEMORY_CONSOLIDATE_TYPE"
  --policy.memory_update_fused="$MEMORY_UPDATE_FUSED"
  --policy.memory_training_layout="$MEMORY_TRAINING_LAYOUT"
  --policy.training_history_mode="$TRAINING_HISTORY_MODE"
  --wandb.enable=false
)

if [[ -n "$DATASET_ROOT" ]]; then
  CMD+=(--dataset.root="$DATASET_ROOT")
fi

if (($# > 0)); then
  CMD+=("$@")
fi

echo "[INFO] project: $PROJECT_DIR"
echo "[INFO] output: $OUT_DIR"
echo "[INFO] log: $LOG_FILE"
echo "[INFO] run_mode: $RUN_MODE"
echo "[INFO] command:"
printf ' %q' "${CMD[@]}"
printf '\n'

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
