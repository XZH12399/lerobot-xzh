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

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

POLICY_TYPE="${POLICY_TYPE:-pi05_spatial}"
POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-lerobot/pi05_libero_base}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/HuggingFaceVLA/libero}"

JOB_NAME="${JOB_NAME:-pi05_spatial_libero_smoke}"
RUN_MODE="${RUN_MODE:-fg}"
DRY_RUN="${DRY_RUN:-0}"

STEPS="${STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-2}"
LOG_FREQ="${LOG_FREQ:-1}"
EVAL_FREQ="${EVAL_FREQ:-0}"

POLICY_DTYPE="${POLICY_DTYPE:-float32}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/$JOB_NAME}"

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

CMD=(
  python -m lerobot.scripts.lerobot_train
  --policy.type="$POLICY_TYPE"
  --policy.pretrained_path="$POLICY_PRETRAINED_PATH"
  --policy.push_to_hub="$PUSH_TO_HUB"
  --policy.device="$POLICY_DEVICE"
  --policy.dtype="$POLICY_DTYPE"
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --output_dir="$OUT_DIR"
  --job_name="$JOB_NAME"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --eval_freq="$EVAL_FREQ"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --wandb.enable=false
)

if (($# > 0)); then
  CMD+=("$@")
fi

echo "[INFO] project: $PROJECT_DIR"
echo "[INFO] output: $OUT_DIR"
echo "[INFO] run_mode: $RUN_MODE"
echo "[INFO] command:"
printf ' %q' "${CMD[@]}"
printf '\n'

cd "$PROJECT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$RUN_MODE" == "fg" || "${1:-}" == "--fg" ]]; then
  "${CMD[@]}"
else
  nohup "${CMD[@]}" >"$OUT_DIR.log" 2>&1 &
  PID=$!
  echo "[OK] started PID: $PID"
  echo "[OK] tail log: tail -f $OUT_DIR.log"
fi
