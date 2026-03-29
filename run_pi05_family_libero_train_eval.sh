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

default_dataset_root() {
  local candidates=(
    "$HOME/data/libero"
    "$HOME/.cache/huggingface/lerobot/HuggingFaceVLA/libero"
    "/mnt/sda/xzh/huggingface/lerobot/HuggingFaceVLA/libero"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$HOME/data/libero"
}

setup_common_env() {
  source "$CONDA_SH"
  conda activate "$ENV_NAME"

  export HF_ENDPOINT
  export CUDA_VISIBLE_DEVICES
  export PYTORCH_CUDA_ALLOC_CONF
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
  export OMP_NUM_THREADS
  export MKL_NUM_THREADS
  export OPENBLAS_NUM_THREADS
  export NUMEXPR_NUM_THREADS
  export TOKENIZERS_PARALLELISM

  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:${LD_LIBRARY_PATH:-}"
  fi
  export __EGL_VENDOR_LIBRARY_DIRS="$HOME/.compat/nvidia470_egl/share/glvnd/egl_vendor.d"
  unset PYOPENGL_PLATFORM || true
  export MUJOCO_GL=egl
}

build_train_cmd() {
  local -n out_cmd_ref="$1"
  if [[ "$MODEL_VARIANT" == "pi05" ]]; then
    out_cmd_ref=(
      python -m lerobot.scripts.lerobot_train
      --policy.path="$PI05_BASE_POLICY_PATH"
      --policy.push_to_hub=false
      --dataset.repo_id="$DATASET_REPO_ID"
      --dataset.root="$DATASET_ROOT"
      --output_dir="$RUN_DIR"
      --job_name="$JOB_NAME"
      --steps="$TRAIN_STEPS"
      --batch_size="$BATCH_SIZE"
      --num_workers="$NUM_WORKERS"
      --eval_freq=0
      --save_freq="$SAVE_FREQ"
      --log_freq="$LOG_FREQ"
      --policy.device="$POLICY_DEVICE"
      --policy.dtype="$POLICY_DTYPE"
      --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
      --wandb.enable=false
    )
    return 0
  fi

  if [[ "$MODEL_VARIANT" == "pi05_spatial" ]]; then
    out_cmd_ref=(
      python -m lerobot.scripts.lerobot_train
      --policy.type=pi05_spatial
      --policy.pretrained_path="$PI05_SPATIAL_PRETRAINED_PATH"
      --policy.push_to_hub=false
      --dataset.repo_id="$DATASET_REPO_ID"
      --dataset.root="$DATASET_ROOT"
      --output_dir="$RUN_DIR"
      --job_name="$JOB_NAME"
      --steps="$TRAIN_STEPS"
      --batch_size="$BATCH_SIZE"
      --num_workers="$NUM_WORKERS"
      --eval_freq=0
      --save_freq="$SAVE_FREQ"
      --log_freq="$LOG_FREQ"
      --policy.device="$POLICY_DEVICE"
      --policy.dtype="$POLICY_DTYPE"
      --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
      --wandb.enable=false
    )
    return 0
  fi

  echo "[ERROR] unsupported MODEL_VARIANT=$MODEL_VARIANT"
  return 1
}

CONDA_SH="${CONDA_SH:-$(detect_conda_sh || true)}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
PROJECT_DIR="${PROJECT_DIR:-$(default_project_dir)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/pi05_family}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/eval_outputs/pi05_family}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/pi05_family}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_VARIANT="${MODEL_VARIANT:-pi05}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
DATASET_ROOT="${DATASET_ROOT:-$(default_dataset_root)}"
BENCHMARK_TASK="${BENCHMARK_TASK:-libero_90}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-$TRAIN_STEPS}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-20}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"

PI05_BASE_POLICY_PATH="${PI05_BASE_POLICY_PATH:-lerobot/pi05_libero_base}"
PI05_SPATIAL_PRETRAINED_PATH="${PI05_SPATIAL_PRETRAINED_PATH:-lerobot/pi05_libero_base}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-${MODEL_VARIANT}_${BENCHMARK_TASK}_step$(printf '%06d' "$TRAIN_STEPS")_${RUN_TAG}}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$JOB_NAME}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_ROOT/${JOB_NAME}.train.log}"
EVAL_DIR="${EVAL_DIR:-$EVAL_ROOT/${JOB_NAME}_eval_${BENCHMARK_TASK}_ep${EVAL_EPISODES}}"
EVAL_LOG="${EVAL_LOG:-$LOG_ROOT/${JOB_NAME}.eval.log}"

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found."
  exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] project dir not found: $PROJECT_DIR"
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] dataset root not found: $DATASET_ROOT"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

setup_common_env
cd "$PROJECT_DIR"

echo "[INFO] model_variant=$MODEL_VARIANT"
echo "[INFO] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[INFO] dataset_root=$DATASET_ROOT"
echo "[INFO] benchmark_task=$BENCHMARK_TASK"
echo "[INFO] train_steps=$TRAIN_STEPS"
echo "[INFO] save_freq=$SAVE_FREQ"
echo "[INFO] num_workers=$NUM_WORKERS"
echo "[INFO] omp_num_threads=$OMP_NUM_THREADS"
echo "[INFO] mkl_num_threads=$MKL_NUM_THREADS"
echo "[INFO] openblas_num_threads=$OPENBLAS_NUM_THREADS"
echo "[INFO] numexpr_num_threads=$NUMEXPR_NUM_THREADS"
echo "[INFO] tokenizers_parallelism=$TOKENIZERS_PARALLELISM"
echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] train_log=$TRAIN_LOG"
echo "[INFO] eval_dir=$EVAL_DIR"
echo "[INFO] eval_log=$EVAL_LOG"

train_cmd=()
build_train_cmd train_cmd

printf '[TRAIN] cmd:'
printf ' %q' "${train_cmd[@]}"
printf '\n'

"${train_cmd[@]}" 2>&1 | tee "$TRAIN_LOG"

step_id="$(printf '%06d' "$TRAIN_STEPS")"
CKPT_POLICY_DIR="$RUN_DIR/checkpoints/$step_id/pretrained_model"
required_ckpt_files=(
  "$CKPT_POLICY_DIR/config.json"
  "$CKPT_POLICY_DIR/model.safetensors"
  "$CKPT_POLICY_DIR/policy_preprocessor.json"
  "$CKPT_POLICY_DIR/policy_postprocessor.json"
)
for required_file in "${required_ckpt_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] missing checkpoint artifact: $required_file"
    exit 1
  fi
done

mkdir -p "$EVAL_DIR"

eval_cmd=(
  python -m lerobot.scripts.lerobot_eval
  --policy.path="$CKPT_POLICY_DIR"
  --policy.device="$POLICY_DEVICE"
  --policy.dtype="$POLICY_DTYPE"
  --env.type=libero
  --env.task="$BENCHMARK_TASK"
  --eval.batch_size="$EVAL_BATCH_SIZE"
  --eval.n_episodes="$EVAL_EPISODES"
  --output_dir="$EVAL_DIR"
  --job_name="${JOB_NAME}_${BENCHMARK_TASK}_ep${EVAL_EPISODES}"
)

printf '[EVAL] cmd:'
printf ' %q' "${eval_cmd[@]}"
printf '\n'

"${eval_cmd[@]}" 2>&1 | tee "$EVAL_LOG"

if [[ ! -f "$EVAL_DIR/eval_info.json" ]]; then
  echo "[ERROR] missing eval_info.json: $EVAL_DIR/eval_info.json"
  exit 1
fi

python - "$EVAL_DIR/eval_info.json" <<'PY'
import json
import sys
from pathlib import Path

eval_info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
overall = eval_info.get("overall", {})
print("[SUMMARY]", json.dumps({
    "pc_success": overall.get("pc_success"),
    "avg_sum_reward": overall.get("avg_sum_reward"),
    "n_episodes": overall.get("n_episodes"),
    "eval_s": overall.get("eval_s"),
}, ensure_ascii=False))
PY
