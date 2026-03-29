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

setup_common_env() {
  source "$CONDA_SH"
  conda activate "$ENV_NAME"

  export HF_ENDPOINT
  export CUDA_VISIBLE_DEVICES
  export PYTORCH_CUDA_ALLOC_CONF
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
}

setup_eval_env() {
  setup_common_env
  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:${LD_LIBRARY_PATH:-}"
  fi
  export __EGL_VENDOR_LIBRARY_DIRS="$HOME/.compat/nvidia470_egl/share/glvnd/egl_vendor.d"
  unset PYOPENGL_PLATFORM || true
  export MUJOCO_GL=egl
}

record_eval_summary() {
  local step="$1"
  local eval_info="$2"
  python - "$SUMMARY_FILE" "$step" "$eval_info" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
step = int(sys.argv[2])
eval_info_path = Path(sys.argv[3])
info = json.loads(eval_info_path.read_text(encoding='utf-8'))
overall = info.get('overall', {})
rows = []
if summary_path.exists():
    rows = [line.rstrip('\n').split('\t') for line in summary_path.read_text(encoding='utf-8').splitlines() if line.strip()]
header = ['step', 'pc_success', 'avg_sum_reward', 'eval_s', 'eval_dir']
body = {}
for row in rows[1:]:
    if not row:
        continue
    body[row[0]] = row
body[str(step)] = [
    str(step),
    str(overall.get('pc_success')),
    str(overall.get('avg_sum_reward')),
    str(overall.get('eval_s')),
    str(eval_info_path.parent),
]
ordered = [header] + [body[k] for k in sorted(body, key=lambda x: int(x))]
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text('\n'.join('\t'.join(row) for row in ordered) + '\n', encoding='utf-8')
print('[SUMMARY]', ordered[-1])
PY
}

run_train_stage() {
  local target_step="$1"
  local previous_step="${2:-}"
  local stage_log="$LOG_ROOT/${JOB_NAME}_train_to_$(printf '%06d' "$target_step").log"

  echo "[TRAIN] target_step=$target_step previous_step=${previous_step:-none}"
  echo "[TRAIN] log=$stage_log"

  setup_common_env
  cd "$PROJECT_DIR"

  local cmd=()
  if [[ -z "$previous_step" ]]; then
    cmd=(
      python -m lerobot.scripts.lerobot_train
      --policy.path="$BASE_POLICY_PATH"
      --dataset.repo_id="$DATASET_REPO_ID"
      --dataset.root="$DATASET_ROOT"
      --output_dir="$RUN_DIR"
      --job_name="$JOB_NAME"
      --steps="$target_step"
      --batch_size="$BATCH_SIZE"
      --num_workers="$NUM_WORKERS"
      --eval_freq=0
      --save_freq="$target_step"
      --log_freq="$LOG_FREQ"
      --policy.device="$POLICY_DEVICE"
      --policy.dtype="$POLICY_DTYPE"
      --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
      --policy.push_to_hub=false
      --wandb.enable=false
    )
  else
    local prev_id
    prev_id="$(printf '%06d' "$previous_step")"
    local config_path="$RUN_DIR/checkpoints/$prev_id/pretrained_model/train_config.json"
    if [[ ! -f "$config_path" ]]; then
      echo "[ERROR] resume config missing: $config_path"
      return 1
    fi
    cmd=(
      python -m lerobot.scripts.lerobot_train
      --config_path="$config_path"
      --resume=true
      --steps="$target_step"
      --batch_size="$BATCH_SIZE"
      --num_workers="$NUM_WORKERS"
      --eval_freq=0
      --save_freq="$target_step"
      --log_freq="$LOG_FREQ"
      --policy.device="$POLICY_DEVICE"
      --policy.dtype="$POLICY_DTYPE"
      --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
      --wandb.enable=false
    )
  fi

  printf '[TRAIN] cmd:' | tee -a "$stage_log"
  printf ' %q' "${cmd[@]}" | tee -a "$stage_log"
  printf '\n' | tee -a "$stage_log"

  "${cmd[@]}" 2>&1 | tee -a "$stage_log"
}

run_eval_stage() {
  local target_step="$1"
  local step_id
  step_id="$(printf '%06d' "$target_step")"
  local ckpt_policy_dir="$RUN_DIR/checkpoints/$step_id/pretrained_model"
  local eval_dir="$EVAL_ROOT/${JOB_NAME}_step${step_id}_libero10_ep${EVAL_EPISODES}"
  local eval_log="$LOG_ROOT/${JOB_NAME}_eval_${step_id}.log"

  if [[ ! -d "$ckpt_policy_dir" ]]; then
    echo "[ERROR] checkpoint policy dir missing: $ckpt_policy_dir"
    return 1
  fi

  if [[ -f "$eval_dir/eval_info.json" ]]; then
    echo "[EVAL] existing eval found for step $step_id, reusing: $eval_dir/eval_info.json"
    record_eval_summary "$target_step" "$eval_dir/eval_info.json"
    return 0
  fi

  echo "[EVAL] target_step=$target_step"
  echo "[EVAL] output=$eval_dir"
  echo "[EVAL] log=$eval_log"

  mkdir -p "$eval_dir"
  setup_eval_env
  cd "$PROJECT_DIR"

  local cmd=(
    python -m lerobot.scripts.lerobot_eval
    --policy.path="$ckpt_policy_dir"
    --policy.device="$POLICY_DEVICE"
    --policy.dtype="$POLICY_DTYPE"
    --env.type=libero
    --env.task=libero_10
    --eval.batch_size="$EVAL_BATCH_SIZE"
    --eval.n_episodes="$EVAL_EPISODES"
    --output_dir="$eval_dir"
    --job_name="${JOB_NAME}_step${step_id}_libero10"
  )

  printf '[EVAL] cmd:' | tee -a "$eval_log"
  printf ' %q' "${cmd[@]}" | tee -a "$eval_log"
  printf '\n' | tee -a "$eval_log"

  "${cmd[@]}" 2>&1 | tee -a "$eval_log"

  if [[ ! -f "$eval_dir/eval_info.json" ]]; then
    echo "[ERROR] eval_info.json missing after eval: $eval_dir/eval_info.json"
    return 1
  fi

  record_eval_summary "$target_step" "$eval_dir/eval_info.json"
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

BASE_POLICY_PATH="${BASE_POLICY_PATH:-$PROJECT_DIR/outputs/pi05_family/pi05_spatial_coord_anchor_smoke_fixcfg_20260326/checkpoints/000002/pretrained_model}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/libero}"
JOB_NAME="${JOB_NAME:-pi05_spatial_fixcfg_continue_20260327}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$JOB_NAME}"
SUMMARY_FILE="${SUMMARY_FILE:-$EVAL_ROOT/${JOB_NAME}_stage_summary.tsv}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_FREQ="${LOG_FREQ:-20}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
STAGE_STEPS_STR="${STAGE_STEPS_STR:-1000 5000 10000 20000 50000 100000}"

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

if [[ ! -d "$BASE_POLICY_PATH" ]]; then
  echo "[ERROR] base policy path not found: $BASE_POLICY_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

IFS=' ' read -r -a STAGE_STEPS <<< "$STAGE_STEPS_STR"
if [[ ${#STAGE_STEPS[@]} -eq 0 ]]; then
  echo "[ERROR] no stage steps configured."
  exit 1
fi

if [[ ! -f "$SUMMARY_FILE" ]]; then
  printf 'step\tpc_success\tavg_sum_reward\teval_s\teval_dir\n' > "$SUMMARY_FILE"
fi

echo "[INFO] project=$PROJECT_DIR"
echo "[INFO] conda_sh=$CONDA_SH"
echo "[INFO] env_name=$ENV_NAME"
echo "[INFO] base_policy_path=$BASE_POLICY_PATH"
echo "[INFO] dataset_root=$DATASET_ROOT"
echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] eval_root=$EVAL_ROOT"
echo "[INFO] summary_file=$SUMMARY_FILE"
echo "[INFO] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[INFO] policy_dtype=$POLICY_DTYPE"
echo "[INFO] batch_size=$BATCH_SIZE"
echo "[INFO] stage_steps=${STAGE_STEPS[*]}"

previous_step=""
for target_step in "${STAGE_STEPS[@]}"; do
  step_id="$(printf '%06d' "$target_step")"
  ckpt_policy_dir="$RUN_DIR/checkpoints/$step_id/pretrained_model"

  if [[ -d "$ckpt_policy_dir" ]]; then
    echo "[SKIP] checkpoint already exists for step $step_id"
  else
    run_train_stage "$target_step" "$previous_step"
  fi

  run_eval_stage "$target_step"
  previous_step="$target_step"
done

echo "[DONE] all staged training/evaluation finished."
