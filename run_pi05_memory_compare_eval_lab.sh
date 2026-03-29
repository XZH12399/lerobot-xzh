#!/usr/bin/env bash
set -euo pipefail

detect_conda_sh() {
  local candidates=(
    "${CONDA_SH:-}"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/.conda/etc/profile.d/conda.sh"
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
    "$HOME/projects/lerobot-xzh"
    "$HOME/XZH/project/lerobot-xzh"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$HOME/projects/lerobot-xzh"
}

CONDA_SH="${CONDA_SH:-$(detect_conda_sh || true)}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
PROJECT_DIR="${PROJECT_DIR:-$(default_project_dir)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/eval_outputs/pi05_family}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/pi05_family}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-$HOME/.codex_runtime}"

POLICY_PATH="${POLICY_PATH:-}"
ENV_TASK="${ENV_TASK:-libero_10}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
RESET_MEMORY_ON_NEW_EPISODE="${RESET_MEMORY_ON_NEW_EPISODE:-true}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-}"
RUN_TAG="${RUN_TAG:-}"
ONLY_CASE="${ONLY_CASE:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found."
  exit 1
fi

if [[ -z "$POLICY_PATH" ]]; then
  echo "[ERROR] POLICY_PATH is required. Point it to a pretrained_model directory or HF repo."
  exit 1
fi

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$(basename "$POLICY_PATH")"
  if [[ "$RUN_TAG" == "pretrained_model" ]]; then
    RUN_TAG="$(basename "$(dirname "$POLICY_PATH")")"
  fi
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export HF_ENDPOINT
export MUJOCO_GL
export PYOPENGL_PLATFORM
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH_EXTRA:${PYTHONPATH:-}"

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

COMMON_CMD=(
  python -m lerobot.scripts.lerobot_eval
  --policy.path="$POLICY_PATH"
  --policy.device="$POLICY_DEVICE"
  --policy.dtype="$POLICY_DTYPE"
  --policy.reset_memory_on_new_episode="$RESET_MEMORY_ON_NEW_EPISODE"
  --env.type=libero
  --env.task="$ENV_TASK"
  --eval.batch_size="$EVAL_BATCH_SIZE"
  --eval.n_episodes="$EVAL_EPISODES"
)

if [[ -n "$POLICY_N_ACTION_STEPS" ]]; then
  COMMON_CMD+=(--policy.n_action_steps="$POLICY_N_ACTION_STEPS")
fi

if [[ -n "$POLICY_EMPTY_CAMERAS" ]]; then
  COMMON_CMD+=(--policy.empty_cameras="$POLICY_EMPTY_CAMERAS")
fi

run_case() {
  local case_name="$1"
  local use_history="$2"
  local out_dir="$OUTPUT_ROOT/${RUN_TAG}_${case_name}"
  local log_file="$LOG_ROOT/${RUN_TAG}_${case_name}.log"

  mkdir -p "$out_dir"
  echo "[INFO] case=$case_name use_history_memory=$use_history"
  echo "[INFO] output=$out_dir"
  echo "[INFO] log=$log_file"

  (
    cd "$PROJECT_DIR"
    "${COMMON_CMD[@]}" \
      --output_dir="$out_dir" \
      --policy.use_history_memory="$use_history"
  ) 2>&1 | tee "$log_file"

  if [[ -f "$out_dir/eval_info.json" ]]; then
    python - <<PY
import json
from pathlib import Path
path = Path(r"$out_dir") / "eval_info.json"
info = json.loads(path.read_text())
overall = info.get("overall", {})
print("[SUMMARY] case=$case_name pc_success=", overall.get("pc_success"))
PY
  fi
}

echo "[INFO] project: $PROJECT_DIR"
echo "[INFO] policy_path: $POLICY_PATH"
echo "[INFO] env.task: $ENV_TASK"
echo "[INFO] eval.n_episodes: $EVAL_EPISODES"
echo "[INFO] eval.batch_size: $EVAL_BATCH_SIZE"
echo "[INFO] policy.n_action_steps: ${POLICY_N_ACTION_STEPS:-<checkpoint default>}"
echo "[INFO] policy.empty_cameras: ${POLICY_EMPTY_CAMERAS:-<checkpoint default>}"
echo "[INFO] run_tag: $RUN_TAG"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

case "${ONLY_CASE:-all}" in
  all)
    run_case "history_on" "true"
    run_case "history_off" "false"
    ;;
  history_on|on|true)
    run_case "history_on" "true"
    ;;
  history_off|off|false)
    run_case "history_off" "false"
    ;;
  *)
    echo "[ERROR] Unsupported ONLY_CASE=$ONLY_CASE"
    exit 1
    ;;
esac
