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

setup_eval_env() {
  source "$CONDA_SH"
  conda activate "$ENV_NAME"

  export HF_ENDPOINT
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:${LD_LIBRARY_PATH:-}"
  fi
  export __EGL_VENDOR_LIBRARY_DIRS="$HOME/.compat/nvidia470_egl/share/glvnd/egl_vendor.d"
  unset PYOPENGL_PLATFORM || true
  export MUJOCO_GL=egl
}

CONDA_SH="${CONDA_SH:-$(detect_conda_sh || true)}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
PROJECT_DIR="${PROJECT_DIR:-$(default_project_dir)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/pi05_family}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/eval_outputs/pi05_family}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/pi05_family}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
JOB_NAME="${JOB_NAME:-pi05_spatial_fixcfg_continue_20260327}"
STEP_RAW="${STEP:-10000}"
STEP_NUM=$((10#$STEP_RAW))
STEP_ID="$(printf '%06d' "$STEP_NUM")"
POLICY_DIR="${POLICY_DIR:-$OUTPUT_ROOT/$JOB_NAME/checkpoints/$STEP_ID/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
GPU_IDS_STR="${GPU_IDS:-0 1}"
GPU_IDS_STR="${GPU_IDS_STR//,/ }"
TASK_IDS_STR="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
TASK_IDS_STR="${TASK_IDS_STR//,/ }"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PARENT_NAME="${PARENT_NAME:-${JOB_NAME}_step${STEP_ID}_libero10_ep${EVAL_EPISODES}_parallel_${RUN_TAG}}"
PARENT_EVAL_DIR="${PARENT_EVAL_DIR:-$EVAL_ROOT/$PARENT_NAME}"
TASK_EVAL_ROOT="$PARENT_EVAL_DIR/tasks"
RUN_LOG_DIR="$LOG_ROOT/$PARENT_NAME"
LAUNCH_MANIFEST="$PARENT_EVAL_DIR/launch_manifest.tsv"
MERGED_INFO="$PARENT_EVAL_DIR/eval_info.json"
START_TS="$(date +%s)"

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found."
  exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] project dir not found: $PROJECT_DIR"
  exit 1
fi

if [[ ! -d "$POLICY_DIR" ]]; then
  echo "[ERROR] policy dir not found: $POLICY_DIR"
  exit 1
fi

required_policy_files=(
  "$POLICY_DIR/model.safetensors"
  "$POLICY_DIR/policy_preprocessor.json"
  "$POLICY_DIR/policy_postprocessor.json"
)
for required_file in "${required_policy_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] checkpoint is incomplete, missing: $required_file"
    exit 1
  fi
done

read -r -a GPU_IDS_ARR <<< "$GPU_IDS_STR"
read -r -a TASK_IDS_ARR <<< "$TASK_IDS_STR"

if [[ ${#GPU_IDS_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] no GPU ids configured."
  exit 1
fi

if [[ ${#TASK_IDS_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] no task ids configured."
  exit 1
fi

mkdir -p "$TASK_EVAL_ROOT" "$RUN_LOG_DIR"
printf 'task_id\tgpu\tpid\tstatus\ttask_dir\tlog_path\n' > "$LAUNCH_MANIFEST"

setup_eval_env
cd "$PROJECT_DIR"

echo "[INFO] project_dir=$PROJECT_DIR"
echo "[INFO] policy_dir=$POLICY_DIR"
echo "[INFO] parent_eval_dir=$PARENT_EVAL_DIR"
echo "[INFO] run_log_dir=$RUN_LOG_DIR"
echo "[INFO] gpu_ids=$GPU_IDS_STR"
echo "[INFO] task_ids=$TASK_IDS_STR"
echo "[INFO] eval_episodes_per_task=$EVAL_EPISODES"

declare -a PIDS=()
declare -a PID_TASKS=()
declare -a PID_GPUS=()

launch_task() {
  local task_id="$1"
  local gpu_id="$2"
  local task_label
  task_label="task_$(printf '%02d' "$task_id")"
  local task_dir="$TASK_EVAL_ROOT/$task_label"
  local task_log="$RUN_LOG_DIR/${task_label}.log"
  local child_job_name="${JOB_NAME}_step${STEP_ID}_${task_label}_libero10"

  mkdir -p "$task_dir"

  if [[ -f "$task_dir/eval_info.json" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task_id" "$gpu_id" "-" "skipped_existing" "$task_dir" "$task_log" >> "$LAUNCH_MANIFEST"
    echo "[SKIP] task_id=$task_id gpu=$gpu_id existing=$task_dir/eval_info.json"
    return 0
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    cmd=(
      python -m lerobot.scripts.lerobot_eval
      --policy.path="$POLICY_DIR"
      --policy.device="$POLICY_DEVICE"
      --policy.dtype="$POLICY_DTYPE"
      --env.type=libero
      --env.task=libero_10
      --env.task_ids="[$task_id]"
      --eval.batch_size="$EVAL_BATCH_SIZE"
      --eval.n_episodes="$EVAL_EPISODES"
      --output_dir="$task_dir"
      --job_name="$child_job_name"
    )

    printf '[TASK %02d] cmd:' "$task_id"
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
  ) > "$task_log" 2>&1 &

  local pid=$!
  PIDS+=("$pid")
  PID_TASKS+=("$task_id")
  PID_GPUS+=("$gpu_id")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task_id" "$gpu_id" "$pid" "launched" "$task_dir" "$task_log" >> "$LAUNCH_MANIFEST"
  echo "[LAUNCH] task_id=$task_id gpu=$gpu_id pid=$pid log=$task_log"
}

for idx in "${!TASK_IDS_ARR[@]}"; do
  task_id="${TASK_IDS_ARR[$idx]}"
  gpu_idx=$((idx % ${#GPU_IDS_ARR[@]}))
  gpu_id="${GPU_IDS_ARR[$gpu_idx]}"
  launch_task "$task_id" "$gpu_id"
done

failures=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  task_id="${PID_TASKS[$idx]}"
  gpu_id="${PID_GPUS[$idx]}"
  if wait "$pid"; then
    echo "[DONE] task_id=$task_id gpu=$gpu_id pid=$pid"
  else
    echo "[FAIL] task_id=$task_id gpu=$gpu_id pid=$pid"
    failures=$((failures + 1))
  fi
done

python - "$TASK_EVAL_ROOT" "$MERGED_INFO" "$START_TS" "$GPU_IDS_STR" "$TASK_IDS_STR" "$JOB_NAME" "$STEP_ID" "$EVAL_EPISODES" "$failures" <<'PY'
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

task_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
start_ts = float(sys.argv[3])
gpu_ids = [x for x in sys.argv[4].split() if x]
expected_task_ids = [int(x) for x in sys.argv[5].split() if x]
job_name = sys.argv[6]
step_id = sys.argv[7]
eval_episodes = int(sys.argv[8])
failures = int(sys.argv[9])

per_task = []
group_acc = defaultdict(lambda: {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": []})
missing = []

for task_id in expected_task_ids:
    info_path = task_root / f"task_{task_id:02d}" / "eval_info.json"
    if not info_path.exists():
        missing.append(task_id)
        continue

    payload = json.loads(info_path.read_text(encoding="utf-8"))
    items = payload.get("per_task") or []
    if not items:
        raise RuntimeError(f"per_task missing in {info_path}")

    entry = items[0]
    per_task.append(entry)
    group_name = entry.get("task_group", "libero_10")
    metrics = entry.get("metrics", {})

    for key in ("sum_rewards", "max_rewards", "successes", "video_paths"):
        value = metrics.get(key, [])
        if isinstance(value, list):
            group_acc[group_name][key].extend(value)
        elif value is not None:
            group_acc[group_name][key].append(value)

if failures or missing:
    status = {
        "status": "incomplete",
        "job_name": job_name,
        "step_id": step_id,
        "gpu_ids": gpu_ids,
        "expected_task_ids": expected_task_ids,
        "missing_task_ids": missing,
        "launcher_failures": failures,
        "task_eval_root": str(task_root),
    }
    out_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    raise SystemExit(1)

per_task.sort(key=lambda item: (item.get("task_group", ""), int(item.get("task_id", -1))))

def summarize(bucket):
    n = len(bucket["sum_rewards"])
    avg_sum_reward = sum(float(x) for x in bucket["sum_rewards"]) / n if n else None
    avg_max_reward = sum(float(x) for x in bucket["max_rewards"]) / n if n else None
    pc_success = 100.0 * sum(1.0 if bool(x) else 0.0 for x in bucket["successes"]) / len(bucket["successes"]) if bucket["successes"] else None
    return {
        "avg_sum_reward": avg_sum_reward,
        "avg_max_reward": avg_max_reward,
        "pc_success": pc_success,
        "n_episodes": n,
        "video_paths": list(bucket["video_paths"]),
    }

per_group = {group_name: summarize(bucket) for group_name, bucket in group_acc.items()}
overall_bucket = {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": []}
for bucket in group_acc.values():
    for key in overall_bucket:
        overall_bucket[key].extend(bucket[key])

overall = summarize(overall_bucket)
wall_time = time.time() - start_ts
overall["eval_s"] = wall_time
overall["eval_ep_s"] = wall_time / max(1, overall["n_episodes"])

merged = {
    "per_task": per_task,
    "per_group": per_group,
    "overall": overall,
    "parallel_meta": {
        "status": "completed",
        "job_name": job_name,
        "step_id": step_id,
        "gpu_ids": gpu_ids,
        "task_ids": expected_task_ids,
        "episodes_per_task": eval_episodes,
        "task_eval_root": str(task_root),
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
}
out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(json.dumps({"merged_eval_info": str(out_path), "overall": overall}, indent=2))
PY

echo "[DONE] merged_eval_info=$MERGED_INFO"