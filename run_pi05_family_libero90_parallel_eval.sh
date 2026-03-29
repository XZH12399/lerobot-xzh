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
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:${LD_LIBRARY_PATH:-}"
  fi
  export __EGL_VENDOR_LIBRARY_DIRS="$HOME/.compat/nvidia470_egl/share/glvnd/egl_vendor.d"
  unset PYOPENGL_PLATFORM || true
  export MUJOCO_GL=egl
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

CONDA_SH="${CONDA_SH:-$(detect_conda_sh || true)}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
EVAL_MODULE="${EVAL_MODULE:-lerobot.scripts.lerobot_eval}"
PROJECT_DIR="${PROJECT_DIR:-$(default_project_dir)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/pi05_family}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/eval_outputs/pi05_family}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/pi05_family}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
JOB_NAME="${JOB_NAME:-pi05_spatial_fixcfg_continue_20260327}"
STEP_RAW="${STEP:-}"
if [[ -n "${POLICY_DIR:-}" && -z "$STEP_RAW" ]]; then
  derived_step="$(basename "$(dirname "$POLICY_DIR")")"
  if [[ "$derived_step" =~ ^[0-9]+$ ]]; then
    STEP_ID="$(printf '%06d' "$((10#$derived_step))")"
  else
    STEP_ID="manual"
  fi
else
  STEP_RAW="${STEP_RAW:-100000}"
  STEP_NUM=$((10#$STEP_RAW))
  STEP_ID="$(printf '%06d' "$STEP_NUM")"
fi
POLICY_DIR="${POLICY_DIR:-$OUTPUT_ROOT/$JOB_NAME/checkpoints/$STEP_ID/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
BENCHMARK_TASK="${BENCHMARK_TASK:-libero_90}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
GPU_IDS_STR="${GPU_IDS:-0 1}"
GPU_IDS_STR="${GPU_IDS_STR//,/ }"
GPU_IDS_STR="${GPU_IDS_STR//[/ }"
GPU_IDS_STR="${GPU_IDS_STR//]/ }"
TASK_IDS_STR="${TASK_IDS:-}"
TASK_IDS_STR="${TASK_IDS_STR//,/ }"
TASK_IDS_STR="${TASK_IDS_STR//[/ }"
TASK_IDS_STR="${TASK_IDS_STR//]/ }"
NUM_SHARDS="${NUM_SHARDS:-}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-}"
DRY_RUN="${DRY_RUN:-false}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PARENT_NAME="${PARENT_NAME:-${JOB_NAME}_step${STEP_ID}_${BENCHMARK_TASK}_ep${EVAL_EPISODES}_sharded_${RUN_TAG}}"
PARENT_EVAL_DIR="${PARENT_EVAL_DIR:-$EVAL_ROOT/$PARENT_NAME}"
SHARD_EVAL_ROOT="$PARENT_EVAL_DIR/shards"
RUN_LOG_DIR="$LOG_ROOT/$PARENT_NAME"
LAUNCH_MANIFEST="$PARENT_EVAL_DIR/launch_manifest.tsv"
SHARD_PLAN="$PARENT_EVAL_DIR/shard_plan.tsv"
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

mkdir -p "$PARENT_EVAL_DIR" "$SHARD_EVAL_ROOT" "$RUN_LOG_DIR"
printf 'shard_id\tgpu\tpid\tstatus\tshard_dir\tlog_path\ttask_ids\n' > "$LAUNCH_MANIFEST"

setup_eval_env
cd "$PROJECT_DIR"

if [[ -z "$TASK_IDS_STR" ]]; then
  TASK_IDS_STR="$(python - "$BENCHMARK_TASK" <<'PY'
import sys
from libero.libero import benchmark
suite_name = sys.argv[1]
bench = benchmark.get_benchmark_dict()
if suite_name not in bench:
    raise SystemExit(f'Unknown LIBERO suite: {suite_name}. Available: {sorted(bench)}')
suite = bench[suite_name]()
print(' '.join(str(i) for i in range(len(suite.tasks))))
PY
)"
fi

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

if [[ -z "$NUM_SHARDS" ]]; then
  NUM_SHARDS="${#GPU_IDS_ARR[@]}"
fi
if (( NUM_SHARDS < 1 )); then
  echo "[ERROR] NUM_SHARDS must be >= 1, got $NUM_SHARDS"
  exit 1
fi
if (( NUM_SHARDS > ${#TASK_IDS_ARR[@]} )); then
  NUM_SHARDS="${#TASK_IDS_ARR[@]}"
fi

if [[ -z "$MAX_CONCURRENT_JOBS" ]]; then
  if (( NUM_SHARDS < ${#GPU_IDS_ARR[@]} )); then
    MAX_CONCURRENT_JOBS="$NUM_SHARDS"
  else
    MAX_CONCURRENT_JOBS="${#GPU_IDS_ARR[@]}"
  fi
fi
if (( MAX_CONCURRENT_JOBS < 1 )); then
  echo "[ERROR] MAX_CONCURRENT_JOBS must be >= 1, got $MAX_CONCURRENT_JOBS"
  exit 1
fi
if (( MAX_CONCURRENT_JOBS > NUM_SHARDS )); then
  MAX_CONCURRENT_JOBS="$NUM_SHARDS"
fi

python - "$SHARD_PLAN" "$NUM_SHARDS" "${TASK_IDS_ARR[@]}" <<'PY'
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
num_shards = int(sys.argv[2])
task_ids = [int(x) for x in sys.argv[3:]]
base, extra = divmod(len(task_ids), num_shards)
cursor = 0
lines = ["shard_id\ttask_count\ttask_ids\ttask_ids_json"]
for shard_id in range(num_shards):
    shard_size = base + (1 if shard_id < extra else 0)
    shard_tasks = task_ids[cursor:cursor + shard_size]
    cursor += shard_size
    task_ids_space = ' '.join(str(x) for x in shard_tasks)
    task_ids_json = '[' + ','.join(str(x) for x in shard_tasks) + ']'
    lines.append(f"{shard_id}\t{len(shard_tasks)}\t{task_ids_space}\t{task_ids_json}")
out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out_path)
PY

mapfile -t SHARD_LINES < <(tail -n +2 "$SHARD_PLAN")

echo "[INFO] project_dir=$PROJECT_DIR"
echo "[INFO] policy_dir=$POLICY_DIR"
echo "[INFO] benchmark_task=$BENCHMARK_TASK"
echo "[INFO] parent_eval_dir=$PARENT_EVAL_DIR"
echo "[INFO] run_log_dir=$RUN_LOG_DIR"
echo "[INFO] gpu_ids=$GPU_IDS_STR"
echo "[INFO] task_ids=${TASK_IDS_ARR[*]}"
echo "[INFO] num_shards=$NUM_SHARDS"
echo "[INFO] max_concurrent_jobs=$MAX_CONCURRENT_JOBS"
echo "[INFO] eval_episodes_per_task=$EVAL_EPISODES"
echo "[INFO] dry_run=$DRY_RUN"

action_status=""
last_pid=""
launch_shard() {
  local shard_id="$1"
  local gpu_id="$2"
  local task_ids_space="$3"
  local task_ids_json="$4"
  local shard_label="shard_$(printf '%02d' "$shard_id")"
  local shard_dir="$SHARD_EVAL_ROOT/$shard_label"
  local shard_log="$RUN_LOG_DIR/${shard_label}.log"
  local child_job_name="${JOB_NAME}_step${STEP_ID}_${BENCHMARK_TASK}_${shard_label}"
  local cmd=(
    python -m "$EVAL_MODULE"
    --policy.path="$POLICY_DIR"
    --policy.device="$POLICY_DEVICE"
    --policy.dtype="$POLICY_DTYPE"
    --env.type=libero
    --env.task="$BENCHMARK_TASK"
    --env.task_ids="$task_ids_json"
    --eval.batch_size="$EVAL_BATCH_SIZE"
    --eval.n_episodes="$EVAL_EPISODES"
    --output_dir="$shard_dir"
    --job_name="$child_job_name"
  )

  action_status=""
  last_pid=""
  mkdir -p "$shard_dir"
  printf '%s\n' "$task_ids_space" > "$shard_dir/task_ids.txt"

  if [[ -f "$shard_dir/eval_info.json" ]]; then
    action_status="skipped_existing"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$shard_id" "$gpu_id" "-" "$action_status" "$shard_dir" "$shard_log" "$task_ids_space" >> "$LAUNCH_MANIFEST"
    echo "[SKIP] shard_id=$shard_id gpu=$gpu_id existing=$shard_dir/eval_info.json task_ids=$task_ids_space"
    return 0
  fi

  if is_truthy "$DRY_RUN"; then
    action_status="dry_run"
    {
      printf '[DRY_RUN SHARD %02d] task_ids=%s\n' "$shard_id" "$task_ids_space"
      printf '[DRY_RUN SHARD %02d] cmd:' "$shard_id"
      printf ' %q' "${cmd[@]}"
      printf '\n'
    } | tee "$shard_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$shard_id" "$gpu_id" "-" "$action_status" "$shard_dir" "$shard_log" "$task_ids_space" >> "$LAUNCH_MANIFEST"
    return 0
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    printf '[SHARD %02d] task_ids=%s\n' "$shard_id" "$task_ids_space"
    printf '[SHARD %02d] cmd:' "$shard_id"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  ) > "$shard_log" 2>&1 &

  last_pid=$!
  action_status="launched"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$shard_id" "$gpu_id" "$last_pid" "$action_status" "$shard_dir" "$shard_log" "$task_ids_space" >> "$LAUNCH_MANIFEST"
  echo "[LAUNCH] shard_id=$shard_id gpu=$gpu_id pid=$last_pid task_ids=$task_ids_space log=$shard_log"
}

failures=0
current=0
total_shards=${#SHARD_LINES[@]}
while (( current < total_shards )); do
  wave_pids=()
  wave_shards=()
  wave_gpus=()
  launched_in_wave=0

  while (( launched_in_wave < MAX_CONCURRENT_JOBS && current < total_shards )); do
    line="${SHARD_LINES[$current]}"
    IFS=$'\t' read -r shard_id task_count task_ids_space task_ids_json <<< "$line"
    gpu_idx=$((shard_id % ${#GPU_IDS_ARR[@]}))
    gpu_id="${GPU_IDS_ARR[$gpu_idx]}"

    launch_shard "$shard_id" "$gpu_id" "$task_ids_space" "$task_ids_json"
    if [[ "$action_status" == "launched" ]]; then
      wave_pids+=("$last_pid")
      wave_shards+=("$shard_id")
      wave_gpus+=("$gpu_id")
    fi

    current=$((current + 1))
    launched_in_wave=$((launched_in_wave + 1))
  done

  if is_truthy "$DRY_RUN"; then
    continue
  fi

  for idx in "${!wave_pids[@]}"; do
    pid="${wave_pids[$idx]}"
    shard_id="${wave_shards[$idx]}"
    gpu_id="${wave_gpus[$idx]}"
    if wait "$pid"; then
      echo "[DONE] shard_id=$shard_id gpu=$gpu_id pid=$pid"
    else
      echo "[FAIL] shard_id=$shard_id gpu=$gpu_id pid=$pid"
      failures=$((failures + 1))
    fi
  done
done

if is_truthy "$DRY_RUN"; then
  echo "[DRY_RUN] shard_plan=$SHARD_PLAN"
  echo "[DRY_RUN] launch_manifest=$LAUNCH_MANIFEST"
  exit 0
fi

python - "$SHARD_EVAL_ROOT" "$MERGED_INFO" "$START_TS" "$GPU_IDS_STR" "$TASK_IDS_STR" "$JOB_NAME" "$STEP_ID" "$EVAL_EPISODES" "$failures" "$BENCHMARK_TASK" "$NUM_SHARDS" "$MAX_CONCURRENT_JOBS" "$SHARD_PLAN" <<'PY'
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

shard_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
start_ts = float(sys.argv[3])
gpu_ids = [x for x in sys.argv[4].split() if x]
expected_task_ids = [int(x) for x in sys.argv[5].split() if x]
job_name = sys.argv[6]
step_id = sys.argv[7]
eval_episodes = int(sys.argv[8])
launcher_failures = int(sys.argv[9])
benchmark_task = sys.argv[10]
num_shards = int(sys.argv[11])
max_concurrent_jobs = int(sys.argv[12])
shard_plan_path = Path(sys.argv[13])
expected_task_set = set(expected_task_ids)

plan_lines = shard_plan_path.read_text(encoding='utf-8').splitlines()[1:]
expected_shards = []
for line in plan_lines:
    shard_id_str, _task_count_str, task_ids_space, _task_ids_json = line.split('\t', 3)
    planned_tasks = [int(x) for x in task_ids_space.split()] if task_ids_space else []
    expected_shards.append((int(shard_id_str), planned_tasks))

per_task_map = {}
duplicate_task_ids = []
missing_shards = []
group_acc = defaultdict(lambda: {'sum_rewards': [], 'max_rewards': [], 'successes': [], 'video_paths': []})
shard_summaries = []

for shard_id, planned_task_ids in expected_shards:
    info_path = shard_root / f'shard_{shard_id:02d}' / 'eval_info.json'
    if not info_path.exists():
        missing_shards.append({'shard_id': shard_id, 'task_ids': planned_task_ids, 'info_path': str(info_path)})
        continue

    payload = json.loads(info_path.read_text(encoding='utf-8'))
    shard_summaries.append({
        'shard_id': shard_id,
        'task_ids': planned_task_ids,
        'info_path': str(info_path),
        'overall': payload.get('overall', {}),
    })

    for entry in payload.get('per_task', []):
        task_id = int(entry.get('task_id'))
        if task_id in per_task_map:
            duplicate_task_ids.append(task_id)
            continue
        per_task_map[task_id] = entry
        group_name = entry.get('task_group', benchmark_task)
        metrics = entry.get('metrics', {})
        for key in ('sum_rewards', 'max_rewards', 'successes', 'video_paths'):
            value = metrics.get(key, [])
            if isinstance(value, list):
                group_acc[group_name][key].extend(value)
            elif value is not None:
                group_acc[group_name][key].append(value)

missing_task_ids = [task_id for task_id in expected_task_ids if task_id not in per_task_map]
extra_task_ids = sorted(task_id for task_id in per_task_map if task_id not in expected_task_set)

def summarize(bucket):
    n = len(bucket['sum_rewards'])
    avg_sum_reward = sum(float(x) for x in bucket['sum_rewards']) / n if n else None
    avg_max_reward = sum(float(x) for x in bucket['max_rewards']) / n if n else None
    pc_success = (
        100.0 * sum(1.0 if bool(x) else 0.0 for x in bucket['successes']) / len(bucket['successes'])
        if bucket['successes'] else None
    )
    return {
        'avg_sum_reward': avg_sum_reward,
        'avg_max_reward': avg_max_reward,
        'pc_success': pc_success,
        'n_episodes': n,
        'video_paths': list(bucket['video_paths']),
    }

if launcher_failures or missing_shards or missing_task_ids or duplicate_task_ids or extra_task_ids:
    status = {
        'status': 'incomplete',
        'job_name': job_name,
        'step_id': step_id,
        'benchmark_task': benchmark_task,
        'gpu_ids': gpu_ids,
        'num_shards': num_shards,
        'max_concurrent_jobs': max_concurrent_jobs,
        'expected_task_ids': expected_task_ids,
        'missing_task_ids': missing_task_ids,
        'extra_task_ids': extra_task_ids,
        'duplicate_task_ids': duplicate_task_ids,
        'missing_shards': missing_shards,
        'launcher_failures': launcher_failures,
        'shard_root': str(shard_root),
        'shard_plan': str(shard_plan_path),
    }
    out_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    raise SystemExit(1)

per_task = [per_task_map[task_id] for task_id in sorted(per_task_map)]
per_group = {group_name: summarize(bucket) for group_name, bucket in group_acc.items()}
overall_bucket = {'sum_rewards': [], 'max_rewards': [], 'successes': [], 'video_paths': []}
for bucket in group_acc.values():
    for key in overall_bucket:
        overall_bucket[key].extend(bucket[key])

overall = summarize(overall_bucket)
wall_time = time.time() - start_ts
overall['eval_s'] = wall_time
overall['eval_ep_s'] = wall_time / max(1, overall['n_episodes'])

merged = {
    'per_task': per_task,
    'per_group': per_group,
    'overall': overall,
    'parallel_meta': {
        'status': 'completed',
        'job_name': job_name,
        'step_id': step_id,
        'benchmark_task': benchmark_task,
        'gpu_ids': gpu_ids,
        'task_ids': expected_task_ids,
        'episodes_per_task': eval_episodes,
        'num_shards': num_shards,
        'max_concurrent_jobs': max_concurrent_jobs,
        'shard_root': str(shard_root),
        'shard_plan': str(shard_plan_path),
        'merged_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'shards': shard_summaries,
    },
}
out_path.write_text(json.dumps(merged, indent=2), encoding='utf-8')
print(json.dumps({'merged_eval_info': str(out_path), 'overall': overall}, indent=2))
PY

echo "[DONE] merged_eval_info=$MERGED_INFO"