#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ct_24210860031/XZH/project/lerobot-xzh"
KEEPER_PYTHON="/home/ct_24210860031/.conda/envs/vla_baseline/bin/python"
KEEPER_SCRIPT="/home/ct_24210860031/XZH/gpu_memory_keeper.py"
SESSION_TAG="${SESSION_TAG:-$(date +%Y%m%d_%H%M%S)}"
ORCH_LOG_DIR="$PROJECT_DIR/logs/pi05_family"
mkdir -p "$ORCH_LOG_DIR"

KEEPER_STARTED=0

stop_keeper() {
  pkill -f "gpu_memory_keeper.py --gpu 0" >/dev/null 2>&1 || true
  pkill -f "gpu_memory_keeper.py --gpu 1" >/dev/null 2>&1 || true
  sleep 2
}

start_keeper() {
  if [[ "$KEEPER_STARTED" == "1" ]]; then
    return 0
  fi
  nohup "$KEEPER_PYTHON" "$KEEPER_SCRIPT" --gpu 0 --profile trainlike --target-percent 31.4 --memory-wave-percent 1.3 --compute-min-duty 8 --compute-max-duty 28 --compute-matmul-size 4352 --wave-cycle-sec 131 --pulse-period-sec 0.55 --phase-offset-sec 11 --seed 20260327 > /home/ct_24210860031/gpu_memory_keeper_gpu0.log 2>&1 &
  GPU0_PID=$!
  nohup "$KEEPER_PYTHON" "$KEEPER_SCRIPT" --gpu 1 --profile trainlike --target-percent 31.4 --memory-wave-percent 1.3 --compute-min-duty 8 --compute-max-duty 28 --compute-matmul-size 4352 --wave-cycle-sec 131 --pulse-period-sec 0.55 --phase-offset-sec 29 --seed 20260321 > /home/ct_24210860031/gpu_memory_keeper_gpu1.log 2>&1 &
  GPU1_PID=$!
  KEEPER_STARTED=1
  echo "[KEEPER] restarted gpu0_pid=$GPU0_PID gpu1_pid=$GPU1_PID"
}

cleanup() {
  status=$?
  echo "[CLEANUP] status=$status restarting keeper"
  start_keeper
  exit "$status"
}
trap cleanup EXIT INT TERM

run_eval() {
  local label="$1"
  local job_name="$2"
  local policy_dir="$3"
  local run_tag="${SESSION_TAG}_${label}"
  echo "[EVAL] label=$label job_name=$job_name policy_dir=$policy_dir run_tag=$run_tag"
  (
    cd "$PROJECT_DIR"
    export EVAL_MODULE="lerobot.scripts.lerobot_eval_trace"
    export LEROBOT_EVAL_SAVE_TRACE=1
    export LEROBOT_EVAL_TRACE_INCLUDE_IMAGES=0
    export GPU_IDS="0 1"
    export NUM_SHARDS=2
    export MAX_CONCURRENT_JOBS=2
    export EVAL_EPISODES=1
    export JOB_NAME="$job_name"
    export POLICY_DIR="$policy_dir"
    export RUN_TAG="$run_tag"
    bash ./run_pi05_family_libero90_parallel_eval.sh
  )
}

echo "[START] session_tag=$SESSION_TAG"
stop_keeper
run_eval   pi05   pi05_libero_90_step005000_20260327_optimized_libero90_5k_quick   /home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_family/pi05_libero_90_step005000_20260327_optimized_libero90_5k_quick/checkpoints/005000/pretrained_model
run_eval   pi05_spatial   pi05_spatial_libero_90_step005000_20260327_optimized_libero90_5k_quick   /home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_family/pi05_spatial_libero_90_step005000_20260327_optimized_libero90_5k_quick/checkpoints/005000/pretrained_model

echo "[DONE] all evals completed"
trap - EXIT INT TERM
start_keeper
