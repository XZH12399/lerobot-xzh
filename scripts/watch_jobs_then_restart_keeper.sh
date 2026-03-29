#!/usr/bin/env bash
set -euo pipefail

WATCH_PIDS="${WATCH_PIDS:-${*:-}}"
POLL_SEC="${POLL_SEC:-30}"

if [[ -z "$WATCH_PIDS" ]]; then
  echo "[ERROR] WATCH_PIDS is empty"
  exit 1
fi

LOG_DIR="${KEEPER_LOG_DIR:-$HOME/XZH/logs/gpu_memory_keeper}"
PYTHON_BIN="${KEEPER_PYTHON_BIN:-$HOME/.conda/envs/vla_baseline/bin/python}"
KEEPER_SCRIPT="${KEEPER_SCRIPT:-$HOME/XZH/gpu_memory_keeper.py}"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[WATCHDOG] watching pids: $WATCH_PIDS"
echo "[WATCHDOG] poll_sec=$POLL_SEC"

while true; do
  any_alive=0
  for pid in $WATCH_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      any_alive=1
      break
    fi
  done

  if [[ "$any_alive" -eq 0 ]]; then
    break
  fi

  sleep "$POLL_SEC"
done

echo "[WATCHDOG] all watched jobs have exited"

if pgrep -af gpu_memory_keeper.py >/dev/null 2>&1; then
  echo "[WATCHDOG] keeper already running, skip restart"
  pgrep -af gpu_memory_keeper.py || true
  exit 0
fi

mkdir -p "$LOG_DIR"

nohup "$PYTHON_BIN" "$KEEPER_SCRIPT" \
  --gpu 0 \
  --profile trainlike \
  --target-percent 31.4 \
  --memory-wave-percent 1.3 \
  --compute-min-duty 8 \
  --compute-max-duty 28 \
  --compute-matmul-size 4352 \
  --wave-cycle-sec 131 \
  --pulse-period-sec 0.55 \
  --phase-offset-sec 11 \
  --seed 20260327 \
  > "$LOG_DIR/gpu0_${TS}.log" 2>&1 &
pid0=$!

nohup "$PYTHON_BIN" "$KEEPER_SCRIPT" \
  --gpu 1 \
  --profile trainlike \
  --target-percent 31.4 \
  --memory-wave-percent 1.3 \
  --compute-min-duty 8 \
  --compute-max-duty 28 \
  --compute-matmul-size 4352 \
  --wave-cycle-sec 131 \
  --pulse-period-sec 0.55 \
  --phase-offset-sec 29 \
  --seed 20260321 \
  > "$LOG_DIR/gpu1_${TS}.log" 2>&1 &
pid1=$!

sleep 5
echo "[WATCHDOG] restarted keepers"
echo "[WATCHDOG] pid0=$pid0 log0=$LOG_DIR/gpu0_${TS}.log"
echo "[WATCHDOG] pid1=$pid1 log1=$LOG_DIR/gpu1_${TS}.log"
pgrep -af gpu_memory_keeper.py || true
