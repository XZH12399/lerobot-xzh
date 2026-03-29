#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-1}"
REPO_DIR="/home/ct_24210860031/XZH/project/lerobot-xzh"
TRAIN_OUT="${2:-$REPO_DIR/outputs/pi05_family/pi05_spatial_coupled_libero_90_minval_5k_20260328}"
EVAL_ROOT="${3:-$REPO_DIR/eval_outputs/pi05_family}"
BENCHMARK_TASK="${4:-libero_90}"
N_EPISODES="${5:-1}"
STEPS_CSV="${6:-000500,001000,005000}"
RUN_PREFIX="${7:-pi05_spatial_coupled_libero_90_minval_5k_20260328}"

TOKENIZER_PATH="/home/ct_24210860031/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
IFS=',' read -r -a STEPS <<< "$STEPS_CSV"

mkdir -p "$EVAL_ROOT"

run_anchor() {
  local ckpt_dir="$1"
  local output_json="$2"

  if [[ -f "$output_json" ]]; then
    echo "[SKIP] anchor exists: $output_json"
    return 0
  fi

  echo "[RUN] anchor: $ckpt_dir -> $output_json"
  source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
  conda activate vla_baseline
  export PYTHONPATH="$REPO_DIR/src"
  export LEROBOT_TOKENIZER_PATH="$TOKENIZER_PATH"
  export LEROBOT_TOKENIZER_LOCAL_FILES_ONLY=1
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES="$GPU_ID"

  python "$REPO_DIR/scripts/pi05_spatial_anchor_corruption.py" \
    --policy-path "$ckpt_dir/pretrained_model" \
    --output-json "$output_json"
}

run_eval() {
  local ckpt_dir="$1"
  local output_dir="$2"
  local job_name="$3"

  if [[ -f "$output_dir/eval_info.json" ]]; then
    echo "[SKIP] eval exists: $output_dir/eval_info.json"
    return 0
  fi

  if [[ -f "$output_dir" ]]; then
    echo "[CLEAN] removing partial eval file: $output_dir"
    rm -f "$output_dir"
  elif [[ -d "$output_dir" && ! -f "$output_dir/eval_info.json" ]]; then
    echo "[CLEAN] removing partial eval dir: $output_dir"
    rm -rf "$output_dir"
  fi

  echo "[RUN] eval: $ckpt_dir -> $output_dir"
  "$REPO_DIR/scripts/pi05_spatial_minval_eval.sh" \
    "$GPU_ID" \
    "$ckpt_dir/pretrained_model" \
    "$BENCHMARK_TASK" \
    "$output_dir" \
    "$job_name" \
    "$N_EPISODES"

  python - <<PY
import json
from pathlib import Path
p = Path("$output_dir") / "eval_info.json"
d = json.loads(p.read_text())
overall = d["overall"]
print("[RESULT] eval success =", overall.get("pc_success"))
PY
}

wait_ckpt() {
  local ckpt_dir="$1"
  while [[ ! -d "$ckpt_dir/pretrained_model" ]]; do
    echo "[WAIT] checkpoint not ready: $ckpt_dir"
    sleep 60
  done
}

for step in "${STEPS[@]}"; do
  ckpt_dir="$TRAIN_OUT/checkpoints/$step"
  anchor_json="$EVAL_ROOT/${RUN_PREFIX}_anchor_step${step}.json"
  eval_dir="$EVAL_ROOT/${RUN_PREFIX}_step${step}_eval_${BENCHMARK_TASK}_ep${N_EPISODES}"
  job_name="${RUN_PREFIX}_step${step}_eval_${BENCHMARK_TASK}_ep${N_EPISODES}"

  wait_ckpt "$ckpt_dir"
  run_anchor "$ckpt_dir" "$anchor_json"
  run_eval "$ckpt_dir" "$eval_dir" "$job_name"
  echo "[DONE] validation finished for step $step"
done

echo "[DONE] all minimal validation jobs completed"
