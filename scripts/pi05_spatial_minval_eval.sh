#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <gpu_id> <policy_path> <benchmark_task> <output_dir> <job_name> [n_episodes]"
  exit 1
fi

GPU_ID="$1"
POLICY_PATH="$2"
BENCHMARK_TASK="$3"
OUTPUT_DIR="$4"
JOB_NAME="$5"
N_EPISODES="${6:-1}"

REPO_DIR="/home/ct_24210860031/XZH/project/lerobot-xzh"
TOKENIZER_PATH="/home/ct_24210860031/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src"
export LEROBOT_TOKENIZER_PATH="$TOKENIZER_PATH"
export LEROBOT_TOKENIZER_LOCAL_FILES_ONLY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$HOME/.compat/nvidia470_egl/lib:${LD_LIBRARY_PATH:-}"
fi
export __EGL_VENDOR_LIBRARY_DIRS="$HOME/.compat/nvidia470_egl/share/glvnd/egl_vendor.d"
unset PYOPENGL_PLATFORM || true
unset MUJOCO_EGL_DEVICE_ID || true
export MUJOCO_GL=egl

cd "$REPO_DIR"

python -m lerobot.scripts.lerobot_eval \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --env.type=libero \
  --env.task="$BENCHMARK_TASK" \
  --eval.batch_size=1 \
  --eval.n_episodes="$N_EPISODES" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME"
