#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <gpu_id> <job_name> <output_dir> <steps> <save_freq> [batch_size] [num_workers]"
  exit 1
fi

GPU_ID="$1"
JOB_NAME="$2"
OUTPUT_DIR="$3"
STEPS="$4"
SAVE_FREQ="$5"
BATCH_SIZE="${6:-4}"
NUM_WORKERS="${7:-4}"

REPO_DIR="/home/ct_24210860031/XZH/project/lerobot-xzh"
TOKENIZER_PATH="/home/ct_24210860031/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export PYTHONPATH="$REPO_DIR/src"
export LEROBOT_TOKENIZER_PATH="$TOKENIZER_PATH"
export LEROBOT_TOKENIZER_LOCAL_FILES_ONLY=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cd "$REPO_DIR"

python -m lerobot.scripts.lerobot_train \
  --policy.type=pi05_spatial \
  --policy.pretrained_path=lerobot/pi05_libero_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/home/ct_24210860031/data/libero \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --eval_freq=0 \
  --save_freq="$SAVE_FREQ" \
  --log_freq=20 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --wandb.enable=false
