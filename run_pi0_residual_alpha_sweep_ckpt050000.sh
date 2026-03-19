#!/usr/bin/env bash
set -euo pipefail
CONDA_SH=$HOME/anaconda3/etc/profile.d/conda.sh
source $CONDA_SH
conda activate vla_baseline
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export LEROBOT_FORCE_TASK_TEXT='Push the T shaped block to the target.'
PROJECT_DIR=/home/XZH/projects/lerobot
CHECKPOINT_DIR=/mnt/sda/xzh/lerobot_outputs/pi0_residual_everystep_alpha1_20260317_015201/checkpoints/050000/pretrained_model
LOG_DIR=$PROJECT_DIR/logs
OUT_ROOT=/mnt/sda/xzh/lerobot_outputs/evals
run_eval() {
  local alpha=$1
  local tag=$2
  local out_dir=$OUT_ROOT/pi0_residual_everystep_alphaSweep_${tag}_ckpt050000_ep50
  local log_file=$LOG_DIR/eval_pi0_residual_everystep_alphaSweep_${tag}_ckpt050000_ep50.log
  echo [$(date '+%F %T')] start alpha=$alpha out=$out_dir
  lerobot-eval \
    --policy.path=$CHECKPOINT_DIR \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=50 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.base_alpha_override=$alpha \
    --output_dir=$out_dir \
    --job_name=pi0_residual_alphaSweep_${tag}_ckpt050000_ep50 \
    >$log_file 2>&1
  echo [$(date '+%F %T')] done alpha=$alpha out=$out_dir
}
run_pair() {
  run_eval $1 $2 &
  pid1=$!
  run_eval $3 $4 &
  pid2=$!
  wait $pid1
  wait $pid2
}
run_pair 0.1 a0p1 0.5 a0p5
run_pair 1.0 a1p0 1.5 a1p5
run_eval 2.0 a2p0
