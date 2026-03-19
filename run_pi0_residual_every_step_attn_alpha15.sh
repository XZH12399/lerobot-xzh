#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-vla_baseline}
PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/lerobot}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/sda/xzh/lerobot_outputs}
CACHE_ROOT=${CACHE_ROOT:-/mnt/sda/xzh}
HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
WANDB_DIR=${WANDB_DIR:-$CACHE_ROOT/wandb}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$CACHE_ROOT/wandb-cache}

HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
HF_TOKEN=${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/lerobot/hf_token}

STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-5000}
EVAL_EPISODES=${EVAL_EPISODES:-10}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-true}
USE_RESIDUAL_BASE=${USE_RESIDUAL_BASE:-true}
BASE_ALPHA_INIT=${BASE_ALPHA_INIT:-0.1}
BASE_POOLING_MODE=${BASE_POOLING_MODE:-attn}
BASE_NUM_QUERIES=${BASE_NUM_QUERIES:-4}
BASE_POOL_NUM_HEADS=${BASE_POOL_NUM_HEADS:-4}
POLICY_TYPE=${POLICY_TYPE:-pi0_residual_every_step}
RUN_PREFIX=${RUN_PREFIX:-pi0_residual_every_step_attn_alpha15}
WANDB_PROJECT=${WANDB_PROJECT:-pi0_neurips_baseline}
JOB_NAME=${JOB_NAME:-pi0_residual_every_step_attn_alpha15_pusht_sanity}
RUN_MODE=${RUN_MODE:-bg}

source $CONDA_SH
conda activate $ENV_NAME

mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/outputs $OUTPUT_ROOT
mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $WANDB_DIR $WANDB_CACHE_DIR

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-$OUTPUT_ROOT/${RUN_PREFIX}_${TS}}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/logs/${RUN_PREFIX}_${TS}.log}

export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export WANDB_DIR
export WANDB_CACHE_DIR

if [[ -z $HF_TOKEN && -f $HF_TOKEN_FILE ]]; then
  HF_TOKEN=$(tr -d  \t\r\n < $HF_TOKEN_FILE)
fi
if [[ -n $HF_TOKEN ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
fi

CMD=(
  lerobot-train
  --policy.type=$POLICY_TYPE
  --dataset.repo_id=lerobot/pusht
  --dataset.video_backend=pyav
  --env.type=pusht
  --env.episode_length=300
  --steps=$STEPS
  --save_freq=$SAVE_FREQ
  --eval.n_episodes=$EVAL_EPISODES
  --eval.batch_size=$EVAL_BATCH_SIZE
  --batch_size=$BATCH_SIZE
  --policy.device=cuda
  --policy.train_expert_only=$TRAIN_EXPERT_ONLY
  --policy.use_residual_base=$USE_RESIDUAL_BASE
  --policy.base_alpha_init=$BASE_ALPHA_INIT
  --policy.base_pooling_mode=$BASE_POOLING_MODE
  --policy.base_num_queries=$BASE_NUM_QUERIES
  --policy.base_pool_num_heads=$BASE_POOL_NUM_HEADS
  --wandb.enable=true
  --wandb.project=$WANDB_PROJECT
  --wandb.disable_artifact=true
  --policy.pretrained_path=lerobot/pi0_base
  --policy.push_to_hub=false
  --resume=false
  --output_dir=$OUT_DIR
  --job_name=$JOB_NAME
)

echo [INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
echo [INFO] POLICY_TYPE: $POLICY_TYPE
echo [INFO] BASE_POOLING_MODE: $BASE_POOLING_MODE
echo [INFO] BASE_NUM_QUERIES: $BASE_NUM_QUERIES
echo [INFO] BASE_POOL_NUM_HEADS: $BASE_POOL_NUM_HEADS
echo [INFO] output: $OUT_DIR
echo [INFO] log: $LOG_FILE

cd $PROJECT_DIR
if [[ $RUN_MODE == fg || ${1:-} == --fg ]]; then
  exec ${CMD[@]}
else
  nohup ${CMD[@]} >$LOG_FILE 2>&1 &
  PID=$!
  echo [OK] started PID: $PID
  echo [OK] tail log: tail -f $LOG_FILE
fi
