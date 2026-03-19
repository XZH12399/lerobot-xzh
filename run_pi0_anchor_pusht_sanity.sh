#!/usr/bin/env bash
set -euo pipefail

# Repro script for LeRobot pi0_anchor + PushT sanity training.
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/sda/xzh/lerobot_outputs}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/sda/xzh}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
WANDB_DIR="${WANDB_DIR:-$CACHE_ROOT/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_ROOT/wandb-cache}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/lerobot/hf_token}"

STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
USE_COARSE_PRIOR="${USE_COARSE_PRIOR:-false}"
COARSE_DETACH="${COARSE_DETACH:-true}"
LAMBDA_PRIOR="${LAMBDA_PRIOR:-}"
WANDB_PROJECT="${WANDB_PROJECT:-pi0_neurips_baseline}"
JOB_NAME="${JOB_NAME:-pi0_anchor_pusht_sanity}"
RESUME="${RESUME:-false}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"

RUN_MODE="${RUN_MODE:-bg}" # bg|fg

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found: $CONDA_SH"
  exit 1
fi

source "$CONDA_SH"
conda activate "$ENV_NAME"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs" "$OUTPUT_ROOT"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$WANDB_DIR" "$WANDB_CACHE_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
if [[ "$RESUME" == "true" ]]; then
  if [[ -z "$TRAIN_RUN_DIR" ]]; then
    TRAIN_RUN_DIR="$(ls -1dt "$OUTPUT_ROOT"/pi0_anchor_pusht_* "$PROJECT_DIR"/outputs/pi0_anchor_pusht_* 2>/dev/null | head -n 1 || true)"
  fi
  OUT_DIR="${OUT_DIR:-$TRAIN_RUN_DIR}"
  LOG_FILE="${LOG_FILE:-$PROJECT_DIR/logs/pi0_anchor_pusht_resume_${TS}.log}"
else
  OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/pi0_anchor_pusht_${TS}}"
  LOG_FILE="${LOG_FILE:-$PROJECT_DIR/logs/pi0_anchor_pusht_${TS}.log}"
fi

if [[ -z "$OUT_DIR" || ( "$RESUME" == "true" && ! -d "$OUT_DIR" ) ]]; then
  echo "[ERROR] RESUME=true but run dir not found: $OUT_DIR"
  exit 1
fi

if [[ "$RESUME" == "true" && -z "$CONFIG_PATH" ]]; then
  if [[ -f "$OUT_DIR/checkpoints/last/pretrained_model/train_config.json" ]]; then
    CONFIG_PATH="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  else
    CONFIG_PATH="$(ls -1t "$OUT_DIR"/checkpoints/*/pretrained_model/train_config.json 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ "$RESUME" == "true" && ( -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ) ]]; then
  echo "[ERROR] RESUME=true requires a valid train config path."
  echo "[ERROR] Could not find train_config.json under: $OUT_DIR/checkpoints"
  exit 1
fi

export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export WANDB_DIR
export WANDB_CACHE_DIR

# Load token from file when env var is not provided.
if [[ -z "$HF_TOKEN" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d " \t\r\n" < "$HF_TOKEN_FILE")"
fi

if [[ -n "$HF_TOKEN" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  echo "[INFO] HF token loaded."
else
  echo "[WARN] HF token not set."
  echo "[WARN] Set HF_TOKEN or put token in $HF_TOKEN_FILE"
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
  echo "[WARN] WANDB_API_KEY is not set and ~/.netrc is missing. W&B login may fail."
fi

CMD=(
  lerobot-train
  --policy.type=pi0_anchor
  --dataset.repo_id=lerobot/pusht
  --dataset.video_backend=pyav
  --env.type=pusht
  --env.episode_length=300
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --eval.n_episodes="$EVAL_EPISODES"
  --eval.batch_size="$EVAL_BATCH_SIZE"
  --batch_size="$BATCH_SIZE"
  --policy.device=cuda
  --policy.train_expert_only="$TRAIN_EXPERT_ONLY"
  --policy.use_coarse_prior="$USE_COARSE_PRIOR"
  --policy.coarse_detach="$COARSE_DETACH"
  --wandb.enable=true
  --wandb.project="$WANDB_PROJECT"
  --wandb.disable_artifact=true
  --policy.pretrained_path=lerobot/pi0_base
  --policy.push_to_hub=false
  --resume="$RESUME"
  --output_dir="$OUT_DIR"
  --job_name="$JOB_NAME"
)

if [[ -n "$LAMBDA_PRIOR" ]]; then
  CMD+=(--policy.lambda_prior="$LAMBDA_PRIOR")
fi

if [[ "$RESUME" == "true" ]]; then
  CMD+=(--config_path="$CONFIG_PATH")
fi

echo "[INFO] project: $PROJECT_DIR"
echo "[INFO] env: $ENV_NAME"
echo "[INFO] HF_ENDPOINT: $HF_ENDPOINT"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] TRAIN_EXPERT_ONLY: $TRAIN_EXPERT_ONLY"
echo "[INFO] USE_COARSE_PRIOR: $USE_COARSE_PRIOR"
echo "[INFO] COARSE_DETACH: $COARSE_DETACH"
if [[ -n "$LAMBDA_PRIOR" ]]; then echo "[INFO] LAMBDA_PRIOR: $LAMBDA_PRIOR"; fi
echo "[INFO] RESUME: $RESUME"
echo "[INFO] OUTPUT_ROOT: $OUTPUT_ROOT"
echo "[INFO] CACHE_ROOT: $CACHE_ROOT"
echo "[INFO] HF_HOME: $HF_HOME"
echo "[INFO] WANDB_DIR: $WANDB_DIR"
echo "[INFO] output: $OUT_DIR"
if [[ "$RESUME" == "true" ]]; then
  echo "[INFO] config_path: $CONFIG_PATH"
fi
echo "[INFO] log: $LOG_FILE"

cd "$PROJECT_DIR"

if [[ "$RUN_MODE" == "fg" || "${1:-}" == "--fg" ]]; then
  echo "[INFO] running in foreground"
  exec "${CMD[@]}"
else
  echo "[INFO] running in background"
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "[OK] started PID: $PID"
  echo "[OK] tail log: tail -f $LOG_FILE"
fi
