#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$ROOT_DIR/requirements_lock.txt"
LOG_DIR="$ROOT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/install_vla_baseline_${TIMESTAMP}.log}"
TMP_DIR="${TMPDIR:-/tmp}/vla_baseline_install"

CONDA_BIN="${CONDA_BIN:-/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-vla_baseline}"
ENV_HOME="${ENV_HOME:-/home/ct_24210860031/.conda/envs}"
ENV_PYTHON="$ENV_HOME/$ENV_NAME/bin/python"

mkdir -p "$LOG_DIR" "$TMP_DIR" "/home/ct_24210860031/.conda/pkgs" "$ENV_HOME"

export CONDA_PKGS_DIRS="/home/ct_24210860031/.conda/pkgs"
export CONDA_ENVS_PATH="$ENV_HOME"
export PIP_DISABLE_PIP_VERSION_CHECK=1

: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] root=$ROOT_DIR"
echo "[$(date '+%F %T')] log=$LOG_FILE"
echo "[$(date '+%F %T')] tmp=$TMP_DIR"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "conda not found at $CONDA_BIN"
  exit 1
fi

if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "[$(date '+%F %T')] creating conda env $ENV_NAME"
  "$CONDA_BIN" env create -f "$ROOT_DIR/environment_min.yml"
else
  echo "[$(date '+%F %T')] reusing existing env $ENV_NAME"
fi

if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "python not found at $ENV_PYTHON after env creation"
  exit 1
fi

cat > "$TMP_DIR/stage_requirements.py" <<'PY'
from pathlib import Path
import os
import re

root = Path(os.environ["ROOT_DIR"])
req_file = Path(os.environ["REQ_FILE"])
tmp_dir = Path(os.environ["TMP_DIR"])

def normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")

stages = {
    "torch_cuda": [
        "torch",
        "torchvision",
        "triton",
        "cuda-bindings",
        "cuda-pathfinder",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-cufile-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-cusparselt-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvshmem-cu12",
        "nvidia-nvtx-cu12",
    ],
    "hf_core": [
        "accelerate",
        "datasets",
        "diffusers",
        "huggingface-hub",
        "hf-xet",
        "safetensors",
        "tokenizers",
        "transformers",
        "av",
        "einops",
        "h5py",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "pandas",
        "pyarrow",
        "scipy",
        "tqdm",
        "wandb",
        "tensorboard",
        "tensorboard-data-server",
        "tensorboardx",
        "thop",
    ],
    "sim_stack": [
        "bddl",
        "dm-control",
        "dm-env",
        "dm-tree",
        "egl_probe",
        "glfw",
        "gym-aloha",
        "gym-pusht",
        "gymnasium",
        "hf-libero",
        "hf_egl_probe",
        "labmaze",
        "mujoco",
        "pygame",
        "pymunk",
        "pynput",
        "python-xlib",
        "robomimic",
        "robosuite",
        "shapely",
    ],
    "editable": [
        "-e .",
    ],
}

lines = []
for raw in req_file.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("-e "):
        lines.append(("editable", line))
        continue
    match = re.match(r"^([A-Za-z0-9_.-]+)", line)
    if not match:
        continue
    package = normalize(match.group(1))
    lines.append((package, line))

assigned = set()
for stage_name, package_names in stages.items():
    out_lines = []
    normalized_names = {normalize(name) for name in package_names if not name.startswith("-e ")}
    for package, line in lines:
        if stage_name == "editable":
            if package == "editable":
                out_lines.append(line)
                assigned.add((package, line))
            continue
        if package in normalized_names:
            out_lines.append(line)
            assigned.add((package, line))
    (tmp_dir / f"{stage_name}.txt").write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

remaining = []
for package, line in lines:
    if (package, line) in assigned:
        continue
    remaining.append(line)
(tmp_dir / "remaining.txt").write_text("\n".join(remaining) + ("\n" if remaining else ""))
PY

export ROOT_DIR REQ_FILE TMP_DIR
"$ENV_PYTHON" "$TMP_DIR/stage_requirements.py"

run_stage() {
  local stage_name="$1"
  local req_path="$TMP_DIR/${stage_name}.txt"

  if [[ ! -s "$req_path" ]]; then
    echo "[$(date '+%F %T')] stage=${stage_name} skipped (empty)"
    return
  fi

  echo
  echo "========== stage: ${stage_name} =========="
  echo "[$(date '+%F %T')] using $req_path"
  sed -n '1,40p' "$req_path"
  "$ENV_PYTHON" -m pip install --no-deps -r "$req_path"
  echo "[$(date '+%F %T')] stage=${stage_name} done"
}

echo
echo "========== stage: bootstrap =========="
"$ENV_PYTHON" -m pip install --upgrade pip setuptools wheel packaging
echo "[$(date '+%F %T')] stage=bootstrap done"

run_stage "torch_cuda"
run_stage "hf_core"
run_stage "sim_stack"
run_stage "remaining"
run_stage "editable"

echo
echo "========== stage: smoke =========="
"$ENV_PYTHON" - <<'PY'
import importlib

mods = ["torch", "transformers", "datasets", "wandb", "mujoco", "robosuite", "lerobot"]
for name in mods:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "no_version"))
PY
echo "[$(date '+%F %T')] install complete"
