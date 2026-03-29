# A100 Server NVIDIA 470 EGL Repair Guide

This note records how we repaired `LIBERO` evaluation on the `A100` server using GPU-accelerated headless EGL under user permissions only.

Project path:

```bash
/home/ct_24210860031/XZH/project/lerobot-xzh
```

This document supersedes the slower `OSMesa` path when the goal is fast local evaluation on `A100`.

Related fallback document:

```bash
/home/ct_24210860031/XZH/project/lerobot-xzh/docs/project_notes/infra/a100_osmesa_eval_fix.md
```

## 1. Why OSMesa Was Slow

The original `A100` image could train normally, but `LIBERO` evaluation failed on the default `EGL` path because `robosuite` could not create an EGL device display.

Typical failure:

```text
Cannot initialize a EGL device display
```

Root cause:

- `robosuite` uses `eglQueryDevicesEXT()` and `EGL_PLATFORM_DEVICE_EXT`
- the server image exposed NVIDIA compute devices
- but it did not provide a usable NVIDIA EGL user-space graphics stack
- so `eglQueryDevicesEXT()` returned zero devices on the original setup

`OSMesa` fixed correctness, but it used CPU software rendering and was too slow for full `libero_10` comparisons.

## 2. Final Working Idea

Instead of relying on the broken system EGL stack, we installed a user-space NVIDIA EGL runtime matching the server driver version.

Driver version on `A100`:

```bash
470.199.02
```

Working idea:

- download the official NVIDIA `470.199.02` Linux driver runfile
- extract only the user-space graphics libraries
- build a private runtime under `~/.compat/nvidia470_egl`
- point `GLVND` and the dynamic loader to this runtime using environment variables
- run `MUJOCO_GL=egl`

This keeps everything in user space and does not require root.

## 3. Final Compatibility Location

Main prefix:

```bash
/home/ct_24210860031/.compat/nvidia470_egl
```

Important subdirectories:

```bash
/home/ct_24210860031/.compat/nvidia470_egl/download
/home/ct_24210860031/.compat/nvidia470_egl/extract
/home/ct_24210860031/.compat/nvidia470_egl/lib
/home/ct_24210860031/.compat/nvidia470_egl/share/glvnd/egl_vendor.d
```

Important files:

```bash
/home/ct_24210860031/.compat/nvidia470_egl/download/NVIDIA-Linux-x86_64-470.199.02.run
/home/ct_24210860031/.compat/nvidia470_egl/lib/libEGL_nvidia.so.0
/home/ct_24210860031/.compat/nvidia470_egl/lib/libnvidia-eglcore.so.470.199.02
/home/ct_24210860031/.compat/nvidia470_egl/share/glvnd/egl_vendor.d/10_nvidia.json
```

## 4. One-Time Reconstruction Steps

If the server resets and this prefix disappears, rebuild it with the following steps.

### 4.1 Create directories

```bash
mkdir -p /home/ct_24210860031/.compat/nvidia470_egl/{download,lib,share/glvnd/egl_vendor.d}
```

### 4.2 Download the official NVIDIA runfile

```bash
cd /home/ct_24210860031/.compat/nvidia470_egl/download
wget -O NVIDIA-Linux-x86_64-470.199.02.run   https://download.nvidia.com/XFree86/Linux-x86_64/470.199.02/NVIDIA-Linux-x86_64-470.199.02.run
```

### 4.3 Extract the runfile

Important: the target directory must not already exist.

```bash
cd /home/ct_24210860031/.compat/nvidia470_egl
rm -rf extract
bash download/NVIDIA-Linux-x86_64-470.199.02.run --extract-only --target extract
```

### 4.4 Build the private EGL runtime

```bash
PREFIX=/home/ct_24210860031/.compat/nvidia470_egl
EXTRACT=$PREFIX/extract
LIB=$PREFIX/lib
VENDOR=$PREFIX/share/glvnd/egl_vendor.d

mkdir -p "$LIB" "$VENDOR"
rm -f "$LIB"/* "$VENDOR"/*.json

for f in   libEGL.so.1.1.0 libGLX.so.0 libGLdispatch.so.0 libOpenGL.so.0   libGLESv1_CM.so.1.2.0 libGLESv2.so.2.1.0   libEGL_nvidia.so.470.199.02 libGLX_nvidia.so.470.199.02   libGLESv1_CM_nvidia.so.470.199.02 libGLESv2_nvidia.so.470.199.02   libnvidia-eglcore.so.470.199.02 libnvidia-glcore.so.470.199.02   libnvidia-glsi.so.470.199.02 libnvidia-glvkspirv.so.470.199.02   libnvidia-tls.so.470.199.02 libnvidia-allocator.so.470.199.02   libnvidia-cfg.so.470.199.02 libnvidia-ml.so.470.199.02
  do
  if [ -f "$EXTRACT/$f" ]; then
    ln -sf "$EXTRACT/$f" "$LIB/$f"
  fi
done

ln -sf "$EXTRACT/libEGL.so.1.1.0" "$LIB/libEGL.so.1"
ln -sf "$EXTRACT/libGLX.so.0" "$LIB/libGLX.so.0"
ln -sf "$EXTRACT/libGLdispatch.so.0" "$LIB/libGLdispatch.so.0"
ln -sf "$EXTRACT/libOpenGL.so.0" "$LIB/libOpenGL.so.0"
ln -sf "$EXTRACT/libGLESv1_CM.so.1.2.0" "$LIB/libGLESv1_CM.so.1"
ln -sf "$EXTRACT/libGLESv2.so.2.1.0" "$LIB/libGLESv2.so.2"
ln -sf "$EXTRACT/libEGL_nvidia.so.470.199.02" "$LIB/libEGL_nvidia.so.0"
ln -sf "$EXTRACT/libGLX_nvidia.so.470.199.02" "$LIB/libGLX_nvidia.so.0"
ln -sf "$EXTRACT/libGLESv1_CM_nvidia.so.470.199.02" "$LIB/libGLESv1_CM_nvidia.so.1"
ln -sf "$EXTRACT/libGLESv2_nvidia.so.470.199.02" "$LIB/libGLESv2_nvidia.so.2"
ln -sf "$EXTRACT/libnvidia-ml.so.470.199.02" "$LIB/libnvidia-ml.so.1"

cp -f "$EXTRACT/10_nvidia.json" "$VENDOR/10_nvidia.json"
```

## 5. Runtime Environment For GPU EGL Evaluation

Before running evaluation on `A100`, activate conda and export:

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export LD_LIBRARY_PATH=/home/ct_24210860031/.compat/nvidia470_egl/lib:/home/ct_24210860031/.conda/envs/vla_baseline/lib:${LD_LIBRARY_PATH:-}
export __EGL_VENDOR_LIBRARY_DIRS=/home/ct_24210860031/.compat/nvidia470_egl/share/glvnd/egl_vendor.d
unset PYOPENGL_PLATFORM
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/home/ct_24210860031/XZH/project/lerobot-xzh/src:/home/ct_24210860031/.codex_runtime:${PYTHONPATH:-}
```

Important meaning:

- `LD_LIBRARY_PATH`: points the loader to the extracted NVIDIA 470 EGL libraries
- `__EGL_VENDOR_LIBRARY_DIRS`: tells GLVND to use the private `10_nvidia.json`
- `unset PYOPENGL_PLATFORM`: prevents forcing `osmesa`
- `MUJOCO_GL=egl`: tells mujoco to use EGL instead of OSMesa

## 6. Sanity Check

Run this before a real eval.

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export LD_LIBRARY_PATH=/home/ct_24210860031/.compat/nvidia470_egl/lib:/home/ct_24210860031/.conda/envs/vla_baseline/lib:${LD_LIBRARY_PATH:-}
export __EGL_VENDOR_LIBRARY_DIRS=/home/ct_24210860031/.compat/nvidia470_egl/share/glvnd/egl_vendor.d
unset PYOPENGL_PLATFORM
export MUJOCO_GL=egl

python - <<'PY'
from mujoco.egl import egl_ext as EGL
from robosuite.renderers.context.egl_context import create_initialized_egl_device_display
print('EGL_DEVICES', len(EGL.eglQueryDevicesEXT()))
print('EGL_DISPLAY_OK', create_initialized_egl_device_display(0))
PY
```

Expected shape of result:

```text
EGL_DEVICES 2
EGL_DISPLAY_OK <OpenGL._opaque.EGLDisplay_pointer object at ...>
```

If `EGL_DEVICES` is `0`, the EGL vendor stack is still not active.

## 7. Real Smoke Test

A minimal real eval that we verified on `A100`:

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export LD_LIBRARY_PATH=/home/ct_24210860031/.compat/nvidia470_egl/lib:/home/ct_24210860031/.conda/envs/vla_baseline/lib:${LD_LIBRARY_PATH:-}
export __EGL_VENDOR_LIBRARY_DIRS=/home/ct_24210860031/.compat/nvidia470_egl/share/glvnd/egl_vendor.d
unset PYOPENGL_PLATFORM
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/home/ct_24210860031/XZH/project/lerobot-xzh/src:/home/ct_24210860031/.codex_runtime:${PYTHONPATH:-}

cd /home/ct_24210860031/XZH/project/lerobot-xzh

CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_eval   --policy.path=/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_family/pi05_libero10_overnight_20260320_overnight_base/checkpoints/010000/pretrained_model   --policy.device=cuda   --policy.dtype=bfloat16   --env.type=libero   --env.task=libero_10   --env.task_ids=[0]   --eval.batch_size=1   --eval.n_episodes=1   --output_dir=/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_egl_smoke_task0_20260321
```

Verified result from this smoke:

- `pc_success = 100.0`
- `n_episodes = 1`
- `eval_s ~= 68.98`

This confirmed:

- EGL device enumeration worked
- robosuite environment creation worked
- mujoco EGL rendering worked
- policy inference on GPU worked
- LeRobot evaluation pipeline worked end-to-end

## 8. Full Evaluation Command Templates

### 8.1 Baseline pi05

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export LD_LIBRARY_PATH=/home/ct_24210860031/.compat/nvidia470_egl/lib:/home/ct_24210860031/.conda/envs/vla_baseline/lib:${LD_LIBRARY_PATH:-}
export __EGL_VENDOR_LIBRARY_DIRS=/home/ct_24210860031/.compat/nvidia470_egl/share/glvnd/egl_vendor.d
unset PYOPENGL_PLATFORM
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/home/ct_24210860031/XZH/project/lerobot-xzh/src:/home/ct_24210860031/.codex_runtime:${PYTHONPATH:-}

cd /home/ct_24210860031/XZH/project/lerobot-xzh

CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_eval   --policy.path=/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_family/pi05_libero10_overnight_20260320_overnight_base/checkpoints/010000/pretrained_model   --policy.device=cuda   --policy.dtype=bfloat16   --env.type=libero   --env.task=libero_10   --eval.batch_size=1   --eval.n_episodes=10   --output_dir=/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_overnight_final_eval_egl_manual
```

### 8.2 pi05_memory with history on

```bash
CUDA_VISIBLE_DEVICES=1 python -m lerobot.scripts.lerobot_eval   --policy.path=/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_family/pi05_memory_libero10_memfix2_full_20260321_memfix2_full/checkpoints/010000/pretrained_model   --policy.device=cuda   --policy.dtype=bfloat16   --policy.use_history_memory=true   --env.type=libero   --env.task=libero_10   --eval.batch_size=1   --eval.n_episodes=10   --output_dir=/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_memory_hist_on_eval_egl_manual
```

### 8.3 pi05_memory with history off

```bash
CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_eval   --policy.path=/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_family/pi05_memory_libero10_memfix2_full_20260321_memfix2_full/checkpoints/010000/pretrained_model   --policy.device=cuda   --policy.dtype=bfloat16   --policy.use_history_memory=false   --env.type=libero   --env.task=libero_10   --eval.batch_size=1   --eval.n_episodes=10   --output_dir=/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_memory_hist_off_eval_egl_manual
```

## 9. How To Watch Logs

```bash
tail -f /home/ct_24210860031/XZH/project/lerobot-xzh/logs/pi05_family/pi05_overnight_final_eval_egl_20260321.log
tail -f /home/ct_24210860031/XZH/project/lerobot-xzh/logs/pi05_family/pi05_memory_memfix2_hist_on_eval_egl_20260321.log
tail -f /home/ct_24210860031/XZH/project/lerobot-xzh/logs/pi05_family/pi05_memory_memfix2_hist_off_eval_egl_20260321.log
```

To stop `tail -f`, press `Ctrl + C`.

## 10. Performance Notes

Observed comparison on `A100`:

- `OSMesa`: correct but very slow, roughly overnight scale for full multi-run comparisons
- `NVIDIA EGL`: much faster and suitable for direct local evaluation on `A100`

Even with GPU EGL, evaluation will not necessarily show training-like GPU utilization because:

- `eval.batch_size=1`
- rollout is environment-interaction heavy
- `LIBERO` evaluation alternates between environment stepping, rendering, and policy forward passes

So it is normal to see:

- GPU memory occupied
- GPU utilization fluctuating
- utilization much lower than training

## 11. Fast Recovery Checklist

If the server resets, restore in this order:

1. Recreate `/home/ct_24210860031/.compat/nvidia470_egl`
2. Redownload `NVIDIA-Linux-x86_64-470.199.02.run`
3. Re-extract the runfile
4. Rebuild the private `lib/` symlink runtime
5. Restore the EGL environment variables
6. Run the sanity check
7. Run the one-task smoke test
8. Launch the real evaluation

## 12. If EGL Breaks Again

Quick diagnosis commands:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1

python - <<'PY'
from mujoco.egl import egl_ext as EGL
print('EGL_DEVICES', len(EGL.eglQueryDevicesEXT()))
PY
```

Interpretation:

- if the driver version changes away from `470.199.02`, this document may need a matching new NVIDIA runfile
- if `EGL_DEVICES` becomes `0`, check `LD_LIBRARY_PATH` and `__EGL_VENDOR_LIBRARY_DIRS`
- if EGL still fails, the OSMesa fallback doc can still be used
