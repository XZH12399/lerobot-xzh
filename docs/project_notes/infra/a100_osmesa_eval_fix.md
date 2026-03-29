# A100 Server LIBERO Evaluation Repair Guide

This note records how we repaired `LIBERO` evaluation on the `A100` server under user permissions only.

Project path:

```bash
/home/ct_24210860031/XZH/project/lerobot-xzh
```

The goal is:

- train on `A100`
- evaluate on `A100`
- avoid checkpoint transfer to `LabServer`

## 1. Problem Summary

`A100` could train `pi05` / `pi05_memory`, but `LIBERO` evaluation failed during environment creation.

The original failure mode was:

- `robosuite`
- `libero`
- `OffScreenRenderEnv`
- `egl_context.py`

Typical error:

```text
Cannot initialize a EGL device display
```

Root cause:

- the server image does not provide a usable NVIDIA EGL device path for `robosuite`
- `eglQueryDevicesEXT()` returned zero devices
- user-level Python package installation alone was not enough

We also tested:

- `EGL`: failed
- `Xvfb + GLX`: partially repaired, but final `GLX` support was still unavailable
- `OSMesa`: this is the path that finally worked

## 2. Final Working Idea

Use software rendering instead of GPU EGL:

- `PYOPENGL_PLATFORM=osmesa`
- `MUJOCO_GL=osmesa`

This avoids the broken `EGL` path and lets `LIBERO` envs render offscreen on CPU.

This is slower than hardware EGL, but it works on the current `A100` image.

## 3. Compatibility Assets We Added

We created three user-space compatibility locations.

### 3.1 OSMesa runtime

Installed from the network into:

```bash
/home/ct_24210860031/.compat/osmesa_env
```

Important library:

```bash
/home/ct_24210860031/.compat/osmesa_env/lib/libOSMesa.so
```

### 3.2 Old libgcrypt compatibility

`OSMesa` depended on an old `libgcrypt.so.11`, which the server did not provide.

Installed into:

```bash
/home/ct_24210860031/.compat/libgcrypt11
```

Important libraries:

```bash
/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda-linux-gnu/sysroot/lib64/libgcrypt.so.11
/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda_cos6-linux-gnu/sysroot/lib64/libgcrypt.so.11
```

### 3.3 Missing LIBERO textures

The `A100` assets cache was incomplete. We synced the full `textures/` directory.

Final location:

```bash
/home/ct_24210860031/.cache/libero/assets/textures
```

## 4. Commands Used To Build The Fix

### 4.1 Install OSMesa

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda create -y -p /home/ct_24210860031/.compat/osmesa_env -c menpo osmesa
```

### 4.2 Install libgcrypt.so.11 compatibility

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda create -y -p /home/ct_24210860031/.compat/libgcrypt11 -c defaults libgcrypt-cos6-x86_64
```

### 4.3 Sync missing textures

From local machine:

```bash
scp -3 -r LabServer:/home/XZH/software/libero/assets/textures A100:/home/ct_24210860031/.cache/libero/assets/
```

If `LabServer` is unavailable, any equivalent complete `LIBERO assets/textures` directory is fine.

## 5. Required Runtime Environment

Before evaluation on `A100`, export:

```bash
export LD_LIBRARY_PATH=/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda-linux-gnu/sysroot/lib64:/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda_cos6-linux-gnu/sysroot/lib64:/home/ct_24210860031/.compat/osmesa_env/lib:/home/ct_24210860031/.conda/envs/vla_baseline/lib:${LD_LIBRARY_PATH:-}
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/home/ct_24210860031/XZH/project/lerobot-xzh/src:/home/ct_24210860031/.codex_runtime:${PYTHONPATH:-}
```

Then activate the environment:

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline
```

## 6. Sanity Check

You can quickly check whether the OSMesa path is alive:

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export LD_LIBRARY_PATH=/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda-linux-gnu/sysroot/lib64:/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda_cos6-linux-gnu/sysroot/lib64:/home/ct_24210860031/.compat/osmesa_env/lib:/home/ct_24210860031/.conda/envs/vla_baseline/lib:${LD_LIBRARY_PATH:-}
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

python -c "from mujoco.osmesa import GLContext; print('OSMESA_IMPORT_OK', GLContext)"
```

Expected result:

```text
OSMESA_IMPORT_OK <class 'mujoco.osmesa.GLContext'>
```

## 7. Evaluation Command Template

### 7.1 Baseline pi05

```bash
source /home/ct_24210860031/miniconda3/etc/profile.d/conda.sh
conda activate vla_baseline

export LD_LIBRARY_PATH=/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda-linux-gnu/sysroot/lib64:/home/ct_24210860031/.compat/libgcrypt11/x86_64-conda_cos6-linux-gnu/sysroot/lib64:/home/ct_24210860031/.compat/osmesa_env/lib:/home/ct_24210860031/.conda/envs/vla_baseline/lib:${LD_LIBRARY_PATH:-}
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/home/ct_24210860031/XZH/project/lerobot-xzh/src:/home/ct_24210860031/.codex_runtime:${PYTHONPATH:-}

cd /home/ct_24210860031/XZH/project/lerobot-xzh

python -m lerobot.scripts.lerobot_eval \
  --policy.path=lerobot/pi05_libero_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/your_eval_name
```

### 7.2 pi05_memory checkpoint

Replace `--policy.path` with your checkpoint or `pretrained_model` directory:

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=/path/to/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_memory_eval_run
```

## 8. Verified Result

We successfully ran a real local evaluation on `A100` using OSMesa.

Output directory:

```bash
/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/a100_osmesa_probe2
```

Result file:

```bash
/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/a100_osmesa_probe2/eval_info.json
```

This run used the baseline `lerobot/pi05_libero_base` policy and completed end-to-end.

Important meaning:

- environment creation worked
- rendering worked
- rollout worked
- video output worked

At this stage, local evaluation on `A100` is no longer blocked by rendering.

## 9. Caveats

- `OSMesa` is software rendering, so it is slower than EGL
- this method depends on compatibility libraries in `~/.compat`
- if the server is reset, these compatibility prefixes may need to be recreated
- if `LIBERO` assets are partially missing again, resync `textures/`

Observed speed from the successful smoke:

- roughly `804s` for `10` episodes of the baseline run
- about `80s` per episode in this setup

## 10. Fast Recovery Checklist

If `A100` is reset, restore in this order:

1. Recreate `osmesa_env`
2. Recreate `libgcrypt11`
3. Ensure `textures/` exists under `~/.cache/libero/assets`
4. Export the runtime environment variables
5. Re-run the sanity check
6. Launch evaluation

## 11. Recommended Next Step

Create a dedicated shell script, for example:

```bash
run_libero_eval_a100_osmesa.sh
```

that:

- activates `vla_baseline`
- exports the OSMesa compatibility paths
- launches `lerobot_eval`

This avoids retyping the environment variables every time.
