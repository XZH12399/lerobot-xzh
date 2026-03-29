#!/usr/bin/env python

import os
import time
import threading
from pathlib import Path
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch

from lerobot.scripts import lerobot_eval as base
from lerobot.utils.constants import ACTION


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _save_rollout_batch_traces(
    trace_dir: Path,
    rollout_data: dict[str, Any],
    done_indices: torch.Tensor,
    *,
    start_episode_index: int,
    max_episodes: int,
    seeds: list[int] | None,
) -> list[str]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_paths: list[str] = []
    batch_count = min(max_episodes, rollout_data[ACTION].shape[0])

    for local_ep_ix in range(batch_count):
        episode_ix = start_episode_index + local_ep_ix
        num_steps = int(done_indices[local_ep_ix].item()) + 1
        payload = {
            "episode_ix": episode_ix,
            "seed": None if seeds is None else int(seeds[local_ep_ix]),
            ACTION: rollout_data[ACTION][local_ep_ix, :num_steps].cpu(),
            "reward": rollout_data["reward"][local_ep_ix, :num_steps].cpu(),
            "success": rollout_data["success"][local_ep_ix, :num_steps].cpu(),
            "done": rollout_data["done"][local_ep_ix, :num_steps].cpu(),
        }
        trace_path = trace_dir / f"trace_episode_{episode_ix}.pt"
        torch.save(payload, trace_path)
        trace_paths.append(str(trace_path))

    return trace_paths


def _eval_policy_with_trace(
    env: gym.vector.VectorEnv,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    start_seed: int | None = None,
    trace_dir: Path | None = None,
) -> dict[str, Any]:
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")
    if trace_dir is None:
        raise ValueError("trace_dir must be provided when trace saving is enabled.")

    start = time.time()
    policy.eval()
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    trace_paths: list[str] = []
    threads = []
    debug_stats_infer = None
    debug_stats_infer_steps = None
    n_episodes_rendered = 0
    video_paths: list[str] = [] if max_episodes_rendered > 0 else []

    def render_frame(vec_env: gym.vector.VectorEnv):
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, vec_env.num_envs)
        if isinstance(vec_env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([vec_env.envs[i].render() for i in range(n_to_render_now)]))
        elif isinstance(vec_env, gym.vector.AsyncVectorEnv):
            ep_frames.append(np.stack(vec_env.call("render")[:n_to_render_now]))

    for batch_ix in range(n_batches):
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = list(range(
                start_seed + (batch_ix * env.num_envs),
                start_seed + ((batch_ix + 1) * env.num_envs),
            ))

        rollout_data = base.rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=seeds,
            return_observations=False,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        if debug_stats_infer is None:
            model = getattr(policy, "model", None)
            stats = getattr(model, "debug_stats_infer", None)
            if stats:
                debug_stats_infer = {
                    k: float(v) if isinstance(v, (int, float)) else v for k, v in stats.items()
                }

        if debug_stats_infer_steps is None:
            model = getattr(policy, "model", None)
            step_stats = getattr(model, "debug_stats_infer_steps", None)
            if step_stats:
                debug_stats_infer_steps = [
                    {k: float(v) if isinstance(v, (int, float)) else v for k, v in item.items()}
                    for item in step_stats
                ]

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()

        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.extend([None] * rollout_data[ACTION].shape[0])

        remaining_episodes = max(0, n_episodes - batch_ix * env.num_envs)
        trace_paths.extend(
            _save_rollout_batch_traces(
                trace_dir,
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                max_episodes=remaining_episodes,
                seeds=seeds,
            )
        )

        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)
            for stacked_frames, done_index in zip(batch_stacked_frames, done_indices.flatten().tolist(), strict=False):
                if n_episodes_rendered >= max_episodes_rendered:
                    break
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=base.write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

    for thread in threads:
        thread.join()

    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
        "trace_paths": trace_paths,
    }

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths
    if debug_stats_infer is not None:
        info["debug_stats_infer"] = debug_stats_infer
        if "base_fusion_mode" in debug_stats_infer:
            info["base_fusion_mode"] = debug_stats_infer["base_fusion_mode"]
    if debug_stats_infer_steps is not None:
        info["debug_stats_infer_steps"] = debug_stats_infer_steps
    return info


def _patched_eval_one(
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    save_trace = _env_flag("LEROBOT_EVAL_SAVE_TRACE", default=False)
    if not save_trace:
        task_result = base.eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            return_episode_data=return_episode_data,
            start_seed=start_seed,
        )
    else:
        if videos_dir is not None:
            trace_dir = videos_dir.parent.parent / "traces" / videos_dir.name
        else:
            trace_dir = Path.cwd() / "traces"
        task_result = _eval_policy_with_trace(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            start_seed=start_seed,
            trace_dir=trace_dir,
        )

    per_episode = task_result["per_episode"]
    metrics = base.TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )
    if "debug_stats_infer" in task_result:
        metrics["debug_stats_infer"] = task_result["debug_stats_infer"]
    if "trace_paths" in task_result:
        metrics["trace_paths"] = task_result["trace_paths"]
    return metrics


base.eval_one = _patched_eval_one


if __name__ == "__main__":
    base.main()
