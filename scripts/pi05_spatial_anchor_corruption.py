#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run denoising-stage anchor corruption sanity checks for pi05_spatial."
    )
    parser.add_argument("--policy-path", required=True, help="Checkpoint pretrained_model directory.")
    parser.add_argument("--dataset-root", default="/home/ct_24210860031/data/libero")
    parser.add_argument("--dataset-repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)

    cfg = TrainPipelineConfig.from_pretrained(policy_path)
    cfg.batch_size = args.batch_size
    cfg.num_workers = 0
    cfg.dataset.root = args.dataset_root
    cfg.dataset.repo_id = args.dataset_repo_id
    cfg.policy.device = args.device
    cfg.policy.dtype = "bfloat16" if args.device.startswith("cuda") else "float32"
    cfg.policy.pretrained_path = policy_path

    dataset = make_dataset(cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preprocessor, _ = make_pre_post_processors(cfg.policy, pretrained_path=str(policy_path))
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    metrics: dict[str, list[float]] = {
        "coord_abs_mean": [],
        "delta_abs_mean": [],
        "delta_to_coord_ratio": [],
        "geom_hidden_delta_abs_mean": [],
        "geom_hidden_delta_ratio": [],
        "valid_main_frac": [],
        "valid_aux_frac": [],
        "shuffle_action_diff_ratio": [],
        "const_action_diff_ratio": [],
    }

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= args.num_batches:
                break

            batch = preprocessor(raw_batch)
            images, img_masks = policy._preprocess_images(batch)
            tokens = batch[OBS_LANGUAGE_TOKENS]
            masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            state = policy.prepare_state(batch)
            current_eef_pos = policy.prepare_current_eef_pos(batch)
            camera_context = policy.extract_camera_context(batch)

            noise_shape = (
                tokens.shape[0],
                policy.config.chunk_size,
                policy.config.max_action_dim,
            )
            noise = policy.model.sample_noise(noise_shape, tokens.device)

            base_action, spatial_context = policy.model.sample_trajectory(
                images,
                img_masks,
                tokens,
                masks,
                state,
                current_eef_pos=current_eef_pos,
                camera_context=camera_context,
                noise=noise,
                return_spatial_context=True,
            )
            shuffle_action = policy.model.sample_actions(
                images,
                img_masks,
                tokens,
                masks,
                state,
                current_eef_pos=current_eef_pos,
                camera_context=camera_context,
                noise=noise,
                coord_corruption_mode="shuffle",
            )
            const_action = policy.model.sample_actions(
                images,
                img_masks,
                tokens,
                masks,
                state,
                current_eef_pos=current_eef_pos,
                camera_context=camera_context,
                noise=noise,
                coord_corruption_mode="constant",
            )

            zero_timestep = torch.zeros(tokens.shape[0], dtype=torch.float32, device=tokens.device)
            geometry_context = policy.model.predict_refined_trajectory_from_action_like(
                base_action,
                state,
                current_eef_pos=current_eef_pos,
                spatial_context=spatial_context,
                camera_context=camera_context,
                timestep=zero_timestep,
            )

            denom = base_action.abs().mean().clamp_min(1e-6)
            coord_abs_mean = geometry_context["coord_state"].abs().mean().item()
            delta_abs_mean = geometry_context["delta_coord"].abs().mean().item()

            metrics["coord_abs_mean"].append(coord_abs_mean)
            metrics["delta_abs_mean"].append(delta_abs_mean)
            metrics["delta_to_coord_ratio"].append(delta_abs_mean / max(coord_abs_mean, 1e-6))
            metrics["geom_hidden_delta_abs_mean"].append(geometry_context["hidden_delta_abs_mean"].mean().item())
            metrics["geom_hidden_delta_ratio"].append(geometry_context["hidden_delta_ratio"].mean().item())
            metrics["valid_main_frac"].append(geometry_context["valid_main"].float().mean().item())
            metrics["valid_aux_frac"].append(geometry_context["valid_aux"].float().mean().item())
            metrics["shuffle_action_diff_ratio"].append(((shuffle_action - base_action).abs().mean() / denom).item())
            metrics["const_action_diff_ratio"].append(((const_action - base_action).abs().mean() / denom).item())

    result = {key: sum(values) / len(values) for key, values in metrics.items()}
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
