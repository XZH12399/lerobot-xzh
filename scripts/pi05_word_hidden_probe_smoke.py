#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05_word.configuration_pi05_word import PI05WordConfig
from lerobot.policies.pi05_word.modeling_pi05_word import PI05WordPolicy, make_att_2d_masks
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

IMAGE_KEY = f"{OBS_IMAGES}.cam_high"
PRIMITIVES = [
    "move_to",
    "approach",
    "retreat",
    "align",
    "follow_path",
    "hold_pose",
    "open",
    "close",
    "maintain",
    "release",
]
TARGETS = ["bowl", "cup", "drawer", "table", "surface", "none"]
RELATIONS = ["above", "near", "contact", "inside", "on", "none"]


class SmokeSemanticsHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.primitive_head = nn.Linear(hidden_dim, len(PRIMITIVES))
        self.target_head = nn.Linear(hidden_dim, len(TARGETS))
        self.relation_head = nn.Linear(hidden_dim, len(RELATIONS))

    def forward(self, pooled_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = pooled_hidden.to(torch.float32)
        return {
            "primitive_logits": self.primitive_head(hidden),
            "target_logits": self.target_head(hidden),
            "relation_logits": self.relation_head(hidden),
        }


def build_prompt(task: str, state: np.ndarray) -> str:
    cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
    discretized_states = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_states.tolist()))
    return f"Task: {cleaned_text}, State: {state_str};\nAction: "


def build_config(device: str, dtype: str) -> PI05WordConfig:
    config = PI05WordConfig(device=device, dtype=dtype)
    config.input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *config.image_resolution)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(config.max_state_dim,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(config.max_action_dim,)),
    }
    return config


def resolve_local_hf_snapshot(model_id_or_path: str) -> str:
    path = Path(model_id_or_path)
    if path.exists():
        return str(path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = cache_root / f"models--{model_id_or_path.replace('/', '--')}"
    if not cache_dir.exists():
        return model_id_or_path

    ref_file = cache_dir / "refs" / "main"
    if ref_file.exists():
        commit = ref_file.read_text(encoding="utf-8").strip()
        snapshot_dir = cache_dir / "snapshots" / commit
        if snapshot_dir.exists():
            return str(snapshot_dir)

    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return str(snapshots[0])

    return model_id_or_path


def tensor_shape(x: torch.Tensor | None) -> list[int] | None:
    return None if x is None else list(x.shape)


def main():
    parser = argparse.ArgumentParser(description="Minimal hidden-state probing smoke test for pi05_word.")
    parser.add_argument("--checkpoint", default="lerobot/pi05_libero_base")
    parser.add_argument("--tokenizer", default="google/paligemma-3b-pt-224")
    parser.add_argument("--task", default="pick up the bowl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--num-steps", type=int, default=1, help="FM denoising steps for action-path smoke test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    device = torch.device(args.device)
    config = build_config(device=str(device), dtype=args.dtype)
    checkpoint_path = resolve_local_hf_snapshot(args.checkpoint)
    tokenizer_path = resolve_local_hf_snapshot(args.tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt = build_prompt(args.task, np.zeros(config.max_state_dim, dtype=np.float32))
    tokenized = tokenizer(
        [prompt],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.tokenizer_max_length,
    )

    policy = PI05WordPolicy.from_pretrained(
        checkpoint_path,
        config=config,
        local_files_only=True,
        strict=False,
    )
    policy = policy.to(device)
    policy.eval()
    policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    batch = {
        IMAGE_KEY: torch.rand(1, 3, *config.image_resolution, device=device, dtype=torch.float32),
        OBS_LANGUAGE_TOKENS: tokenized["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].to(device=device, dtype=torch.bool),
    }

    with torch.inference_mode():
        images, img_masks = policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        (prefix_hidden, _), past_key_values = policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        hidden_mask = prefix_pad_masks.to(prefix_hidden.dtype).unsqueeze(-1)
        pooled_hidden = (prefix_hidden * hidden_mask).sum(dim=1) / hidden_mask.sum(dim=1).clamp_min(1.0)

        semantics_head = SmokeSemanticsHead(prefix_hidden.shape[-1]).to(device)
        semantics_outputs = semantics_head(pooled_hidden)

        action_chunk = policy.predict_action_chunk(batch, num_steps=args.num_steps)

    summary = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "resolved_checkpoint_path": checkpoint_path,
        "tokenizer": args.tokenizer,
        "resolved_tokenizer_path": tokenizer_path,
        "device": str(device),
        "dtype": args.dtype,
        "prompt": prompt,
        "image_key": IMAGE_KEY,
        "token_shape": list(batch[OBS_LANGUAGE_TOKENS].shape),
        "prefix_embs_shape": tensor_shape(prefix_embs),
        "prefix_pad_masks_shape": tensor_shape(prefix_pad_masks),
        "prefix_att_masks_shape": tensor_shape(prefix_att_masks),
        "prefix_hidden_shape": tensor_shape(prefix_hidden),
        "pooled_hidden_shape": tensor_shape(pooled_hidden),
        "pooled_hidden_l2_norm": float(pooled_hidden.float().norm(dim=-1).mean().item()),
        "past_key_values_layers": 0 if past_key_values is None else len(past_key_values),
        "primitive_logits_shape": tensor_shape(semantics_outputs["primitive_logits"]),
        "target_logits_shape": tensor_shape(semantics_outputs["target_logits"]),
        "relation_logits_shape": tensor_shape(semantics_outputs["relation_logits"]),
        "primitive_logits_finite": bool(torch.isfinite(semantics_outputs["primitive_logits"]).all().item()),
        "target_logits_finite": bool(torch.isfinite(semantics_outputs["target_logits"]).all().item()),
        "relation_logits_finite": bool(torch.isfinite(semantics_outputs["relation_logits"]).all().item()),
        "action_chunk_shape": tensor_shape(action_chunk),
        "action_chunk_finite": bool(torch.isfinite(action_chunk).all().item()),
        "action_chunk_mean_abs": float(action_chunk.float().abs().mean().item()),
    }

    output = json.dumps(summary, indent=2, ensure_ascii=True)
    print(output)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
