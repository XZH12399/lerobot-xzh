#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05_spatial.configuration_pi05_spatial import DEFAULT_IMAGE_SIZE, PI05SpatialConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def slice_action_trajectory(actions: Tensor, start_index: int, trajectory_dim: int) -> Tensor:
    """Extract the trajectory-related action subspace used for fallback supervision."""
    end_index = start_index + trajectory_dim
    if actions.shape[-1] < end_index:
        raise ValueError(
            f"Cannot slice trajectory dims [{start_index}:{end_index}] from actions with shape {tuple(actions.shape)}"
        )
    return actions[..., start_index:end_index]


def integrate_delta_trajectory(current_eef_pos: Tensor, delta_actions: Tensor) -> Tensor:
    """Approximate a future trajectory by integrating relative action deltas from the current EE position."""
    if current_eef_pos.dim() != 2:
        raise ValueError(
            f"current_eef_pos must have shape (B, D), got shape {tuple(current_eef_pos.shape)}"
        )
    if delta_actions.dim() != 3:
        raise ValueError(f"delta_actions must have shape (B, T, D), got shape {tuple(delta_actions.shape)}")
    cumulative_delta = torch.cumsum(delta_actions, dim=1)
    return current_eef_pos[:, None, :] + cumulative_delta


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05Spatial."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class SpatialFeatureExtractor(nn.Module):
    """Lightweight multi-scale CNN used for geometry-aware local sampling."""

    def __init__(self, feature_dims: tuple[int, int, int]):
        super().__init__()
        c1, c2, c3 = feature_dims
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(self._group_count(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self._group_count(c1), c1),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(self._group_count(c2), c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self._group_count(c2), c2),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(self._group_count(c3), c3),
            nn.SiLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self._group_count(c3), c3),
            nn.SiLU(),
        )

    @staticmethod
    def _group_count(num_channels: int) -> int:
        for group_count in range(min(8, num_channels), 0, -1):
            if num_channels % group_count == 0:
                return group_count
        return 1

    def forward(self, image: Tensor) -> list[Tensor]:
        high = self.stage1(image)
        mid = self.stage2(high)
        low = self.stage3(mid)
        return [low, mid, high]


class Trajectory3DTokenizer(nn.Module):
    """Builds trajectory tokens from explicit 3D points plus state/time context."""

    def __init__(
        self,
        trajectory_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_fourier_bands: int,
    ):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.num_fourier_bands = num_fourier_bands
        coord_input_dim = trajectory_dim * num_fourier_bands * 2
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.token_fuse = nn.Sequential(
            nn.Linear(model_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def fourier_encode(self, trajectory_xyz: Tensor) -> Tensor:
        freq_bands = torch.pow(
            trajectory_xyz.new_tensor(2.0),
            torch.arange(self.num_fourier_bands, device=trajectory_xyz.device, dtype=trajectory_xyz.dtype),
        )
        scaled = trajectory_xyz.unsqueeze(-1) * freq_bands * math.pi
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(start_dim=-2)

    def forward(
        self,
        trajectory_emb: Tensor,
        state_emb: Tensor,
        time_emb: Tensor,
        trajectory_xyz: Tensor,
    ) -> tuple[Tensor, Tensor]:
        coord_emb = self.coord_proj(self.fourier_encode(trajectory_xyz))
        state_ctx = state_emb[:, None, :].expand(-1, trajectory_emb.shape[1], -1)
        time_ctx = time_emb[:, None, :].expand_as(state_ctx)
        token_delta = self.token_fuse(torch.cat([trajectory_emb, coord_emb, state_ctx, time_ctx], dim=-1))
        tokens = trajectory_emb + self.residual_gate * token_delta
        return tokens, coord_emb


class ProjectAndSample(nn.Module):
    """Projects 3D points to an image plane and samples anchor-centered local windows."""

    def __init__(
        self,
        image_resolution: tuple[int, int],
        projection_fallback_scale: float,
        local_window_radius: int = 0,
        local_window_sigma: float = 1.0,
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.projection_fallback_scale = projection_fallback_scale
        self.local_window_radius = local_window_radius
        self.local_window_sigma = local_window_sigma

    def _apply_transform(self, points: Tensor, transform: Tensor | None) -> Tensor:
        if transform is None:
            return points
        ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
        points_h = torch.cat([points, ones], dim=-1)
        transformed = torch.matmul(transform[:, None, :, :], points_h.unsqueeze(-1)).squeeze(-1)
        return transformed[..., :3]

    def project_points(
        self,
        points: Tensor,
        intrinsics: Tensor | None = None,
        transform: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        points_cam = self._apply_transform(points, transform)
        z = points_cam[..., 2].clamp_min(1e-4)

        if intrinsics is not None:
            fx = intrinsics[:, 0, 0][:, None]
            fy = intrinsics[:, 1, 1][:, None]
            cx = intrinsics[:, 0, 2][:, None]
            cy = intrinsics[:, 1, 2][:, None]
            u = fx * (points_cam[..., 0] / z) + cx
            v = fy * (points_cam[..., 1] / z) + cy
            width = max(self.image_resolution[1] - 1, 1)
            height = max(self.image_resolution[0] - 1, 1)
            grid_x = 2.0 * (u / width) - 1.0
            grid_y = 2.0 * (v / height) - 1.0
        else:
            denom = points_cam[..., 2].abs().clamp_min(self.projection_fallback_scale)
            grid_x = torch.tanh(points_cam[..., 0] / denom)
            grid_y = torch.tanh(points_cam[..., 1] / denom)

        grid = torch.stack([grid_x, grid_y], dim=-1)
        valid = (z > 1e-4) & (grid.abs() <= 1.0).all(dim=-1)
        return grid, valid

    def _sample_local_window(self, feature_map: Tensor, grid: Tensor) -> Tensor:
        base_grid = grid.to(dtype=feature_map.dtype)
        if self.local_window_radius <= 0:
            sampled = F.grid_sample(
                feature_map,
                base_grid.unsqueeze(2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            return sampled.squeeze(-1).transpose(1, 2)

        height = max(feature_map.shape[-2], 1)
        width = max(feature_map.shape[-1], 1)
        step_x = 2.0 / max(width - 1, 1)
        step_y = 2.0 / max(height - 1, 1)
        sigma_sq = max(self.local_window_sigma, 1e-6) ** 2

        offsets = []
        weights = []
        for delta_y in range(-self.local_window_radius, self.local_window_radius + 1):
            for delta_x in range(-self.local_window_radius, self.local_window_radius + 1):
                offsets.append((delta_x * step_x, delta_y * step_y))
                weights.append(math.exp(-((delta_x**2 + delta_y**2) / (2.0 * sigma_sq))))

        offsets_t = base_grid.new_tensor(offsets).view(1, 1, -1, 2)
        weights_t = feature_map.new_tensor(weights).view(1, 1, -1)
        local_grid = base_grid.unsqueeze(2) + offsets_t
        sampled = F.grid_sample(
            feature_map,
            local_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        valid_window = (local_grid.abs() <= 1.0).all(dim=-1)
        weights_t = weights_t * valid_window.to(dtype=weights_t.dtype)
        weights_sum = weights_t.sum(dim=-1).clamp_min(1e-6)
        pooled = (sampled * weights_t[:, None, :, :]).sum(dim=-1) / weights_sum[:, None, :]
        return pooled.transpose(1, 2)

    def forward(
        self,
        points: Tensor,
        feature_maps: list[Tensor],
        intrinsics: Tensor | None = None,
        transform: Tensor | None = None,
        image_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        grid, valid = self.project_points(points, intrinsics=intrinsics, transform=transform)
        sampled_features = []
        for feature_map in feature_maps:
            sampled_features.append(self._sample_local_window(feature_map, grid))

        sampled = torch.cat(sampled_features, dim=-1)
        sampled = sampled * valid[:, :, None].to(dtype=sampled.dtype)
        if image_mask is not None:
            sampled = sampled * image_mask[:, None, None].to(dtype=sampled.dtype)
        return sampled, grid, valid


class GeometryRefinementBlock(nn.Module):
    """Injects local dual-view geometry evidence into trajectory tokens."""

    def __init__(
        self,
        model_dim: int,
        sampled_feature_dim: int,
        hidden_dim: int,
        trajectory_dim: int,
        hidden_residual_scale: float,
        coord_delta_scale: float,
    ):
        super().__init__()
        geom_input_dim = model_dim + 2 * sampled_feature_dim + 2 * model_dim
        self.geometry_proj = nn.Sequential(
            nn.Linear(geom_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.token_refine = nn.Sequential(
            nn.Linear(model_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )
        self.hidden_residual_scale = hidden_residual_scale
        self.coord_delta_scale = coord_delta_scale

    def forward(
        self,
        trajectory_tokens: Tensor,
        coord_emb: Tensor,
        sampled_main: Tensor,
        sampled_aux: Tensor,
        state_emb: Tensor,
        time_emb: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state_ctx = state_emb[:, None, :].expand(-1, trajectory_tokens.shape[1], -1)
        time_ctx = time_emb[:, None, :].expand_as(state_ctx)
        geometry_inputs = torch.cat([coord_emb, sampled_main, sampled_aux, state_ctx, time_ctx], dim=-1)
        geom_emb = self.geometry_proj(geometry_inputs)
        token_delta = self.token_refine(torch.cat([trajectory_tokens, geom_emb], dim=-1))
        hidden_delta = self.hidden_residual_scale * token_delta
        refined_tokens = trajectory_tokens + hidden_delta
        delta_xyz = self.coord_delta_scale * self.delta_head(geom_emb)
        return refined_tokens, delta_xyz, hidden_delta


class TrajectoryToActionDecoder(nn.Module):
    """Maps a denoised 3D trajectory back to the original action space."""

    def __init__(self, trajectory_dim: int, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trajectory_dim + state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Keep the decoder active from step 0 so gradients reach its MLP immediately.
        self.residual_gate = nn.Parameter(torch.ones(1))

    def forward(self, trajectory_xyz: Tensor, state: Tensor) -> Tensor:
        state_ctx = state[:, None, :].expand(-1, trajectory_xyz.shape[1], -1)
        return self.residual_gate * self.net(torch.cat([trajectory_xyz, state_ctx], dim=-1))


class PI05SpatialPytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05Spatial PyTorch model."""

    def __init__(self, config: PI05SpatialConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.spatial_feature_extractor = SpatialFeatureExtractor(config.spatial_feature_dims)
        self.trajectory_tokenizer = Trajectory3DTokenizer(
            trajectory_dim=config.trajectory_dim,
            model_dim=action_expert_config.width,
            hidden_dim=config.geometry_hidden_dim,
            num_fourier_bands=config.geometry_num_fourier_bands,
        )
        self.project_and_sample = ProjectAndSample(
            image_resolution=config.image_resolution,
            projection_fallback_scale=config.projection_fallback_scale,
            local_window_radius=config.geometry_local_window_radius,
            local_window_sigma=config.geometry_local_window_sigma,
        )
        self.geometry_refinement = GeometryRefinementBlock(
            model_dim=action_expert_config.width,
            sampled_feature_dim=sum(config.spatial_feature_dims),
            hidden_dim=config.geometry_hidden_dim,
            trajectory_dim=config.trajectory_dim,
            hidden_residual_scale=config.geometry_hidden_residual_scale,
            coord_delta_scale=config.geometry_coord_delta_scale,
        )
        self.trajectory_decoder = TrajectoryToActionDecoder(
            trajectory_dim=config.trajectory_dim,
            state_dim=config.max_state_dim,
            action_dim=config.max_action_dim,
            hidden_dim=config.trajectory_decoder_hidden_dim,
        )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05SpatialPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05SpatialPytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def encode_time_embedding(self, timestep: Tensor) -> Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        return self._apply_checkpoint(time_mlp_func, time_emb)

    def extract_spatial_context(self, images, img_masks) -> dict[str, Any]:
        if len(images) == 0:
            raise ValueError("At least one image is required for PI05Spatial.")

        max_index = len(images) - 1
        main_index = min(self.config.main_camera_index, max_index)
        aux_index = min(self.config.aux_camera_index, max_index)

        main_image = images[main_index]
        main_mask = img_masks[main_index]
        main_features = self.spatial_feature_extractor(main_image)

        if len(images) == 1:
            aux_features = [torch.zeros_like(feature_map) for feature_map in main_features]
            aux_mask = torch.zeros_like(main_mask)
        else:
            aux_image = images[aux_index]
            aux_features = self.spatial_feature_extractor(aux_image)
            aux_mask = img_masks[aux_index]

        return {
            "main_features": main_features,
            "aux_features": aux_features,
            "main_mask": main_mask,
            "aux_mask": aux_mask,
        }

    def initialize_coordinate_state(
        self,
        action_like: Tensor,
        current_eef_pos: Tensor | None,
    ) -> Tensor:
        coord_delta = slice_action_trajectory(
            action_like,
            start_index=self.config.trajectory_action_start_index,
            trajectory_dim=self.config.trajectory_dim,
        )
        if current_eef_pos is None:
            return torch.cumsum(coord_delta, dim=1)

        current_coord = current_eef_pos[..., : self.config.trajectory_dim]
        if current_coord.dim() == 1:
            current_coord = current_coord.unsqueeze(0)
        return integrate_delta_trajectory(current_coord, coord_delta)

    def apply_coord_state_corruption(
        self,
        coord_state: Tensor,
        corruption_mode: Literal["shuffle", "constant"] | None = None,
    ) -> Tensor:
        if corruption_mode is None:
            return coord_state
        if corruption_mode == "shuffle":
            if coord_state.shape[0] <= 1:
                return coord_state
            return torch.roll(coord_state, shifts=1, dims=0)
        if corruption_mode == "constant":
            return coord_state.mean(dim=0, keepdim=True).expand_as(coord_state)
        raise ValueError(f"Unsupported coord corruption mode: {corruption_mode}")

    def embed_suffix(self, noisy_actions, coord_state, timestep, state, spatial_context, camera_context):
        """Embed noisy action tokens together with explicit coordinate anchors."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state_value):
            return self.state_proj(state_value)

        def action_proj_func(noisy_action_value):
            return self.action_in_proj(noisy_action_value)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        time_emb = self.encode_time_embedding(timestep)
        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        action_tokens, coord_emb = self.trajectory_tokenizer(
            action_emb, state_emb, time_emb, coord_state
        )

        sampled_main, uv_main, valid_main = self.project_and_sample(
            coord_state,
            spatial_context["main_features"],
            intrinsics=camera_context.get("main_intrinsics"),
            transform=None,
            image_mask=spatial_context["main_mask"],
        )
        sampled_aux, uv_aux, valid_aux = self.project_and_sample(
            coord_state,
            spatial_context["aux_features"],
            intrinsics=camera_context.get("aux_intrinsics"),
            transform=camera_context.get("aux_from_main"),
            image_mask=spatial_context["aux_mask"],
        )
        refined_tokens, delta_coord, hidden_delta = self.geometry_refinement(
            action_tokens,
            coord_emb,
            sampled_main,
            sampled_aux,
            state_emb,
            time_emb,
        )
        pred_coord_state = coord_state + delta_coord
        adarms_cond = time_emb
        hidden_delta_abs_mean = hidden_delta.abs().mean(dim=(1, 2))
        base_token_abs_mean = action_tokens.abs().mean(dim=(1, 2)).clamp_min(1e-6)

        embs.append(refined_tokens)
        bsize, action_token_dim = refined_tokens.shape[:2]
        action_token_mask = torch.ones(bsize, action_token_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_token_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        geometry_context = {
            "coord_state": coord_state,
            "pred_coord_state": pred_coord_state,
            "delta_coord": delta_coord,
            "uv_main": uv_main,
            "uv_aux": uv_aux,
            "valid_main": valid_main,
            "valid_aux": valid_aux,
            "hidden_delta_abs_mean": hidden_delta_abs_mean,
            "hidden_delta_ratio": hidden_delta_abs_mean / base_token_abs_mean,
            "hidden_residual_scale": hidden_delta_abs_mean.new_full(
                hidden_delta_abs_mean.shape, self.geometry_refinement.hidden_residual_scale
            ),
            "coord_delta_scale": hidden_delta_abs_mean.new_full(
                hidden_delta_abs_mean.shape, self.geometry_refinement.coord_delta_scale
            ),
        }

        return embs, pad_masks, att_masks, adarms_cond, geometry_context

    def combine_velocity_prediction(self, suffix_out: Tensor) -> Tensor:
        v_t = self.action_out_proj(suffix_out)
        return v_t.to(dtype=torch.float32)

    def predict_refined_trajectory_from_action_like(
        self,
        action_like: Tensor,
        state: Tensor,
        *,
        current_eef_pos: Tensor | None = None,
        spatial_context: dict[str, Any] | None = None,
        camera_context: dict[str, Tensor | None] | None = None,
        images=None,
        img_masks=None,
        timestep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Refine a denoised action-like sequence into a geometry-conditioned 3D trajectory."""
        if camera_context is None:
            camera_context = {}
        if spatial_context is None:
            if images is None or img_masks is None:
                raise ValueError("images and img_masks are required when spatial_context is not provided.")
            spatial_context = self.extract_spatial_context(images, img_masks)
        if timestep is None:
            timestep = torch.zeros(action_like.shape[0], dtype=torch.float32, device=action_like.device)

        coord_state = self.initialize_coordinate_state(action_like, current_eef_pos)
        _, _, _, _, geometry_context = self.embed_suffix(
            action_like,
            coord_state,
            timestep,
            state,
            spatial_context,
            camera_context,
        )
        return geometry_context

    def decode_trajectory_to_actions(
        self,
        pred_clean_trajectory: Tensor,
        state: Tensor,
        denoised_actions: Tensor | None = None,
    ) -> Tensor:
        del pred_clean_trajectory, state
        if denoised_actions is None:
            raise ValueError("denoised_actions is required when pi05_spatial keeps the FM final action branch.")
        return denoised_actions.to(dtype=torch.float32)

    def apply_sequence_mask(self, value: Tensor, valid_mask: Tensor | None) -> Tensor:
        if valid_mask is None:
            return value

        mask = valid_mask
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        return value * mask.to(dtype=value.dtype)

    def reduce_sequence_loss(self, value: Tensor, valid_mask: Tensor | None) -> Tensor:
        if valid_mask is None:
            return value.mean(dim=tuple(range(1, value.dim())))

        mask = valid_mask
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=value.dtype).expand_as(value)
        denom = mask.sum(dim=tuple(range(1, value.dim()))).clamp_min(1.0)
        return (value * mask).sum(dim=tuple(range(1, value.dim()))) / denom

    def compute_projection_loss(
        self,
        pred_trajectory: Tensor,
        target_trajectory: Tensor,
        camera_context: dict[str, Tensor | None],
        spatial_context: dict[str, Any],
        trajectory_valid_mask: Tensor | None = None,
    ) -> Tensor:
        view_specs = (
            (
                camera_context.get("main_intrinsics"),
                None,
                spatial_context["main_mask"],
            ),
            (
                camera_context.get("aux_intrinsics"),
                camera_context.get("aux_from_main"),
                spatial_context["aux_mask"],
            ),
        )

        total_loss = pred_trajectory.new_zeros(pred_trajectory.shape[0])
        valid_views = pred_trajectory.new_zeros(pred_trajectory.shape[0])

        for intrinsics, transform, image_mask in view_specs:
            pred_grid, pred_valid = self.project_and_sample.project_points(
                pred_trajectory,
                intrinsics=intrinsics,
                transform=transform,
            )
            target_grid, target_valid = self.project_and_sample.project_points(
                target_trajectory,
                intrinsics=intrinsics,
                transform=transform,
            )
            valid = pred_valid & target_valid
            if image_mask is not None:
                valid = valid & image_mask[:, None]
            if trajectory_valid_mask is not None:
                valid = valid & trajectory_valid_mask

            view_loss = torch.abs(pred_grid - target_grid).mean(dim=-1)
            view_loss = torch.where(valid, view_loss, torch.zeros_like(view_loss))
            denom = valid.to(dtype=view_loss.dtype).sum(dim=1).clamp_min(1.0)
            total_loss = total_loss + view_loss.sum(dim=1) / denom
            valid_views = valid_views + valid.any(dim=1).to(dtype=total_loss.dtype)

        return total_loss / valid_views.clamp_min(1.0)

    def compute_smoothness_loss(
        self, trajectory: Tensor, trajectory_valid_mask: Tensor | None = None
    ) -> Tensor:
        if trajectory.shape[1] < 2:
            return trajectory.new_zeros(trajectory.shape[0])

        velocity = trajectory[:, 1:] - trajectory[:, :-1]
        velocity_valid_mask = None
        if trajectory_valid_mask is not None:
            velocity_valid_mask = trajectory_valid_mask[:, 1:] & trajectory_valid_mask[:, :-1]
        vel_loss = self.reduce_sequence_loss(velocity.abs(), velocity_valid_mask)

        if trajectory.shape[1] < 3:
            return vel_loss

        acceleration = trajectory[:, 2:] - 2 * trajectory[:, 1:-1] + trajectory[:, :-2]
        acceleration_valid_mask = None
        if trajectory_valid_mask is not None:
            acceleration_valid_mask = (
                trajectory_valid_mask[:, 2:]
                & trajectory_valid_mask[:, 1:-1]
                & trajectory_valid_mask[:, :-2]
            )
        acc_loss = self.reduce_sequence_loss(acceleration.abs(), acceleration_valid_mask)
        return vel_loss + acc_loss

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        state,
        trajectory,
        current_eef_pos=None,
        action_targets=None,
        camera_context=None,
        trajectory_is_pad=None,
        action_is_pad=None,
        noise=None,
        time=None,
    ) -> dict[str, Tensor | None]:
        """Do a full training forward pass and compute all spatial MVP losses."""
        if camera_context is None:
            camera_context = {}

        diffusion_target = trajectory if action_targets is None else action_targets

        if noise is None:
            noise = self.sample_noise(diffusion_target.shape, diffusion_target.device)

        if time is None:
            time = self.sample_time(diffusion_target.shape[0], diffusion_target.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * diffusion_target
        u_t = noise - diffusion_target
        coord_state = self.initialize_coordinate_state(x_t, current_eef_pos)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        spatial_context = self.extract_spatial_context(images, img_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, geometry_context = self.embed_suffix(
            x_t,
            coord_state,
            time,
            state,
            spatial_context,
            camera_context,
        )

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.combine_velocity_prediction(suffix_out)

        trajectory_valid_mask = None
        if trajectory_is_pad is not None:
            trajectory_valid_mask = ~trajectory_is_pad.bool()
        elif action_is_pad is not None:
            trajectory_valid_mask = ~action_is_pad.bool()

        action_valid_mask = None if action_is_pad is None else ~action_is_pad.bool()
        if action_targets is None:
            diffusion_loss_tensor = F.mse_loss(
                slice_action_trajectory(
                    u_t,
                    start_index=self.config.trajectory_action_start_index,
                    trajectory_dim=self.config.trajectory_dim,
                ),
                slice_action_trajectory(
                    v_t,
                    start_index=self.config.trajectory_action_start_index,
                    trajectory_dim=self.config.trajectory_dim,
                ),
                reduction="none",
            )
            diffusion_valid_mask = trajectory_valid_mask
        else:
            diffusion_loss_tensor = F.mse_loss(u_t, v_t, reduction="none")
            diffusion_valid_mask = action_valid_mask

        diffusion_loss_tensor = self.apply_sequence_mask(diffusion_loss_tensor, diffusion_valid_mask)
        diffusion_loss = self.reduce_sequence_loss(diffusion_loss_tensor, diffusion_valid_mask)

        pred_clean_action = x_t - time_expanded * v_t
        action_geometry_context = self.predict_refined_trajectory_from_action_like(
            pred_clean_action,
            state,
            current_eef_pos=current_eef_pos,
            spatial_context=spatial_context,
            camera_context=camera_context,
        )
        pred_clean_trajectory = action_geometry_context["pred_coord_state"]
        target_trajectory = trajectory[..., : self.config.trajectory_dim]
        trajectory_loss_tensor = torch.abs(pred_clean_trajectory - target_trajectory)
        trajectory_loss_tensor = self.apply_sequence_mask(trajectory_loss_tensor, trajectory_valid_mask)
        trajectory_loss = self.reduce_sequence_loss(trajectory_loss_tensor, trajectory_valid_mask)
        projection_loss = self.compute_projection_loss(
            pred_clean_trajectory,
            target_trajectory,
            camera_context,
            spatial_context,
            trajectory_valid_mask=trajectory_valid_mask,
        )
        smoothness_loss = self.compute_smoothness_loss(
            pred_clean_trajectory, trajectory_valid_mask=trajectory_valid_mask
        )

        pred_action = self.decode_trajectory_to_actions(
            pred_clean_trajectory,
            state,
            denoised_actions=pred_clean_action,
        )
        action_loss_tensor = None
        action_loss = None
        if action_targets is not None:
            original_action_dim = self.config.output_features[ACTION].shape[0]
            action_loss_tensor = F.l1_loss(
                pred_action[..., :original_action_dim],
                action_targets[..., :original_action_dim],
                reduction="none",
            )
            action_loss_tensor = self.apply_sequence_mask(action_loss_tensor, action_valid_mask)
            action_loss = self.reduce_sequence_loss(action_loss_tensor, action_valid_mask)

        return {
            "diffusion_loss_tensor": diffusion_loss_tensor,
            "diffusion_loss": diffusion_loss,
            "trajectory_loss_tensor": trajectory_loss_tensor,
            "trajectory_loss": trajectory_loss,
            "projection_loss": projection_loss,
            "smoothness_loss": smoothness_loss,
            "action_loss_tensor": action_loss_tensor,
            "action_loss": action_loss,
            "pred_clean_trajectory": pred_clean_trajectory,
            "pred_action": pred_action,
            "coord_abs_mean": action_geometry_context["coord_state"].abs().mean(),
            "delta_abs_mean": action_geometry_context["delta_coord"].abs().mean(),
            "pred_clean_trajectory_std": pred_clean_trajectory.std(unbiased=False),
            "geom_hidden_delta_abs_mean": action_geometry_context["hidden_delta_abs_mean"],
            "geom_hidden_delta_ratio": action_geometry_context["hidden_delta_ratio"],
            "geometry_hidden_residual_scale": action_geometry_context["hidden_residual_scale"],
            "geometry_coord_delta_scale": action_geometry_context["coord_delta_scale"],
            "valid_main_frac": action_geometry_context["valid_main"].to(dtype=torch.float32).mean(),
            "valid_aux_frac": action_geometry_context["valid_aux"].to(dtype=torch.float32).mean(),
        }

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_trajectory(
        self,
        images,
        img_masks,
        tokens,
        masks,
        state,
        current_eef_pos=None,
        camera_context=None,
        noise=None,
        num_steps=None,
        coord_corruption_mode: Literal["shuffle", "constant"] | None = None,
        return_spatial_context: bool = False,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Run denoising and optionally return the cached spatial context for geometry decoding."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if camera_context is None:
            camera_context = {}

        if noise is None:
            trajectory_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(trajectory_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        spatial_context = self.extract_spatial_context(images, img_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    state=state,
                    spatial_context=spatial_context,
                    camera_context=camera_context,
                    current_eef_pos=current_eef_pos,
                    coord_corruption_mode=coord_corruption_mode,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        if return_spatial_context:
            return x_t, spatial_context
        return x_t

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        state,
        current_eef_pos=None,
        camera_context=None,
        noise=None,
        num_steps=None,
        coord_corruption_mode: Literal["shuffle", "constant"] | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        denoised_actions = self.sample_trajectory(
            images,
            img_masks,
            tokens,
            masks,
            state,
            current_eef_pos=current_eef_pos,
            camera_context=camera_context,
            noise=noise,
            num_steps=num_steps,
            coord_corruption_mode=coord_corruption_mode,
            **kwargs,
        )
        return denoised_actions.to(dtype=torch.float32)

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        state,
        spatial_context,
        camera_context,
        current_eef_pos=None,
        coord_corruption_mode: Literal["shuffle", "constant"] | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        coord_state = self.initialize_coordinate_state(x_t, current_eef_pos)
        coord_state = self.apply_coord_state_corruption(coord_state, coord_corruption_mode)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, geometry_context = self.embed_suffix(
            x_t,
            coord_state,
            timestep,
            state,
            spatial_context,
            camera_context,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.combine_velocity_prediction(suffix_out)


class PI05SpatialPolicy(PreTrainedPolicy):
    """PI05Spatial Policy for LeRobot."""

    config_class = PI05SpatialConfig
    name = "pi05_spatial"

    def __init__(
        self,
        config: PI05SpatialConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05Spatial model
        self.init_rtc_processor()
        self.model = PI05SpatialPytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def _inherit_compatible_pretrained_settings(
        cls,
        config: PI05SpatialConfig,
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> PI05SpatialConfig:
        """Reuse compatible runtime settings from a source checkpoint when bootstrapping from base pi05."""
        try:
            source_config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
        except Exception as exc:  # best effort only
            logging.warning(f"Could not inspect pretrained config for PI05Spatial inheritance: {exc}")
            return config

        if isinstance(source_config, PI05SpatialConfig):
            return config

        default_config = cls.config_class()
        inherited_fields = []
        field_names = (
            "empty_cameras",
            "n_action_steps",
            "chunk_size",
            "image_resolution",
            "max_state_dim",
            "max_action_dim",
            "num_inference_steps",
            "tokenizer_max_length",
        )
        for field_name in field_names:
            if not hasattr(source_config, field_name):
                continue
            if getattr(config, field_name) != getattr(default_config, field_name):
                continue
            setattr(config, field_name, copy.deepcopy(getattr(source_config, field_name)))
            inherited_fields.append(field_name)

        if inherited_fields:
            logging.info(
                "PI05Spatial inherited compatible pretrained settings from %s: %s",
                pretrained_name_or_path,
                ", ".join(inherited_fields),
            )

        return config

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05Spatial model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        elif isinstance(config, PI05SpatialConfig):
            config = cls._inherit_compatible_pretrained_settings(
                config,
                pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05_spatial
            # pi05_spatial model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _as_batched_tensor(self, value: Any) -> Tensor:
        device = next(self.parameters()).device
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
        else:
            value = value.to(device=device)
            if value.dtype != torch.float32:
                value = value.to(torch.float32)
        return value

    def _as_batched_mask(self, value: Any) -> Tensor:
        device = next(self.parameters()).device
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=torch.bool, device=device)
        else:
            value = value.to(device=device, dtype=torch.bool)
        return value

    def _prepare_optional_matrix(self, batch: dict[str, Tensor], key: str) -> Tensor | None:
        value = batch.get(key)
        if value is None:
            return None
        value = self._as_batched_tensor(value)
        if value.dim() == 2:
            value = value.unsqueeze(0)
        return value

    def extract_camera_context(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        return {
            "main_intrinsics": self._prepare_optional_matrix(batch, self.config.camera_main_intrinsics_key),
            "aux_intrinsics": self._prepare_optional_matrix(batch, self.config.camera_aux_intrinsics_key),
            "aux_from_main": self._prepare_optional_matrix(batch, self.config.camera_aux_from_main_key),
        }

    def prepare_state(self, batch):
        """Pad state and ensure a batch dimension exists."""
        state = self._as_batched_tensor(batch[OBS_STATE])
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = self._as_batched_tensor(batch[ACTION])
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        actions = pad_vector(actions, self.config.max_action_dim)
        return actions

    def prepare_optional_pad_mask(self, batch: dict[str, Tensor], key: str) -> Tensor | None:
        value = batch.get(key)
        if value is None:
            return None
        value = self._as_batched_mask(value)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        return value[:, : self.config.chunk_size]

    def prepare_trajectory(self, batch, actions: Tensor | None = None) -> Tensor:
        trajectory = batch.get(self.config.trajectory_key)
        if trajectory is None:
            if actions is None:
                actions = self.prepare_action(batch)

            delta_actions = slice_action_trajectory(
                actions,
                start_index=self.config.trajectory_action_start_index,
                trajectory_dim=self.config.trajectory_dim,
            )

            current_eef_pos = batch.get(self.config.current_eef_pos_key)
            if current_eef_pos is not None and self.config.derive_trajectory_from_eef_delta:
                current_eef_pos = self._as_batched_tensor(current_eef_pos)
                if current_eef_pos.dim() == 1:
                    current_eef_pos = current_eef_pos.unsqueeze(0)
                current_eef_pos = current_eef_pos[..., : self.config.trajectory_dim]
                trajectory = integrate_delta_trajectory(current_eef_pos, delta_actions)
            elif not self.config.use_action_as_trajectory_fallback:
                raise ValueError(
                    f"Trajectory key '{self.config.trajectory_key}' is missing and action fallback is disabled."
                )
            else:
                trajectory = delta_actions
        else:
            trajectory = self._as_batched_tensor(trajectory)
            if trajectory.dim() == 2:
                trajectory = trajectory.unsqueeze(0)
            trajectory = trajectory[..., : self.config.trajectory_dim]

        return trajectory

    def prepare_current_eef_pos(self, batch):
        current_eef_pos = batch.get(self.config.current_eef_pos_key)
        if current_eef_pos is None:
            current_eef_pos = batch.get(OBS_STATE)
        if current_eef_pos is None:
            return None
        current_eef_pos = self._as_batched_tensor(current_eef_pos)
        if current_eef_pos.dim() == 1:
            current_eef_pos = current_eef_pos.unsqueeze(0)
        if current_eef_pos.dim() == 3:
            current_eef_pos = current_eef_pos[:, 0]
        return current_eef_pos[..., : self.config.trajectory_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_trajectory_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a future 3D trajectory in the main camera frame."""
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        current_eef_pos = self.prepare_current_eef_pos(batch)
        camera_context = self.extract_camera_context(batch)
        sampled_actions = self.model.sample_trajectory(
            images,
            img_masks,
            tokens,
            masks,
            state,
            current_eef_pos=current_eef_pos,
            camera_context=camera_context,
            **kwargs,
        )
        return self.model.initialize_coordinate_state(sampled_actions, current_eef_pos)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        current_eef_pos = self.prepare_current_eef_pos(batch)
        camera_context = self.extract_camera_context(batch)

        # Sample actions using the model (pass through RTC kwargs)
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            state,
            current_eef_pos=current_eef_pos,
            camera_context=camera_context,
            **kwargs,
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        trajectory = self.prepare_trajectory(batch, actions=actions)
        current_eef_pos = self.prepare_current_eef_pos(batch)
        trajectory_is_pad = self.prepare_optional_pad_mask(batch, self.config.trajectory_pad_key)
        action_is_pad = self.prepare_optional_pad_mask(batch, self.config.action_pad_key)
        camera_context = self.extract_camera_context(batch)

        outputs = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            state,
            trajectory,
            current_eef_pos=current_eef_pos,
            action_targets=actions,
            camera_context=camera_context,
            trajectory_is_pad=trajectory_is_pad if self.config.mask_padding_loss else None,
            action_is_pad=action_is_pad if self.config.mask_padding_loss else None,
        )
        total_loss = (
            outputs["diffusion_loss"]
            + self.config.lambda_trajectory * outputs["trajectory_loss"]
            + self.config.lambda_projection * outputs["projection_loss"]
            + self.config.lambda_smooth * outputs["smoothness_loss"]
        )
        if outputs["action_loss"] is not None:
            total_loss = total_loss + self.config.lambda_action * outputs["action_loss"]

        loss_tensor_for_reporting = outputs["action_loss_tensor"]
        if loss_tensor_for_reporting is None:
            loss_tensor_for_reporting = outputs["diffusion_loss_tensor"]

        loss_dict = {
            "loss_per_dim": loss_tensor_for_reporting.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "diffusion_loss": outputs["diffusion_loss"].mean().item(),
            "trajectory_loss": outputs["trajectory_loss"].mean().item(),
            "projection_loss": outputs["projection_loss"].mean().item(),
            "smoothness_loss": outputs["smoothness_loss"].mean().item(),
            "action_loss": 0.0 if outputs["action_loss"] is None else outputs["action_loss"].mean().item(),
            "coord_abs_mean": outputs["coord_abs_mean"].mean().item(),
            "delta_abs_mean": outputs["delta_abs_mean"].mean().item(),
            "pred_clean_trajectory_std": outputs["pred_clean_trajectory_std"].mean().item(),
            "geom_hidden_delta_abs_mean": outputs["geom_hidden_delta_abs_mean"].mean().item(),
            "geom_hidden_delta_ratio": outputs["geom_hidden_delta_ratio"].mean().item(),
            "geometry_hidden_residual_scale": outputs["geometry_hidden_residual_scale"].mean().item(),
            "geometry_coord_delta_scale": outputs["geometry_coord_delta_scale"].mean().item(),
            "valid_main_frac": outputs["valid_main_frac"].mean().item(),
            "valid_aux_frac": outputs["valid_aux_frac"].mean().item(),
        }

        if reduction == "none":
            loss_dict["loss"] = total_loss.mean().item()
            return total_loss, loss_dict
        else:
            loss = total_loss.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Return default PEFT target modules for PI05Spatial fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|"
            "trajectory_tokenizer|geometry_refinement|trajectory_decoder|spatial_feature_extractor"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}).*)"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
