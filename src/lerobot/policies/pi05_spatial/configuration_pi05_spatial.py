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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05_spatial")
@dataclass
class PI05SpatialConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Trajectory representation and observation wiring.
    trajectory_dim: int = 3
    trajectory_key: str = f"{OBS_STATE}.future_eef_pos"
    trajectory_source_key: str = OBS_STATE
    trajectory_source_start_index: int = 0
    trajectory_pad_key: str = f"{OBS_STATE}.future_eef_pos_is_pad"
    trajectory_source_pad_key: str = f"{OBS_STATE}_is_pad"
    current_eef_pos_key: str = f"{OBS_STATE}.eef_pos"
    use_observation_trajectory_supervision: bool = True
    derive_trajectory_from_eef_delta: bool = True
    trajectory_action_start_index: int = 0
    use_action_as_trajectory_fallback: bool = True
    action_pad_key: str = "action_is_pad"
    mask_padding_loss: bool = True

    # Camera and projection wiring. The target trajectory is assumed to live in the main camera frame.
    camera_main_intrinsics_key: str = "observation.camera.main.intrinsics"
    camera_aux_intrinsics_key: str = "observation.camera.aux.intrinsics"
    camera_aux_from_main_key: str = "observation.camera.aux.T_main"
    main_camera_index: int = 0
    aux_camera_index: int = 1
    projection_fallback_scale: float = 0.25

    # Spatial geometry branch.
    spatial_feature_dims: tuple[int, int, int] = (32, 64, 128)
    geometry_hidden_dim: int = 512
    geometry_num_fourier_bands: int = 8
    geometry_hidden_residual_scale: float = 0.25
    geometry_coord_delta_scale: float = 0.1
    geometry_local_window_radius: int = 1
    geometry_local_window_sigma: float = 1.0

    # Lightweight trajectory-to-action decoder that keeps the current LeRobot API intact.
    trajectory_decoder_hidden_dim: int = 512
    predict_gripper: bool = False

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Auxiliary trajectory losses for the spatial MVP.
    lambda_trajectory: float = 1.0
    lambda_projection: float = 0.5
    lambda_smooth: float = 0.1
    lambda_action: float = 0.5

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.trajectory_dim <= 0:
            raise ValueError(f"trajectory_dim must be positive, got {self.trajectory_dim}")

        if self.trajectory_dim > self.max_action_dim:
            raise ValueError(
                f"trajectory_dim ({self.trajectory_dim}) cannot be greater than max_action_dim ({self.max_action_dim})"
            )

        if self.trajectory_action_start_index < 0:
            raise ValueError(
                f"trajectory_action_start_index must be non-negative, got {self.trajectory_action_start_index}"
            )

        if self.trajectory_action_start_index + self.trajectory_dim > self.max_action_dim:
            raise ValueError(
                "trajectory_action_start_index + trajectory_dim must stay within max_action_dim, "
                f"got start={self.trajectory_action_start_index}, trajectory_dim={self.trajectory_dim}, "
                f"max_action_dim={self.max_action_dim}"
            )

        if len(self.spatial_feature_dims) != 3:
            raise ValueError(
                f"spatial_feature_dims must contain exactly 3 feature widths, got {self.spatial_feature_dims}"
            )

        if self.geometry_hidden_residual_scale < 0:
            raise ValueError(
                "geometry_hidden_residual_scale must be non-negative, "
                f"got {self.geometry_hidden_residual_scale}"
            )

        if self.geometry_coord_delta_scale < 0:
            raise ValueError(
                f"geometry_coord_delta_scale must be non-negative, got {self.geometry_coord_delta_scale}"
            )

        if self.geometry_local_window_radius < 0:
            raise ValueError(
                f"geometry_local_window_radius must be non-negative, got {self.geometry_local_window_radius}"
            )

        if self.geometry_local_window_sigma <= 0:
            raise ValueError(
                f"geometry_local_window_sigma must be positive, got {self.geometry_local_window_sigma}"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices_by_key(self) -> dict[str, list[int]]:
        if not self.use_observation_trajectory_supervision:
            return {}
        # Query the current EE position plus the next chunk_size future positions, then
        # let the processor split them into a current state and an explicit future trajectory.
        return {self.trajectory_source_key: list(range(self.chunk_size + 1))}

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
