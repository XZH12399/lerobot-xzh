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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05_spatial.configuration_pi05_spatial import PI05SpatialConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="pi05_spatial_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05SpatialPrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05Spatial")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="pi05_spatial_prepare_trajectory_processor_step")
@dataclass
class Pi05SpatialPrepareTrajectoryProcessorStep(ProcessorStep):
    """Derive an explicit future EE trajectory from the queried EE-position sequence."""

    trajectory_source_key: str = OBS_STATE
    trajectory_source_start_index: int = 0
    trajectory_source_pad_key: str = f"{OBS_STATE}_is_pad"
    current_eef_pos_key: str = f"{OBS_STATE}.eef_pos"
    trajectory_key: str = f"{OBS_STATE}.future_eef_pos"
    trajectory_pad_key: str = f"{OBS_STATE}.future_eef_pos_is_pad"
    chunk_size: int = 50
    trajectory_dim: int = 3

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("Observation is required for PI05Spatial trajectory preparation")

        if self.trajectory_key in observation:
            return transition

        eef_pos_sequence = observation.get(self.trajectory_source_key)
        if eef_pos_sequence is None or not isinstance(eef_pos_sequence, torch.Tensor):
            return transition

        # Training batches with key-specific delta timestamps arrive as (B, T + 1, D).
        if eef_pos_sequence.dim() != 3:
            return transition

        required_steps = self.chunk_size + 1
        if eef_pos_sequence.shape[1] < required_steps:
            raise ValueError(
                f"Expected '{self.trajectory_source_key}' to contain at least {required_steps} steps, "
                f"got shape {tuple(eef_pos_sequence.shape)}"
            )

        observation = observation.copy()
        source_end_index = self.trajectory_source_start_index + self.trajectory_dim
        if eef_pos_sequence.shape[-1] < source_end_index:
            raise ValueError(
                f"Expected '{self.trajectory_source_key}' to have at least {source_end_index} features, "
                f"got shape {tuple(eef_pos_sequence.shape)}"
            )

        trajectory_sequence = eef_pos_sequence[:, :, self.trajectory_source_start_index:source_end_index]
        if self.trajectory_source_key == OBS_STATE:
            observation[OBS_STATE] = eef_pos_sequence[:, 0]
        observation[self.current_eef_pos_key] = trajectory_sequence[:, 0]
        observation[self.trajectory_key] = trajectory_sequence[:, 1:required_steps]

        source_pad = observation.get(self.trajectory_source_pad_key)
        if source_pad is not None:
            if not isinstance(source_pad, torch.Tensor):
                source_pad = torch.as_tensor(source_pad, dtype=torch.bool, device=eef_pos_sequence.device)
            else:
                source_pad = source_pad.to(device=eef_pos_sequence.device, dtype=torch.bool)
            if source_pad.dim() == 1:
                source_pad = source_pad.unsqueeze(0)
            if source_pad.shape[1] < required_steps:
                raise ValueError(
                    f"Expected '{self.trajectory_source_pad_key}' to contain at least {required_steps} steps, "
                    f"got shape {tuple(source_pad.shape)}"
                )
            observation[self.trajectory_pad_key] = source_pad[:, 1:required_steps]

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi05_spatial_pre_post_processors(
    config: PI05SpatialConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI05Spatial policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI05Spatial policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05SpatialPrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05SpatialPrepareTrajectoryProcessorStep(
            trajectory_source_key=config.trajectory_source_key,
            trajectory_source_start_index=config.trajectory_source_start_index,
            trajectory_source_pad_key=config.trajectory_source_pad_key,
            current_eef_pos_key=config.current_eef_pos_key,
            trajectory_key=config.trajectory_key,
            trajectory_pad_key=config.trajectory_pad_key,
            chunk_size=config.chunk_size,
            trajectory_dim=config.trajectory_dim,
        ),
        Pi05SpatialPrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
