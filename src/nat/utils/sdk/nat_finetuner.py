# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SDK wrapper for finetuning configuration.

This module provides a Pythonic interface for configuring finetuning
in NeMo Agent Toolkit workflows.

Example:
    ```python
    from nat.utils.sdk.nat_finetuner import NatFinetuner, CurriculumLearning

    # With plugin-specific implementations (e.g., OpenPipe ART):
    from nat.plugins.openpipe.register import ARTTrainer, ARTTrajectoryBuilder, ARTTrainerAdapter

    # Configure finetuning with objects (required)
    finetuning = NatFinetuner(
        trainer=ARTTrainer(),
        trajectory_builder=ARTTrajectoryBuilder(num_generations=2),
        trainer_adapter=ARTTrainerAdapter(backend=...),
        reward_function=my_evaluator,  # A NatEvaluator instance
        num_epochs=10,
        curriculum_learning=CurriculumLearning(enabled=True),
    )

    # Add to workflow
    workflow.add_finetuning(finetuning)
    ```
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.utils.sdk.nat_evaluator import NatEvaluator
from nat.utils.sdk.nat_trainer import NatTrainer
from nat.utils.sdk.nat_trainer import NatTrainerAdapter
from nat.utils.sdk.nat_trainer import NatTrajectoryBuilder


class CurriculumLearning(CurriculumLearningConfig):
    """SDK wrapper for curriculum learning configuration.

    Inherits all fields from CurriculumLearningConfig. Curriculum learning
    progressively introduces harder training examples to improve model
    learning and convergence.

    Inherited fields:
        - enabled: Whether to enable curriculum learning (default: False)
        - initial_percentile: Fraction of examples to start with (default: 0.3)
        - increment_percentile: Fraction to add at each expansion (default: 0.2)
        - expansion_interval: Epochs between expansions (default: 5)
        - min_reward_diff: Minimum reward variance threshold (default: 0.1)
        - sort_ascending: Sort direction (default: False = easy-to-hard)
        - random_subsample: Optional subsampling fraction

    Example:
        ```python
        from nat.utils.sdk.nat_finetuner import CurriculumLearning

        curriculum = CurriculumLearning(
            enabled=True,
            initial_percentile=0.3,
            increment_percentile=0.2,
            expansion_interval=5,
            min_reward_diff=0.1,
            sort_ascending=False,  # Easy-to-hard
        )
        ```
    """


class NatFinetuner(BaseModel):
    """SDK wrapper for finetuning configuration.

    Components MUST be passed as objects. The name fields are automatically
    populated from the objects when converting to FinetuneConfig.

    Required fields:
        - trainer: Trainer object (NatTrainer subclass)
        - trajectory_builder: TrajectoryBuilder object (NatTrajectoryBuilder subclass)
        - trainer_adapter: TrainerAdapter object (NatTrainerAdapter subclass)
        - reward_function: Evaluator object for rewards (NatEvaluator subclass)

    Optional fields:
        - target_function_names: Functions to extract trajectories from
        - target_model_name: Specific model to target
        - curriculum_learning: Curriculum learning configuration
        - num_epochs: Number of training epochs
        - output_dir: Directory for outputs

    Example:
        ```python
        from nat.utils.sdk.nat_finetuner import NatFinetuner, CurriculumLearning

        finetuning = NatFinetuner(
            trainer=ARTTrainer(),
            trajectory_builder=ARTTrajectoryBuilder(num_generations=2),
            trainer_adapter=ARTTrainerAdapter(backend=backend_config),
            reward_function=my_accuracy_evaluator,
            num_epochs=10,
            curriculum_learning=CurriculumLearning(enabled=True),
        )
        ```
    """

    # Required object fields
    trainer: NatTrainer = Field(description="Trainer object for finetuning")
    trajectory_builder: NatTrajectoryBuilder = Field(
        description="TrajectoryBuilder object for collecting training data")
    trainer_adapter: NatTrainerAdapter = Field(description="TrainerAdapter object for submitting to training backend")
    reward_function: NatEvaluator = Field(description="Evaluator object for computing rewards")

    # Optional configuration fields
    target_function_names: list[str] = Field(default=["<workflow>"],
                                             description="Functions to extract trajectories from")
    target_model_name: str | None = Field(default=None, description="Target model name to fine-tune")
    curriculum_learning: CurriculumLearning | CurriculumLearningConfig | None = Field(
        default=None, description="Curriculum learning configuration")
    num_epochs: int = Field(default=1, ge=1, description="Number of epochs to run")
    output_dir: Path = Field(default=Path("./.tmp/nat/finetuning/"),
                             description="Directory for outputs and checkpoints")

    model_config = {"arbitrary_types_allowed": True}

    def _get_trainer_name(self) -> str:
        """Get the trainer name from the object."""
        name, _ = self.trainer.compute_name_and_config(TrainerConfig)
        return name

    def _get_trajectory_builder_name(self) -> str:
        """Get the trajectory builder name from the object."""
        name, _ = self.trajectory_builder.compute_name_and_config(TrajectoryBuilderConfig)
        return name

    def _get_trainer_adapter_name(self) -> str:
        """Get the trainer adapter name from the object."""
        name, _ = self.trainer_adapter.compute_name_and_config(TrainerAdapterConfig)
        return name

    def _get_reward_function_name(self) -> str:
        """Get the reward function (evaluator) name from the object."""
        from nat.data_models.evaluator import EvaluatorBaseConfig
        name, _ = self.reward_function.compute_name_and_config(EvaluatorBaseConfig)
        return name

    @property
    def trainer_name(self) -> str:
        """The trainer name (computed from trainer object)."""
        return self._get_trainer_name()

    @property
    def trajectory_builder_name(self) -> str:
        """The trajectory builder name (computed from trajectory_builder object)."""
        return self._get_trajectory_builder_name()

    @property
    def trainer_adapter_name(self) -> str:
        """The trainer adapter name (computed from trainer_adapter object)."""
        return self._get_trainer_adapter_name()

    @property
    def reward_function_name(self) -> str:
        """The reward function name (computed from reward_function object)."""
        return self._get_reward_function_name()

    def get_components(self) -> dict:
        """Get all component objects for discovery by NatWorkflow.

        Returns:
            Dict with trainer, trajectory_builder, trainer_adapter, and reward_function
            objects.
        """
        return {
            "trainer": self.trainer,
            "trajectory_builder": self.trajectory_builder,
            "trainer_adapter": self.trainer_adapter,
            "reward_function": self.reward_function,
        }

    def to_finetune_config(self) -> FinetuneConfig:
        """Convert to FinetuneConfig for serialization.

        Returns:
            FinetuneConfig: The configuration object for finetuning.
        """
        curriculum = self.curriculum_learning
        if curriculum is None:
            curriculum = CurriculumLearningConfig()

        return FinetuneConfig(
            enabled=True,
            trainer_name=self.trainer_name,
            trajectory_builder_name=self.trajectory_builder_name,
            trainer_adapter_name=self.trainer_adapter_name,
            reward_function_name=self.reward_function_name,
            target_function_names=self.target_function_names,
            target_model_name=self.target_model_name,
            curriculum_learning=curriculum,
            num_epochs=self.num_epochs,
            output_dir=self.output_dir,
        )


# Re-export commonly used types for convenience
__all__ = [
    "NatFinetuner",
    "CurriculumLearning",  # Re-export from finetuning module for convenience
    "CurriculumLearningConfig",
    "FinetuneConfig",
]
