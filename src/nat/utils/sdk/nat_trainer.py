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
SDK wrappers for trainer, trajectory builder, and trainer adapter configurations.

These classes provide a Pythonic interface for configuring finetuning components
in NeMo Agent Toolkit workflows. They follow the same pattern as NatLLM, NatFunction, etc.

Example:
    ```python
    # Plugin implementations inherit from these base classes
    # For example, with OpenPipe ART plugin installed:
    from nat.plugins.openpipe.config import ARTTrainerConfig
    from nat.utils.sdk.nat_trainer import NatTrainer

    # ARTTrainerConfig would inherit from both TrainerConfig and NatTrainer
    # allowing it to be passed as an object to NatFinetuner
    ```
"""

from __future__ import annotations

from typing import ClassVar

from nat.data_models.common import TypedBaseModel
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.utils.sdk.nat_base import NatBase


class NatTrainer(TypedBaseModel, NatBase):
    """Base SDK class for trainer configurations.

    Trainers orchestrate the finetuning loop across epochs. They coordinate
    between trajectory builders (data collection) and trainer adapters
    (training submission).

    Plugin-specific trainer implementations should inherit from both their
    config class and this SDK class:

    Example:
        ```python
        # In a plugin like nvidia-nat-openpipe-art:
        class ARTTrainer(ARTTrainerConfig, NatTrainer):
            pass
        ```
    """

    _marker_class: ClassVar[type] = TrainerConfig


class NatTrajectoryBuilder(TypedBaseModel, NatBase):
    """Base SDK class for trajectory builder configurations.

    Trajectory builders collect training data by running evaluations
    and recording agent interactions as trajectories.

    Plugin-specific trajectory builder implementations should inherit from both
    their config class and this SDK class:

    Example:
        ```python
        # In a plugin like nvidia-nat-openpipe-art:
        class ARTTrajectoryBuilder(ARTTrajectoryBuilderConfig, NatTrajectoryBuilder):
            pass
        ```
    """

    _marker_class: ClassVar[type] = TrajectoryBuilderConfig


class NatTrainerAdapter(TypedBaseModel, NatBase):
    """Base SDK class for trainer adapter configurations.

    Trainer adapters handle the submission of trajectories to training
    backends and monitor training progress.

    Plugin-specific trainer adapter implementations should inherit from both
    their config class and this SDK class:

    Example:
        ```python
        # In a plugin like nvidia-nat-openpipe-art:
        class ARTTrainerAdapter(ARTTrainerAdapterConfig, NatTrainerAdapter):
            pass
        ```
    """

    _marker_class: ClassVar[type] = TrainerAdapterConfig


# Re-export commonly used types for convenience
__all__ = [
    "NatTrainer",
    "NatTrajectoryBuilder",
    "NatTrainerAdapter",
]
