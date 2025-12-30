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
SDK classes for OpenPipe ART plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from nat.utils.sdk.nat_trainer import NatTrainer
from nat.utils.sdk.nat_trainer import NatTrainerAdapter
from nat.utils.sdk.nat_trainer import NatTrajectoryBuilder

from .config import ARTTrainerAdapterConfig
from .config import ARTTrainerConfig
from .config import ARTTrajectoryBuilderConfig


class ARTTrajectoryBuilder(ARTTrajectoryBuilderConfig, NatTrajectoryBuilder):
    """OpenPipe ART Trajectory Builder for collecting training data.

    ![Icon](https://cdn.simpleicons.org/pytorch/EE4C2C)

    Collects training trajectories by running evaluations and recording
    agent interactions.

    Example:
        ```python
        from nat.plugins.openpipe.sdk import ARTTrajectoryBuilder

        trajectory_builder = ARTTrajectoryBuilder(
            num_generations=2,
        )
        ```
    """


class ARTTrainer(ARTTrainerConfig, NatTrainer):
    """OpenPipe ART Trainer for finetuning.

    ![Icon](https://cdn.simpleicons.org/pytorch/EE4C2C)

    Orchestrates the finetuning loop across epochs using the OpenPipe ART
    backend.

    Example:
        ```python
        from nat.plugins.openpipe.sdk import ARTTrainer

        trainer = ARTTrainer()
        ```
    """


class ARTTrainerAdapter(ARTTrainerAdapterConfig, NatTrainerAdapter):
    """OpenPipe ART Trainer Adapter for submitting training jobs.

    ![Icon](https://cdn.simpleicons.org/pytorch/EE4C2C)

    Handles submission of trajectories to the ART training backend and
    monitors training progress.

    Example:
        ```python
        from nat.plugins.openpipe.sdk import ARTTrainerAdapter

        adapter = ARTTrainerAdapter(
            ...
        )
        ```
    """
