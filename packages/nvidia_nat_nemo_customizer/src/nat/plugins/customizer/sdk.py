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
SDK classes for NeMo Customizer plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from nat.utils.sdk.nat_trainer import NatTrainer
from nat.utils.sdk.nat_trainer import NatTrainerAdapter
from nat.utils.sdk.nat_trainer import NatTrajectoryBuilder

from .dpo.config import DPOTrajectoryBuilderConfig
from .dpo.config import NeMoCustomizerTrainerAdapterConfig
from .dpo.config import NeMoCustomizerTrainerConfig


class DPOTrajectoryBuilder(DPOTrajectoryBuilderConfig, NatTrajectoryBuilder):
    """DPO Trajectory Builder for collecting preference pairs.

    ![Icon](https://cdn.simpleicons.org/nvidia/76B900)

    Collects preference data from workflows that produce TTC_END intermediate
    steps with TTCEventData. Groups candidates by turn_id and creates preference
    pairs based on score differences.

    Example:
        ```python
        from nat.plugins.customizer.sdk import DPOTrajectoryBuilder

        trajectory_builder = DPOTrajectoryBuilder(
            ttc_step_name="dpo_candidate_move",
            exhaustive_pairs=True,
            min_score_diff=0.05,
            max_pairs_per_turn=5,
        )
        ```
    """


class NeMoCustomizerTrainer(NeMoCustomizerTrainerConfig, NatTrainer):
    """NeMo Customizer Trainer for DPO/SFT finetuning.

    ![Icon](https://cdn.simpleicons.org/nvidia/76B900)

    Orchestrates DPO data collection and training job submission. Runs the
    trajectory builder multiple times to collect data, then submits a single
    training job to NeMo Customizer.

    Example:
        ```python
        from nat.plugins.customizer.sdk import NeMoCustomizerTrainer

        trainer = NeMoCustomizerTrainer(
            num_runs=5,
            wait_for_completion=True,
            deduplicate_pairs=True,
            max_pairs=10000,
        )
        ```
    """


class NeMoCustomizerTrainerAdapter(NeMoCustomizerTrainerAdapterConfig, NatTrainerAdapter):
    """NeMo Customizer Trainer Adapter for submitting training jobs.

    ![Icon](https://cdn.simpleicons.org/nvidia/76B900)

    Submits DPO/SFT training jobs to NeMo Customizer and optionally deploys
    the trained model.

    Example:
        ```python
        from nat.plugins.customizer.sdk import NeMoCustomizerTrainerAdapter

        adapter = NeMoCustomizerTrainerAdapter(
            backend=NeMoCustomizerBackendConfig(
                nemo_customizer_url="https://your-customizer-instance.nvidia.com",
            ),
            config=DPOFinetuneConfig(
                epochs=1,
                learning_rate=1e-5,
            ),
            deploy_after_training=True,
        )
        ```
    """
