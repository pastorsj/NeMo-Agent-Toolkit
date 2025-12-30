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
DPO (Direct Preference Optimization) components for NAT.

This module provides:
- DPO Trajectory Builder: Collects preference data from scored TTC intermediate steps
- NeMo Customizer TrainerAdapter: Submits DPO training jobs to NeMo Customizer
"""

from nat.plugins.customizer.sdk import DPOTrajectoryBuilder
from nat.plugins.customizer.sdk import NeMoCustomizerTrainer
from nat.plugins.customizer.sdk import NeMoCustomizerTrainerAdapter

from .config import DPOSpecificHyperparameters
from .config import DPOTrajectoryBuilderConfig
from .config import NeMoCustomizerHyperparameters
from .config import NeMoCustomizerTrainerAdapterConfig
from .config import NeMoCustomizerTrainerConfig
from .config import NIMDeploymentConfig

# Runtime implementation classes (for internal use)
from .trainer import NeMoCustomizerTrainerImpl
from .trainer_adapter import NeMoCustomizerTrainerAdapterImpl
from .trajectory_builder import DPOTrajectoryBuilderImpl

__all__ = [
    # SDK classes (for user-facing configuration)
    "DPOTrajectoryBuilderConfig",
    "DPOTrajectoryBuilder",
    "NeMoCustomizerTrainerConfig",
    "NeMoCustomizerTrainer",
    "NeMoCustomizerTrainerAdapterConfig",
    "NeMoCustomizerTrainerAdapter",  # Runtime implementation classes
    "DPOTrajectoryBuilderImpl",
    "NeMoCustomizerTrainerImpl",
    "NeMoCustomizerTrainerAdapterImpl",  # Other configs
    "NeMoCustomizerHyperparameters",
    "DPOSpecificHyperparameters",
    "NIMDeploymentConfig",
]
