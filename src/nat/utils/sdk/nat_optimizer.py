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
NatOptimizer SDK module.

Provides the NatOptimizer class for workflow optimization, along with
re-exports of the underlying configuration classes for convenience.

Example:
    ```python
    from nat.utils.sdk.nat_optimizer import (
        NatOptimizer,
        OptimizerMetric,
        NumericOptimizationConfig,
    )

    optimizer = NatOptimizer(
        output_path=Path("./optimizer_results"),
        eval_metrics={
            "accuracy": OptimizerMetric(
                evaluator_name="accuracy",
                direction="maximize",
            ),
        },
        numeric=NumericOptimizationConfig(enabled=True, n_trials=20),
    )

    nat_workflow.add_optimizer(optimizer)
    await nat_workflow.optimize(endpoint="http://localhost:8000")
    ```
"""

# Re-export underlying config classes for convenience
from nat.data_models.optimizer import NumericOptimizationConfig
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerMetric
from nat.data_models.optimizer import OptimizerRunConfig
from nat.data_models.optimizer import PromptGAOptimizationConfig
from nat.data_models.optimizer import SamplerType

__all__ = [
    "NatOptimizer",
    "OptimizerConfig",
    "OptimizerMetric",
    "OptimizerRunConfig",
    "NumericOptimizationConfig",
    "PromptGAOptimizationConfig",
    "SamplerType",
]


class NatOptimizer(OptimizerConfig):
    """SDK optimizer configuration for workflow optimization.

    Extends OptimizerConfig directly. All optimizer settings are inherited
    from OptimizerConfig. Runtime parameters (endpoint, dataset override, etc.)
    are passed to workflow.optimize() when executing.

    Inherited fields from OptimizerConfig:
        - output_path: Path to save optimization results
        - eval_metrics: Dictionary of metrics to optimize (OptimizerMetric)
        - reps_per_param_set: Number of evaluation repetitions per parameter set
        - target: Target value to stop optimization when reached
        - multi_objective_combination_mode: Method to combine multiple objectives
        - numeric: NumericOptimizationConfig for Optuna-based tuning
        - prompt: PromptGAOptimizationConfig for GA-based prompt optimization

    Example:
        ```python
        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(
                    evaluator_name="accuracy",
                    direction="maximize",
                ),
            },
            numeric=NumericOptimizationConfig(enabled=True, n_trials=20),
        )

        nat_workflow.add_optimizer(optimizer)
        # Runtime params passed to optimize()
        await nat_workflow.optimize(endpoint="http://localhost:8000")
        ```
    """

    def to_optimizer_config(self) -> OptimizerConfig:
        """Return self as OptimizerConfig for serialization.

        Since NatOptimizer extends OptimizerConfig directly with no additional
        fields, this just returns the parent class representation.
        """
        return OptimizerConfig(
            output_path=self.output_path,
            eval_metrics=self.eval_metrics,
            reps_per_param_set=self.reps_per_param_set,
            target=self.target,
            multi_objective_combination_mode=self.multi_objective_combination_mode,
            numeric=self.numeric,
            prompt=self.prompt,
        )
