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
NatEvaluation SDK module.

Provides the NatEvaluation class for workflow evaluation, along with
re-exports of the underlying configuration classes for convenience.

Example:
    ```python
    from nat.utils.sdk.nat_evaluation import NatEvaluation, EvalDatasetJsonConfig

    evaluation = NatEvaluation(
        output_dir=Path("./eval_results"),
        dataset=EvalDatasetJsonConfig(
            file_path=Path("./data/test.json"),
            id_key="id",
            question_key="question",
            answer_key="answer",
        ),
        evaluators=[accuracy_evaluator],
    )

    nat_workflow.add_evaluator(evaluation)
    await nat_workflow.evaluate(reps=3)
    ```
"""

from pydantic import Field

# Re-export underlying config classes for convenience
from nat.data_models.dataset_handler import EvalDatasetConfig
from nat.data_models.dataset_handler import EvalDatasetCsvConfig
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.evaluate import EvalConfig
from nat.data_models.evaluate import EvalGeneralConfig
from nat.data_models.evaluate import EvalOutputConfig
from nat.data_models.evaluate import JobManagementConfig
from nat.data_models.profiler import ProfilerConfig
from nat.eval.config import EvaluationRunConfig
from nat.utils.sdk.nat_evaluator import NatEvaluator

__all__ = [
    "NatEvaluation",
    "EvalConfig",
    "EvalGeneralConfig",
    "EvalDatasetConfig",
    "EvalDatasetCsvConfig",
    "EvalDatasetJsonConfig",
    "EvalOutputConfig",
    "EvaluationRunConfig",
    "JobManagementConfig",
    "ProfilerConfig",
    "NatEvaluator",
]


class NatEvaluation(EvalGeneralConfig):
    """SDK evaluation configuration for workflow evaluation.

    Extends EvalGeneralConfig with evaluators list. All evaluation settings
    are inherited from EvalGeneralConfig. Runtime parameters (reps, endpoint,
    etc.) are passed to workflow.evaluate() when executing.

    Inherited fields from EvalGeneralConfig:
        - max_concurrency: Maximum concurrent evaluations (default: 8)
        - workflow_alias: Display alias for the workflow
        - output_dir: Path to save evaluation results
        - output: Detailed output configuration (EvalOutputConfig)
        - dataset: Dataset configuration (EvalDatasetConfig)
        - profiler: Profiler configuration (ProfilerConfig)

    Added fields:
        - evaluators: List of NatEvaluator instances for evaluation metrics

    Example:
        ```python
        evaluation = NatEvaluation(
            output_dir=Path("./results"),
            dataset=EvalDatasetJsonConfig(
                file_path=Path("./data.json"),
                id_key="id",
                question_key="question",
                answer_key="answer",
            ),
            evaluators=[accuracy_evaluator],
        )

        nat_workflow.add_evaluator(evaluation)
        # Runtime params passed to evaluate()
        await nat_workflow.evaluate(reps=3, endpoint="http://localhost:8000")
        ```
    """

    # Evaluators list - discovered by NatWorkflow during traversal
    evaluators: list[NatEvaluator] = Field(
        default_factory=list,
        description="List of evaluators for specific aspects (accuracy, latency, etc.).",
    )

    def to_eval_general_config(self) -> EvalGeneralConfig:
        """Convert to EvalGeneralConfig for serialization."""
        return EvalGeneralConfig(
            max_concurrency=self.max_concurrency,
            workflow_alias=self.workflow_alias,
            output_dir=self.output_dir,
            output=self.output,
            dataset=self.dataset,
            profiler=self.profiler,
        )
