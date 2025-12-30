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
SDK classes for Evaluators.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_evaluator import NatEvaluator
from nat.utils.sdk.nat_llm import NatLLM

from .rag_evaluator.register import RagasEvaluatorConfig
from .swe_bench_evaluator.register import SweBenchEvaluatorConfig
from .trajectory_evaluator.register import TrajectoryEvaluatorConfig
from .tunable_rag_evaluator.register import TunableRagEvaluatorConfig


class RagasEvaluator(RagasEvaluatorConfig, NatEvaluator):
    """RAGAS Evaluator"""

    llm: NatLLM = Field(exclude=True)
    llm_name: LLMRef = Field(description="", default="", init=False)  # type: ignore[assignment]

    @model_validator(mode="after")
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm is not None:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class TrajectoryEvaluator(TrajectoryEvaluatorConfig, NatEvaluator):
    """Trajectory Evaluator"""


class TunableRagEvaluator(TunableRagEvaluatorConfig, NatEvaluator):
    """Tunable RAG Evaluator"""

    llm: NatLLM = Field(exclude=True)
    llm_name: LLMRef = Field(description="", default=LLMRef(value=""), init=False)

    @model_validator(mode="after")
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm is not None:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class SweBenchEvaluator(SweBenchEvaluatorConfig, NatEvaluator):
    """SWE Bench Evaluator"""
