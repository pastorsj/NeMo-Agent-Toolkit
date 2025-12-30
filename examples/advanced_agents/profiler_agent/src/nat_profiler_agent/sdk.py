# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SDK classes for this example package."""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

from .register import ProfilerAgentConfig


class ProfilerAgent(ProfilerAgentConfig, NatFunction):
    """Profiler Agent"""

    llm_name: LLMRef = Field(description="", default=LLMRef(value=""), init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self
