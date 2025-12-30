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

from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

from .agno_personal_finance_function import AgnoPersonalFinanceFunctionConfig


class AgnoPersonalFinanceFunction(AgnoPersonalFinanceFunctionConfig, NatFunction):
    """AGNO Personal Finance Function"""

    llm_name: LLMRef = Field(description="The name of the LLM to use for the financial research and planner agents.",
                             default=LLMRef(value=""),
                             init=False)
    tools: list[FunctionRef] = Field(default_factory=list,
                                     description="The tools to use for the financial research and planner agents.",
                                     init=False)

    llm: NatLLM = Field(exclude=True)
    tool_objects: list[NatFunction] = Field(default_factory=list, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.tool_objects and len(self.tool_objects) > 0:
            self.tools = [FunctionRef(value=tool.computed_name) for tool in self.tool_objects]
        return self
