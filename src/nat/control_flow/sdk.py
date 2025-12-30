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
SDK classes for Control Flow.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_agent import NatAgent
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

from .router_agent.register import RouterAgentWorkflowConfig
from .sequential_executor import SequentialExecutorConfig


class RouterAgentWorkflow(RouterAgentWorkflowConfig, NatAgent):
    """Router Agent Workflow"""

    llm_name: LLMRef = Field(description="The LLM model to use with the agent.", default=LLMRef(value=""), init=False)
    branches: list[FunctionRef] = Field(default_factory=list,
                                        description="The list of branches to provide to the router agent.",
                                        init=False)

    llm: NatLLM = Field(exclude=True)
    branch_functions: list[NatFunction] | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.branch_functions and len(self.branch_functions) > 0:
            self.branches = [FunctionRef(value=fn.computed_name) for fn in self.branch_functions]
        return self


class SequentialExecutor(SequentialExecutorConfig, NatFunction):
    """Sequential Executor"""

    tool_list: list[FunctionRef] = Field(default_factory=list,
                                         description="A list of functions to execute sequentially.",
                                         init=False)

    tools: list[NatFunction] | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set tool names from tool objects if tools are provided."""
        if self.tools and len(self.tools) > 0:
            self.tool_list = [FunctionRef(value=tool.computed_name) for tool in self.tools]
        return self
