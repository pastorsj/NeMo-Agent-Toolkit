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
SDK classes for Agents.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_agent import NatAgent
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_function_group import NatFunctionGroup
from nat.utils.sdk.nat_llm import NatLLM

from .react_agent.register import ReActAgentWorkflowConfig
from .reasoning_agent.reasoning_agent import ReasoningFunctionConfig
from .responses_api_agent.register import ResponsesAPIAgentWorkflowConfig
from .rewoo_agent.register import ReWOOAgentWorkflowConfig
from .tool_calling_agent.register import ToolCallAgentWorkflowConfig


class NatReActAgent(ReActAgentWorkflowConfig, NatAgent):
    """ReAct Agent Workflow"""

    llm_name: LLMRef = Field(description="The LLM model to use with the agent.", default=LLMRef(value=""), init=False)
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="The list of tools to provide to the react agent.", init=False)

    llm: NatLLM = Field(exclude=True)
    tools: list[NatFunction | NatFunctionGroup | NatAgent] = Field(exclude=True, default=[])

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        self.llm_name = LLMRef(value=self.llm.computed_name)
        refs = []
        for tool in self.tools:
            if isinstance(tool, NatFunctionGroup):
                refs.append(FunctionGroupRef(value=tool.computed_name))
            else:
                # NatFunction and NatAgent both use FunctionRef
                refs.append(FunctionRef(value=tool.computed_name))
        self.tool_names = refs
        return self


class ToolCallingAgent(ToolCallAgentWorkflowConfig, NatAgent):
    """Tool Calling Agent Workflow"""

    llm_name: LLMRef = Field(description="The LLM model to use with the agent.", default=LLMRef(value=""), init=False)
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="The list of tools to provide to the tool calling agent.", init=False)
    return_direct: list[FunctionRef] | None = Field(
        default=None,
        description="List of tool names that should return responses directly without LLM processing.",
        init=False)

    llm: NatLLM = Field(exclude=True)
    tools: list[NatFunction | NatFunctionGroup | NatAgent] | None = Field(default=None, exclude=True)
    return_direct_functions: list[NatFunction] | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.tools and len(self.tools) > 0:
            refs = []
            for tool in self.tools:
                if isinstance(tool, NatFunctionGroup):
                    refs.append(FunctionGroupRef(value=tool.computed_name))
                else:
                    # NatFunction and NatAgent both use FunctionRef
                    refs.append(FunctionRef(value=tool.computed_name))
            self.tool_names = refs
        if self.return_direct_functions and len(self.return_direct_functions) > 0:
            self.return_direct = [FunctionRef(value=fn.computed_name) for fn in self.return_direct_functions]
        return self


# Backward compatibility alias
ToolCallAgentWorkflow = ToolCallingAgent


class ReWOOAgentWorkflow(ReWOOAgentWorkflowConfig, NatAgent):
    """ReWOO Agent Workflow"""

    llm_name: LLMRef = Field(description="The LLM model to use with the agent.", default=LLMRef(value=""), init=False)
    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="The list of tools to provide to the rewoo agent.", init=False)

    llm: NatLLM = Field(exclude=True)
    tools: list[NatFunction | NatFunctionGroup | NatAgent] | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.tools and len(self.tools) > 0:
            refs = []
            for tool in self.tools:
                if isinstance(tool, NatFunctionGroup):
                    refs.append(FunctionGroupRef(value=tool.computed_name))
                else:
                    # NatFunction and NatAgent both use FunctionRef
                    refs.append(FunctionRef(value=tool.computed_name))
            self.tool_names = refs
        return self


class ReasoningFunction(ReasoningFunctionConfig, NatAgent):
    """Reasoning Function"""

    llm_name: LLMRef = Field(description="The LLM model to use with the agent.", default=LLMRef(value=""), init=False)
    augmented_fn: FunctionRef = Field(default=FunctionRef(value=""),
                                      description="The name of the function to reason on.",
                                      init=False)

    llm: NatLLM = Field(exclude=True)
    augmented_function: NatFunction = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.augmented_function:
            self.augmented_fn = FunctionRef(value=self.augmented_function.computed_name)
        return self


class ResponsesAPIAgentWorkflow(ResponsesAPIAgentWorkflowConfig, NatFunction):
    """Responses API Agent Workflow"""

    llm_name: LLMRef = Field(description="The LLM model to use with the agent.", default=LLMRef(value=""), init=False)
    nat_tools: list[FunctionRef] = Field(default_factory=list,
                                         description="The list of tools to provide to the agent.",
                                         init=False)

    llm: NatLLM = Field(exclude=True)
    tools: list[NatFunction] | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.tools and len(self.tools) > 0:
            self.nat_tools = [FunctionRef(value=tool.computed_name) for tool in self.tools]
        return self
