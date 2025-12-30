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
SDK classes for LangChain plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

from .tools.code_generation_tool import CodeGenerationToolConfig
from .tools.tavily_internet_search import TavilyInternetSearchToolConfig
from .tools.wikipedia_search import WikiSearchToolConfig


class TavilyInternetSearchTool(TavilyInternetSearchToolConfig, NatFunction):
    """Tavily Internet Search Tool"""


class CodeGenerationTool(CodeGenerationToolConfig, NatFunction):
    """Code Generation Tool"""

    llm_name: LLMRef = Field(description="LLM to use for code generation.", default=LLMRef(value=""), init=False)
    llm: NatLLM = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class WikiSearchTool(WikiSearchToolConfig, NatFunction):
    """Wikipedia Search Tool"""
