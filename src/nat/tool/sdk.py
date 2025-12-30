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
SDK classes for Tools.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import RetrieverRef
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_function_group import NatFunctionGroup
from nat.utils.sdk.nat_llm import NatLLM
from nat.utils.sdk.nat_memory import NatMemory
from nat.utils.sdk.nat_retriever import NatRetriever

from .chat_completion import ChatCompletionConfig
from .code_execution.register import CodeExecutionToolConfig
from .datetime_tools import CurrentTimeToolConfig
from .document_search import MilvusDocumentSearchToolConfig
from .github_tools import GithubFilesGroupConfig
from .github_tools import GithubGroupConfig
from .memory_tools.add_memory_tool import AddToolConfig
from .memory_tools.delete_memory_tool import DeleteToolConfig
from .memory_tools.get_memory_tool import GetToolConfig
from .nvidia_rag import NVIDIARAGToolConfig
from .retriever import RetrieverToolConfig


class CurrentTimeTool(CurrentTimeToolConfig, NatFunction):
    """Current Time Tool"""


class ChatCompletion(ChatCompletionConfig, NatFunction):
    """Chat Completion Tool"""

    llm: NatLLM = Field(exclude=True)
    llm_name: LLMRef = Field(description="", default=LLMRef(value=""), init=False)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class NatRetrieverTool(RetrieverToolConfig, NatFunction):
    """NAT Retriever Tool"""

    retriever: RetrieverRef = Field(description="", default=RetrieverRef(value=""), init=False)
    nat_retriever: NatRetriever = Field(exclude=True)

    @model_validator(mode='after')
    def set_retriever_name_from_retriever(self):
        """Set retriever name from retriever object if retriever is provided."""
        if self.nat_retriever:
            self.retriever = RetrieverRef(value=self.nat_retriever.computed_name)
        return self


class NVIDIARAGTool(NVIDIARAGToolConfig, NatFunction):
    """NVIDIA RAG Tool"""


class MilvusDocumentSearchTool(MilvusDocumentSearchToolConfig, NatFunction):
    """Milvus Document Search Tool"""

    llm: NatLLM = Field(exclude=True)
    llm_name: LLMRef = Field(description="", default=LLMRef(value=""), init=False)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class CodeExecutionTool(CodeExecutionToolConfig, NatFunction):
    """Code Execution Tool"""


class GithubGroup(GithubGroupConfig, NatFunctionGroup):
    """GitHub Function Group"""


class GithubFilesTool(GithubFilesGroupConfig, NatFunction):
    """Github Files Tool"""


class GetMemoryTool(GetToolConfig, NatFunction):
    """Get Memory Tool"""

    nat_memory: NatMemory = Field(exclude=True)
    memory: MemoryRef = Field(description="", default=MemoryRef(value=""), init=False)

    @model_validator(mode='after')
    def set_memory_name_from_memory(self):
        """Set memory name from memory object if memory is provided."""
        if self.nat_memory:
            self.memory = MemoryRef(value=self.nat_memory.computed_name)
        return self


class AddMemoryTool(AddToolConfig, NatFunction):
    """Add Memory Tool"""

    nat_memory: NatMemory = Field(exclude=True)
    memory: MemoryRef = Field(description="", default=MemoryRef(value=""), init=False)

    @model_validator(mode='after')
    def set_memory_name_from_memory(self):
        """Set memory name from memory object if memory is provided."""
        if self.nat_memory:
            self.memory = MemoryRef(value=self.nat_memory.computed_name)
        return self


class DeleteMemoryTool(DeleteToolConfig, NatFunction):
    """Delete Memory Tool"""

    nat_memory: NatMemory = Field(exclude=True)
    memory: MemoryRef = Field(description="", default=MemoryRef(value=""), init=False)

    @model_validator(mode='after')
    def set_memory_name_from_memory(self):
        """Set memory name from memory object if memory is provided."""
        if self.nat_memory:
            self.memory = MemoryRef(value=self.nat_memory.computed_name)
        return self
