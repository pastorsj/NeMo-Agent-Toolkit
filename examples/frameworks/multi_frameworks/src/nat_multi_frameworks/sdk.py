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

from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

from .haystack_agent import HaystackChitchatConfig
from .langchain_research_tool import LangChainResearchConfig
from .llama_index_rag_tool import LlamaIndexRAGConfig
from .register import MultiFrameworksWorkflowConfig


class MultiFrameworksWorkflowTool(MultiFrameworksWorkflowConfig, NatFunction):
    """Multi Frameworks Workflow Tool"""

    llm: LLMRef = Field(description="LLM to use for the multi frameworks workflow.",
                        default=LLMRef(value="nim_llm"),
                        init=False)
    research_tool: FunctionRef = Field(description="Research tool to use for the multi frameworks workflow.",
                                       default=FunctionRef(value=""),
                                       init=False)
    rag_tool: FunctionRef = Field(description="RAG tool to use for the multi frameworks workflow.",
                                  default=FunctionRef(value=""),
                                  init=False)
    chitchat_agent: FunctionRef = Field(description="Chitchat agent to use for the multi frameworks workflow.",
                                        default=FunctionRef(value=""),
                                        init=False)

    nat_llm: NatLLM = Field(exclude=True)
    nat_research_tool: NatFunction = Field(exclude=True)
    nat_rag_tool: NatFunction = Field(exclude=True)
    nat_chitchat_agent: NatFunction = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.nat_llm:
            self.llm = LLMRef(value=self.nat_llm.computed_name)
        if self.nat_research_tool:
            self.research_tool = FunctionRef(value=self.nat_research_tool.computed_name)
        if self.nat_rag_tool:
            self.rag_tool = FunctionRef(value=self.nat_rag_tool.computed_name)
        if self.nat_chitchat_agent:
            self.chitchat_agent = FunctionRef(value=self.nat_chitchat_agent.computed_name)
        return self


class LangChainResearchTool(LangChainResearchConfig, NatFunction):
    """LangChain Research Tool"""

    llm_name: LLMRef = Field(description="LLM to use for the research task.", default=LLMRef(value=""), init=False)

    web_tool: FunctionRef = Field(description="Web search tool to use for the research task.",
                                  default=FunctionRef(value=""),
                                  init=False)

    llm: NatLLM = Field(exclude=True)
    web_tool_fn: NatFunction = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.web_tool_fn:
            self.web_tool = FunctionRef(value=self.web_tool_fn.computed_name)
        return self


class HaystackChitchatTool(HaystackChitchatConfig, NatFunction):
    """Haystack Chitchat Tool"""

    llm_name: LLMRef = Field(description="LLM to use for the chitchat agent.", default=LLMRef(value=""), init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class LlamaIndexRAGTool(LlamaIndexRAGConfig, NatFunction):
    """Llama Index RAG Tool"""

    llm_name: LLMRef = Field(description="LLM to use for the RAG task.", default=LLMRef(value=""), init=False)
    embedding_name: EmbedderRef = Field(description="Embedder to use for the RAG task.",
                                        default=EmbedderRef(value=""),
                                        init=False)

    llm: NatLLM = Field(exclude=True)
    embedder: NatEmbedder = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.embedder:
            self.embedding_name = EmbedderRef(value=self.embedder.computed_name)
        return self
