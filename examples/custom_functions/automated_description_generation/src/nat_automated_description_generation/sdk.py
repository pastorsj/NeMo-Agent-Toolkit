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
from nat.data_models.component_ref import RetrieverRef
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM
from nat.utils.sdk.nat_retriever import NatRetriever

from .register import AutomatedDescriptionMilvusWorkflowConfig


class AutomatedDescriptionMilvusWorkflow(AutomatedDescriptionMilvusWorkflowConfig, NatFunction):
    """Automated Description Generation Workflow for Milvus Collections"""

    llm_name: LLMRef = Field(description="LLM to use for summarizing documents and generating a description.",
                             default=LLMRef(value=""),
                             init=False)
    retriever_name: RetrieverRef = Field(description="Name of the retriever to use for fetching documents.",
                                         default=RetrieverRef(value=""),
                                         init=False)
    retrieval_tool_name: FunctionRef = Field(description="Name of the retrieval tool to use for fetching documents.",
                                             default=FunctionRef(value=""),
                                             init=False)

    llm: NatLLM = Field(exclude=True)
    retriever: NatRetriever = Field(exclude=True)
    retrieval_tool: NatFunction = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.retriever:
            self.retriever_name = RetrieverRef(value=self.retriever.computed_name)
        if self.retrieval_tool:
            self.retrieval_tool_name = FunctionRef(value=self.retrieval_tool.computed_name)
        return self
