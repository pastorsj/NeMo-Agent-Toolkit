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
SDK classes for Vanna plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import RetrieverRef
from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM
from nat.utils.sdk.nat_retriever import NatRetriever

from .execute_db_query import ExecuteDBQueryConfig
from .text2sql import Text2SQLConfig


class Text2SQLTool(Text2SQLConfig, NatFunction):
    """Text2SQL Tool"""

    llm_name: LLMRef = Field(description="LLM for SQL generation", default=LLMRef(value=""), init=False)
    embedder_name: EmbedderRef = Field(
        description="Embedder for vector operations",
        default=EmbedderRef(value=""),
        init=False,
    )
    milvus_retriever: RetrieverRef = Field(
        description="Milvus retriever reference for vector operations.",
        default=RetrieverRef(value=""),
        init=False,
    )

    llm: NatLLM = Field(exclude=True)
    embedder: NatEmbedder = Field(exclude=True)
    retriever: NatRetriever = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        if self.embedder:
            self.embedder_name = EmbedderRef(value=self.embedder.computed_name)
        if self.retriever:
            self.milvus_retriever = RetrieverRef(value=self.retriever.computed_name)
        return self


class ExecuteDBQueryTool(ExecuteDBQueryConfig, NatFunction):
    """Execute DB Query Tool"""
