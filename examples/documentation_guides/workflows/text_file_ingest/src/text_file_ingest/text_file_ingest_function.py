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

import logging
import os

from pydantic import Field
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_function import NatFunction

logger = logging.getLogger(__name__)


class TextFileIngestFunctionConfig(FunctionBaseConfig, name="text_file_ingest"):
    ingest_glob: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = Field(description="Embedder to use for text file ingest.",
                                       default=EmbedderRef(value="nvidia/nv-embedqa-e5-v5"))


class TextFileIngestTool(TextFileIngestFunctionConfig, NatFunction):

    embedder_name: EmbedderRef = Field(description="Embedder to use for text file ingest.",
                                       default=EmbedderRef(value="nvidia/nv-embedqa-e5-v5"),
                                       init=False)

    embedder: NatEmbedder = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set embedder name from embedder object if embedder is provided."""
        if self.embedder:
            self.embedder_name = EmbedderRef(value=self.embedder.compute_name(EmbedderBaseConfig))
        return self


@register_function(config_type=TextFileIngestFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def text_file_ingest_tool(config: TextFileIngestFunctionConfig, builder: Builder):

    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import USearch
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings: Embeddings = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    logger.info("Ingesting documents from: %s", config.ingest_glob)
    (ingest_dir, ingest_glob) = os.path.split(config.ingest_glob)
    loader = DirectoryLoader(ingest_dir, glob=ingest_glob, loader_cls=TextLoader)

    docs = [document async for document in loader.alazy_load()]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size)
    documents = text_splitter.split_documents(docs)
    vector = await USearch.afrom_documents(documents, embeddings)

    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "text_file_ingest",
        config.description,
    )

    async def _inner(query: str) -> str:

        return await retriever_tool.arun(query)

    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
    yield FunctionInfo.from_fn(_inner, description=config.description)
