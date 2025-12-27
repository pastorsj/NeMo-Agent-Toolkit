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

from pydantic import Field
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

log = logging.getLogger(__name__)


class CodeGenerationToolConfig(FunctionBaseConfig, name="code_generation"):
    """
    Tool for generating code using the configured LLM.

    ## Details
    Name: Code Generation Tool
    Icon: N/A
    """
    llm_name: LLMRef = Field(description="LLM to use for code generation.")
    verbose: bool = False
    programming_language: str = "Python"
    description: str = ("Useful to generate Python code. For any questions about code generation, you must only use "
                        "this tool!")


class CodeGenerationTool(CodeGenerationToolConfig, NatFunction):
    """Code Generation Tool"""

    llm_name: LLMRef = Field(description="LLM to use for code generation.", default=LLMRef(value=""), init=False)
    llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


@register_function(config_type=CodeGenerationToolConfig)
async def code_generation_tool(config: CodeGenerationToolConfig, builder: Builder):
    from langchain_core.prompts.chat import ChatPromptTemplate

    log.info('Initializing code generation tool\nGetting tool LLM from config')
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_prompt = """
You are a helpful code assistant that can teach a junior developer how to code.  Your language of
 choice is {programming_language}. Don't explain the code, just generate the code block itself.
"""
    user_prompt = """
{question}
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    log.info("Filling tool's prompt variable from config")
    prompt = prompt.partial(programming_language=config.programming_language)
    tool = prompt | llm
    log.info('Initialized code generation tool')

    async def _inner(query: str) -> str:
        log.info('Running code generation tool')
        response = await tool.ainvoke({"question": query})
        if config.verbose:
            log.debug('Tool input was: %s\nTool output is: \n%s', query, response)
        return response.text()

    yield FunctionInfo.from_fn(_inner, description=config.description)
