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
from functools import cached_property
from pathlib import Path
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from nat.agent.react_agent.register import ReActAgentWorkflowConfig
from nat.agent.rewoo_agent.register import ReWOOAgentWorkflowConfig
from nat.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from nat.cli.commands.evaluate import run_and_evaluate
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.evaluate import EvalConfig
from nat.data_models.evaluate import EvalGeneralConfig
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.front_end import FrontEndBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.logging import LoggingBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.eval.config import EvaluationRunConfig
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils import run_workflow
from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_evaluation import NatEvaluation
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_function_group import NatFunctionGroup
from nat.utils.sdk.nat_general_configuraton import NatGeneralConfiguration
from nat.utils.sdk.nat_llm import NatLLM
from nat.utils.sdk.nat_memory import NatMemory
from nat.utils.sdk.nat_object_store import NatObjectStore
from nat.utils.sdk.nat_retriever import NatRetriever

logger = logging.getLogger(__name__)


class NatAgent(BaseModel):
    configuration: NatGeneralConfiguration | None = Field(
        description="The general configuration of the agent, including loggers, tracers, and a front end", default=None)
    agent_name: str = Field(description="The name of the agent if it is being used as a tool", default="")
    llm_name: LLMRef = Field(description="The LLM model to use with the agent.", default=LLMRef(value=""))

    memory: NatMemory | list[NatMemory] | None = Field(description="The memory configuration for the agent.",
                                                       default=None)
    tools: list[Union[NatFunction, "NatAgent"]] = Field(description="List of tools to be used by the agent.",
                                                        default=[])  # noqa: UP007
    tool_groups: list[NatFunctionGroup] = Field(description="List of tool groups to be used by the agent.", default=[])
    object_stores: list[NatObjectStore] = Field(description="List of object stores used across the agent", default=[])
    llm: NatLLM = Field(description="The LLM model to use with the agent.")
    referenced_embedders: list[NatEmbedder] = Field(
        description="List of embedders referenced by tools used by the agent", default=[])
    referenced_llms: list[NatLLM] = Field(description="List of LLMs referenced by the tools used by the agent.",
                                          default=[])
    retrievers: list[NatRetriever] = Field(description="List of retrievers referenced by the tools used by the agent.",
                                           default=[])
    evaluator: NatEvaluation | None = Field(description="The evaluator to use with the agent", default=None)

    @cached_property
    def _config(self) -> Config:
        discover_and_register_plugins(PluginTypes.ALL)

        return self.__build_config_object()

    async def prompt(self, prompt: str):
        # Use the agent to process the prompt
        return await run_workflow(config=self._config, prompt=prompt)

    async def evaluate(self):
        if self.evaluator is None:
            raise ValueError("No evaluator has been set. Please set an evaluator before running this.")

        config = EvaluationRunConfig(
            config_file=self._config,
            dataset=str(self.evaluator.dataset) if self.evaluator.dataset else None,
            result_json_path=self.evaluator.result_json_path,
            skip_workflow=self.evaluator.skip_workflow,
            skip_completed_entries=self.evaluator.skip_completed_entries,
            endpoint=self.evaluator.endpoint,
            endpoint_timeout=self.evaluator.endpoint_timeout,
            reps=self.evaluator.reps,
            override=self.evaluator.override,
        )
        return await run_and_evaluate(config)

    def add_evaluator(self, evaluator: NatEvaluation):
        self.evaluator = evaluator

        # Reset the _config property cache
        del self._config

    def save_to_config_file(self, file_path: str | Path) -> None:
        self._config.save_to_file(file_path)

    def __build_config_object(self) -> Config:
        # Build config using tools, llms...

        config_args = {}

        general_configuration = self.build_general_configuration()
        if general_configuration is not None:
            config_args["general"] = general_configuration

        config_functions = self.build_functions()
        if config_functions is not None and len(config_functions) > 0:
            config_args["functions"] = config_functions

        config_function_groups = self.build_function_groups()
        if config_function_groups is not None and len(config_function_groups) > 0:
            config_args["function_groups"] = config_function_groups

        config_llms = self.build_llms()
        if config_llms is not None and len(config_llms) > 0:
            config_args["llms"] = config_llms

        config_embedders = self.build_embedders()
        if config_embedders is not None and len(config_embedders) > 0:
            config_args["embedders"] = config_embedders

        config_memory = self.build_memory()
        if config_memory is not None:
            config_args["memory"] = config_memory

        config_object_stores = self.build_object_stores()
        if config_object_stores is not None and len(config_object_stores) > 0:
            config_args["object_stores"] = config_object_stores

        config_retrievers = self.build_retrievers()
        if config_retrievers is not None and len(config_retrievers) > 0:
            config_args["retrievers"] = config_retrievers

        workflow = self.build_workflow()
        config_args["workflow"] = workflow

        evaluator = self.build_evaluator()
        if evaluator is not None:
            config_args["eval"] = evaluator

        try:
            config = Config(**config_args)
        except ValidationError as e:
            logger.error(e.errors())
            raise

        return config

    def build_memory(self) -> dict[str, MemoryBaseConfig] | None:
        if self.memory is None or (isinstance(self.memory, list) and len(self.memory) == 0):
            return None
        if isinstance(self.memory, NatMemory):
            return dict(self.memory.compute_name_and_config(MemoryBaseConfig))
        return dict(mem.compute_name_and_config(MemoryBaseConfig) for mem in self.memory)

    def build_general_configuration(self) -> GeneralConfig | None:
        if self.configuration is None:
            return None
        else:
            general_configuration = {}

            loggers = self.configuration.loggers
            if len(loggers) > 0:
                general_configuration["telemetry"] = {
                    "logging": dict(lgr.compute_name_and_config(LoggingBaseConfig) for lgr in loggers)
                }

            telemetry_exporters = self.configuration.telemetry_exporters
            if len(telemetry_exporters) > 0:
                if general_configuration.get("telemetry") is not None:
                    general_configuration["telemetry"]["tracing"] = dict(
                        tracer.compute_name_and_config(TelemetryExporterBaseConfig) for tracer in telemetry_exporters)
                else:
                    general_configuration["telemetry"] = {
                        "tracing":
                            dict(
                                tracer.compute_name_and_config(TelemetryExporterBaseConfig)
                                for tracer in telemetry_exporters)
                    }

            front_end = self.configuration.front_end_configuration
            if front_end is not None:
                general_configuration["front_end"] = front_end.compute_config(FrontEndBaseConfig)

            return GeneralConfig(**general_configuration)

    def build_functions(self) -> dict[str, NatFunction] | None:
        if self.tools is None or len(self.tools) == 0:
            return None

        registered_functions = {}
        for tool in self.tools:
            if isinstance(tool, NatFunction):
                computed_name, config_type = tool.compute_name_and_config(FunctionBaseConfig)
                registered_functions[computed_name] = config_type
            elif isinstance(tool, NatAgent):
                if tool.agent_name == "":
                    raise ValueError(
                        "Please set an agent_name since this agent will act as a tool for another agent to use.")
                else:
                    # If this is an agent, register the agent at the top level, then recurse into the agent and
                    # register the other functions or agents.
                    registered_functions[tool.agent_name] = tool.build_workflow()

                    agent_functions = tool.build_functions()
                    if agent_functions is not None and len(agent_functions) > 0:
                        registered_functions = {**registered_functions, **agent_functions}
            else:
                raise ValueError("Tools need to either be instances of NatTool or NatAgent")

        return registered_functions

    def build_function_groups(self) -> dict[str, FunctionGroupBaseConfig] | None:
        if (self.tool_groups is None or len(self.tool_groups) == 0) and (self.tools is None or len(self.tools) == 0):
            return None

        registered_function_groups = dict(
            tool_group.compute_name_and_config(FunctionGroupBaseConfig) for tool_group in self.tool_groups)

        # Check if an agent registered as a function contains tool groups that were not registered at the top level
        if self.tools is not None or len(self.tools) > 0:
            for tool in self.tools:
                if isinstance(tool, NatAgent):
                    agent_function_groups = tool.build_function_groups()
                    if agent_function_groups is not None and len(agent_function_groups) > 0:
                        registered_function_groups = {**registered_function_groups, **agent_function_groups}

        return registered_function_groups

    def build_llms(self) -> dict[str, LLMBaseConfig] | None:
        if ((self.referenced_llms is None or len(self.referenced_llms) == 0) and (self.llm is None)
                and (self.evaluator is None or len(self.evaluator.evaluation_llms) == 0)
                and (self.tools is None or len(self.tools) == 0)):
            return None

        llm_name, llm_config = self.llm.compute_name_and_config(LLMBaseConfig)
        registered_llms = {
            llm_name: llm_config, **dict(llm.compute_name_and_config(LLMBaseConfig) for llm in self.referenced_llms)
        }

        if self.evaluator is not None:
            registered_llms = {
                **registered_llms,
                **dict(llm.compute_name_and_config(LLMBaseConfig) for llm in self.evaluator.evaluation_llms)
            }

        # Check if an agent registered as a function contains llms that were not registered at the top level
        if self.tools is not None or len(self.tools) > 0:
            for tool in self.tools:
                if isinstance(tool, NatAgent):
                    agent_llms = tool.build_llms()
                    if agent_llms is not None and len(agent_llms) > 0:
                        registered_llms = {**registered_llms, **agent_llms}

        return registered_llms

    def build_embedders(self) -> dict[str, EmbedderBaseConfig] | None:
        if (self.referenced_embedders is None or len(self.referenced_embedders) == 0) and (self.tools is None
                                                                                           or len(self.tools) == 0):
            return None
        registered_embedders = dict(
            embedder.compute_name_and_config(EmbedderBaseConfig) for embedder in self.referenced_embedders)

        # Check if an agent registered as a function contains embedders that were not registered at the top level
        if self.tools is not None or len(self.tools) > 0:
            for tool in self.tools:
                if isinstance(tool, NatAgent):
                    agent_embedders = tool.build_embedders()
                    if agent_embedders is not None and len(agent_embedders) > 0:
                        registered_embedders = {**registered_embedders, **agent_embedders}

        return registered_embedders

    def build_retrievers(self) -> dict[str, RetrieverBaseConfig] | None:
        # Gather retrievers from tools if any
        if (self.retrievers is None or len(self.retrievers) == 0) and (self.tools is None or len(self.tools) == 0):
            return None

        registered_retrievers = dict(
            retriever.compute_name_and_config(RetrieverBaseConfig) for retriever in self.retrievers)

        # Check if an agent registered as a function contains embedders that were not registered at the top level
        if self.tools is not None or len(self.tools) > 0:
            for tool in self.tools:
                if isinstance(tool, NatAgent):
                    agent_retrievers = tool.build_retrievers()
                    if agent_retrievers is not None and len(agent_retrievers) > 0:
                        registered_retrievers = {**registered_retrievers, **agent_retrievers}

        return registered_retrievers

    def build_object_stores(self) -> dict[str, ObjectStoreBaseConfig] | None:
        # Gather retrievers from tools if any
        if (self.object_stores is None or len(self.object_stores) == 0):
            return None

        registered_object_stores = dict(
            object_store.compute_name_and_config(ObjectStoreBaseConfig) for object_store in self.object_stores)

        return registered_object_stores

    def build_evaluator(self) -> EvalConfig | None:
        if self.evaluator is None:
            return None

        eval_config = {}

        if self.evaluator.general_evaluator is not None:
            eval_config["general"] = EvalGeneralConfig(**self.evaluator.general_evaluator.model_dump(
                exclude_unset=True))
        if self.evaluator.evaluators is not None and len(self.evaluator.evaluators) > 0:
            eval_config["evaluators"] = dict(
                ev.compute_name_and_config(EvaluatorBaseConfig) for ev in self.evaluator.evaluators)

        eval_config = EvalConfig(**eval_config)

        return eval_config

    def _build_tool_names(self) -> list[FunctionRef | FunctionGroupRef]:
        tool_names = []
        for tool in self.tools:
            if isinstance(tool, NatFunction):
                tool_names.append(FunctionRef(value=tool.compute_name(FunctionBaseConfig)))
            elif isinstance(tool, NatAgent):
                if tool.agent_name == "":
                    raise ValueError(
                        "Please set an agent_name since this agent will act as a tool for another agent to use.")
                else:
                    tool_names.append(FunctionRef(value=tool.agent_name))
            else:
                raise ValueError("Tools need to either be instances of NatTool or NatAgent")

        tool_group_names = [
            FunctionGroupRef(value=tool_group.compute_name(FunctionGroupBaseConfig)) for tool_group in self.tool_groups
        ]

        return tool_names + tool_group_names

    def build_workflow(self) -> AgentBaseConfig:
        raise NotImplementedError("Subclasses must implement build_workflow method.")


class NatReactAgent(NatAgent, ReActAgentWorkflowConfig):

    def build_workflow(self) -> ReActAgentWorkflowConfig:
        return ReActAgentWorkflowConfig(
            tool_names=self._build_tool_names(),
            llm_name=LLMRef(value=self.llm.compute_name(LLMBaseConfig)),
            **self.model_dump(
                exclude_unset=True,
                exclude={
                    "tool_names",
                    "llm_name",
                    "tools",
                    "tool_groups",
                    "llm",
                    "referenced_embedders",
                    "referenced_llms",
                },
            ),
        )


class NatRewooAgent(NatAgent, ReWOOAgentWorkflowConfig):

    def build_workflow(self) -> ReWOOAgentWorkflowConfig:
        return ReWOOAgentWorkflowConfig(
            tool_names=self._build_tool_names(),
            llm_name=LLMRef(value=self.llm.compute_name(LLMBaseConfig)),
            **self.model_dump(
                exclude_unset=True,
                exclude={
                    "tool_names",
                    "llm_name",
                    "tools",
                    "tool_groups",
                    "llm",
                    "referenced_embedders",
                    "referenced_llms",
                },
            ),
        )


class NatToolCallingAgent(NatAgent, ToolCallAgentWorkflowConfig):

    def build_workflow(self) -> ToolCallAgentWorkflowConfig:
        return ToolCallAgentWorkflowConfig(
            tool_names=self._build_tool_names(),
            llm_name=LLMRef(value=self.llm.compute_name(LLMBaseConfig)),
            **self.model_dump(
                exclude_unset=True,
                exclude={
                    "tool_names",
                    "llm_name",
                    "tools",
                    "tool_groups",
                    "llm",
                    "referenced_embedders",
                    "referenced_llms",
                },
            ),
        )
