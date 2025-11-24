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

import json
import logging
import re
import textwrap
import traceback
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Union

import yaml
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
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.eval.config import EvaluationRunConfig
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils import run_workflow
from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_evaluator import NatEvaluator
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
    evaluator: NatEvaluator | None = Field(description="The evaluator to use with the agent", default=None)

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

    def add_evaluator(self, evaluator: NatEvaluator):
        self.evaluator = evaluator

        # Reset the _config property cache
        del self._config

    def save_to_config_file(self, file_path: str | Path) -> None:
        """Saves the agent's configuration to a YAML file.

        Args:
            file_path (str): The path to the YAML file where the configuration will be saved.
        """

        # Check if path exists
        file_path = Path(file_path)
        if file_path.exists() and file_path.is_dir():
            raise ValueError(f"file_path '{file_path}' is a directory, expected a file path")
        if not file_path.parent.exists():
            raise ValueError(f"Directory '{file_path.parent}' does not exist.")
        if file_path.suffix.lower() != ".yaml" and file_path.suffix.lower() != ".yml":
            raise ValueError(f"file_path '{file_path}' does not have a .yaml or .yml extension.")

        # Example usage: serialize agent._config and print YAML
        try:
            serialized = self._config.model_dump(exclude_unset=True, by_alias=True, round_trip=True)
            self.__add_type_field_for_model(serialized, self._config)

            # Prepare long strings: wrap long single-line strings at word boundaries
            # and leave existing newlines intact. Then dump using a dumper that
            # represents multiline strings with the block scalar `|` style.
            self.__prepare_multiline_strings(serialized, width=100)

            # Custom dumper that will use our representation for str values.
            # Use SafeDumper subclass
            class _BlockSafeDumper(yaml.SafeDumper):
                pass

            def _str_representer(dumper, data):
                # If the string contains a newline, or contains explicit line breaks
                # (we prepared long strings to contain line breaks), represent
                # using the literal block style `|` so YAML preserves formatting.
                if isinstance(data, str) and ("\n" in data):
                    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
                # Otherwise, use default representation (flow style)
                return dumper.represent_scalar("tag:yaml.org,2002:str", data)

            _BlockSafeDumper.add_representer(str, _str_representer)

            # Dump with our custom dumper and preserve key order
            # Dump to a string so we can post-process the YAML output
            # to insert a blank line between top-level declarations.
            yaml_str = yaml.dump(
                json.loads(json.dumps(serialized, sort_keys=False)),
                sort_keys=False,
                Dumper=_BlockSafeDumper,
            )

            # Insert a blank line before each top-level mapping key.
            # Top-level keys start at column 0 (no leading whitespace). We look
            # for a newline followed by a non-space character and a colon later
            # on the line, and insert an extra newline to separate sections.
            yaml_str = re.sub(r"\n(?=[^ \t].+?:)", "\n\n", yaml_str)

            with open(file_path, "w") as f:
                f.write(yaml_str)
        except Exception as e:
            # Fallback debugging output
            logger.error(f"Serialization failed: {e}")
            print(traceback.format_exc())

    def __wrap_long_string(self, s: str, width: int) -> str:
        """Wrap a single-line string at word boundaries to a maximum width.
        If the string already contains newline characters, return it unchanged.
        """
        if s is None:
            return s
        if "\n" in s:
            # already multiline — preserve natural breaks
            return s
        # Use a TextWrapper that avoids breaking long words or inserting hyphenation.
        # This avoids inserting `-` into wrapped lines.
        wrapper = textwrap.TextWrapper(width=width, break_long_words=False, break_on_hyphens=False)
        return "\n".join(wrapper.wrap(s))

    def __prepare_multiline_strings(self, obj: Any, width: int = 100) -> None:
        """Recursively walk the data structure and replace long strings with
        wrapped versions that include line breaks, so the YAML dumper will
        emit them using block scalars.
        This function mutates `obj` in place.
        """
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str):
                    obj[k] = self.__wrap_long_string(v, width)
                else:
                    self.__prepare_multiline_strings(v, width)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    obj[i] = self.__wrap_long_string(v, width)
                else:
                    self.__prepare_multiline_strings(v, width)

    def __add_type_field_for_model(self, data: Any, model: Any) -> None:
        """Recursively add a `_type` key to dict nodes based on the corresponding pydantic BaseModel instance.
        - `data` is the dict/list/primitive produced by `model.model_dump(...)`.
        - `model` is the pydantic BaseModel instance (if available) that produced `data`.
        This function mutates `data` in-place.
        """
        # Only operate when we have a BaseModel and a dict to annotate
        if not isinstance(model, BaseModel):
            return

        # Try to get a `type` attribute from the model instance. Many Config classes in this project
        # expose a `type` attribute which we want to serialize as `_type`.
        t = getattr(model, "type", None)
        if t is None:
            # fallback: some classes may use `type_` or `_type` or a class-level mapping; try a couple common names
            t = getattr(model, "type_", None) or getattr(model, "_type", None)

        if t is not None and isinstance(data, dict):
            # only add `_type` when something meaningful is present
            # Insert `_type` as the first key in the mapping so it always
            # appears first in the emitted YAML.
            # We mutate `data` in-place so callers keep the same object.
            existing_items = list(data.items())
            data.clear()
            data["_type"] = t
            for k, v in existing_items:
                if k == "_type":
                    # if `_type` existed already, skip duplicate
                    continue
                data[k] = v

        # Walk model fields to recurse into nested BaseModel instances, lists and dicts
        # `model.model_fields` is provided by pydantic v2 and describes declared fields.
        try:
            model_fields = getattr(model.__class__, "model_fields", None) or getattr(model, "model_fields", None)
        except Exception:
            model_fields = None

        if not model_fields or not isinstance(data, dict):
            return

        # For each declared field, find corresponding dumped value and the actual attribute on the model
        for field_name in model_fields:
            if field_name not in data:
                continue

            dumped_value = data[field_name]
            try:
                real_value = getattr(model, field_name)
            except Exception:
                real_value = None

            # Recurse for BaseModel child
            if isinstance(real_value, BaseModel) and isinstance(dumped_value, dict):
                self.__add_type_field_for_model(dumped_value, real_value)

            # Recurse for lists whose elements may be BaseModel instances
            elif isinstance(real_value, list) and isinstance(dumped_value, list):
                for i, elem in enumerate(dumped_value):
                    try:
                        real_elem = real_value[i]
                    except Exception:
                        real_elem = None
                    if isinstance(real_elem, BaseModel) and isinstance(elem, dict):
                        self.__add_type_field_for_model(elem, real_elem)

            # Recurse for dicts where values are BaseModel instances (mapping-of-models)
            elif isinstance(real_value, dict) and isinstance(dumped_value, dict):
                for k, v in dumped_value.items():
                    corresponding = None
                    try:
                        corresponding = real_value.get(k)
                    except Exception:
                        corresponding = None
                    if isinstance(corresponding, BaseModel) and isinstance(v, dict):
                        self.__add_type_field_for_model(v, corresponding)

        # Ensure that if `_type` was added by recursion on children or previously
        # present, it sits as the first key in this mapping as well.
        if isinstance(data, dict) and "_type" in data:
            existing_items = list(data.items())
            data.clear()
            # Re-insert `_type` first, then the rest in original order skipping duplicate
            data["_type"] = None
            for k, v in existing_items:
                if k == "_type":
                    # set the real value for `_type`
                    data["_type"] = v
                    break
            for k, v in existing_items:
                if k == "_type":
                    continue
                data[k] = v

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
            return {self.memory.memory_name: self.memory.config}
        return {mem.memory_name: mem.config for mem in self.memory}

    def build_general_configuration(self) -> GeneralConfig | None:
        if self.configuration is None:
            return None

        general_configuration = {}

        loggers = self.configuration.loggers
        if len(loggers) > 0:
            general_configuration["telemetry"] = {"logging": {lgr.logger_name: lgr.config for lgr in loggers}}

        telemetry_exporters = self.configuration.tracers
        if len(telemetry_exporters) > 0:
            if general_configuration.get("telemetry") is not None:
                general_configuration["telemetry"]["tracing"] = {
                    tracer.tracer_name: tracer.config
                    for tracer in telemetry_exporters
                }
            else:
                general_configuration["telemetry"] = {
                    "tracing": {
                        tracer.tracer_name: tracer.config
                        for tracer in telemetry_exporters
                    }
                }

        front_end = self.configuration.front_end_configuration
        if front_end is not None:
            general_configuration["front_end"] = front_end

        return GeneralConfig(**general_configuration)

    def build_functions(self) -> dict[str, FunctionBaseConfig] | None:
        if self.tools is None or len(self.tools) == 0:
            return None

        registered_functions = {}
        for tool in self.tools:
            if isinstance(tool, NatFunction):
                registered_functions[tool.tool_name] = tool.config
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

        registered_function_groups = {
            tool_group.function_group_name: tool_group.config
            for tool_group in self.tool_groups
        }

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

        registered_llms = {
            self.llm.llm_name: self.llm.config,
            **{
                llm.llm_name: llm.config
                for llm in self.referenced_llms
            },
        }

        if self.evaluator is not None:
            registered_llms = {
                **registered_llms,
                **{
                    llm.llm_name: llm.config
                    for llm in self.evaluator.evaluation_llms
                },
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
        registered_embedders = {embedder.embedder_name: embedder.config for embedder in self.referenced_embedders}

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

        registered_retrievers = {retriever.retriever_name: retriever.config for retriever in self.retrievers}

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

        registered_object_stores = {
            object_store.object_store_name: object_store.config
            for object_store in self.object_stores
        }

        return registered_object_stores

    def build_evaluator(self) -> EvalConfig | None:
        if self.evaluator is None:
            return None

        eval_config = {}

        if self.evaluator.general_evaluator is not None:
            eval_config["general"] = self.evaluator.general_evaluator
        if self.evaluator.evaluators is not None and len(self.evaluator.evaluators) > 0:
            eval_config["evaluators"] = {ev.evaluator_name: ev.config for ev in self.evaluator.evaluators}

        eval_config = EvalConfig(**eval_config)

        return eval_config

    def _build_tool_names(self) -> list[FunctionRef | FunctionGroupRef]:
        tool_names = []
        for tool in self.tools:
            if isinstance(tool, NatFunction):
                tool_names.append(FunctionRef(value=tool.tool_name))
            elif isinstance(tool, NatAgent):
                if tool.agent_name == "":
                    raise ValueError(
                        "Please set an agent_name since this agent will act as a tool for another agent to use.")
                else:
                    tool_names.append(FunctionRef(value=tool.agent_name))
            else:
                raise ValueError("Tools need to either be instances of NatTool or NatAgent")

        tool_group_names = [FunctionGroupRef(value=tool_group.function_group_name) for tool_group in self.tool_groups]

        return tool_names + tool_group_names

    def build_workflow(self) -> AgentBaseConfig:
        raise NotImplementedError("Subclasses must implement build_workflow method.")


class NatReactAgent(NatAgent, ReActAgentWorkflowConfig):

    def build_workflow(self) -> ReActAgentWorkflowConfig:
        return ReActAgentWorkflowConfig(
            tool_names=self._build_tool_names(),
            llm_name=LLMRef(value=self.llm.llm_name),
            **self.model_dump(
                exclude_unset=True,
                exclude={
                    "tool_namesllm_name",
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
            llm_name=LLMRef(value=self.llm.llm_name),
            **self.model_dump(
                exclude_unset=True,
                exclude={
                    "tool_namesllm_name",
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
            llm_name=LLMRef(value=self.llm.llm_name),
            **self.model_dump(
                exclude_unset=True,
                exclude={
                    "tool_namesllm_name",
                    "tools",
                    "tool_groups",
                    "llm",
                    "referenced_embedders",
                    "referenced_llms",
                },
            ),
        )
