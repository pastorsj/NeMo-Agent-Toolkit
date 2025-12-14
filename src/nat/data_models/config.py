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

import json
import logging
import re
import sys
import textwrap
import traceback
import typing
from datetime import timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import ValidationError
from pydantic import ValidationInfo
from pydantic import ValidatorFunctionWrapHandler
from pydantic import field_validator

from nat.data_models.evaluate import EvalConfig
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.data_models.front_end import FrontEndBaseConfig
from nat.data_models.function import EmptyFunctionConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.logging import LoggingBaseConfig
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

from .authentication import AuthProviderBaseConfig
from .common import HashableBaseModel
from .common import TypedBaseModel
from .embedder import EmbedderBaseConfig
from .llm import LLMBaseConfig
from .memory import MemoryBaseConfig
from .middleware import FunctionMiddlewareBaseConfig
from .object_store import ObjectStoreBaseConfig
from .retriever import RetrieverBaseConfig

logger = logging.getLogger(__name__)


def _process_validation_error(err: ValidationError, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):
    from nat.cli.type_registry import GlobalTypeRegistry

    new_errors = []
    logged_once = False
    needs_reraise = False
    for e in err.errors():

        error_type = e['type']
        if error_type == 'union_tag_invalid' and "ctx" in e and not logged_once:
            requested_type = e["ctx"]["tag"]
            if (info.field_name in ('workflow', 'functions')):
                registered_keys = GlobalTypeRegistry.get().get_registered_functions()
            elif (info.field_name == "function_groups"):
                registered_keys = GlobalTypeRegistry.get().get_registered_function_groups()
            elif (info.field_name == "authentication"):
                registered_keys = GlobalTypeRegistry.get().get_registered_auth_providers()
            elif (info.field_name == "llms"):
                registered_keys = GlobalTypeRegistry.get().get_registered_llm_providers()
            elif (info.field_name == "embedders"):
                registered_keys = GlobalTypeRegistry.get().get_registered_embedder_providers()
            elif (info.field_name == "memory"):
                registered_keys = GlobalTypeRegistry.get().get_registered_memorys()
            elif (info.field_name == "object_stores"):
                registered_keys = GlobalTypeRegistry.get().get_registered_object_stores()
            elif (info.field_name == "retrievers"):
                registered_keys = GlobalTypeRegistry.get().get_registered_retriever_providers()
            elif (info.field_name == "tracing"):
                registered_keys = GlobalTypeRegistry.get().get_registered_telemetry_exporters()
            elif (info.field_name == "logging"):
                registered_keys = GlobalTypeRegistry.get().get_registered_logging_method()
            elif (info.field_name == "evaluators"):
                registered_keys = GlobalTypeRegistry.get().get_registered_evaluators()
            elif (info.field_name == "front_ends"):
                registered_keys = GlobalTypeRegistry.get().get_registered_front_ends()
            elif (info.field_name == "ttc_strategies"):
                registered_keys = GlobalTypeRegistry.get().get_registered_ttc_strategies()
            elif (info.field_name == "middleware"):
                registered_keys = GlobalTypeRegistry.get().get_registered_middleware()
            elif (info.field_name == "trainers"):
                registered_keys = GlobalTypeRegistry.get().get_registered_trainers()
            elif (info.field_name == "trainer_adapters"):
                registered_keys = GlobalTypeRegistry.get().get_registered_trainer_adapters()
            elif (info.field_name == "trajectory_builders"):
                registered_keys = GlobalTypeRegistry.get().get_registered_trajectory_builders()

            else:
                assert False, f"Unknown field name {info.field_name} in validator"

            # Check and see if the there are multiple full types which match this short type
            matching_keys = [k for k in registered_keys if k.local_name == requested_type]

            assert len(matching_keys) != 1, "Exact match should have been found. Contact developers"

            matching_key_names = [x.full_type for x in matching_keys]
            registered_key_names = [x.full_type for x in registered_keys]

            if (len(matching_keys) == 0):
                # This is a case where the requested type is not found. Show a helpful message about what is
                # available
                logger.error(("Requested %s type `%s` not found. "
                              "Have you ensured the necessary package has been installed with `uv pip install`?"
                              "\nAvailable %s names:\n - %s\n"),
                             info.field_name,
                             requested_type,
                             info.field_name,
                             '\n - '.join(registered_key_names))
            else:
                # This is a case where the requested type is ambiguous.
                logger.error(("Requested %s type `%s` is ambiguous. "
                              "Matched multiple %s by their local name: %s. "
                              "Please use the fully qualified %s name."
                              "\nAvailable %s names:\n - %s\n"),
                             info.field_name,
                             requested_type,
                             info.field_name,
                             matching_key_names,
                             info.field_name,
                             info.field_name,
                             '\n - '.join(registered_key_names))

            # Only show one error
            logged_once = True

        elif error_type == 'missing':
            location = e["loc"]
            if len(location) > 1:  # remove the _type field from the location
                e['loc'] = (location[0], ) + location[2:]
                needs_reraise = True

        new_errors.append(e)

    if needs_reraise:
        raise ValidationError.from_exception_data(title=err.title, line_errors=new_errors)


class TelemetryConfig(BaseModel):

    logging: dict[str, LoggingBaseConfig] = Field(default_factory=dict)
    tracing: dict[str, TelemetryExporterBaseConfig] = Field(default_factory=dict)

    @field_validator("logging", "tracing", mode="wrap")
    @classmethod
    def validate_components(cls, value: typing.Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):

        try:
            return handler(value)
        except ValidationError as err:
            _process_validation_error(err, handler, info)
            raise

    @classmethod
    def rebuild_annotations(cls):

        from nat.cli.type_registry import GlobalTypeRegistry

        type_registry = GlobalTypeRegistry.get()

        TracingAnnotation = dict[str,
                                 typing.Annotated[type_registry.compute_annotation(TelemetryExporterBaseConfig),
                                                  Discriminator(TypedBaseModel.discriminator)]]

        LoggingAnnotation = dict[str,
                                 typing.Annotated[type_registry.compute_annotation(LoggingBaseConfig),
                                                  Discriminator(TypedBaseModel.discriminator)]]

        should_rebuild = False

        tracing_field = cls.model_fields.get("tracing")
        if tracing_field is not None and tracing_field.annotation != TracingAnnotation:
            tracing_field.annotation = TracingAnnotation
            should_rebuild = True

        logging_field = cls.model_fields.get("logging")
        if logging_field is not None and logging_field.annotation != LoggingAnnotation:
            logging_field.annotation = LoggingAnnotation
            should_rebuild = True

        if (should_rebuild):
            return cls.model_rebuild(force=True)

        return False


class GeneralConfig(BaseModel):

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    use_uvloop: bool | None = Field(
        default=None,
        deprecated=
        "`use_uvloop` field is deprecated and will be removed in a future release. The use of `uv_loop` is now" +
        "automatically determined based on platform")
    """
    This field is deprecated and ignored. It previously controlled whether to use uvloop as the event loop. uvloop
    usage is now determined automatically based on the platform.
    """

    telemetry: TelemetryConfig = TelemetryConfig()

    default_user_id: str = Field(
        default="default_user_id",
        description="Default user ID for per-user workflows when "
        "no session is available (for example, when using 'nat run'). This value identifies "
        "the workflow instances. For multi-user deployments with 'nat serve', the 'nat-session' "
        "cookie overrides this value. Must be a non-empty string when used as a fallback user ID.")
    per_user_workflow_timeout: timedelta = Field(
        default=timedelta(minutes=30),
        description="Time after which inactive per-user workflows are cleaned up. "
        "Only applies when workflow is per-user. Defaults to 30 minutes.")
    per_user_workflow_cleanup_interval: timedelta = Field(
        default=timedelta(minutes=5),
        description="Interval for running cleanup of inactive per-user workflows. "
        "Only applies when workflow is per-user. Defaults to 5 minutes.")

    # FrontEnd Configuration
    front_end: FrontEndBaseConfig = FastApiFrontEndConfig()

    @field_validator("front_end", mode="wrap")
    @classmethod
    def validate_components(cls, value: typing.Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):

        try:
            return handler(value)
        except ValidationError as err:
            _process_validation_error(err, handler, info)
            raise

    @classmethod
    def rebuild_annotations(cls):

        from nat.cli.type_registry import GlobalTypeRegistry

        type_registry = GlobalTypeRegistry.get()

        FrontEndAnnotation = typing.Annotated[type_registry.compute_annotation(FrontEndBaseConfig),
                                              Discriminator(TypedBaseModel.discriminator)]

        should_rebuild = False

        front_end_field = cls.model_fields.get("front_end")
        if front_end_field is not None and front_end_field.annotation != FrontEndAnnotation:
            front_end_field.annotation = FrontEndAnnotation
            should_rebuild = True

        if (TelemetryConfig.rebuild_annotations()):
            should_rebuild = True

        if (should_rebuild):
            return cls.model_rebuild(force=True)

        return False


class Config(HashableBaseModel):

    model_config = ConfigDict(extra="forbid")

    # Global Options
    general: GeneralConfig = GeneralConfig()

    # Functions Configuration
    functions: dict[str, FunctionBaseConfig] = Field(default_factory=dict)

    # Function Groups Configuration
    function_groups: dict[str, FunctionGroupBaseConfig] = Field(default_factory=dict)

    # Middleware Configuration
    middleware: dict[str, FunctionMiddlewareBaseConfig] = Field(default_factory=dict)

    # LLMs Configuration
    llms: dict[str, LLMBaseConfig] = Field(default_factory=dict)

    # Embedders Configuration
    embedders: dict[str, EmbedderBaseConfig] = Field(default_factory=dict)

    # Memory Configuration
    memory: dict[str, MemoryBaseConfig] = Field(default_factory=dict)

    # Object Stores Configuration
    object_stores: dict[str, ObjectStoreBaseConfig] = Field(default_factory=dict)

    # Optimizer Configuration
    optimizer: OptimizerConfig = OptimizerConfig()

    # Retriever Configuration
    retrievers: dict[str, RetrieverBaseConfig] = Field(default_factory=dict)

    # TTC Strategies
    ttc_strategies: dict[str, TTCStrategyBaseConfig] = Field(default_factory=dict)

    # Workflow Configuration
    workflow: FunctionBaseConfig = EmptyFunctionConfig()

    # Authentication Configuration
    authentication: dict[str, AuthProviderBaseConfig] = Field(default_factory=dict)

    # Evaluation Options
    eval: EvalConfig = EvalConfig()

    # Finetuning Options
    trainers: dict[str, TrainerConfig] = Field(default_factory=dict)
    trainer_adapters: dict[str, TrainerAdapterConfig] = Field(default_factory=dict)
    trajectory_builders: dict[str, TrajectoryBuilderConfig] = Field(default_factory=dict)
    finetuning: FinetuneConfig = FinetuneConfig()

    def print_summary(self, stream: typing.TextIO = sys.stdout):
        """Print a summary of the configuration"""

        stream.write("\nConfiguration Summary:\n")
        stream.write("-" * 20 + "\n")
        if self.workflow:
            stream.write(f"Workflow Type: {self.workflow.type}\n")

        stream.write(f"Number of Functions: {len(self.functions)}\n")
        stream.write(f"Number of Function Groups: {len(self.function_groups)}\n")
        stream.write(f"Number of LLMs: {len(self.llms)}\n")
        stream.write(f"Number of Embedders: {len(self.embedders)}\n")
        stream.write(f"Number of Memory: {len(self.memory)}\n")
        stream.write(f"Number of Object Stores: {len(self.object_stores)}\n")
        stream.write(f"Number of Retrievers: {len(self.retrievers)}\n")
        stream.write(f"Number of TTC Strategies: {len(self.ttc_strategies)}\n")
        stream.write(f"Number of Authentication Providers: {len(self.authentication)}\n")

    @field_validator("functions",
                     "function_groups",
                     "middleware",
                     "llms",
                     "embedders",
                     "memory",
                     "retrievers",
                     "workflow",
                     "ttc_strategies",
                     "authentication",
                     "trainers",
                     "trainer_adapters",
                     "trajectory_builders",
                     mode="wrap")
    @classmethod
    def validate_components(cls, value: typing.Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):

        try:
            return handler(value)
        except ValidationError as err:
            _process_validation_error(err, handler, info)
            raise

    @classmethod
    def rebuild_annotations(cls):

        from nat.cli.type_registry import GlobalTypeRegistry

        type_registry = GlobalTypeRegistry.get()

        LLMsAnnotation = dict[str,
                              typing.Annotated[type_registry.compute_annotation(LLMBaseConfig),
                                               Discriminator(TypedBaseModel.discriminator)]]

        AuthenticationProviderAnnotation = dict[str,
                                                typing.Annotated[
                                                    type_registry.compute_annotation(AuthProviderBaseConfig),
                                                    Discriminator(TypedBaseModel.discriminator)]]

        EmbeddersAnnotation = dict[str,
                                   typing.Annotated[type_registry.compute_annotation(EmbedderBaseConfig),
                                                    Discriminator(TypedBaseModel.discriminator)]]

        FunctionsAnnotation = dict[str,
                                   typing.Annotated[type_registry.compute_annotation(FunctionBaseConfig),
                                                    Discriminator(TypedBaseModel.discriminator)]]

        FunctionGroupsAnnotation = dict[str,
                                        typing.Annotated[type_registry.compute_annotation(FunctionGroupBaseConfig),
                                                         Discriminator(TypedBaseModel.discriminator)]]

        MiddlewareAnnotation = dict[str,
                                    typing.Annotated[type_registry.compute_annotation(FunctionMiddlewareBaseConfig),
                                                     Discriminator(TypedBaseModel.discriminator)]]

        MemoryAnnotation = dict[str,
                                typing.Annotated[type_registry.compute_annotation(MemoryBaseConfig),
                                                 Discriminator(TypedBaseModel.discriminator)]]

        ObjectStoreAnnotation = dict[str,
                                     typing.Annotated[type_registry.compute_annotation(ObjectStoreBaseConfig),
                                                      Discriminator(TypedBaseModel.discriminator)]]
        RetrieverAnnotation = dict[str,
                                   typing.Annotated[type_registry.compute_annotation(RetrieverBaseConfig),
                                                    Discriminator(TypedBaseModel.discriminator)]]

        TTCStrategyAnnotation = dict[str,
                                     typing.Annotated[type_registry.compute_annotation(TTCStrategyBaseConfig),
                                                      Discriminator(TypedBaseModel.discriminator)]]

        WorkflowAnnotation = typing.Annotated[(type_registry.compute_annotation(FunctionBaseConfig)),
                                              Discriminator(TypedBaseModel.discriminator)]

        TrainersAnnotation = dict[str,
                                  typing.Annotated[type_registry.compute_annotation(TrainerConfig),
                                                   Discriminator(TypedBaseModel.discriminator)]]

        TrainerAdaptersAnnotation = dict[str,
                                         typing.Annotated[type_registry.compute_annotation(TrainerAdapterConfig),
                                                          Discriminator(TypedBaseModel.discriminator)]]

        TrajectoryBuildersAnnotation = dict[str,
                                            typing.Annotated[type_registry.compute_annotation(TrajectoryBuilderConfig),
                                                             Discriminator(TypedBaseModel.discriminator)]]

        should_rebuild = False

        auth_providers_field = cls.model_fields.get("authentication")
        if auth_providers_field is not None and auth_providers_field.annotation != AuthenticationProviderAnnotation:
            auth_providers_field.annotation = AuthenticationProviderAnnotation
            should_rebuild = True

        llms_field = cls.model_fields.get("llms")
        if llms_field is not None and llms_field.annotation != LLMsAnnotation:
            llms_field.annotation = LLMsAnnotation
            should_rebuild = True

        embedders_field = cls.model_fields.get("embedders")
        if embedders_field is not None and embedders_field.annotation != EmbeddersAnnotation:
            embedders_field.annotation = EmbeddersAnnotation
            should_rebuild = True

        functions_field = cls.model_fields.get("functions")
        if functions_field is not None and functions_field.annotation != FunctionsAnnotation:
            functions_field.annotation = FunctionsAnnotation
            should_rebuild = True

        function_groups_field = cls.model_fields.get("function_groups")
        if function_groups_field is not None and function_groups_field.annotation != FunctionGroupsAnnotation:
            function_groups_field.annotation = FunctionGroupsAnnotation
            should_rebuild = True

        middleware_field = cls.model_fields.get("middleware")
        if (middleware_field is not None and middleware_field.annotation != MiddlewareAnnotation):
            middleware_field.annotation = MiddlewareAnnotation
            should_rebuild = True

        memory_field = cls.model_fields.get("memory")
        if memory_field is not None and memory_field.annotation != MemoryAnnotation:
            memory_field.annotation = MemoryAnnotation
            should_rebuild = True

        object_stores_field = cls.model_fields.get("object_stores")
        if object_stores_field is not None and object_stores_field.annotation != ObjectStoreAnnotation:
            object_stores_field.annotation = ObjectStoreAnnotation
            should_rebuild = True

        retrievers_field = cls.model_fields.get("retrievers")
        if retrievers_field is not None and retrievers_field.annotation != RetrieverAnnotation:
            retrievers_field.annotation = RetrieverAnnotation
            should_rebuild = True

        ttc_strategies_field = cls.model_fields.get("ttc_strategies")
        if ttc_strategies_field is not None and ttc_strategies_field.annotation != TTCStrategyAnnotation:
            ttc_strategies_field.annotation = TTCStrategyAnnotation
            should_rebuild = True

        workflow_field = cls.model_fields.get("workflow")
        if workflow_field is not None and workflow_field.annotation != WorkflowAnnotation:
            workflow_field.annotation = WorkflowAnnotation
            should_rebuild = True

        trainers_field = cls.model_fields.get("trainers")
        if trainers_field is not None and trainers_field.annotation != TrainersAnnotation:
            trainers_field.annotation = TrainersAnnotation
            should_rebuild = True

        trainer_adapters_field = cls.model_fields.get("trainer_adapters")
        if trainer_adapters_field is not None and trainer_adapters_field.annotation != TrainerAdaptersAnnotation:
            trainer_adapters_field.annotation = TrainerAdaptersAnnotation
            should_rebuild = True

        trajectory_builders_field = cls.model_fields.get("trajectory_builders")
        if (trajectory_builders_field is not None
                and trajectory_builders_field.annotation != TrajectoryBuildersAnnotation):
            trajectory_builders_field.annotation = TrajectoryBuildersAnnotation
            should_rebuild = True

        if (GeneralConfig.rebuild_annotations()):
            should_rebuild = True

        if (EvalConfig.rebuild_annotations()):
            should_rebuild = True

        if (should_rebuild):
            return cls.model_rebuild(force=True)

        return False

    def save_to_file(self, file_path: str | Path) -> None:
        """Saves the agent's configuration to a YAML file.

        Args:
            file_path (str | Path): The path to the YAML file where the configuration will be saved.
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
            serialized = self.model_dump(exclude_unset=True, by_alias=True, round_trip=True)
            self.__add_type_field_for_model(serialized, self)

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

    def __prepare_multiline_strings(self, obj: typing.Any, width: int = 100) -> None:
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

    def __add_type_field_for_model(self, data: typing.Any, model: typing.Any) -> None:
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


# Compatibility aliases with previous releases
AIQConfig = Config
