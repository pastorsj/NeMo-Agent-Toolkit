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

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import ValidationError

from nat.cli.commands.evaluate import run_and_evaluate
from nat.cli.commands.optimize import run_optimizer
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.evaluate import EvalConfig
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.data_models.front_end import FrontEndBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.logging import LoggingBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.middleware import MiddlewareBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.optimizer import OptimizerRunConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.eval.config import EvaluationRunConfig
from nat.finetuning.finetuning_runtime import finetuning_main
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils import run_workflow
from nat.utils.sdk.nat_agent import NatAgent
from nat.utils.sdk.nat_auth_provider import NatAuthProvider
from nat.utils.sdk.nat_base import NatBase
from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_evaluation import NatEvaluation
from nat.utils.sdk.nat_evaluator import NatEvaluator
from nat.utils.sdk.nat_finetuner import NatFinetuner
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_function_group import NatFunctionGroup
from nat.utils.sdk.nat_general_configuraton import NatGeneralConfiguration
from nat.utils.sdk.nat_llm import NatLLM
from nat.utils.sdk.nat_memory import NatMemory
from nat.utils.sdk.nat_middleware import NatMiddleware
from nat.utils.sdk.nat_object_store import NatObjectStore
from nat.utils.sdk.nat_optimizer import NatOptimizer
from nat.utils.sdk.nat_retriever import NatRetriever
from nat.utils.sdk.nat_trainer import NatTrainer
from nat.utils.sdk.nat_trainer import NatTrainerAdapter
from nat.utils.sdk.nat_trainer import NatTrajectoryBuilder
from nat.utils.sdk.nat_ttc_strategy import NatTTCStrategy

logger = logging.getLogger(__name__)


class DiscoveredComponents:
    """Container for all discovered NAT components during traversal."""

    def __init__(self):
        self.functions: dict[str, FunctionBaseConfig] = {}
        self.function_groups: dict[str, FunctionGroupBaseConfig] = {}
        self.llms: dict[str, LLMBaseConfig] = {}
        self.embedders: dict[str, EmbedderBaseConfig] = {}
        self.retrievers: dict[str, RetrieverBaseConfig] = {}
        self.memory: dict[str, MemoryBaseConfig] = {}
        self.middleware: dict[str, MiddlewareBaseConfig] = {}
        self.object_stores: dict[str, ObjectStoreBaseConfig] = {}
        self.ttc_strategies: dict[str, TTCStrategyBaseConfig] = {}
        self.auth_providers: dict[str, AuthProviderBaseConfig] = {}
        self.evaluators: dict[str, EvaluatorBaseConfig] = {}
        # Finetuning components
        self.trainers: dict[str, TrainerConfig] = {}
        self.trajectory_builders: dict[str, TrajectoryBuilderConfig] = {}
        self.trainer_adapters: dict[str, TrainerAdapterConfig] = {}
        # Track visited objects to avoid infinite recursion
        self._visited: set[int] = set()
        # Track environment variable references for serialization
        # Format: {component_name: {field_name: env_var_name}}
        self.env_var_refs: dict[str, dict[str, str]] = {}

    def add_llm(self, llm: NatLLM) -> None:
        name, config = llm.compute_name_and_config(LLMBaseConfig)
        self.llms[name] = config
        # Collect env var refs if present
        env_refs = getattr(llm, '_env_var_refs', None)
        if env_refs:
            self.env_var_refs[name] = dict(env_refs)

    def add_embedder(self, embedder: NatEmbedder) -> None:
        name, config = embedder.compute_name_and_config(EmbedderBaseConfig)
        self.embedders[name] = config

    def add_retriever(self, retriever: NatRetriever) -> None:
        name, config = retriever.compute_name_and_config(RetrieverBaseConfig)
        self.retrievers[name] = config

    def add_memory(self, memory: NatMemory) -> None:
        name, config = memory.compute_name_and_config(MemoryBaseConfig)
        self.memory[name] = config

    def add_middleware(self, middleware: NatMiddleware) -> None:
        name, config = middleware.compute_name_and_config(MiddlewareBaseConfig)
        self.middleware[name] = config

    def add_object_store(self, object_store: NatObjectStore) -> None:
        name, config = object_store.compute_name_and_config(ObjectStoreBaseConfig)
        self.object_stores[name] = config

    def add_ttc_strategy(self, ttc_strategy: NatTTCStrategy) -> None:
        name, config = ttc_strategy.compute_name_and_config(TTCStrategyBaseConfig)
        self.ttc_strategies[name] = config

    def add_auth_provider(self, auth_provider: NatAuthProvider) -> None:
        name, config = auth_provider.compute_name_and_config(AuthProviderBaseConfig)
        self.auth_providers[name] = config

    def add_function(self, function: NatFunction | NatBase) -> None:
        name, config = function.compute_name_and_config(FunctionBaseConfig)
        self.functions[name] = config
        # Collect env var refs if present
        env_refs = getattr(function, '_env_var_refs', None)
        if env_refs:
            self.env_var_refs[name] = dict(env_refs)

    def add_function_group(self, function_group: NatFunctionGroup) -> None:
        name, config = function_group.compute_name_and_config(FunctionGroupBaseConfig)
        self.function_groups[name] = config

    def add_evaluator(self, evaluator: NatEvaluator) -> None:
        name, config = evaluator.compute_name_and_config(EvaluatorBaseConfig)
        self.evaluators[name] = config

    def add_trainer(self, trainer: NatTrainer) -> None:
        name, config = trainer.compute_name_and_config(TrainerConfig)
        self.trainers[name] = config

    def add_trajectory_builder(self, trajectory_builder: NatTrajectoryBuilder) -> None:
        name, config = trajectory_builder.compute_name_and_config(TrajectoryBuilderConfig)
        self.trajectory_builders[name] = config

    def add_trainer_adapter(self, trainer_adapter: NatTrainerAdapter) -> None:
        name, config = trainer_adapter.compute_name_and_config(TrainerAdapterConfig)
        self.trainer_adapters[name] = config

    def was_visited(self, obj: object) -> bool:
        """Check if an object was already visited."""
        return id(obj) in self._visited

    def mark_visited(self, obj: object) -> None:
        """Mark an object as visited."""
        self._visited.add(id(obj))


class NatWorkflow(BaseModel):
    """Main workflow class that takes an entrypoint and discovers all NAT components.

    The NatWorkflow recursively traverses the entrypoint to discover all NAT components
    (LLMs, embedders, retrievers, tools, etc.) and builds a complete configuration that
    can be saved to a YAML file or used to run the workflow.

    ## Details
    Name: Workflow
    Icon: N/A

    Example:
        ```python
        from nat.llm.nim_llm import NimLLM
        from nat.agent.react_agent.register import NatReActAgent
        from nat.utils.sdk.nat_workflow import NatWorkflow

        llm = NimLLM(model_name="meta/llama-3.3-70b-instruct", name="my_llm")
        agent = NatReActAgent(llm=llm, tools=[...])
        workflow = NatWorkflow(entrypoint=agent)

        # Run the workflow
        response = await workflow.prompt("Hello!")

        # Save to config file
        workflow.save_to_config_file("config.yaml")
        ```
    """

    entrypoint: NatFunction | NatAgent = Field(description="The entrypoint function or agent for this workflow.")
    configuration: NatGeneralConfiguration | None = Field(
        description="The general configuration of the workflow, including loggers, tracers, and a front end",
        default=None)
    evaluator: NatEvaluation | None = Field(description="The evaluator configuration for this workflow",
                                            default=None,
                                            init=False)
    optimizer: NatOptimizer | None = Field(description="The optimizer configuration for this workflow",
                                           default=None,
                                           init=False)
    finetuning: NatFinetuner | None = Field(description="The finetuning configuration for this workflow",
                                            default=None,
                                            init=False)

    # Private attributes for caching discovered components
    _discovered: DiscoveredComponents | None = PrivateAttr(default=None)

    @cached_property
    def _config(self) -> Config:
        discover_and_register_plugins(PluginTypes.ALL)
        return self._build_config_object()

    async def prompt(self, prompt: str, *, conversation_id: str | None = None):
        """Run the workflow with a prompt.

        Args:
            prompt: The input prompt to process.
            conversation_id: Optional conversation ID for memory persistence.
                If provided, memory operations will use this ID to store/retrieve context.

        Returns:
            The response from the workflow.
        """
        session_kwargs = {}
        if conversation_id is not None:
            session_kwargs["conversation_id"] = conversation_id
        return await run_workflow(config=self._config, prompt=prompt, session_kwargs=session_kwargs or None)

    async def evaluate(
            self,
            *,
            dataset: str | None = None,
            result_json_path: str = "$",
            skip_workflow: bool = False,
            skip_completed_entries: bool = False,
            endpoint: str | None = None,
            endpoint_timeout: int = 300,
            reps: int = 1,
            override: tuple[tuple[str, str], ...] = (),
    ):
        """Run evaluation on the workflow.

        Args:
            dataset: Path to override the dataset in config.
            result_json_path: JSON path to extract results from workflow output.
            skip_workflow: Skip workflow execution, use existing results.
            skip_completed_entries: Skip entries that already have generated answers.
            endpoint: Remote endpoint URL for running the workflow.
            endpoint_timeout: HTTP response timeout in seconds.
            reps: Number of repetitions for the evaluation.
            override: Config overrides as key-value tuples.

        Raises:
            ValueError: If no evaluator has been set.

        Returns:
            The evaluation results.
        """
        if self.evaluator is None:
            raise ValueError("No evaluator has been set. Please set an evaluator before running this.")

        config = EvaluationRunConfig(
            config_file=self._config,
            dataset=dataset,
            result_json_path=result_json_path,
            skip_workflow=skip_workflow,
            skip_completed_entries=skip_completed_entries,
            endpoint=endpoint,
            endpoint_timeout=endpoint_timeout,
            reps=reps,
            override=override,
        )
        return await run_and_evaluate(config)

    def add_evaluator(self, evaluator: NatEvaluation):
        """Add an evaluator to the workflow.

        Args:
            evaluator: The NatEvaluation configuration to add.
        """
        self.evaluator = evaluator
        # Reset the cached config
        if "_config" in self.__dict__:
            del self.__dict__["_config"]
        # Reset discovered components
        self._discovered = None

    async def optimize(
            self,
            *,
            dataset: str | None = None,
            result_json_path: str = "$",
            endpoint: str | None = None,
            endpoint_timeout: int = 300,
            override: tuple[tuple[str, str], ...] = (),
    ):
        """Run optimization on the workflow.

        Optimizes the workflow parameters based on the configured optimizer settings.
        This requires both an evaluator and optimizer to be set.

        Args:
            dataset: Path to override the dataset in config.
            result_json_path: JSON path to extract results from workflow output.
            endpoint: Remote endpoint URL for running the workflow.
            endpoint_timeout: HTTP response timeout in seconds.
            override: Config overrides as key-value tuples.

        Raises:
            ValueError: If no optimizer or evaluator has been set.

        Returns:
            The optimized configuration.
        """
        if self.optimizer is None:
            raise ValueError("No optimizer has been set. Please set an optimizer before running this.")
        if self.evaluator is None:
            raise ValueError("No evaluator has been set. Optimization requires evaluation metrics.")

        config = OptimizerRunConfig(
            config_file=self._config,
            dataset=dataset,
            result_json_path=result_json_path,
            endpoint=endpoint,
            endpoint_timeout=endpoint_timeout,
            override=override,
        )
        return await run_optimizer(config)

    def add_optimizer(self, optimizer: NatOptimizer):
        """Add an optimizer to the workflow.

        Args:
            optimizer: The NatOptimizer configuration to add.
        """
        self.optimizer = optimizer
        # Reset the cached config
        if "_config" in self.__dict__:
            del self.__dict__["_config"]
        # Reset discovered components
        self._discovered = None

    def add_finetuning(self, finetuning: NatFinetuner):
        """Add finetuning configuration to the workflow.

        Args:
            finetuning: The NatFinetuner configuration to add.
        """
        self.finetuning = finetuning
        # Reset the cached config
        if "_config" in self.__dict__:
            del self.__dict__["_config"]
        # Reset discovered components
        self._discovered = None

    async def finetune(
        self,
        *,
        dataset: str | None = None,
        result_json_path: str = "$",
        endpoint: str | None = None,
        endpoint_timeout: int = 300,
        override: tuple[tuple[str, str], ...] = (),
        validation_dataset: str | None = None,
        validation_interval: int = 5,
        validation_config_file: str | None = None,
    ):
        """Run finetuning on the workflow.

        This runs the finetuning harness to collect trajectories and train
        the model using the configured trainer, trajectory builder, and
        trainer adapter.

        Args:
            dataset: Path to override the dataset in config.
            result_json_path: JSON path to extract results from workflow output.
            endpoint: Remote endpoint URL for running the workflow.
            endpoint_timeout: HTTP response timeout in seconds.
            override: Config overrides as key-value tuples.
            validation_dataset: Path to validation dataset for periodic validation.
            validation_interval: Run validation every N epochs.
            validation_config_file: Optional separate config file for validation runs.

        Raises:
            ValueError: If no finetuning configuration or evaluator has been set.

        Returns:
            The finetuning results.
        """
        if self.finetuning is None:
            raise ValueError("No finetuning configuration has been set. "
                             "Please call add_finetuning() before running finetune().")
        if self.evaluator is None:
            raise ValueError("No evaluator has been set. Finetuning requires evaluation metrics.")

        config = FinetuneRunConfig(
            config_file=self._config,
            dataset=dataset,
            result_json_path=result_json_path,
            endpoint=endpoint,
            endpoint_timeout=endpoint_timeout,
            override=override,
            validation_dataset=validation_dataset,
            validation_interval=validation_interval,
            validation_config_file=validation_config_file,
        )
        return await finetuning_main(config)

    def save_to_config_file(self, file_path: str | Path) -> None:
        """Save the workflow configuration to a YAML file.

        Environment variable references (from NatEnvironmentVariable) are preserved
        as ${VAR_NAME} format instead of the resolved values.

        Args:
            file_path: The path to save the configuration file to.
        """
        # Get the discovered components to access env var refs
        discovered = self._discover_components()

        # Save the config
        self._config.save_to_file(file_path)

        # If there are env var refs, post-process the file to replace values with ${VAR}
        if discovered.env_var_refs:
            self._apply_env_var_refs_to_file(file_path, discovered.env_var_refs)

    def _apply_env_var_refs_to_file(self, file_path: str | Path, env_var_refs: dict[str, dict[str, str]]) -> None:
        """Apply environment variable references to a saved config file.

        This replaces serialized secret values with ${VAR_NAME} format.

        Args:
            file_path: Path to the config file.
            env_var_refs: Mapping of component_name -> {field_name: env_var_name}.
        """
        import os
        import re

        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")

        # For each component with env var refs
        for field_refs in env_var_refs.values():
            for field_name, env_var_name in field_refs.items():
                # Get the actual value from the environment
                actual_value = os.environ.get(env_var_name, "")
                if not actual_value:
                    continue

                # Replace the actual value with ${ENV_VAR_NAME}
                # We need to be careful to only replace within the correct context
                # Look for patterns like "field_name: actual_value"
                env_ref = f"${{{env_var_name}}}"

                # Escape special regex characters in the actual value
                escaped_value = re.escape(actual_value)

                # Replace the value - handle both quoted and unquoted values
                # Pattern: field_name: "value" or field_name: value
                patterns = [
                    (rf'({field_name}:\s*)"{escaped_value}"', rf'\g<1>{env_ref}'),
                    (rf"({field_name}:\s*)'{escaped_value}'", rf'\g<1>{env_ref}'),
                    (rf'({field_name}:\s*){escaped_value}(\s*(?:#|$|\n))', rf'\g<1>{env_ref}\g<2>'),
                ]

                for pattern, replacement in patterns:
                    content = re.sub(pattern, replacement, content)

        # Write back the modified content
        file_path.write_text(content, encoding="utf-8")

    def _discover_components(self) -> DiscoveredComponents:
        """Discover all NAT components from the entrypoint recursively.

        Returns:
            A DiscoveredComponents object containing all found components.
        """
        if self._discovered is not None:
            return self._discovered

        self._discovered = DiscoveredComponents()
        self._traverse_nat_object(self.entrypoint, self._discovered)

        # Also traverse evaluators if present
        if self.evaluator is not None and self.evaluator.evaluators:
            for ev in self.evaluator.evaluators:
                self._traverse_nat_object(ev, self._discovered)

        # Discover finetuning components if present
        if self.finetuning is not None:
            finetuning_components = self.finetuning.get_components()
            if "trainer" in finetuning_components:
                self._discovered.add_trainer(finetuning_components["trainer"])
            if "trajectory_builder" in finetuning_components:
                self._discovered.add_trajectory_builder(finetuning_components["trajectory_builder"])
            if "trainer_adapter" in finetuning_components:
                self._discovered.add_trainer_adapter(finetuning_components["trainer_adapter"])
            if "reward_function" in finetuning_components:
                self._discovered.add_evaluator(finetuning_components["reward_function"])

        return self._discovered

    def _traverse_nat_object(self, obj: NatBase, discovered: DiscoveredComponents) -> None:
        """Recursively traverse a NAT object and discover all nested NAT components.

        Args:
            obj: The NAT object to traverse.
            discovered: The container to add discovered components to.
        """
        if obj is None or discovered.was_visited(obj):
            return

        discovered.mark_visited(obj)

        # Process the object based on its type
        self._add_component_to_discovered(obj, discovered)

        # Get all field values and recursively process NAT objects
        for field_name in obj.model_fields:
            try:
                field_value = getattr(obj, field_name, None)
            except AttributeError:
                # Some fields may not be accessible
                continue

            if field_value is None:
                continue

            # Handle lists of NAT objects
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, NatBase):
                        self._traverse_nat_object(item, discovered)
            # Handle single NAT objects
            elif isinstance(field_value, NatBase):
                self._traverse_nat_object(field_value, discovered)

    def _add_component_to_discovered(self, obj: NatBase, discovered: DiscoveredComponents) -> None:
        """Add a NAT component to the appropriate collection in discovered.

        Args:
            obj: The NAT object to add.
            discovered: The container to add the component to.
        """
        # Check type and add to appropriate collection
        # Order matters - check more specific types first
        if isinstance(obj, NatLLM):
            discovered.add_llm(obj)
        elif isinstance(obj, NatEmbedder):
            discovered.add_embedder(obj)
        elif isinstance(obj, NatRetriever):
            discovered.add_retriever(obj)
        elif isinstance(obj, NatMemory):
            discovered.add_memory(obj)
        elif isinstance(obj, NatMiddleware):
            discovered.add_middleware(obj)
        elif isinstance(obj, NatObjectStore):
            discovered.add_object_store(obj)
        elif isinstance(obj, NatTTCStrategy):
            discovered.add_ttc_strategy(obj)
        elif isinstance(obj, NatAuthProvider):
            discovered.add_auth_provider(obj)
        elif isinstance(obj, NatFunctionGroup):
            discovered.add_function_group(obj)
        elif isinstance(obj, NatEvaluator):
            discovered.add_evaluator(obj)
        elif isinstance(obj, NatAgent):
            # NatAgent is a NatFunction, but we want to add it as a function
            discovered.add_function(obj)
        elif isinstance(obj, NatFunction):
            discovered.add_function(obj)

    def _build_config_object(self) -> Config:
        """Build the Config object from discovered components.

        Returns:
            A Config object ready for use or serialization.
        """
        discovered = self._discover_components()

        config_args: dict = {}

        # Build general configuration
        general_config = self._build_general_configuration()
        if general_config is not None:
            config_args["general"] = general_config

        # Build functions (excluding the entrypoint, which goes in workflow)
        # Use compute_name_and_config to get the entrypoint name consistently
        # pylint: disable=no-member
        entrypoint_name, _ = self.entrypoint.compute_name_and_config(FunctionBaseConfig)
        # pylint: enable=no-member
        functions = {k: v for k, v in discovered.functions.items() if k != entrypoint_name}
        if functions:
            config_args["functions"] = functions

        # Build function groups
        if discovered.function_groups:
            config_args["function_groups"] = discovered.function_groups

        # Build LLMs
        if discovered.llms:
            config_args["llms"] = discovered.llms

        # Build embedders
        if discovered.embedders:
            config_args["embedders"] = discovered.embedders

        # Build memory
        if discovered.memory:
            config_args["memory"] = discovered.memory

        # Build middleware
        if discovered.middleware:
            config_args["middleware"] = discovered.middleware

        # Build object stores
        if discovered.object_stores:
            config_args["object_stores"] = discovered.object_stores

        # Build retrievers
        if discovered.retrievers:
            config_args["retrievers"] = discovered.retrievers

        # Build TTC strategies
        if discovered.ttc_strategies:
            config_args["ttc_strategies"] = discovered.ttc_strategies

        # Build authentication providers
        if discovered.auth_providers:
            config_args["authentication"] = discovered.auth_providers

        # Build workflow (the entrypoint)
        config_args["workflow"] = self._build_workflow()

        # Build eval configuration
        eval_config = self._build_evaluator(discovered)
        if eval_config is not None:
            config_args["eval"] = eval_config

        # Build optimizer configuration
        optimizer_config = self._build_optimizer()
        if optimizer_config is not None:
            config_args["optimizer"] = optimizer_config

        # Build trainers
        if discovered.trainers:
            config_args["trainers"] = discovered.trainers

        # Build trajectory builders
        if discovered.trajectory_builders:
            config_args["trajectory_builders"] = discovered.trajectory_builders

        # Build trainer adapters
        if discovered.trainer_adapters:
            config_args["trainer_adapters"] = discovered.trainer_adapters

        # Build finetuning configuration
        finetuning_config = self._build_finetuning()
        if finetuning_config is not None:
            config_args["finetuning"] = finetuning_config

        try:
            config = Config(**config_args)
        except ValidationError as e:
            logger.error(e.errors())
            raise

        return config

    def _build_general_configuration(self) -> GeneralConfig | None:
        """Build the general configuration from NatGeneralConfiguration.

        Returns:
            A GeneralConfig object or None if no configuration is set.
        """
        config = self.configuration
        if config is None:
            return None

        # pylint: disable=no-member
        general_configuration: dict = {}
        telemetry_config: dict = {}

        loggers = config.loggers
        if loggers and len(loggers) > 0:
            telemetry_config["logging"] = dict[str, LoggingBaseConfig](lgr.compute_name_and_config(LoggingBaseConfig)
                                                                       for lgr in loggers)

        telemetry_exporters = config.telemetry_exporters
        if telemetry_exporters and len(telemetry_exporters) > 0:
            telemetry_config["tracing"] = dict[str, TelemetryExporterBaseConfig](
                tracer.compute_name_and_config(TelemetryExporterBaseConfig) for tracer in telemetry_exporters)

        if telemetry_config:
            general_configuration["telemetry"] = telemetry_config

        front_end = config.front_end_configuration
        if front_end is not None:
            general_configuration["front_end"] = front_end.compute_name_and_config(FrontEndBaseConfig)[1]
        # pylint: enable=no-member

        if not general_configuration:
            return None

        return GeneralConfig(**general_configuration)

    def _build_workflow(self) -> FunctionBaseConfig:
        """Build the workflow configuration from the entrypoint.

        Returns:
            The FunctionBaseConfig for the workflow entrypoint.
        """
        # pylint: disable=no-member
        _, config = self.entrypoint.compute_name_and_config(FunctionBaseConfig)
        # pylint: enable=no-member
        return config

    def _build_evaluator(self, discovered: DiscoveredComponents) -> EvalConfig | None:
        """Build the eval configuration.

        Args:
            discovered: The discovered components container.

        Returns:
            An EvalConfig object or None if no evaluator is set.
        """
        if self.evaluator is None:
            return None

        eval_config: dict = {}

        # Convert NatEvaluation to EvalGeneralConfig for serialization
        eval_config["general"] = self.evaluator.to_eval_general_config()

        # Use evaluators from discovered components
        if discovered.evaluators:
            eval_config["evaluators"] = discovered.evaluators

        return EvalConfig(**eval_config)

    def _build_optimizer(self):
        """Build the optimizer configuration.

        Returns:
            An OptimizerConfig object or None if no optimizer is set.
        """
        if self.optimizer is None:
            return None

        return self.optimizer.to_optimizer_config()

    def _build_finetuning(self):
        """Build the finetuning configuration.

        Returns:
            A FinetuneConfig object or None if no finetuning is set.
        """
        if self.finetuning is None:
            return None

        # NatFinetuner inherits from FinetuneConfig, so we can use it directly
        # But to_finetune_config() gives us a clean base config without SDK fields
        return self.finetuning.to_finetune_config()
