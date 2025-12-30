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
"""
Config parser for importing YAML configurations into the workflow builder.

Parses a validated NAT Config object and converts it into a workflow state
that can be rendered on the UI canvas with components and connections.

The approach:
1. Validate the config dict to get a proper Config Pydantic object
2. Use type introspection to determine component categories from base classes
3. Use the type registry to get full type information
4. Detect connections by scanning for ComponentRef fields in model annotations
"""

import logging
import types
import typing
import uuid
from collections.abc import Mapping
from typing import Any
from typing import get_args
from typing import get_origin

from pydantic import BaseModel

from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.component import ComponentGroup
from nat.data_models.component_ref import ComponentRef
from nat.data_models.config import Config
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.evaluator import EvaluatorBaseConfig
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
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils.data_models.schema_validator import validate_schema
from nat.workflow_builder_api.models import ConnectionPort
from nat.workflow_builder_api.models import ImportedComponent
from nat.workflow_builder_api.models import ImportedConnection
from nat.workflow_builder_api.models import ImportedWorkflowState
from nat.workflow_builder_api.models import Position
from nat.workflow_builder_api.models import RefType
from nat.workflow_builder_api.utils.connections import extract_connection_ports
from nat.workflow_builder_api.utils.schema import extract_fields
from nat.workflow_builder_api.utils.schema import extract_json_schema
from nat.workflow_builder_api.utils.schema import get_display_name_from_docstring
from nat.workflow_builder_api.utils.schema import get_icon_url_from_docstring
from nat.workflow_builder_api.utils.schema import get_sdk_excluded_fields

logger = logging.getLogger(__name__)

# =============================================================================
# BASE CLASS TO COMPONENT TYPE MAPPING
# =============================================================================

# Map base config classes to UI component types
# Order matters for subclass checks (more specific first)
BASE_CLASS_TO_COMPONENT_TYPE: list[tuple[type, str]] = [
    # Agents before functions (AgentBaseConfig extends FunctionBaseConfig)
    (AgentBaseConfig, "agent"),
    (FunctionBaseConfig, "function"),
    (FunctionGroupBaseConfig, "function_group"),
    (LLMBaseConfig, "llm"),
    (EmbedderBaseConfig, "embedder"),
    (MemoryBaseConfig, "memory"),
    (ObjectStoreBaseConfig, "object_store"),
    (RetrieverBaseConfig, "retriever"),
    (AuthProviderBaseConfig, "authentication"),
    (MiddlewareBaseConfig, "middleware"),
    (TTCStrategyBaseConfig, "ttc_strategy"),
    (FrontEndBaseConfig, "front_end"),
    (LoggingBaseConfig, "logger"),
    (TelemetryExporterBaseConfig, "telemetry_exporter"),
    (EvaluatorBaseConfig, "evaluator"),
    (TrainerConfig, "trainer"),
    (TrainerAdapterConfig, "trainer_adapter"),
    (TrajectoryBuilderConfig, "trajectory_builder"),
]

# Map ComponentGroup to UI component type for connections
COMPONENT_GROUP_TO_TYPE: dict[ComponentGroup, str] = {
    ComponentGroup.LLMS: "llm",
    ComponentGroup.EMBEDDERS: "embedder",
    ComponentGroup.FUNCTIONS: "function",
    ComponentGroup.FUNCTION_GROUPS: "function_group",
    ComponentGroup.MEMORY: "memory",
    ComponentGroup.OBJECT_STORES: "object_store",
    ComponentGroup.RETRIEVERS: "retriever",
    ComponentGroup.AUTHENTICATION: "authentication",
    ComponentGroup.MIDDLEWARE: "middleware",
    ComponentGroup.TTC_STRATEGIES: "ttc_strategy",
    ComponentGroup.TRAINERS: "trainer",
    ComponentGroup.TRAINER_ADAPTERS: "trainer_adapter",
    ComponentGroup.TRAJECTORY_BUILDERS: "trajectory_builder",
}

# Map ComponentGroup to RefType for connections
COMPONENT_GROUP_TO_REF_TYPE: dict[ComponentGroup, RefType] = {
    ComponentGroup.LLMS: RefType.LLM,
    ComponentGroup.EMBEDDERS: RefType.EMBEDDER,
    ComponentGroup.FUNCTIONS: RefType.FUNCTION,
    ComponentGroup.FUNCTION_GROUPS: RefType.FUNCTION_GROUP,
    ComponentGroup.MEMORY: RefType.MEMORY,
    ComponentGroup.OBJECT_STORES: RefType.OBJECT_STORE,
    ComponentGroup.RETRIEVERS: RefType.RETRIEVER,
    ComponentGroup.AUTHENTICATION: RefType.AUTHENTICATION,
    ComponentGroup.MIDDLEWARE: RefType.MIDDLEWARE,
    ComponentGroup.TTC_STRATEGIES: RefType.FUNCTION,  # TTC strategies provide function-like behavior
    ComponentGroup.TRAINERS: RefType.TRAINER,
    ComponentGroup.TRAINER_ADAPTERS: RefType.TRAINER_ADAPTER,
    ComponentGroup.TRAJECTORY_BUILDERS: RefType.TRAJECTORY_BUILDER,
}

# Layout constants
COMPONENT_WIDTH = 280
COMPONENT_MIN_HEIGHT = 180  # Minimum height - generous to avoid overlaps
HORIZONTAL_GAP = 150
VERTICAL_GAP = 60  # Gap between components
CANVAS_MARGIN = 120

# Height calculation constants (matching NATNode CSS with generous padding)
HEADER_HEIGHT = 80  # Icon, name, type badge, buttons + padding
INPUT_SECTION_HEADER = 30  # "Inputs" label + padding
INPUT_PORT_ROW = 32  # Each port row
INPUT_CONNECTION_ROW = 22  # Each connection under a port
OUTPUT_SECTION = 50  # Output section + padding
HEIGHT_BUFFER = 20  # Extra buffer for safety


def calculate_component_height(
    component: "ImportedComponent",
    connections: list["ImportedConnection"],
) -> int:
    """
    Calculate the predicted height of a component based on its ports and connections.

    Args:
        component: The component to calculate height for
        connections: All connections in the workflow

    Returns:
        Predicted height in pixels
    """
    height = HEADER_HEIGHT

    # Add input ports section if component has ports
    input_ports = component.input_ports
    if input_ports:
        height += INPUT_SECTION_HEADER

        for port in input_ports:
            height += INPUT_PORT_ROW

            # Count connections to this port
            port_connections = [
                c for c in connections if c.target_id == component.id and c.target_field == port.field_name
            ]
            height += len(port_connections) * INPUT_CONNECTION_ROW

    # Add output section (most components have an output)
    height += OUTPUT_SECTION

    # Add buffer for safety
    height += HEIGHT_BUFFER

    return max(height, COMPONENT_MIN_HEIGHT)


# =============================================================================
# TYPE INTROSPECTION UTILITIES
# =============================================================================


def get_component_type_from_class(config_class: type) -> str:
    """
    Determine the UI component type from a config class using base class checks.

    Args:
        config_class: The config class (e.g., NIMModelConfig, ReActAgentConfig)

    Returns:
        UI component type string (e.g., "llm", "agent", "function")
    """
    for base_class, component_type in BASE_CLASS_TO_COMPONENT_TYPE:
        if issubclass(config_class, base_class):
            return component_type

    return "function"  # Default fallback


def get_full_type_from_registry(config_instance: BaseModel) -> str:
    """
    Get the full type string from the type registry for a config instance.

    Args:
        config_instance: A validated config instance (e.g., NIMModelConfig instance)

    Returns:
        Full type string (e.g., "nim/NIMModelConfig") or empty string if not found
    """
    config_class = type(config_instance)
    registry = GlobalTypeRegistry.get()

    # Try to look up in each registry based on base class
    try:
        if issubclass(config_class, LLMBaseConfig):
            info = registry.get_llm_provider(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, EmbedderBaseConfig):
            info = registry.get_embedder_provider(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, FunctionBaseConfig):
            info = registry.get_function(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, FunctionGroupBaseConfig):
            info = registry.get_function_group(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, MemoryBaseConfig):
            info = registry.get_memory(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, ObjectStoreBaseConfig):
            info = registry.get_object_store(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, RetrieverBaseConfig):
            info = registry.get_retriever_provider(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, AuthProviderBaseConfig):
            info = registry.get_auth_provider(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, MiddlewareBaseConfig):
            info = registry.get_middleware(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, FrontEndBaseConfig):
            info = registry.get_front_end(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, LoggingBaseConfig):
            info = registry.get_logging_method(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, TelemetryExporterBaseConfig):
            info = registry.get_telemetry_exporter(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, EvaluatorBaseConfig):
            info = registry.get_evaluator(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, TrainerConfig):
            info = registry.get_trainer(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, TrainerAdapterConfig):
            info = registry.get_trainer_adapter(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, TrajectoryBuilderConfig):
            info = registry.get_trajectory_builder(config_class)
            return info.full_type
    except KeyError:
        pass

    try:
        if issubclass(config_class, TTCStrategyBaseConfig):
            info = registry.get_ttc_strategy(config_class)
            return info.full_type
    except KeyError:
        pass

    # Fallback: use class name
    return config_class.__name__


def get_component_ref_from_annotation(annotation: Any) -> type[ComponentRef] | None:
    """
    Check if an annotation is a ComponentRef type or contains one.

    Args:
        annotation: A type annotation

    Returns:
        The ComponentRef subclass if found, None otherwise
    """
    if annotation is None:
        return None

    # Direct ComponentRef subclass
    if isinstance(annotation, type) and issubclass(annotation, ComponentRef):
        return annotation

    # Check for list/sequence types
    origin = get_origin(annotation)
    if origin in (list, typing.Sequence):
        args = get_args(annotation)
        if args:
            # Check for union inside list (e.g., list[FunctionRef | FunctionGroupRef])
            inner = args[0]
            inner_origin = get_origin(inner)
            is_union = inner_origin is typing.Union or isinstance(inner, types.UnionType)
            if is_union:
                inner_args = get_args(inner)
                for arg in inner_args:
                    if isinstance(arg, type) and issubclass(arg, ComponentRef):
                        return arg  # Return first one found
            elif isinstance(inner, type) and issubclass(inner, ComponentRef):
                return inner

    # Check for Optional/Union types
    is_union = origin is typing.Union or isinstance(annotation, types.UnionType)
    if is_union:
        args = get_args(annotation)
        for arg in args:
            if arg is type(None):
                continue
            ref_type = get_component_ref_from_annotation(arg)
            if ref_type:
                return ref_type

    return None


def is_list_annotation(annotation: Any) -> bool:
    """Check if an annotation is a list type."""
    origin = get_origin(annotation)
    return origin in (list, typing.Sequence)


# =============================================================================
# COMPONENT EXTRACTION
# =============================================================================


def extract_component(
    component_id: str,
    config_instance: BaseModel,
    component_type_override: str | None = None,
) -> ImportedComponent:
    """
    Extract an ImportedComponent from a validated config instance.

    Args:
        component_id: The component's name/key from the config
        config_instance: The validated Pydantic config instance
        component_type_override: Override the detected component type

    Returns:
        ImportedComponent ready for UI rendering
    """
    config_class = type(config_instance)
    component_type = component_type_override or get_component_type_from_class(config_class)
    full_type = get_full_type_from_registry(config_instance)

    # Get config values as dict - only include explicitly set fields
    # Use exclude_unset=True to only include fields that were explicitly provided
    # Don't use by_alias - some YAML files use Python names (model_name), others use aliases (model)
    # The comparison logic normalizes these differences
    config_dict = config_instance.model_dump(exclude_unset=True)

    # Preserve the original type value for export
    # The exporter needs this to recreate the exact same _type format
    original_type = config_dict.get("type") or config_dict.get("_type")
    if original_type:
        config_dict["_original_type"] = original_type

    # Remove discriminator fields - the exporter adds _type based on _original_type
    config_dict.pop("_type", None)
    config_dict.pop("type", None)

    # Extract input ports (connection fields)
    input_ports = extract_connection_ports(config_class)

    # Extract fields for the configuration form
    try:
        json_schema = extract_json_schema(config_class)
        sdk_excluded = get_sdk_excluded_fields(config_class)
        fields = extract_fields(json_schema, sdk_excluded, config_class)
    except Exception as e:
        logger.warning("Failed to extract fields for %s: %s", config_class.__name__, e)
        fields = []

    # Get icon URL and display name from docstring
    icon_url = get_icon_url_from_docstring(config_class)
    display_name = get_display_name_from_docstring(config_class)

    # Use display_name if available, otherwise use the component_id
    component_name = display_name if display_name else component_id

    return ImportedComponent(
        id=component_id,
        component_type=component_type,
        name=component_name,
        position=Position(x=0, y=0),  # Will be set during layout
        full_type=full_type,
        config=config_dict,
        input_ports=input_ports,
        fields=fields,
        icon_url=icon_url,
        display_name=display_name,
    )


def parse_dict_section(
    section_data: Mapping[str, BaseModel],
    component_type_override: str | None = None,
) -> list[ImportedComponent]:
    """
    Parse a config section that contains a dict of named components.

    Args:
        section_data: Mapping of names to config instances
        component_type_override: Override detected component type

    Returns:
        List of ImportedComponents
    """
    components = []
    for name, config_instance in section_data.items():
        if isinstance(config_instance, BaseModel):
            component = extract_component(name, config_instance, component_type_override)
            components.append(component)
    return components


def parse_components_from_config(config: Config) -> list[ImportedComponent]:
    """
    Parse all components from a validated Config object.

    Args:
        config: Validated Config Pydantic object

    Returns:
        List of all ImportedComponents
    """
    components: list[ImportedComponent] = []

    # NatWorkflow container component (always present)
    # This links to the workflow function via entrypoint connection
    from nat.utils.sdk.nat_workflow import NatWorkflow
    nat_workflow_component = ImportedComponent(
        id="nat_workflow",
        component_type="nat_workflow",
        name="Workflow",
        position=Position(x=0, y=0),
        full_type="nat/NatWorkflow",
        config={"_selected_type": "nat/NatWorkflow"},  # Mark as configured
        input_ports=[
            ConnectionPort(
                field_name="entrypoint",
                ref_type=RefType.FUNCTION,
                accepts_ref_types=[RefType.FUNCTION],
                required=True,
                is_list=False,
                description="The entrypoint function or agent for this workflow",
                title="Entrypoint",
            ),
            ConnectionPort(
                field_name="configuration",
                ref_type=RefType.GENERAL_CONFIG,
                accepts_ref_types=[RefType.GENERAL_CONFIG],
                required=False,
                is_list=False,
                description="General configuration for the workflow",
                title="Configuration",
            ),
            ConnectionPort(
                field_name="evaluation",
                ref_type=RefType.EVALUATION_CONFIG,
                accepts_ref_types=[RefType.EVALUATION_CONFIG],
                required=False,
                is_list=False,
                description="Evaluation configuration",
                title="Evaluation",
            ),
            ConnectionPort(
                field_name="optimizer",
                ref_type=RefType.OPTIMIZER_CONFIG,
                accepts_ref_types=[RefType.OPTIMIZER_CONFIG],
                required=False,
                is_list=False,
                description="Optimizer configuration",
                title="Optimizer",
            ),
            ConnectionPort(
                field_name="finetuning",
                ref_type=RefType.FINETUNER_CONFIG,
                accepts_ref_types=[RefType.FINETUNER_CONFIG],
                required=False,
                is_list=False,
                description="Finetuning configuration",
                title="Finetuning",
            ),
        ],
        fields=[],
        icon_url=None,
        display_name=get_display_name_from_docstring(NatWorkflow) or "Workflow",
    )
    components.append(nat_workflow_component)

    # Workflow function (the entrypoint function that NatWorkflow connects to)
    if config.workflow and not isinstance(config.workflow, type):
        workflow_component = extract_component("workflow", config.workflow)
        components.append(workflow_component)

    # Core component sections
    components.extend(parse_dict_section(config.functions))
    components.extend(parse_dict_section(config.function_groups))
    components.extend(parse_dict_section(config.llms))
    components.extend(parse_dict_section(config.embedders))
    components.extend(parse_dict_section(config.memory))
    components.extend(parse_dict_section(config.object_stores))
    components.extend(parse_dict_section(config.retrievers))
    components.extend(parse_dict_section(config.authentication))
    components.extend(parse_dict_section(config.middleware))
    components.extend(parse_dict_section(config.ttc_strategies))

    # Finetuning components
    components.extend(parse_dict_section(config.trainers))
    components.extend(parse_dict_section(config.trainer_adapters))
    components.extend(parse_dict_section(config.trajectory_builders))

    # Track IDs of general section components we actually create
    general_section_component_ids: list[str] = []

    # General section: telemetry (loggers and telemetry exporters)
    # These are explicitly configured, so we can safely add them
    if hasattr(config.general, "telemetry"):
        telemetry = config.general.telemetry
        if hasattr(telemetry, "logging"):
            for name, logger_config in telemetry.logging.items():
                if isinstance(logger_config, BaseModel):
                    component = extract_component(f"logger_{name}", logger_config, "logger")
                    # Don't override display_name - keep the one from docstring
                    components.append(component)
                    general_section_component_ids.append(component.id)

        if hasattr(telemetry, "tracing"):
            for name, tracer_config in telemetry.tracing.items():
                if isinstance(tracer_config, BaseModel):
                    component = extract_component(f"telemetry_{name}", tracer_config, "telemetry_exporter")
                    # Don't override display_name - keep the one from docstring
                    components.append(component)
                    general_section_component_ids.append(component.id)

    # Note: We DON'T add front_end here because:
    # 1. front_end always has a default value (FastApiFrontEndConfig)
    # 2. We can't distinguish between "explicitly configured" and "default"
    # 3. Most configs don't include front_end explicitly
    # If you need front_end support in round-trip, store its values during import

    # Eval section: evaluators
    if hasattr(config.eval, "evaluators"):
        for name, eval_config in config.eval.evaluators.items():
            if isinstance(eval_config, BaseModel):
                component = extract_component(f"evaluator_{name}", eval_config, "evaluator")
                # Don't override display_name - keep the one from docstring (e.g., "RAGAS Evaluator")
                components.append(component)

    # Check if general config has any content (loggers or telemetry exporters we actually created)
    # We don't include front_end because it always has a default value
    has_general_content = len(general_section_component_ids) > 0

    # General Config container component (only if has content)
    if has_general_content:
        from nat.utils.sdk.nat_general_configuraton import NatGeneralConfiguration

        # Build config dict, including telemetry settings (like enabled)
        general_config_dict: dict[str, Any] = {"_selected_type": "nat/NatGeneralConfiguration"}

        # Store telemetry settings (enabled, etc.) for export
        if hasattr(config.general, "telemetry"):
            telemetry = config.general.telemetry
            # Dump telemetry settings excluding the logging/tracing dicts (those become components)
            telemetry_dict = telemetry.model_dump(by_alias=True, exclude={"logging", "tracing"}, exclude_unset=True)
            if telemetry_dict:
                general_config_dict["_telemetry_settings"] = telemetry_dict

        general_config_component = ImportedComponent(
            id="general_config",
            component_type="general_config",
            name="General Configuration",
            position=Position(x=0, y=0),
            full_type="nat/NatGeneralConfiguration",
            config=general_config_dict,
            input_ports=[
                ConnectionPort(
                    field_name="loggers",
                    ref_type=RefType.LOGGER,
                    accepts_ref_types=[RefType.LOGGER],
                    required=False,
                    is_list=True,
                    description="Loggers for the workflow",
                    title="Loggers",
                ),
                ConnectionPort(
                    field_name="telemetry_exporters",
                    ref_type=RefType.TELEMETRY_EXPORTER,
                    accepts_ref_types=[RefType.TELEMETRY_EXPORTER],
                    required=False,
                    is_list=True,
                    description="Telemetry exporters for the workflow",
                    title="Telemetry Exporters",
                ),
                ConnectionPort(
                    field_name="front_end",
                    ref_type=RefType.FRONT_END,
                    accepts_ref_types=[RefType.FRONT_END],
                    required=False,
                    is_list=False,
                    description="Front-end configuration",
                    title="Front End",
                ),
            ],
            fields=[],
            icon_url=None,
            display_name=get_display_name_from_docstring(NatGeneralConfiguration) or "General Configuration",
        )
        components.append(general_config_component)

    # Check if evaluation config has any evaluators or explicitly set general settings
    # Note: Pydantic creates default EvalGeneralConfig even when no eval section exists,
    # so we must check if any fields were actually set by the user
    has_evaluators = hasattr(config.eval, "evaluators") and len(config.eval.evaluators) > 0
    has_eval_general = (hasattr(config.eval, "general") and config.eval.general is not None
                        and len(config.eval.general.model_dump(exclude_unset=True)) > 0)

    # Evaluation Config container component (if has evaluators or general settings)
    if has_evaluators or has_eval_general:
        from nat.utils.sdk.nat_evaluation import NatEvaluation

        # Build config dict for evaluation_config, including general settings
        eval_config_dict: dict[str, Any] = {"_selected_type": "nat/NatEvaluation"}

        # Store eval.general settings in the component config for export
        if has_eval_general:
            # Dump the general settings preserving only explicitly set fields
            # Use by_alias=True to get _type instead of type for discriminators
            general = config.eval.general
            general_dict = general.model_dump(by_alias=True, exclude_unset=True)
            if general_dict:
                eval_config_dict["_eval_general"] = general_dict

        evaluation_config_component = ImportedComponent(
            id="evaluation_config",
            component_type="evaluation_config",
            name="Evaluation",
            position=Position(x=0, y=0),
            full_type="nat/NatEvaluation",
            config=eval_config_dict,
            input_ports=[
                ConnectionPort(
                    field_name="evaluators",
                    ref_type=RefType.EVALUATOR,
                    accepts_ref_types=[RefType.EVALUATOR],
                    required=False,
                    is_list=True,
                    description="Evaluators for testing workflow quality",
                    title="Evaluators",
                ),
            ],
            fields=[],
            icon_url=None,
            display_name=get_display_name_from_docstring(NatEvaluation) or "Evaluation Configuration",
        )
        components.append(evaluation_config_component)

    # Check if optimizer config has meaningful settings (not just defaults)
    has_optimizer_content = (config.optimizer is not None and
                             (config.optimizer.eval_metrics is not None or config.optimizer.output_path is not None))

    # Optimizer Config container component (only if has content)
    if has_optimizer_content:
        from nat.utils.sdk.nat_optimizer import NatOptimizer

        # Store the optimizer settings for export
        optimizer_settings = config.optimizer.model_dump(exclude_unset=True) if config.optimizer else {}

        optimizer_config_component = ImportedComponent(
            id="optimizer_config",
            component_type="optimizer_config",
            name="Optimizer",
            position=Position(x=0, y=0),
            full_type="nat/NatOptimizer",
            config={
                "_selected_type": "nat/NatOptimizer",
                "_optimizer_settings": optimizer_settings,  # Store for export
            },
            input_ports=[],
            fields=[],
            icon_url=None,
            display_name=get_display_name_from_docstring(NatOptimizer) or "Optimizer Configuration",
        )
        components.append(optimizer_config_component)

    # Check if finetuning config has any content
    has_finetuning_content = (config.finetuning is not None and config.finetuning.enabled) or len(
        config.trainers) > 0 or len(config.trainer_adapters) > 0 or len(config.trajectory_builders) > 0

    # Finetuner Config container component (only if has content)
    if has_finetuning_content:
        from nat.utils.sdk.nat_finetuner import NatFinetuner

        # Store the finetuning settings for export
        finetuning_settings = config.finetuning.model_dump(exclude_unset=True) if config.finetuning else {}

        finetuner_config_component = ImportedComponent(
            id="finetuner_config",
            component_type="finetuner_config",
            name="Finetuner",
            position=Position(x=0, y=0),
            full_type="nat/NatFinetuner",
            config={
                "_selected_type": "nat/NatFinetuner",
                "_finetuning_settings": finetuning_settings,  # Store for export
            },
            input_ports=[
                ConnectionPort(
                    field_name="trainer",
                    ref_type=RefType.TRAINER,
                    accepts_ref_types=[RefType.TRAINER],
                    required=False,
                    is_list=False,
                    description="Trainer for finetuning",
                    title="Trainer",
                ),
                ConnectionPort(
                    field_name="trajectory_builder",
                    ref_type=RefType.TRAJECTORY_BUILDER,
                    accepts_ref_types=[RefType.TRAJECTORY_BUILDER],
                    required=False,
                    is_list=False,
                    description="Trajectory builder for finetuning",
                    title="Trajectory Builder",
                ),
                ConnectionPort(
                    field_name="trainer_adapter",
                    ref_type=RefType.TRAINER_ADAPTER,
                    accepts_ref_types=[RefType.TRAINER_ADAPTER],
                    required=False,
                    is_list=False,
                    description="Trainer adapter for finetuning",
                    title="Trainer Adapter",
                ),
            ],
            fields=[],
            icon_url=None,
            display_name=get_display_name_from_docstring(NatFinetuner) or "Finetuning Configuration",
        )
        components.append(finetuner_config_component)

    return components


# =============================================================================
# CONNECTION EXTRACTION
# =============================================================================


def _extract_refs_from_model(
    model_instance: BaseModel,
    component_lookup: dict[str, ImportedComponent],
    component_group_lookup: dict[ComponentGroup, dict[str, ImportedComponent]],
    field_path: str = "",
) -> list[tuple[str, str, RefType]]:
    """
    Recursively extract ComponentRef fields from a model and its nested BaseModels.

    Args:
        model_instance: The model to scan
        component_lookup: Map of component_id -> ImportedComponent
        component_group_lookup: Map of ComponentGroup -> dict of name -> ImportedComponent
        field_path: Dot-separated path to the current field (for nested models)

    Returns:
        List of tuples: (source_id, field_name_for_connection, ref_type)
    """
    refs: list[tuple[str, str, RefType]] = []
    model_class = type(model_instance)

    for field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation
        value = getattr(model_instance, field_name, None)
        if value is None:
            continue

        current_path = f"{field_path}.{field_name}" if field_path else field_name

        # Check if this field is a ComponentRef
        component_ref_class = get_component_ref_from_annotation(annotation)

        if component_ref_class:
            # This is a ComponentRef field - extract the reference
            try:
                if isinstance(value, str):
                    ref_instance = component_ref_class(value)
                    target_group = ref_instance.component_group
                    ref_names = [value]
                elif isinstance(value, list | tuple):
                    ref_names = [str(v) for v in value if v]
                    if ref_names:
                        ref_instance = component_ref_class(ref_names[0])
                        target_group = ref_instance.component_group
                    else:
                        continue
                else:
                    continue

                ref_type = COMPONENT_GROUP_TO_REF_TYPE.get(target_group, RefType.FUNCTION)

                # Find target components
                for ref_name in ref_names:
                    group_components = component_group_lookup.get(target_group, {})
                    source_id = None

                    if ref_name in group_components:
                        source_id = group_components[ref_name].id
                    elif ref_name in component_lookup:
                        source_id = ref_name

                    if source_id:
                        # Use the nested path as the field name for connection display
                        refs.append((source_id, current_path, ref_type))
                    else:
                        logger.debug("Reference '%s' at %s not found", ref_name, current_path)

            except Exception as e:
                logger.debug("Error extracting ref from %s: %s", current_path, e)
                continue

        # Check if this is a nested BaseModel to scan recursively
        elif isinstance(value, BaseModel):
            nested_refs = _extract_refs_from_model(value, component_lookup, component_group_lookup, current_path)
            refs.extend(nested_refs)

    return refs


def extract_connections_from_component(
    component: ImportedComponent,
    config_instance: BaseModel,
    component_lookup: dict[str, ImportedComponent],
    component_group_lookup: dict[ComponentGroup, dict[str, ImportedComponent]],
) -> list[ImportedConnection]:
    """
    Extract connections from a component by scanning its ComponentRef fields.

    This function recursively scans nested BaseModel fields to find ComponentRefs
    in nested structures (e.g., server.auth_provider in MCP configs).

    Args:
        component: The ImportedComponent
        config_instance: The validated config instance
        component_lookup: Map of component_id -> ImportedComponent
        component_group_lookup: Map of ComponentGroup -> dict of name -> ImportedComponent

    Returns:
        List of connections from this component
    """
    connections: list[ImportedConnection] = []

    # Recursively extract all refs from the config instance
    refs = _extract_refs_from_model(config_instance, component_lookup, component_group_lookup)

    # Create connections for each ref found
    for source_id, field_path, ref_type in refs:
        connections.append(
            ImportedConnection(
                id=f"conn_{uuid.uuid4().hex[:8]}",
                source_id=source_id,
                target_id=component.id,
                target_field=field_path,
                ref_type=ref_type,
            ))

    return connections


def build_component_group_lookup(
    config: Config,
    components: list[ImportedComponent],
) -> dict[ComponentGroup, dict[str, ImportedComponent]]:
    """
    Build a lookup table mapping ComponentGroup to components.

    Args:
        config: The validated Config object
        components: List of parsed components

    Returns:
        Dict mapping ComponentGroup to dict of name -> component
    """
    # Build component_id -> component lookup
    id_lookup = {c.id: c for c in components}

    # Map component groups to their config sections
    group_to_section: dict[ComponentGroup, dict] = {
        ComponentGroup.LLMS: config.llms,
        ComponentGroup.EMBEDDERS: config.embedders,
        ComponentGroup.FUNCTIONS: config.functions,
        ComponentGroup.FUNCTION_GROUPS: config.function_groups,
        ComponentGroup.MEMORY: config.memory,
        ComponentGroup.OBJECT_STORES: config.object_stores,
        ComponentGroup.RETRIEVERS: config.retrievers,
        ComponentGroup.AUTHENTICATION: config.authentication,
        ComponentGroup.MIDDLEWARE: config.middleware,
        ComponentGroup.TTC_STRATEGIES: config.ttc_strategies,
        ComponentGroup.TRAINERS: config.trainers,
        ComponentGroup.TRAINER_ADAPTERS: config.trainer_adapters,
        ComponentGroup.TRAJECTORY_BUILDERS: config.trajectory_builders,
    }

    result: dict[ComponentGroup, dict[str, ImportedComponent]] = {}

    for group, section in group_to_section.items():
        result[group] = {}
        for name in section.keys():
            if name in id_lookup:
                result[group][name] = id_lookup[name]

    return result


def parse_connections_from_config(
    config: Config,
    components: list[ImportedComponent],
) -> list[ImportedConnection]:
    """
    Parse all connections from a validated Config object.

    Args:
        config: Validated Config Pydantic object
        components: List of parsed components

    Returns:
        List of all connections
    """
    connections: list[ImportedConnection] = []

    # Build lookup tables
    component_lookup = {c.id: c for c in components}
    group_lookup = build_component_group_lookup(config, components)

    # Map component IDs to their config instances
    config_instances: dict[str, BaseModel] = {}

    # Collect all config instances
    if config.workflow:
        config_instances["workflow"] = config.workflow

    for name, cfg in config.functions.items():
        config_instances[name] = cfg
    for name, cfg in config.function_groups.items():
        config_instances[name] = cfg
    for name, cfg in config.llms.items():
        config_instances[name] = cfg
    for name, cfg in config.embedders.items():
        config_instances[name] = cfg
    for name, cfg in config.memory.items():
        config_instances[name] = cfg
    for name, cfg in config.object_stores.items():
        config_instances[name] = cfg
    for name, cfg in config.retrievers.items():
        config_instances[name] = cfg
    for name, cfg in config.authentication.items():
        config_instances[name] = cfg
    for name, cfg in config.middleware.items():
        config_instances[name] = cfg
    for name, cfg in config.ttc_strategies.items():
        config_instances[name] = cfg
    for name, cfg in config.trainers.items():
        config_instances[name] = cfg
    for name, cfg in config.trainer_adapters.items():
        config_instances[name] = cfg
    for name, cfg in config.trajectory_builders.items():
        config_instances[name] = cfg

    # Add evaluators (they use prefixed IDs)
    if hasattr(config.eval, "evaluators"):
        for name, cfg in config.eval.evaluators.items():
            config_instances[f"evaluator_{name}"] = cfg

    # Add loggers and telemetry exporters (they also use prefixed IDs)
    if hasattr(config.general, "telemetry"):
        telemetry = config.general.telemetry
        if hasattr(telemetry, "logging"):
            for name, cfg in telemetry.logging.items():
                config_instances[f"logger_{name}"] = cfg
        if hasattr(telemetry, "tracing"):
            for name, cfg in telemetry.tracing.items():
                config_instances[f"telemetry_{name}"] = cfg

    # Extract connections for each component
    for component in components:
        config_instance = config_instances.get(component.id)
        if config_instance and isinstance(config_instance, BaseModel):
            component_connections = extract_connections_from_component(
                component,
                config_instance,
                component_lookup,
                group_lookup,
            )
            connections.extend(component_connections)

    # Add container component connections
    connections.extend(_create_container_connections(config, component_lookup))

    return connections


def _create_container_connections(
    config: Config,
    component_lookup: dict[str, ImportedComponent],
) -> list[ImportedConnection]:
    """
    Create connections for container components (NatWorkflow, GeneralConfig, etc.).

    Args:
        config: The validated Config object
        component_lookup: Dict mapping component ID to component

    Returns:
        List of connections for container components
    """
    connections: list[ImportedConnection] = []

    # NatWorkflow -> workflow function (entrypoint)
    if "workflow" in component_lookup:
        connections.append(
            ImportedConnection(
                id=f"conn_{uuid.uuid4().hex[:8]}",
                source_id="workflow",
                target_id="nat_workflow",
                target_field="entrypoint",
                ref_type=RefType.FUNCTION,
            ))

    # NatWorkflow -> config containers
    if "general_config" in component_lookup:
        connections.append(
            ImportedConnection(
                id=f"conn_{uuid.uuid4().hex[:8]}",
                source_id="general_config",
                target_id="nat_workflow",
                target_field="configuration",
                ref_type=RefType.GENERAL_CONFIG,
            ))

    if "evaluation_config" in component_lookup:
        connections.append(
            ImportedConnection(
                id=f"conn_{uuid.uuid4().hex[:8]}",
                source_id="evaluation_config",
                target_id="nat_workflow",
                target_field="evaluation",
                ref_type=RefType.EVALUATION_CONFIG,
            ))

    if "optimizer_config" in component_lookup:
        connections.append(
            ImportedConnection(
                id=f"conn_{uuid.uuid4().hex[:8]}",
                source_id="optimizer_config",
                target_id="nat_workflow",
                target_field="optimizer",
                ref_type=RefType.OPTIMIZER_CONFIG,
            ))

    if "finetuner_config" in component_lookup:
        connections.append(
            ImportedConnection(
                id=f"conn_{uuid.uuid4().hex[:8]}",
                source_id="finetuner_config",
                target_id="nat_workflow",
                target_field="finetuning",
                ref_type=RefType.FINETUNER_CONFIG,
            ))

    # GeneralConfig connections (only if general_config exists)
    if "general_config" in component_lookup:
        # front_end
        if "front_end" in component_lookup:
            connections.append(
                ImportedConnection(
                    id=f"conn_{uuid.uuid4().hex[:8]}",
                    source_id="front_end",
                    target_id="general_config",
                    target_field="front_end",
                    ref_type=RefType.FRONT_END,
                ))

        # loggers
        if hasattr(config.general, "telemetry") and hasattr(config.general.telemetry, "logging"):
            for name in config.general.telemetry.logging.keys():
                logger_id = f"logger_{name}"
                if logger_id in component_lookup:
                    connections.append(
                        ImportedConnection(
                            id=f"conn_{uuid.uuid4().hex[:8]}",
                            source_id=logger_id,
                            target_id="general_config",
                            target_field="loggers",
                            ref_type=RefType.LOGGER,
                        ))

        # telemetry_exporters
        if hasattr(config.general, "telemetry") and hasattr(config.general.telemetry, "tracing"):
            for name in config.general.telemetry.tracing.keys():
                telemetry_id = f"telemetry_{name}"
                if telemetry_id in component_lookup:
                    connections.append(
                        ImportedConnection(
                            id=f"conn_{uuid.uuid4().hex[:8]}",
                            source_id=telemetry_id,
                            target_id="general_config",
                            target_field="telemetry_exporters",
                            ref_type=RefType.TELEMETRY_EXPORTER,
                        ))

    # EvaluationConfig connections (evaluators) - only if evaluation_config exists
    if "evaluation_config" in component_lookup and hasattr(config.eval, "evaluators"):
        for name in config.eval.evaluators.keys():
            evaluator_id = f"evaluator_{name}"
            if evaluator_id in component_lookup:
                connections.append(
                    ImportedConnection(
                        id=f"conn_{uuid.uuid4().hex[:8]}",
                        source_id=evaluator_id,
                        target_id="evaluation_config",
                        target_field="evaluators",
                        ref_type=RefType.EVALUATOR,
                    ))

    # FinetunerConfig connections - only if finetuner_config exists
    if "finetuner_config" in component_lookup:
        finetuning = config.finetuning
        if finetuning:
            # trainer
            if finetuning.trainer_name:
                trainer_id = finetuning.trainer_name
                if trainer_id in component_lookup:
                    connections.append(
                        ImportedConnection(
                            id=f"conn_{uuid.uuid4().hex[:8]}",
                            source_id=trainer_id,
                            target_id="finetuner_config",
                            target_field="trainer",
                            ref_type=RefType.TRAINER,
                        ))

            # trajectory_builder
            if finetuning.trajectory_builder_name:
                tb_id = finetuning.trajectory_builder_name
                if tb_id in component_lookup:
                    connections.append(
                        ImportedConnection(
                            id=f"conn_{uuid.uuid4().hex[:8]}",
                            source_id=tb_id,
                            target_id="finetuner_config",
                            target_field="trajectory_builder",
                            ref_type=RefType.TRAJECTORY_BUILDER,
                        ))

            # trainer_adapter
            if finetuning.trainer_adapter_name:
                ta_id = finetuning.trainer_adapter_name
                if ta_id in component_lookup:
                    connections.append(
                        ImportedConnection(
                            id=f"conn_{uuid.uuid4().hex[:8]}",
                            source_id=ta_id,
                            target_id="finetuner_config",
                            target_field="trainer_adapter",
                            ref_type=RefType.TRAINER_ADAPTER,
                        ))

    return connections


# =============================================================================
# LAYOUT CALCULATION
# =============================================================================


def calculate_layout(
    components: list[ImportedComponent],
    connections: list[ImportedConnection],
) -> None:
    """
    Calculate layout positions for components using longest-path layering.

    Layout strategy (Sugiyama-style layered graph):
    - Layer 0 (Column 0): NatWorkflow (far left, the workflow container)
    - Layer 1 (Column 1): Workflow entrypoint function/agent + config containers
    - Layer 2+: Components flow left-to-right based on reference chains

    Connection semantics:
    - Connection(source_id=X, target_id=Y, target_field=F) means:
      "Y has field F whose value references X" i.e., Y references X.
    - In visual terms: Y --F--> X (Y points to X)
    - For layout: X should be to the RIGHT of Y (referenced after referencer)

    The layer of each component is determined by the longest path from the
    referencers to the referenced. This ensures that shared dependencies
    (e.g., an LLM used by both an agent and a function) are placed at the
    appropriate depth.

    Example (from react agent config):
        workflow (react_agent) references: [functions, llm]
        code_generation (function) references: [llm]
        evaluators reference: [llm]

        Results in:
        - Layer 0: nat_workflow
        - Layer 1: workflow, evaluation_config
        - Layer 2: functions (wikipedia_search, current_datetime, code_generation), evaluators
        - Layer 3: llm (longest path through code_generation or evaluators)
    """
    if not components:
        return

    component_lookup = {c.id: c for c in components}

    # Special components that get fixed positions
    workflow_id = "nat_workflow"
    entrypoint_id = "workflow"  # The actual function/agent
    config_containers = ["general_config", "evaluation_config", "optimizer_config", "finetuner_config"]

    # Separate special components from regular ones
    special_ids = {workflow_id, entrypoint_id} | set(config_containers)
    regular_components = [c for c in components if c.id not in special_ids]
    regular_ids = {c.id for c in regular_components}

    # Build reference graph based on connection semantics:
    # Connection(source_id=X, target_id=Y) means Y references X
    # So X "is referenced by" Y
    #
    # For layout: layer[X] = max(layer[all Ys that reference X]) + 1
    # We need to find: for each component X, who references it (all Ys where source_id=X)
    is_referenced_by: dict[str, set[str]] = {c.id: set() for c in regular_components}

    for conn in connections:
        if conn.source_id in is_referenced_by:
            # source_id is being referenced BY target_id
            # target_id has a field that points to source_id
            referencer = conn.target_id
            # Include both special IDs (entrypoint, config containers) and regular components
            if referencer in regular_ids or referencer in special_ids:
                is_referenced_by[conn.source_id].add(referencer)

    # Calculate layers using longest path from referencers
    # A component's layer = max(layer of all components that reference it) + 1
    layers: dict[str, int] = {}

    def calculate_layer(component_id: str, visited: set[str]) -> int:
        """Calculate layer based on who references this component."""
        if component_id in layers:
            return layers[component_id]

        if component_id in visited:
            # Cycle detected - return a default
            return 2

        visited.add(component_id)

        # Find all components that reference this one
        referencers = is_referenced_by.get(component_id, set())

        if not referencers:
            # No one references this - orphan component, default to Layer 2
            layers[component_id] = 2
        else:
            max_referencer_layer = 0
            for referencer_id in referencers:
                if referencer_id == entrypoint_id:
                    # Entrypoint is at Layer 1
                    referencer_layer = 1
                elif referencer_id in config_containers:
                    # Config containers are also at Layer 1
                    referencer_layer = 1
                elif referencer_id == workflow_id:
                    # nat_workflow is at Layer 0
                    referencer_layer = 0
                elif referencer_id in regular_ids:
                    # Recursively calculate layer of this referencer
                    referencer_layer = calculate_layer(referencer_id, visited.copy())
                else:
                    # Unknown referencer, skip
                    continue
                max_referencer_layer = max(max_referencer_layer, referencer_layer)

            layers[component_id] = max_referencer_layer + 1

        return layers[component_id]

    # Calculate layers for all regular components
    for component in regular_components:
        calculate_layer(component.id, set())

    # Pre-calculate heights for all components
    component_heights: dict[str, int] = {}
    for component in components:
        component_heights[component.id] = calculate_component_height(component, connections)

    # Position special components first
    # Column 0: NatWorkflow
    if workflow_id in component_lookup:
        component_lookup[workflow_id].position = Position(
            x=CANVAS_MARGIN,
            y=CANVAS_MARGIN,
        )

    # Column 1: Entrypoint and config containers stacked vertically
    col1_x = CANVAS_MARGIN + COMPONENT_WIDTH + HORIZONTAL_GAP
    col1_y = CANVAS_MARGIN

    if entrypoint_id in component_lookup:
        component_lookup[entrypoint_id].position = Position(
            x=col1_x,
            y=col1_y,
        )
        col1_y += component_heights.get(entrypoint_id, COMPONENT_MIN_HEIGHT) + VERTICAL_GAP

    # Stack config containers below the entrypoint
    for container_id in config_containers:
        if container_id in component_lookup:
            component_lookup[container_id].position = Position(
                x=col1_x,
                y=col1_y,
            )
            col1_y += component_heights.get(container_id, COMPONENT_MIN_HEIGHT) + VERTICAL_GAP

    # Group regular components by layer
    columns: dict[int, list[ImportedComponent]] = {}
    for component in regular_components:
        layer = layers.get(component.id, 2)
        if layer not in columns:
            columns[layer] = []
        columns[layer].append(component)

    # Sort within columns by component type for visual consistency
    # Place agents/functions first (they're the "main" components), then dependencies
    type_order = [
        "agent",
        "function",
        "function_group",
        "retriever",
        "memory",
        "llm",
        "embedder",
        "object_store",
        "front_end",
        "authentication",
        "logger",
        "telemetry_exporter",
        "evaluator",
        "trainer",
        "trajectory_builder",
        "trainer_adapter",
        "ttc_strategy",
    ]

    def sort_key(c: ImportedComponent) -> tuple[int, str]:
        try:
            idx = type_order.index(c.component_type)
        except ValueError:
            idx = len(type_order)
        return (idx, c.name)

    for col_components in columns.values():
        col_components.sort(key=sort_key)

    # Assign positions for regular components starting at column 2
    # (column 0 = nat_workflow, column 1 = entrypoint + containers, column 2+ = layers)
    sorted_layers = sorted(columns.keys())
    for layer in sorted_layers:
        col_components = columns[layer]
        # Layer 2 goes to column 2, layer 3 to column 3, etc.
        col_x = CANVAS_MARGIN + layer * (COMPONENT_WIDTH + HORIZONTAL_GAP)

        # Stack components vertically with dynamic heights
        col_y = CANVAS_MARGIN
        for component in col_components:
            component.position = Position(
                x=col_x,
                y=col_y,
            )
            col_y += component_heights.get(component.id, COMPONENT_MIN_HEIGHT) + VERTICAL_GAP


# =============================================================================
# MAIN PARSER
# =============================================================================


def get_validated_config(config_dict: dict[str, Any]) -> Config:
    """
    Validate a config dict and return the typed Config object.

    Args:
        config_dict: Raw config dictionary from YAML

    Returns:
        Validated Config Pydantic object
    """
    # Ensure ALL plugins are loaded (including external packages like nvidia-nat-langchain)
    discover_and_register_plugins(PluginTypes.ALL)

    # Rebuild annotations to include all registered types
    Config.rebuild_annotations()

    # Validate and return typed Config
    return validate_schema(config_dict, Config)


def parse_config_to_workflow_state(config_dict: dict[str, Any]) -> ImportedWorkflowState:
    """
    Parse a config dictionary into a workflow state for the UI.

    Args:
        config_dict: The raw config dictionary from YAML

    Returns:
        ImportedWorkflowState with components and connections ready for UI rendering
    """
    # Get validated Config object with properly typed fields
    config = get_validated_config(config_dict)

    # Parse components using type introspection
    components = parse_components_from_config(config)

    # Parse connections by scanning ComponentRef fields
    connections = parse_connections_from_config(config, components)

    # Calculate layout positions
    calculate_layout(components, connections)

    return ImportedWorkflowState(
        components=components,
        connections=connections,
    )
