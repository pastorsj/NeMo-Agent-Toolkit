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
FastAPI routes for the Workflow Builder API.

Provides per-category endpoints for efficient parallel loading of component types.
"""

import logging

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel

from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import RegisteredInfo
from nat.data_models.agent import AgentBaseConfig
from nat.utils.sdk.nat_evaluation import NatEvaluation
from nat.utils.sdk.nat_finetuner import NatFinetuner
from nat.utils.sdk.nat_general_configuraton import NatGeneralConfiguration
from nat.utils.sdk.nat_optimizer import NatOptimizer

# SDK workflow classes (not registered in type registry)
from nat.utils.sdk.nat_workflow import NatWorkflow
from nat.workflow_builder_api.component_metadata import apply_field_options
from nat.workflow_builder_api.models import ComponentCategory
from nat.workflow_builder_api.models import ComponentTypeInfo
from nat.workflow_builder_api.models import ConnectionPort
from nat.workflow_builder_api.models import FieldInfo
from nat.workflow_builder_api.models import HealthResponse
from nat.workflow_builder_api.models import RefType
from nat.workflow_builder_api.models import RegisteredTypeInfo
from nat.workflow_builder_api.schema_extractor import extract_all_connection_ports
from nat.workflow_builder_api.schema_extractor import extract_fields
from nat.workflow_builder_api.schema_extractor import extract_json_schema
from nat.workflow_builder_api.schema_extractor import get_icon_url_from_docstring
from nat.workflow_builder_api.schema_extractor import get_model_description
from nat.workflow_builder_api.schema_extractor import get_sdk_excluded_fields

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["registry"])

# =============================================================================
# CATEGORY CONFIGURATION
# =============================================================================

# Categories used by the workflow builder UI
UI_CATEGORIES: set[ComponentCategory] = {
    ComponentCategory.LLM,
    ComponentCategory.EMBEDDER,
    ComponentCategory.AGENT,
    ComponentCategory.FUNCTION,
    ComponentCategory.FUNCTION_GROUP,
    ComponentCategory.RETRIEVER,
    ComponentCategory.MEMORY,
    ComponentCategory.OBJECT_STORE,
    ComponentCategory.AUTHENTICATION,
    ComponentCategory.MIDDLEWARE,
    # Front-end and observability
    ComponentCategory.FRONT_END,
    ComponentCategory.LOGGER,
    ComponentCategory.TELEMETRY_EXPORTER,
    # Evaluation
    ComponentCategory.EVALUATOR,
    # Finetuning components
    ComponentCategory.TRAINER,
    ComponentCategory.TRAJECTORY_BUILDER,
    ComponentCategory.TRAINER_ADAPTER,
    # Workflow-level configuration containers
    ComponentCategory.NAT_WORKFLOW,
    ComponentCategory.GENERAL_CONFIG,
    ComponentCategory.EVALUATION_CONFIG,
    ComponentCategory.OPTIMIZER_CONFIG,
    ComponentCategory.FINETUNER_CONFIG,
}

# Mapping from ComponentCategory to RefType (what output port type each category provides)
CATEGORY_TO_REF_TYPE: dict[ComponentCategory, RefType] = {
    ComponentCategory.LLM: RefType.LLM,
    ComponentCategory.EMBEDDER: RefType.EMBEDDER,
    ComponentCategory.FUNCTION: RefType.FUNCTION,
    ComponentCategory.FUNCTION_GROUP: RefType.FUNCTION_GROUP,
    ComponentCategory.AGENT: RefType.FUNCTION,  # Agents can be used as functions/tools
    ComponentCategory.RETRIEVER: RefType.RETRIEVER,
    ComponentCategory.MEMORY: RefType.MEMORY,
    ComponentCategory.OBJECT_STORE: RefType.OBJECT_STORE,
    ComponentCategory.AUTHENTICATION: RefType.AUTHENTICATION,
    ComponentCategory.MIDDLEWARE: RefType.MIDDLEWARE,
    # Front-end and observability
    ComponentCategory.FRONT_END: RefType.FRONT_END,
    ComponentCategory.LOGGER: RefType.LOGGER,
    ComponentCategory.TELEMETRY_EXPORTER: RefType.TELEMETRY_EXPORTER,
    # Evaluation
    ComponentCategory.EVALUATOR: RefType.EVALUATOR,
    # Finetuning components
    ComponentCategory.TRAINER: RefType.TRAINER,
    ComponentCategory.TRAJECTORY_BUILDER: RefType.TRAJECTORY_BUILDER,
    ComponentCategory.TRAINER_ADAPTER: RefType.TRAINER_ADAPTER,
    # Workflow-level configuration containers
    ComponentCategory.NAT_WORKFLOW: RefType.NAT_WORKFLOW,
    ComponentCategory.GENERAL_CONFIG: RefType.GENERAL_CONFIG,
    ComponentCategory.EVALUATION_CONFIG: RefType.EVALUATION_CONFIG,
    ComponentCategory.OPTIMIZER_CONFIG: RefType.OPTIMIZER_CONFIG,
    ComponentCategory.FINETUNER_CONFIG: RefType.FINETUNER_CONFIG,
}

# Category display metadata
CATEGORY_METADATA: dict[ComponentCategory, tuple[str, str]] = {
    ComponentCategory.LLM: ("LLM Providers", "Large Language Model providers for reasoning and generation"),
    ComponentCategory.EMBEDDER: ("Embedders", "Text embedding models for semantic search and retrieval"),
    ComponentCategory.AGENT: ("Agents", "AI agents that can reason, use tools, and complete complex tasks"),
    ComponentCategory.FUNCTION: ("Functions", "Custom functions and tools that agents can use"),
    ComponentCategory.FUNCTION_GROUP: ("Function Groups", "Groups of related functions"),
    ComponentCategory.RETRIEVER: ("Retrievers", "Document retrievers for RAG workflows"),
    ComponentCategory.MEMORY: ("Memory", "Persistent memory storage for agent context"),
    ComponentCategory.OBJECT_STORE: ("Object Stores", "Object storage backends (S3, etc.)"),
    ComponentCategory.AUTHENTICATION: ("Authentication", "API authentication providers"),
    ComponentCategory.MIDDLEWARE: ("Middleware", "Request/response middleware components"),
    # Front-end and observability
    ComponentCategory.FRONT_END: ("Front Ends", "Deployment front ends (FastAPI, Console, MCP)"),
    ComponentCategory.LOGGER: ("Loggers", "Logging configurations for runtime observability"),
    ComponentCategory.TELEMETRY_EXPORTER: ("Telemetry", "Telemetry exporters for tracing and metrics"),
    # Evaluation
    ComponentCategory.EVALUATOR: ("Evaluators", "Evaluation metrics for testing workflow quality"),
    # Finetuning components
    ComponentCategory.TRAINER: ("Trainers", "Training loop orchestrators for finetuning"),
    ComponentCategory.TRAJECTORY_BUILDER: ("Trajectory Builders", "Training data collectors"),
    ComponentCategory.TRAINER_ADAPTER: ("Trainer Adapters", "Training backend adapters"),
    # Workflow-level configuration containers
    ComponentCategory.NAT_WORKFLOW: ("Workflow", "Main workflow entry point"),
    ComponentCategory.GENERAL_CONFIG: ("General Config", "Loggers, telemetry, and front-end configuration"),
    ComponentCategory.EVALUATION_CONFIG: ("Evaluation", "Evaluation configuration with evaluators"),
    ComponentCategory.OPTIMIZER_CONFIG: ("Optimizer", "Hyperparameter optimization configuration"),
    ComponentCategory.FINETUNER_CONFIG: ("Finetuner", "Model finetuning configuration"),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _is_agent_config(config_type: type) -> bool:
    """Check if a config type inherits from AgentBaseConfig."""
    try:
        return issubclass(config_type, AgentBaseConfig)
    except TypeError:
        return False


def _is_test_type(registered_info: RegisteredInfo) -> bool:
    """Check if a registered type is from a tests folder and should be excluded."""
    module_name = registered_info.module_name or ""
    parts = module_name.lower().split(".")
    for part in parts:
        if part in ("tests", "test", "testing"):
            return True
        if part.startswith("test_") or part.endswith("_test"):
            return True
    return False


# Workflow SDK class metadata - each is a single-type category (no variants)
# These are NOT registered in the type registry but are key workflow components
SDK_CLASS_METADATA: dict[ComponentCategory, tuple[type[BaseModel], str, str | None]] = {
    ComponentCategory.NAT_WORKFLOW: (
        NatWorkflow,
        "nat.utils.sdk.nat_workflow",
        "https://cdn.simpleicons.org/nvidia/76B900",  # NVIDIA icon for workflow
    ),
    ComponentCategory.GENERAL_CONFIG: (
        NatGeneralConfiguration,
        "nat.utils.sdk.nat_general_configuraton",
        "https://cdn.simpleicons.org/gnubash/4EAA25",  # Config icon
    ),
    ComponentCategory.EVALUATION_CONFIG: (
        NatEvaluation,
        "nat.utils.sdk.nat_evaluation",
        "https://cdn.simpleicons.org/pytest/0A9EDC",  # Test icon for evaluation
    ),
    ComponentCategory.OPTIMIZER_CONFIG: (
        NatOptimizer,
        "nat.utils.sdk.nat_optimizer",
        "https://cdn.simpleicons.org/apachespark/E25A1C",  # Spark icon for optimizer
    ),
    ComponentCategory.FINETUNER_CONFIG: (
        NatFinetuner,
        "nat.utils.sdk.nat_finetuner",
        "https://cdn.simpleicons.org/pytorch/EE4C2C",  # PyTorch icon for finetuning
    ),
    # Note: TRAINER, TRAJECTORY_BUILDER, TRAINER_ADAPTER are NOT in SDK_CLASS_METADATA
    # because they have registered implementations (from plugins) that should be
    # fetched from the type registry instead. See _get_category_types() for handling.
}


def _mark_connection_fields(fields: list[FieldInfo], input_ports: list[ConnectionPort]) -> list[FieldInfo]:
    """
    Mark fields as component refs if they have a corresponding connection port.

    This ensures that fields for SDK types (NatLogger, NatTelemetryExporter, etc.)
    are properly marked as connection fields and hidden from the config form.

    Args:
        fields: List of extracted fields.
        input_ports: List of connection ports.

    Returns:
        Filtered list of fields, excluding those that are connection ports.
    """
    # Get the set of field names that are connection ports
    port_field_names = {port.field_name for port in input_ports}

    # Filter out fields that are connection ports - they should not appear in the config form
    # Connection ports are configured via visual connections, not text input
    return [field for field in fields if field.name not in port_field_names]


def _build_workflow_sdk_type_info(
    sdk_class: type[BaseModel],
    module_name: str,
    icon_url: str | None,
) -> RegisteredTypeInfo:
    """Build a RegisteredTypeInfo from a workflow SDK class (not a registered type)."""
    json_schema = extract_json_schema(sdk_class)

    # For SDK classes, we don't have a config-level exclusion, so we pass empty set
    fields = extract_fields(json_schema, set())
    description = get_model_description(sdk_class)

    # Extract connection ports from the SDK class
    input_ports = extract_all_connection_ports(sdk_class)

    # Remove fields that are connection ports (they're configured via connections, not text input)
    fields = _mark_connection_fields(fields, input_ports)

    # Use class name as local name
    local_name = sdk_class.__name__
    full_type = f"{module_name}/{local_name}"

    return RegisteredTypeInfo(
        full_type=full_type,
        module_name=module_name,
        local_name=local_name,
        description=description,
        json_schema=json_schema,
        fields=fields,
        is_per_user=False,
        input_ports=input_ports,
        icon_url=icon_url,
    )


def _get_sdk_class_type(category: ComponentCategory) -> RegisteredTypeInfo | None:
    """Get RegisteredTypeInfo for a single SDK class category."""
    if category not in SDK_CLASS_METADATA:
        return None
    sdk_class, module_name, icon_url = SDK_CLASS_METADATA[category]
    try:
        return _build_workflow_sdk_type_info(sdk_class, module_name, icon_url)
    except Exception as e:
        logger.warning(f"Failed to build type info for {sdk_class.__name__}: {e}")
        return None


def _build_registered_type_info(registered_info: RegisteredInfo) -> RegisteredTypeInfo:
    """Build a RegisteredTypeInfo from a RegisteredInfo object."""
    config_type = registered_info.config_type
    json_schema = extract_json_schema(config_type)

    # Get fields that should be excluded based on SDK class (init=False fields)
    sdk_excluded = get_sdk_excluded_fields(config_type)
    fields = extract_fields(json_schema, sdk_excluded)
    description = get_model_description(config_type)

    # Extract connection ports from both config and SDK class
    input_ports = extract_all_connection_ports(config_type)

    # Remove fields that are connection ports (they're configured via connections, not text input)
    fields = _mark_connection_fields(fields, input_ports)

    # Apply field options from the metadata registry
    apply_field_options(fields, registered_info.module_name)

    # Get icon URL from the model's docstring (e.g., ![Icon](https://...))
    icon_url = get_icon_url_from_docstring(config_type)

    return RegisteredTypeInfo(
        full_type=registered_info.full_type,
        module_name=registered_info.module_name,
        local_name=registered_info.local_name,
        description=description,
        json_schema=json_schema,
        fields=fields,
        is_per_user=registered_info.is_per_user,
        input_ports=input_ports,
        icon_url=icon_url,
    )


def _build_type_infos(registered_items: list[RegisteredInfo]) -> list[RegisteredTypeInfo]:
    """Build RegisteredTypeInfo list from registered items, filtering out test types."""
    result = []
    for item in registered_items:
        if _is_test_type(item):
            continue
        try:
            result.append(_build_registered_type_info(item))
        except Exception as e:
            logger.warning(f"Failed to build type info for {item.full_type}: {e}")
    return result


def _get_category_types(category: ComponentCategory) -> ComponentTypeInfo:
    """
    Get component types for a specific category.

    This is the optimized version that only fetches data for the requested category.
    """
    registry = GlobalTypeRegistry.get()

    # Get the raw registered items based on category
    if category == ComponentCategory.LLM:
        items = registry.get_registered_llm_providers()
    elif category == ComponentCategory.EMBEDDER:
        items = registry.get_registered_embedder_providers()
    elif category == ComponentCategory.AGENT:
        # Agents are functions that use AgentBaseConfig
        all_functions = registry.get_registered_functions()
        items = [f for f in all_functions if _is_agent_config(f.config_type)]
    elif category == ComponentCategory.FUNCTION:
        # Regular functions (not agents)
        all_functions = registry.get_registered_functions()
        items = [f for f in all_functions if not _is_agent_config(f.config_type)]
    elif category == ComponentCategory.FUNCTION_GROUP:
        items = registry.get_registered_function_groups()
    elif category == ComponentCategory.RETRIEVER:
        items = registry.get_registered_retriever_providers()
    elif category == ComponentCategory.MEMORY:
        items = registry.get_registered_memorys()
    elif category == ComponentCategory.OBJECT_STORE:
        items = registry.get_registered_object_stores()
    elif category == ComponentCategory.AUTHENTICATION:
        items = registry.get_registered_auth_providers()
    elif category == ComponentCategory.MIDDLEWARE:
        items = registry.get_registered_middleware()
    # Front-end and observability
    elif category == ComponentCategory.FRONT_END:
        items = registry.get_registered_front_ends()
    elif category == ComponentCategory.LOGGER:
        items = registry.get_registered_logging_method()
    elif category == ComponentCategory.TELEMETRY_EXPORTER:
        items = registry.get_registered_telemetry_exporters()
    # Evaluation
    elif category == ComponentCategory.EVALUATOR:
        items = registry.get_registered_evaluators()
    # Finetuning components
    elif category == ComponentCategory.TRAINER:
        items = registry.get_registered_trainers()
    elif category == ComponentCategory.TRAJECTORY_BUILDER:
        items = registry.get_registered_trajectory_builders()
    elif category == ComponentCategory.TRAINER_ADAPTER:
        items = registry.get_registered_trainer_adapters()
    # Workflow SDK classes (single-type categories, not registered in type registry)
    elif category in SDK_CLASS_METADATA:
        # Get metadata
        display_name, description = CATEGORY_METADATA.get(category, (category.value.title(), ""))
        provides_ref_type = CATEGORY_TO_REF_TYPE.get(category)

        # Get the single type for this category
        sdk_type = _get_sdk_class_type(category)
        registered_types = [sdk_type] if sdk_type else []

        return ComponentTypeInfo(
            category=category,
            display_name=display_name,
            description=description,
            registered_types=registered_types,
            provides_ref_type=provides_ref_type,
        )
    else:
        raise HTTPException(status_code=404, detail=f"Category {category} not supported")

    # Build type infos
    registered_types = _build_type_infos(items)

    # Get metadata
    display_name, description = CATEGORY_METADATA.get(category, (category.value.title(), ""))
    provides_ref_type = CATEGORY_TO_REF_TYPE.get(category)

    return ComponentTypeInfo(
        category=category,
        display_name=display_name,
        description=description,
        registered_types=registered_types,
        provides_ref_type=provides_ref_type,
    )


# =============================================================================
# API ROUTES
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health of the API and registry."""
    try:
        registry = GlobalTypeRegistry.get()
        # Try to access one registry to verify it's loaded
        _ = registry.get_registered_llm_providers()
        return HealthResponse(status="healthy", registry_loaded=True)
    except Exception:
        return HealthResponse(status="healthy", registry_loaded=False)


@router.get("/categories", response_model=list[ComponentCategory])
async def get_available_categories() -> list[ComponentCategory]:
    """
    Get list of available component categories.

    Returns the categories that the workflow builder UI supports.
    Use this to know which category endpoints to call.
    """
    return list(UI_CATEGORIES)


@router.get("/components/{category}", response_model=ComponentTypeInfo)
async def get_category_components(category: ComponentCategory) -> ComponentTypeInfo:
    """
    Get all registered component types for a specific category.

    This is the primary endpoint for loading component data.
    Call multiple category endpoints in parallel for faster loading.

    Args:
        category: The component category (e.g., 'llm', 'embedder', 'agent')

    Returns:
        ComponentTypeInfo with all registered types in that category.
    """
    if category not in UI_CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' is not available. Use /categories to see available categories.",
        )

    try:
        return _get_category_types(category)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading category {category}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load category: {str(e)}")
