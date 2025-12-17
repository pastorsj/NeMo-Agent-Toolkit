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
"""

import logging
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException

from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import RegisteredInfo
from nat.data_models.agent import AgentBaseConfig
from nat.workflow_builder_api.models import ComponentCategory
from nat.workflow_builder_api.models import ComponentTypeInfo
from nat.workflow_builder_api.models import HealthResponse
from nat.workflow_builder_api.models import RefType
from nat.workflow_builder_api.models import RegisteredTypeInfo
from nat.workflow_builder_api.models import RegistryResponse
from nat.workflow_builder_api.schema_extractor import extract_all_connection_ports
from nat.workflow_builder_api.schema_extractor import extract_fields
from nat.workflow_builder_api.schema_extractor import extract_json_schema
from nat.workflow_builder_api.schema_extractor import get_model_description
from nat.workflow_builder_api.schema_extractor import get_sdk_excluded_fields

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["registry"])

# Mapping from ComponentCategory to RefType (categories that provide output ports)
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
    ComponentCategory.TTC_STRATEGY: RefType.TTC_STRATEGY,
}


def _is_agent_config(config_type: type) -> bool:
    """Check if a config type inherits from AgentBaseConfig."""
    try:
        return issubclass(config_type, AgentBaseConfig)
    except TypeError:
        return False


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

    return RegisteredTypeInfo(
        full_type=registered_info.full_type,
        module_name=registered_info.module_name,
        local_name=registered_info.local_name,
        description=description,
        json_schema=json_schema,
        fields=fields,
        is_per_user=registered_info.is_per_user,
        input_ports=input_ports,
    )


def _get_component_types() -> list[ComponentTypeInfo]:
    """Get all component types from the registry."""
    registry = GlobalTypeRegistry.get()
    components = []

    # Get all functions first, then split into agents and regular functions
    all_functions = registry.get_registered_functions()
    agent_functions = [f for f in all_functions if _is_agent_config(f.config_type)]
    regular_functions = [f for f in all_functions if not _is_agent_config(f.config_type)]

    # Define all component categories with their getters
    category_configs = [
        {
            "category": ComponentCategory.LLM,
            "display_name": "LLM Providers",
            "description": "Large Language Model providers for reasoning and generation",
            "getter": lambda: registry.get_registered_llm_providers(),
        },
        {
            "category": ComponentCategory.EMBEDDER,
            "display_name": "Embedders",
            "description": "Text embedding models for semantic search and retrieval",
            "getter": lambda: registry.get_registered_embedder_providers(),
        },
        {
            "category": ComponentCategory.AGENT,
            "display_name": "Agents",
            "description": "AI agents that can reason, use tools, and complete complex tasks",
            "getter": lambda: agent_functions,
        },
        {
            "category": ComponentCategory.FUNCTION,
            "display_name": "Functions",
            "description": "Custom functions and tools that agents can use",
            "getter": lambda: regular_functions,
        },
        {
            "category": ComponentCategory.FUNCTION_GROUP,
            "display_name": "Function Groups",
            "description": "Groups of related functions",
            "getter": lambda: registry.get_registered_function_groups(),
        },
        {
            "category": ComponentCategory.RETRIEVER,
            "display_name": "Retrievers",
            "description": "Document retrievers for RAG workflows",
            "getter": lambda: registry.get_registered_retriever_providers(),
        },
        {
            "category": ComponentCategory.MEMORY,
            "display_name": "Memory",
            "description": "Persistent memory storage for agent context",
            "getter": lambda: registry.get_registered_memorys(),
        },
        {
            "category": ComponentCategory.OBJECT_STORE,
            "display_name": "Object Stores",
            "description": "Object storage backends (S3, etc.)",
            "getter": lambda: registry.get_registered_object_stores(),
        },
        {
            "category": ComponentCategory.AUTHENTICATION,
            "display_name": "Authentication Providers",
            "description": "API authentication providers",
            "getter": lambda: registry.get_registered_auth_providers(),
        },
        {
            "category": ComponentCategory.MIDDLEWARE,
            "display_name": "Middleware",
            "description": "Request/response middleware components",
            "getter": lambda: registry.get_registered_middleware(),
        },
        {
            "category": ComponentCategory.TTC_STRATEGY,
            "display_name": "TTC Strategies",
            "description": "Time-to-complete strategies for agent execution",
            "getter": lambda: registry.get_registered_ttc_strategies(),
        },
        {
            "category": ComponentCategory.TRAINER,
            "display_name": "Trainers",
            "description": "Model training components",
            "getter": lambda: registry.get_registered_trainers(),
        },
        {
            "category": ComponentCategory.TRAINER_ADAPTER,
            "display_name": "Trainer Adapters",
            "description": "Adapters for training integrations",
            "getter": lambda: registry.get_registered_trainer_adapters(),
        },
        {
            "category": ComponentCategory.TRAJECTORY_BUILDER,
            "display_name": "Trajectory Builders",
            "description": "Builders for training trajectories",
            "getter": lambda: registry.get_registered_trajectory_builders(),
        },
        {
            "category": ComponentCategory.FRONT_END,
            "display_name": "Front Ends",
            "description": "Workflow front-end entry points",
            "getter": lambda: registry.get_registered_front_ends(),
        },
        {
            "category": ComponentCategory.EVALUATOR,
            "display_name": "Evaluators",
            "description": "Workflow evaluation components",
            "getter": lambda: registry.get_registered_evaluators(),
        },
        {
            "category": ComponentCategory.TELEMETRY_EXPORTER,
            "display_name": "Telemetry Exporters",
            "description": "Telemetry and observability exporters",
            "getter": lambda: registry.get_registered_telemetry_exporters(),
        },
        {
            "category": ComponentCategory.LOGGING,
            "display_name": "Logging Methods",
            "description": "Logging configuration methods",
            "getter": lambda: registry.get_registered_logging_method(),
        },
        {
            "category": ComponentCategory.REGISTRY_HANDLER,
            "display_name": "Registry Handlers",
            "description": "Component registry handlers",
            "getter": lambda: registry.get_registered_registry_handlers(),
        },
    ]

    for config in category_configs:
        try:
            registered_items = config["getter"]()
            registered_types = []

            for item in registered_items:
                try:
                    type_info = _build_registered_type_info(item)
                    registered_types.append(type_info)
                except Exception as e:
                    logger.warning(f"Failed to build type info for {item.full_type}: {e}")

            # Determine what RefType this category provides (if any)
            provides_ref_type = CATEGORY_TO_REF_TYPE.get(config["category"])

            components.append(
                ComponentTypeInfo(
                    category=config["category"],
                    display_name=config["display_name"],
                    description=config["description"],
                    registered_types=registered_types,
                    provides_ref_type=provides_ref_type,
                ))
        except Exception as e:
            logger.warning(f"Failed to get registered types for {config['category']}: {e}")

    return components


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


@router.get("/registry", response_model=RegistryResponse)
async def get_registry() -> RegistryResponse:
    """
    Get all registered component types from the NAT TypeRegistry.

    Returns a comprehensive list of all component categories and their
    registered implementations, including JSON Schemas for configuration.
    """
    components = _get_component_types()
    total_types = sum(len(c.registered_types) for c in components)

    return RegistryResponse(
        components=components,
        total_types=total_types,
    )


@router.get("/registry/{category}", response_model=ComponentTypeInfo)
async def get_category(category: ComponentCategory) -> ComponentTypeInfo:
    """
    Get registered types for a specific component category.

    Args:
        category: The component category to retrieve.

    Returns:
        ComponentTypeInfo with all registered types in that category.
    """
    components = _get_component_types()

    for component in components:
        if component.category == category:
            return component

    raise HTTPException(status_code=404, detail=f"Category {category} not found")


@router.get("/registry/{category}/{full_type:path}", response_model=RegisteredTypeInfo)
async def get_type_info(category: ComponentCategory, full_type: str) -> RegisteredTypeInfo:
    """
    Get detailed information about a specific registered type.

    Args:
        category: The component category.
        full_type: The full type identifier (e.g., "nvidia/nim_llm").

    Returns:
        RegisteredTypeInfo with full schema and field information.
    """
    components = _get_component_types()

    for component in components:
        if component.category == category:
            for reg_type in component.registered_types:
                if reg_type.full_type == full_type:
                    return reg_type

    raise HTTPException(status_code=404, detail=f"Type {full_type} not found in category {category}")


@router.get("/schema/{category}/{full_type:path}")
async def get_type_schema(category: ComponentCategory, full_type: str) -> dict[str, Any]:
    """
    Get just the JSON Schema for a specific registered type.

    This is useful for form generation in the UI.

    Args:
        category: The component category.
        full_type: The full type identifier.

    Returns:
        The JSON Schema dictionary.
    """
    type_info = await get_type_info(category, full_type)
    return type_info.json_schema
