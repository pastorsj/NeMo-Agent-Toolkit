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
Type info builders for the Workflow Builder API.

Builds RegisteredTypeInfo objects from:
- Registry registered types (LLM providers, functions, etc.)
- SDK workflow classes (NatWorkflow, NatGeneralConfiguration, etc.)
"""

import logging

from pydantic import BaseModel

from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import RegisteredInfo
from nat.data_models.agent import AgentBaseConfig
from nat.workflow_builder_api.constants.categories import CATEGORY_METADATA
from nat.workflow_builder_api.constants.categories import CATEGORY_TO_REF_TYPE
from nat.workflow_builder_api.constants.categories import SDK_CLASS_METADATA
from nat.workflow_builder_api.models import ComponentCategory
from nat.workflow_builder_api.models import ComponentTypeInfo
from nat.workflow_builder_api.models import ConnectionPort
from nat.workflow_builder_api.models import FieldInfo
from nat.workflow_builder_api.models import RegisteredTypeInfo
from nat.workflow_builder_api.utils.connections import extract_all_connection_ports
from nat.workflow_builder_api.utils.schema import extract_fields
from nat.workflow_builder_api.utils.schema import extract_json_schema
from nat.workflow_builder_api.utils.schema import get_display_name_from_docstring
from nat.workflow_builder_api.utils.schema import get_icon_url_from_docstring
from nat.workflow_builder_api.utils.schema import get_model_description
from nat.workflow_builder_api.utils.schema import get_sdk_excluded_fields

logger = logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def is_agent_config(config_type: type) -> bool:
    """Check if a config type inherits from AgentBaseConfig."""
    try:
        return issubclass(config_type, AgentBaseConfig)
    except TypeError:
        return False


def is_test_type(registered_info: RegisteredInfo) -> bool:
    """Check if a registered type is from a tests folder.

    Excludes known legitimate modules that happen to contain 'test' in their name,
    such as 'test_time_compute' (TTC strategies).
    """
    module_name = registered_info.module_name or ""
    parts = module_name.lower().split(".")

    # Allowlist for legitimate modules that contain 'test' in their name
    allowlisted_patterns = ("test_time_compute", )

    for part in parts:
        # Skip allowlisted module names
        if any(allowed in part for allowed in allowlisted_patterns):
            continue
        if part in ("tests", "test", "testing"):
            return True
        if part.startswith("test_") or part.endswith("_test"):
            return True
    return False


def mark_connection_fields(fields: list[FieldInfo], input_ports: list[ConnectionPort]) -> list[FieldInfo]:
    """Filter out fields that are connection ports."""
    port_field_names = {port.field_name for port in input_ports}
    return [field for field in fields if field.name not in port_field_names]


# =============================================================================
# TYPE INFO BUILDERS
# =============================================================================


def build_workflow_sdk_type_info(
    sdk_class: type[BaseModel],
    module_name: str,
    icon_url: str | None,
) -> RegisteredTypeInfo:
    """Build a RegisteredTypeInfo from a workflow SDK class."""
    json_schema = extract_json_schema(sdk_class)
    fields = extract_fields(json_schema, set(), sdk_class)
    description = get_model_description(sdk_class)
    input_ports = extract_all_connection_ports(sdk_class)
    fields = mark_connection_fields(fields, input_ports)
    display_name = get_display_name_from_docstring(sdk_class)

    local_name = sdk_class.__name__
    full_type = f"{module_name}/{local_name}"

    return RegisteredTypeInfo(
        full_type=full_type,
        module_name=module_name,
        local_name=local_name,
        display_name=display_name,
        description=description,
        json_schema=json_schema,
        fields=fields,
        is_per_user=False,
        input_ports=input_ports,
        icon_url=icon_url,
    )


def get_sdk_class_type(category: ComponentCategory) -> RegisteredTypeInfo | None:
    """Get RegisteredTypeInfo for a single SDK class category."""
    if category not in SDK_CLASS_METADATA:
        return None
    sdk_class, module_name, icon_url = SDK_CLASS_METADATA[category]
    try:
        return build_workflow_sdk_type_info(sdk_class, module_name, icon_url)
    except Exception as e:
        logger.warning("Failed to build type info for %s: %s", sdk_class.__name__, e)
        return None


def build_registered_type_info(registered_info: RegisteredInfo) -> RegisteredTypeInfo:
    """Build a RegisteredTypeInfo from a RegisteredInfo object."""
    config_type = registered_info.config_type
    json_schema = extract_json_schema(config_type)
    sdk_excluded = get_sdk_excluded_fields(config_type)
    fields = extract_fields(json_schema, sdk_excluded, config_type)
    description = get_model_description(config_type)
    input_ports = extract_all_connection_ports(config_type)
    fields = mark_connection_fields(fields, input_ports)
    icon_url = get_icon_url_from_docstring(config_type)
    display_name = get_display_name_from_docstring(config_type)

    return RegisteredTypeInfo(
        full_type=registered_info.full_type,
        module_name=registered_info.module_name,
        local_name=registered_info.local_name,
        display_name=display_name,
        description=description,
        json_schema=json_schema,
        fields=fields,
        is_per_user=registered_info.is_per_user,
        input_ports=input_ports,
        icon_url=icon_url,
    )


def build_type_infos(registered_items: list[RegisteredInfo]) -> list[RegisteredTypeInfo]:
    """Build RegisteredTypeInfo list from registered items, filtering out test types."""
    result = []
    for item in registered_items:
        if is_test_type(item):
            continue
        try:
            result.append(build_registered_type_info(item))
        except Exception as e:
            logger.warning("Failed to build type info for %s: %s", item.full_type, e)
    return result


# =============================================================================
# CATEGORY TYPE FETCHER
# =============================================================================


def get_category_types(category: ComponentCategory) -> ComponentTypeInfo:
    """
    Get component types for a specific category.

    Raises:
        ValueError: If the category is not supported.
    """
    registry = GlobalTypeRegistry.get()
    items: list[RegisteredInfo] = []

    if category == ComponentCategory.LLM:
        items = registry.get_registered_llm_providers()
    elif category == ComponentCategory.EMBEDDER:
        items = registry.get_registered_embedder_providers()
    elif category == ComponentCategory.AGENT:
        all_functions = registry.get_registered_functions()
        items = [f for f in all_functions if is_agent_config(f.config_type)]
    elif category == ComponentCategory.FUNCTION:
        all_functions = registry.get_registered_functions()
        items = [f for f in all_functions if not is_agent_config(f.config_type)]
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
    elif category == ComponentCategory.FRONT_END:
        items = registry.get_registered_front_ends()
    elif category == ComponentCategory.LOGGER:
        items = registry.get_registered_logging_method()
    elif category == ComponentCategory.TELEMETRY_EXPORTER:
        items = registry.get_registered_telemetry_exporters()
    elif category == ComponentCategory.EVALUATOR:
        items = registry.get_registered_evaluators()
    elif category == ComponentCategory.TRAINER:
        items = registry.get_registered_trainers()
    elif category == ComponentCategory.TRAJECTORY_BUILDER:
        items = registry.get_registered_trajectory_builders()
    elif category == ComponentCategory.TRAINER_ADAPTER:
        items = registry.get_registered_trainer_adapters()
    elif category == ComponentCategory.TTC_STRATEGY:
        items = registry.get_registered_ttc_strategies()
    elif category in SDK_CLASS_METADATA:
        display_name, description = CATEGORY_METADATA.get(category, (category.value.title(), ""))
        provides_ref_type = CATEGORY_TO_REF_TYPE.get(category)
        sdk_type = get_sdk_class_type(category)
        registered_types = [sdk_type] if sdk_type else []
        return ComponentTypeInfo(
            category=category,
            display_name=display_name,
            description=description,
            registered_types=registered_types,
            provides_ref_type=provides_ref_type,
        )
    else:
        raise ValueError(f"Category {category} not supported")

    registered_types = build_type_infos(items)
    display_name, description = CATEGORY_METADATA.get(category, (category.value.title(), ""))
    provides_ref_type = CATEGORY_TO_REF_TYPE.get(category)

    return ComponentTypeInfo(
        category=category,
        display_name=display_name,
        description=description,
        registered_types=registered_types,
        provides_ref_type=provides_ref_type,
    )
