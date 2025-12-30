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
Secret field type detection and environment variable enrichment.

Provides functions to detect SerializableSecretStr/OptionalSecretStr field types
and enrich environment variables with this type information.
"""

import logging
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from nat.workflow_builder_api.models import EnvironmentVariable as ModelEnvironmentVariable

logger = logging.getLogger(__name__)

# Maps component types in YAML to their config class path lookups
COMPONENT_TYPE_TO_REGISTRY_KEY = {
    "llms": "llm",
    "functions": "function",
    "function_groups": "function_group",
    "authentication": "authentication",
    "retrievers": "retriever",
    "embedders": "embedder",
    "memory": "memory",
    "object_stores": "object_store",
    "middleware": "middleware",
    "front_end": "front_end",
    "loggers": "logger",
    "telemetry_exporters": "telemetry_exporter",
}


def get_config_class_for_component(
    component_type: str,
    component_id: str,
    parsed_config: dict[str, Any],
) -> type[BaseModel] | None:
    """
    Get the Pydantic config class for a component from the parsed config.

    Uses the GlobalTypeRegistry to find the config class based on the _type field.

    Args:
        component_type: The YAML section (e.g., 'llms', 'authentication')
        component_id: The component ID (e.g., 'nim_llm', 'my_auth')
        parsed_config: The parsed config dictionary

    Returns:
        The Pydantic config class, or None if not found
    """
    try:
        from nat.cli.type_registry import GlobalTypeRegistry

        registry = GlobalTypeRegistry.get()

        # Get the section from config
        section = parsed_config.get(component_type, {})
        if not section or component_id not in section:
            return None

        component_config = section[component_id]
        if not isinstance(component_config, dict):
            return None

        # Get the _type field
        type_name = component_config.get("_type")
        if not type_name:
            return None

        # Map YAML section to registry lookup
        registry_key = COMPONENT_TYPE_TO_REGISTRY_KEY.get(component_type)
        if not registry_key:
            return None

        # Look up in the registry based on component type
        if registry_key == "llm":
            items = registry.get_registered_llm_providers()
        elif registry_key == "authentication":
            items = registry.get_registered_auth_providers()
        elif registry_key == "function":
            items = registry.get_registered_functions()
        elif registry_key == "function_group":
            items = registry.get_registered_function_groups()
        elif registry_key == "retriever":
            items = registry.get_registered_retriever_providers()
        elif registry_key == "embedder":
            items = registry.get_registered_embedder_providers()
        elif registry_key == "memory":
            items = registry.get_registered_memorys()
        elif registry_key == "object_store":
            items = registry.get_registered_object_stores()
        elif registry_key == "middleware":
            items = registry.get_registered_middleware()
        else:
            return None

        # Find the item with matching local_name
        for item in items:
            if item.local_name == type_name:
                return item.config_type

        return None

    except Exception as e:
        logger.debug("Failed to get config class for %s.%s: %s", component_type, component_id, e)
        return None


def enrich_env_vars_with_field_types(
    env_vars: "list[ModelEnvironmentVariable]",
    parsed_config: dict[str, Any],
) -> "tuple[list[ModelEnvironmentVariable], list[str]]":
    """
    Enrich environment variables with field type information.

    For each env var location, determines if the field is a secret type
    (SerializableSecretStr or OptionalSecretStr) and updates the env var
    metadata accordingly. Also generates warnings for non-secret fields.

    Args:
        env_vars: List of detected environment variables
        parsed_config: The parsed config dictionary (used for type lookup)

    Returns:
        Tuple of (enriched env vars, list of warnings)
    """
    from nat.workflow_builder_api.models import EnvironmentVariable as ModelEnvVar
    from nat.workflow_builder_api.models import EnvVarLocation as ModelLocation
    from nat.workflow_builder_api.utils.schema import get_list_fields
    from nat.workflow_builder_api.utils.schema import get_secret_fields

    warnings: list[str] = []
    enriched_vars: list[ModelEnvVar] = []

    for env_var in env_vars:
        has_secret = False
        has_non_secret = False
        has_list_field = False
        non_secret_locations: list[str] = []

        for location in env_var.locations:
            component_type = location.component_type
            component_id = location.component_id
            field_name = location.field_name

            if not component_type or not component_id:
                # Can't determine field type - assume non-secret
                has_non_secret = True
                non_secret_locations.append(location.path)
                continue

            # Get the config class for this component
            config_class = get_config_class_for_component(component_type, component_id, parsed_config)

            if config_class is None:
                # Can't find config class - assume non-secret
                has_non_secret = True
                non_secret_locations.append(location.path)
                continue

            # Check if this field is a secret type
            secret_fields = get_secret_fields(config_class)
            is_secret = field_name in secret_fields

            # Check if this field is a list type
            list_fields = get_list_fields(config_class)
            is_list = field_name in list_fields

            # Update location
            location.is_secret_field = is_secret
            location.is_list_field = is_list

            if is_secret:
                has_secret = True
            else:
                has_non_secret = True
                non_secret_locations.append(location.path)

            if is_list:
                has_list_field = True

        # Create enriched env var with updated flags
        warning_msg = None
        if has_non_secret and non_secret_locations:
            locations_str = ", ".join(non_secret_locations)
            warning_msg = (f"${{{env_var.name}}} used in non-secret field(s): {locations_str}\n"
                           f"       → Values will be exported as plain text (not SecretStr type)")
            warnings.append(warning_msg)

        # Convert locations to model type with is_secret_field and is_list_field info
        model_locations = [
            ModelLocation(
                path=loc.path,
                component_type=loc.component_type,
                component_id=loc.component_id,
                field_name=loc.field_name,
                is_secret_field=loc.is_secret_field,
                is_list_field=getattr(loc, "is_list_field", False),
            ) for loc in env_var.locations
        ]

        enriched_var = ModelEnvVar(
            name=env_var.name,
            locations=model_locations,
            value=env_var.value,
            is_sensitive=env_var.is_sensitive,
            export_as_variable=env_var.export_as_variable,
            original_reference=env_var.original_reference,
            has_secret_field_usage=has_secret,
            has_non_secret_field_usage=has_non_secret,
            is_list_field=has_list_field,
            warning=warning_msg,
        )
        enriched_vars.append(enriched_var)

    return enriched_vars, warnings
