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
Config exporter utility for converting UI workflow state to YAML configuration.

Converts the workflow builder UI state (components, connections) into a valid
NAT configuration YAML file.
"""

import logging
from typing import Any

import yaml

from nat.cli.type_registry import GlobalTypeRegistry
from nat.workflow_builder_api.constants.categories import CATEGORY_TO_CONFIG_SECTION
from nat.workflow_builder_api.models import ComponentCategory
from nat.workflow_builder_api.models import ExportComponent
from nat.workflow_builder_api.models import ExportConfigResponse
from nat.workflow_builder_api.models import ExportConnection
from nat.workflow_builder_api.models import ExportWorkflowRequest
from nat.workflow_builder_api.models import SecretFieldExportConfig

logger = logging.getLogger(__name__)


def _get_config_section(component_type: str) -> str | None:
    """Get the config section name for a component type using the central mapping."""
    try:
        category = ComponentCategory(component_type)
        return CATEGORY_TO_CONFIG_SECTION.get(category)
    except ValueError:
        return None


# Types that go under nested sections (not top-level)
NESTED_SECTION_TYPES: dict[str, tuple[str, str]] = {
    # (parent_section, child_section)
    "evaluator": ("eval", "evaluators"),
}

# Container component types that are handled specially
CONTAINER_TYPES = {"nat_workflow", "general_config", "evaluation_config", "optimizer_config", "finetuner_config"}

# Component types that should be handled by general_config, not as top-level sections
GENERAL_CONFIG_TYPES = {"front_end", "logger", "telemetry_exporter"}

# Component types that should be handled by evaluation_config
EVAL_CONFIG_TYPES = {"evaluator"}

# Prefixes that should be stripped from component IDs when exporting to sections
# These are added during import to make IDs unique
COMPONENT_ID_PREFIXES: dict[str, str] = {
    "evaluator": "evaluator_",
    "logger": "logger_",
    "telemetry_exporter": "telemetry_",
}

# Fields that are always internal/UI-only and should never be exported
INTERNAL_FIELDS = {
    "_selected_type",
    "_modified_fields",
    "_eval_general",
    "_telemetry_settings",
    "_original_type",
    "_optimizer_settings",
    "_finetuning_settings",
}

# Fields that are connections (component references) - always export these if set
CONNECTION_FIELDS = {
    "llm_name",
    "llm",
    "embedder",
    "embedding_model",
    "retriever",
    "memory",
    "object_store",
    "authentication",
    "middleware",
    "tool_names",
    "tools",
    "function_names",
    "functions",
    "function_groups",
    "loggers",
    "telemetry_exporters",
    "front_ends",
    "evaluators",
    "trainer",
    "trajectory_builder",
    "trainer_adapter",
}

# Fields that are lists of references - values should be wrapped in a list
LIST_CONNECTION_FIELDS = {
    "tool_names",
    "tools",
    "function_names",
    "functions",
    "function_groups",
    "loggers",
    "telemetry_exporters",
    "front_ends",
    "evaluators",
    "middleware",
}

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_workflow_to_yaml(request: ExportWorkflowRequest) -> ExportConfigResponse:
    """
    Export a workflow from the UI to a YAML configuration string.

    Args:
        request: The export request containing components and connections.

    Returns:
        ExportConfigResponse with the YAML content, optional .env file, or error message.
    """
    warnings: list[str] = []
    env_file_content: str | None = None

    try:
        # Build the config dictionary
        config_dict = _build_config_dict(request, warnings)

        # Serialize all values to YAML-compatible types
        config_dict = _serialize_value(config_dict)

        # Apply secret field export configurations
        env_vars = _apply_secret_field_configs(
            config_dict,
            request.secret_field_configs,
            request.components,
        )

        # Generate .env file content if there are env vars to export
        if env_vars:
            env_file_content = _generate_env_file(env_vars)

        # Validate the generated config
        validation_errors = _validate_config(config_dict)
        if validation_errors:
            # Add as warnings but still allow export
            for error in validation_errors:
                warnings.append(f"Validation: {error}")

        # Convert to YAML (without anchors/aliases for cleaner output)
        yaml_content = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
            Dumper=_CleanYamlDumper,
        )

        return ExportConfigResponse(
            success=True,
            yaml_content=yaml_content,
            env_file_content=env_file_content,
            warnings=warnings,
        )

    except Exception as e:
        logger.exception("Error exporting workflow to YAML")
        return ExportConfigResponse(
            success=False,
            error_message=f"Failed to export workflow: {str(e)}",
            warnings=warnings,
        )


class _CleanYamlDumper(yaml.SafeDumper):
    """Custom YAML dumper that avoids anchors and aliases for cleaner output."""

    def ignore_aliases(self, data: Any) -> bool:
        return True

    def represent_data(self, data: Any) -> Any:
        """Override to handle special types that can't be directly serialized."""
        # Handle Pydantic URL types
        if hasattr(data, "__class__"):
            class_name = data.__class__.__name__
            # HttpUrl, AnyUrl, and other Pydantic URL types
            if "Url" in class_name or class_name in ("HttpUrl", "AnyUrl", "AnyHttpUrl"):
                return self.represent_str(str(data))
            # Handle Path objects
            if class_name in ("Path", "PosixPath", "WindowsPath", "PurePath"):
                return self.represent_str(str(data))
            # Handle SecretStr (should show placeholder or be empty)
            if class_name == "SecretStr":
                # Don't export secrets - they should be environment variables
                secret_val = data.get_secret_value() if hasattr(data, "get_secret_value") else str(data)
                return self.represent_str(secret_val)
            # Handle SerializableSecretStr
            if "SecretStr" in class_name:
                if hasattr(data, "get_secret_value"):
                    return self.represent_str(data.get_secret_value())
                return self.represent_str(str(data))

        return super().represent_data(data)


# =============================================================================
# CONFIG BUILDING
# =============================================================================


def _serialize_value(value: Any) -> Any:
    """
    Recursively serialize a value to YAML-compatible types.

    Handles Pydantic types like HttpUrl, Path, SecretStr, Enums, etc.
    """
    from enum import Enum

    if value is None:
        return None

    # Handle Enums first (convert to their value)
    if isinstance(value, Enum):
        return value.value

    # Handle string subclasses (like LLMRef, EmbedderRef, ComponentRef) early
    # to prevent them from hitting the fallback path
    if isinstance(value, str):
        return str(value)  # Convert any str subclass to plain str

    # Get class name for type checking
    class_name = value.__class__.__name__ if hasattr(value, "__class__") else ""

    # Handle Pydantic URL types
    if "Url" in class_name or class_name in ("HttpUrl", "AnyUrl", "AnyHttpUrl"):
        return str(value)

    # Handle Path objects
    if class_name in ("Path", "PosixPath", "WindowsPath", "PurePath"):
        return str(value)

    # Handle SecretStr types
    if "SecretStr" in class_name:
        if hasattr(value, "get_secret_value"):
            return value.get_secret_value()
        return str(value)

    # Handle dicts - recurse (serialize keys too for safety)
    if isinstance(value, dict):
        return {_serialize_value(k): _serialize_value(v) for k, v in value.items()}

    # Handle lists/tuples - recurse
    if isinstance(value, list | tuple):
        return [_serialize_value(v) for v in value]

    # Handle sets - convert to list
    if isinstance(value, set):
        return [_serialize_value(v) for v in value]

    # Handle frozenset
    if isinstance(value, frozenset):
        return [_serialize_value(v) for v in value]

    # Handle Pydantic models - convert to dict
    if hasattr(value, "model_dump"):
        return _serialize_value(value.model_dump())

    # Basic numeric/bool types pass through
    if isinstance(value, int | float | bool):
        return value

    # Fallback - try string conversion for unknown types
    try:
        return str(value)
    except Exception:
        return repr(value)


def _build_config_dict(request: ExportWorkflowRequest, warnings: list[str]) -> dict[str, Any]:
    """
    Build the config dictionary from the workflow request.

    Args:
        request: The export request.
        warnings: List to append warnings to.

    Returns:
        Dictionary that can be serialized to YAML.
    """
    config: dict[str, Any] = {}

    # Build connection lookup: target_id -> {field_name: source_id(s)}
    connections_by_target = _build_connections_lookup(request.connections)

    # First pass: identify components used as workflow entrypoint or general_config items
    # These should NOT be added to their normal sections
    workflow_component: ExportComponent | None = None
    general_config_component: ExportComponent | None = None
    excluded_from_sections: set[str] = set()

    for component in request.components:
        if component.component_type == "nat_workflow":
            workflow_component = component
            # The entrypoint function should be excluded from functions section
            connections = connections_by_target.get(component.id, {})
            entrypoint_id = connections.get("entrypoint")
            if entrypoint_id:
                if isinstance(entrypoint_id, list):
                    excluded_from_sections.update(entrypoint_id)
                else:
                    excluded_from_sections.add(entrypoint_id)
        elif component.component_type == "general_config":
            general_config_component = component
            # Front-end, loggers, telemetry should be excluded from top-level
            connections = connections_by_target.get(component.id, {})
            for field in ["front_ends", "loggers", "telemetry_exporters"]:
                ids = connections.get(field)
                if ids:
                    if isinstance(ids, list):
                        excluded_from_sections.update(ids)
                    else:
                        excluded_from_sections.add(ids)

    # Second pass: add components to their sections
    for component in request.components:
        # Skip container types
        if component.component_type in CONTAINER_TYPES:
            continue

        # Skip components handled by general_config
        if component.component_type in GENERAL_CONFIG_TYPES:
            continue

        # Skip components that are excluded (workflow entrypoint, etc.)
        if component.id in excluded_from_sections:
            continue

        _add_component_to_config(component, config, connections_by_target, warnings)

    # Handle workflow entrypoint
    if workflow_component:
        _handle_workflow_component(workflow_component, config, connections_by_target, request.components, warnings)

    # Handle general config (front_end, logging, tracing)
    if general_config_component:
        _handle_general_config(general_config_component, config, connections_by_target, request.components, warnings)

    # Handle evaluation_config (eval.general settings)
    for component in request.components:
        if component.component_type == "evaluation_config":
            _handle_evaluation_config(component, config, warnings)
            break

    # Handle optimizer_config (optimizer settings)
    for component in request.components:
        if component.component_type == "optimizer_config":
            _handle_optimizer_config(component, config, warnings)
            break

    # Handle finetuner_config (finetuning settings)
    for component in request.components:
        if component.component_type == "finetuner_config":
            _handle_finetuner_config(component, config, warnings)
            break

    return config


def _build_connections_lookup(connections: list[ExportConnection]) -> dict[str, dict[str, str | list[str]]]:
    """
    Build a lookup dictionary from connections.

    Returns:
        Dict mapping target_id to {field_name: source_id or list of source_ids}
    """
    lookup: dict[str, dict[str, str | list[str]]] = {}
    for conn in connections:
        if conn.target_id not in lookup:
            lookup[conn.target_id] = {}

        existing = lookup[conn.target_id].get(conn.target_field)
        if existing is not None:
            # Multiple connections to same field - make it a list
            if isinstance(existing, list):
                existing.append(conn.source_id)
            else:
                lookup[conn.target_id][conn.target_field] = [existing, conn.source_id]
        else:
            lookup[conn.target_id][conn.target_field] = conn.source_id

    return lookup


def _add_component_to_config(
    component: ExportComponent,
    config: dict[str, Any],
    connections_by_target: dict[str, dict[str, str | list[str]]],
    warnings: list[str],
) -> None:
    """
    Add a component to the appropriate config section.

    Args:
        component: The component to add.
        config: The config dictionary to modify.
        connections_by_target: Connection lookup.
        warnings: List to append warnings to.
    """
    # Check for nested section types first (e.g., evaluators under eval.evaluators)
    nested_info = NESTED_SECTION_TYPES.get(component.component_type)
    if nested_info:
        parent_section, child_section = nested_info
        if parent_section not in config:
            config[parent_section] = {}
        if child_section not in config[parent_section]:
            config[parent_section][child_section] = {}

        component_config = _build_component_config(component, connections_by_target)
        component_name = _get_clean_component_name(component.id, component.component_type)
        config[parent_section][child_section][component_name] = component_config
        return

    # Regular top-level sections
    section = _get_config_section(component.component_type)
    if not section:
        warnings.append(f"Unknown component type '{component.component_type}' for component '{component.id}'")
        return

    # Initialize section if needed
    if section not in config:
        config[section] = {}

    # Build component config with only non-default fields
    component_config = _build_component_config(component, connections_by_target)

    # Get the clean component name by stripping any import-time prefixes
    component_name = _get_clean_component_name(component.id, component.component_type)

    # Add to section
    config[section][component_name] = component_config


def _get_clean_component_name(component_id: str, component_type: str) -> str:
    """
    Get the clean component name by stripping import-time prefixes.

    During import, some components get prefixes added (e.g., evaluator_accuracy).
    When exporting, we need to strip these to get back the original names.

    Args:
        component_id: The component's ID which may have a prefix.
        component_type: The component type (e.g., 'evaluator').

    Returns:
        The clean name without the prefix.
    """
    prefix = COMPONENT_ID_PREFIXES.get(component_type)
    if prefix and component_id.startswith(prefix):
        return component_id[len(prefix):]
    return component_id


def _build_component_config(
    component: ExportComponent,
    connections_by_target: dict[str, dict[str, str | list[str]]],
) -> dict[str, Any]:
    """
    Build the configuration dictionary for a single component.

    Includes all fields from the component's config (which were explicitly set
    during import via exclude_unset=True), plus the _type field.

    Note: We do NOT apply alias transformations here because we don't know
    which form (field name vs alias) was used in the original YAML.
    Pydantic normalizes aliases to field names during parsing, so we export
    using the canonical field names.

    Args:
        component: The component to build config for.
        connections_by_target: Connection lookup.

    Returns:
        Component configuration dictionary ready for YAML export.
    """
    component_config: dict[str, Any] = {}

    # Use the original type value if available, otherwise extract from full_type
    original_type = component.config.get("_original_type")
    if original_type:
        component_config["_type"] = original_type
    else:
        type_name = _extract_type_name(component.full_type)
        if type_name:
            component_config["_type"] = type_name

    # Add all fields from the component's config
    # The import phase already filters to only explicitly set fields (exclude_unset=True)
    # so we should export all of them without comparing to defaults
    for field_name, value in component.config.items():
        # Skip internal fields
        if field_name in INTERNAL_FIELDS:
            continue

        # Skip discriminator fields since we add _type above
        if field_name in ("type", "_type"):
            continue

        # Serialize value to YAML-compatible types
        serialized_value = _serialize_value(value)

        if serialized_value is not None:
            component_config[field_name] = serialized_value

    # Apply connections - these override any existing values
    connections = connections_by_target.get(component.id, {})
    for field_name, source_id in connections.items():
        # Ensure list fields are properly formatted as lists
        if field_name in LIST_CONNECTION_FIELDS:
            if isinstance(source_id, list):
                component_config[field_name] = source_id
            else:
                component_config[field_name] = [source_id]
        else:
            component_config[field_name] = source_id

    return component_config


def _get_component_defaults(full_type: str) -> dict[str, Any]:
    """
    Get default values for a component type from its JSON schema.

    Args:
        full_type: Full type string (e.g., "nim/NIMModelConfig")

    Returns:
        Dictionary of field_name -> default_value
    """
    defaults: dict[str, Any] = {}

    try:
        registry = GlobalTypeRegistry.get()

        # Parse full_type to find the registered type
        if "/" not in full_type:
            return defaults

        # Try to find the type in the registry
        type_name = full_type.split("/")[-1]

        # Search through all registered types to find a match
        for getter in [
                registry.get_registered_llm_providers,
                registry.get_registered_embedder_providers,
                registry.get_registered_functions,
                registry.get_registered_function_groups,
                registry.get_registered_retriever_providers,
                registry.get_registered_memorys,
                registry.get_registered_object_stores,
                registry.get_registered_auth_providers,
                registry.get_registered_middleware,
                registry.get_registered_front_ends,
                registry.get_registered_logging_method,
                registry.get_registered_telemetry_exporters,
                registry.get_registered_evaluators,
                registry.get_registered_trainers,
                registry.get_registered_trajectory_builders,
                registry.get_registered_trainer_adapters,
                registry.get_registered_ttc_strategies,
        ]:
            try:
                for registered_info in getter():
                    if registered_info.full_type == full_type or registered_info.local_name == type_name:
                        # Found the type - extract defaults from its model
                        config_type = registered_info.config_type
                        for field_name, field_info in config_type.model_fields.items():
                            if field_info.default is not None:
                                defaults[field_name] = field_info.default
                            elif field_info.default_factory is not None:
                                try:
                                    defaults[field_name] = field_info.default_factory()
                                except Exception:
                                    pass
                        return defaults
            except Exception:
                continue

    except Exception as e:
        logger.debug("Could not get defaults for %s: %s", full_type, e)

    return defaults


def _values_equal(value1: Any, value2: Any) -> bool:
    """
    Check if two values are equal, handling special cases.

    Args:
        value1: First value
        value2: Second value

    Returns:
        True if values are considered equal
    """
    # Handle None comparison
    if value1 is None and value2 is None:
        return True
    if value1 is None or value2 is None:
        return False

    # Handle empty collections
    if isinstance(value1, list | dict) and isinstance(value2, list | dict):
        if len(value1) == 0 and len(value2) == 0:
            return True

    # Standard equality
    try:
        return value1 == value2
    except Exception:
        return False


def _extract_type_name(full_type: str) -> str | None:
    """
    Extract the type name from a full_type string.

    For most types, this returns the short name after the slash.
    For external plugins (those with package paths like nat.plugins.xxx/),
    we return the full path to preserve compatibility with configs that use it.

    Examples:
        "nim/NIMModelConfig" -> "nim"
        "nat.llm/litellm" -> "litellm"
        "nat.plugins.redis/redis_memory" -> "nat.plugins.redis/redis_memory"
    """
    if "/" not in full_type:
        return None

    # Check if this is an external plugin (package path before the slash)
    prefix = full_type.split("/")[0]
    if "." in prefix and prefix.startswith("nat.plugins."):
        # External plugin - preserve full path
        return full_type
    else:
        # Built-in type - use short name
        return full_type.split("/")[-1]


def _handle_workflow_component(
    workflow_component: ExportComponent,
    config: dict[str, Any],
    connections_by_target: dict[str, dict[str, str | list[str]]],
    all_components: list[ExportComponent],
    warnings: list[str],
) -> None:
    """
    Handle the workflow component to set the workflow entrypoint.

    The workflow component connects to a function/agent via 'entrypoint'.
    """
    connections = connections_by_target.get(workflow_component.id, {})
    entrypoint_id = connections.get("entrypoint")

    if entrypoint_id:
        if isinstance(entrypoint_id, list):
            entrypoint_id = entrypoint_id[0]  # Use first if multiple

        # Find the entrypoint component
        entrypoint_component = next((c for c in all_components if c.id == entrypoint_id), None)
        if entrypoint_component:
            # Build the workflow config from the entrypoint function
            workflow_config = _build_component_config(entrypoint_component, connections_by_target)
            config["workflow"] = workflow_config
        else:
            warnings.append(f"Entrypoint component '{entrypoint_id}' not found")
    else:
        warnings.append("No entrypoint connected to workflow")


def _handle_general_config(
    general_component: ExportComponent,
    config: dict[str, Any],
    connections_by_target: dict[str, dict[str, str | list[str]]],
    all_components: list[ExportComponent],
    _warnings: list[str],
) -> None:
    """
    Handle the general config component for front_end, logging, tracing.

    Structure should be:
    - general.front_end: single FrontEndConfig
    - general.telemetry.logging: dict of logger name -> logger config
    - general.telemetry.tracing: dict of tracer name -> tracer config
    """
    connections = connections_by_target.get(general_component.id, {})

    # Handle front_end - goes in general.front_end
    front_end_ids = connections.get("front_ends") or connections.get("front_end")
    if front_end_ids:
        if isinstance(front_end_ids, list) and len(front_end_ids) > 0:
            front_end_id = front_end_ids[0]
        else:
            front_end_id = front_end_ids

        front_end_component = next((c for c in all_components if c.id == front_end_id), None)
        if front_end_component:
            if "general" not in config:
                config["general"] = {}
            config["general"]["front_end"] = _build_component_config(front_end_component, connections_by_target)

    # Handle loggers - goes in general.telemetry.logging
    logger_ids = connections.get("loggers")
    if logger_ids:
        if not isinstance(logger_ids, list):
            logger_ids = [logger_ids]
        for logger_id in logger_ids:
            logger_component = next((c for c in all_components if c.id == logger_id), None)
            if logger_component:
                if "general" not in config:
                    config["general"] = {}
                if "telemetry" not in config["general"]:
                    config["general"]["telemetry"] = {}
                if "logging" not in config["general"]["telemetry"]:
                    config["general"]["telemetry"]["logging"] = {}
                # Strip logger_ prefix from name
                logger_name = _get_clean_component_name(logger_id, "logger")
                config["general"]["telemetry"]["logging"][logger_name] = _build_component_config(
                    logger_component, connections_by_target)

    # Handle telemetry exporters - goes in general.telemetry.tracing
    telemetry_ids = connections.get("telemetry_exporters")
    if telemetry_ids:
        if not isinstance(telemetry_ids, list):
            telemetry_ids = [telemetry_ids]
        for telemetry_id in telemetry_ids:
            telemetry_component = next((c for c in all_components if c.id == telemetry_id), None)
            if telemetry_component:
                if "general" not in config:
                    config["general"] = {}
                if "telemetry" not in config["general"]:
                    config["general"]["telemetry"] = {}
                if "tracing" not in config["general"]["telemetry"]:
                    config["general"]["telemetry"]["tracing"] = {}
                # Strip telemetry_ prefix from name
                tracer_name = _get_clean_component_name(telemetry_id, "telemetry_exporter")
                config["general"]["telemetry"]["tracing"][tracer_name] = _build_component_config(
                    telemetry_component, connections_by_target)

    # Handle telemetry settings (enabled, etc.) from general_config component
    telemetry_settings = general_component.config.get("_telemetry_settings")
    if telemetry_settings and isinstance(telemetry_settings, dict):
        if "general" not in config:
            config["general"] = {}
        if "telemetry" not in config["general"]:
            config["general"]["telemetry"] = {}
        # Merge telemetry settings (like enabled) into the telemetry section
        config["general"]["telemetry"].update(telemetry_settings)


def _handle_evaluation_config(
    evaluation_component: ExportComponent,
    config: dict[str, Any],
    _warnings: list[str],
) -> None:
    """
    Handle the evaluation_config component to export eval.general settings.

    During import, eval.general settings are stored in the component's config
    under the key '_eval_general'. This function extracts them and places them
    in the correct config location.
    """
    eval_general = evaluation_component.config.get("_eval_general")
    if eval_general and isinstance(eval_general, dict):
        # Ensure eval section exists
        if "eval" not in config:
            config["eval"] = {}

        # Add general settings
        config["eval"]["general"] = eval_general


def _handle_optimizer_config(
    optimizer_component: ExportComponent,
    config: dict[str, Any],
    _warnings: list[str],
) -> None:
    """
    Handle the optimizer_config component to export optimizer settings.

    During import, optimizer settings are stored in the component's config
    under the key '_optimizer_settings'. This function extracts them and places them
    in the correct config location (top-level 'optimizer' section).
    """
    optimizer_settings = optimizer_component.config.get("_optimizer_settings")
    if optimizer_settings and isinstance(optimizer_settings, dict):
        # Serialize to ensure YAML-compatible types
        config["optimizer"] = _serialize_value(optimizer_settings)


def _get_preferred_alias_mappings(model_class: type) -> dict[str, str]:
    """
    Extract preferred alias mappings from a Pydantic model's validation_alias fields.

    For fields with AliasChoices, this returns a mapping from the actual field name
    to the preferred (shorter) alias. The preferred alias is the last choice that
    differs from the field name.

    Args:
        model_class: A Pydantic model class to introspect.

    Returns:
        A dictionary mapping field names to their preferred aliases.
    """
    from pydantic import AliasChoices

    mappings = {}
    for field_name, field_info in model_class.model_fields.items():
        if field_info.validation_alias and isinstance(field_info.validation_alias, AliasChoices):
            choices = field_info.validation_alias.choices
            # Find the preferred alias (a choice that differs from the field name)
            for choice in choices:
                if choice != field_name:
                    mappings[field_name] = choice
                    break
    return mappings


def _apply_alias_mappings(data: dict[str, Any], mappings: dict[str, str]) -> dict[str, Any]:
    """
    Transform dictionary keys using alias mappings.

    Args:
        data: The dictionary to transform.
        mappings: A mapping from original keys to preferred keys.

    Returns:
        A new dictionary with transformed keys.
    """
    return {mappings.get(key, key): value for key, value in data.items()}


def _handle_finetuner_config(
    finetuner_component: ExportComponent,
    config: dict[str, Any],
    _warnings: list[str],
) -> None:
    """
    Handle the finetuner_config component to export finetuning settings.

    During import, finetuning settings are stored in the component's config
    under the key '_finetuning_settings'. This function extracts them and places them
    in the correct config location (top-level 'finetuning' section).

    The FinetuneConfig model uses validation_alias to accept both forms:
    - trainer_name / trainer
    - trajectory_builder_name / trajectory_builder
    - etc.

    We export using the shorter alias form to match the common YAML convention.
    """
    finetuning_settings = finetuner_component.config.get("_finetuning_settings")
    if finetuning_settings and isinstance(finetuning_settings, dict):
        from nat.data_models.finetuning import FinetuneConfig

        # Dynamically extract alias mappings from the Pydantic model
        alias_mappings = _get_preferred_alias_mappings(FinetuneConfig)

        # Transform field names to use the alias form
        transformed_settings = _apply_alias_mappings(finetuning_settings, alias_mappings)

        # Serialize to ensure YAML-compatible types
        config["finetuning"] = _serialize_value(transformed_settings)


# =============================================================================
# SECRET FIELD HANDLING
# =============================================================================


def _apply_secret_field_configs(
    config_dict: dict[str, Any],
    secret_configs: list[SecretFieldExportConfig],
    components: list[ExportComponent],
) -> dict[str, str]:
    """
    Apply secret field export configurations to the config dictionary.

    For each secret field configured to export as env_var, replace the value
    in the config with ${ENV_VAR_NAME} and collect the actual values for .env.

    Args:
        config_dict: The config dictionary to modify in place.
        secret_configs: List of secret field export configurations.
        components: Original components for finding component locations.

    Returns:
        Dictionary mapping env var names to their values (for .env file).
    """
    env_vars: dict[str, str] = {}

    for secret_config in secret_configs:
        if secret_config.export_mode == "plain":
            # Plain text export - do nothing, value stays as-is
            continue

        # Export as environment variable
        component_id = secret_config.component_id
        field_name = secret_config.field_name
        env_var_name = secret_config.env_var_name
        value = secret_config.value

        if not env_var_name or not value:
            continue

        # Find the component to determine its section
        component = next((c for c in components if c.id == component_id), None)
        if not component:
            continue

        # Navigate to the correct location in config_dict and replace the value
        section = _get_config_section(component.component_type)
        if not section:
            continue

        # Handle special cases like workflow entrypoint
        if section == "workflow" or component.component_type in CONTAINER_TYPES:
            continue  # Skip workflow-level configs for now

        # Navigate to the field and replace with env var reference
        if section in config_dict and component_id in config_dict[section]:
            if field_name in config_dict[section][component_id]:
                config_dict[section][component_id][field_name] = f"${{{env_var_name}}}"
                env_vars[env_var_name] = value

    return env_vars


def _generate_env_file(env_vars: dict[str, str]) -> str:
    """
    Generate a .env file content from environment variable mappings.

    Args:
        env_vars: Dictionary mapping env var names to their values.

    Returns:
        String content for a .env file.
    """
    lines = [
        "# Environment variables for NAT workflow configuration",
        "# Generated by NAT Workflow Builder",
        "",
    ]

    for name, value in sorted(env_vars.items()):
        # Escape special characters in value
        escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
        # Quote values that contain spaces or special characters
        if " " in value or "\n" in value or '"' in value:
            lines.append(f'{name}="{escaped_value}"')
        else:
            lines.append(f"{name}={escaped_value}")

    lines.append("")  # Trailing newline
    return "\n".join(lines)


# =============================================================================
# VALIDATION
# =============================================================================


def _validate_config(config_dict: dict[str, Any]) -> list[str]:
    """
    Validate the generated config against the NAT Config schema.

    Args:
        config_dict: The generated config dictionary.

    Returns:
        List of validation error messages (empty if valid).
    """
    from pydantic import ValidationError

    from nat.data_models.config import Config
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    errors: list[str] = []

    try:
        # Ensure plugins are loaded
        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
        Config.rebuild_annotations()

        # Try to validate
        Config.model_validate(config_dict)

    except ValidationError as e:
        # Extract user-friendly error messages from Pydantic
        for error in e.errors():
            location = " → ".join(str(x) for x in error.get("loc", []))
            message = error.get("msg", "Unknown error")
            error_type = error.get("type", "")

            # Format nice error message
            if error_type == "extra_forbidden":
                # Parse the input value to show what was unexpected
                ctx = error.get("ctx", {})
                errors.append(f"Unexpected field at '{location}': {message}")
            elif error_type == "union_tag_invalid":
                # Missing or invalid type
                ctx = error.get("ctx", {})
                tag = ctx.get("tag", "unknown")
                errors.append(f"Unknown type '{tag}' at '{location}'. Make sure the required plugin is installed.")
            elif error_type == "missing":
                errors.append(f"Missing required field '{location}'")
            elif location:
                errors.append(f"{location}: {message}")
            else:
                errors.append(message)

    except Exception as e:
        # Generic error handling
        error_str = str(e)
        if "validation error" in error_str.lower():
            # Parse Pydantic validation errors from string
            lines = error_str.split("\n")
            for line in lines[1:]:  # Skip first line which is just "X validation errors"
                line = line.strip()
                if line and not line.startswith("For further"):
                    errors.append(line)
        else:
            errors.append(f"Validation failed: {error_str}")

    return errors
