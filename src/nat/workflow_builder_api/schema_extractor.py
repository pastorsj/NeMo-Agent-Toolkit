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
Schema extraction utilities for Pydantic models.

This module handles extraction of JSON schemas from NAT Pydantic models,
including special handling for ComponentRef types (LLMRef, FunctionRef, etc.)
that use custom validators not directly compatible with JSON schema.
"""

import logging
import re
import types
import typing
from typing import Any
from typing import get_args
from typing import get_origin

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from nat.workflow_builder_api.models import ConnectionPort
from nat.workflow_builder_api.models import FieldInfo
from nat.workflow_builder_api.models import RefType

logger = logging.getLogger(__name__)

# Pattern to match markdown image syntax for icons: ![Icon](URL)
_ICON_PATTERN = re.compile(r'!\[Icon\]\(([^)]+)\)', re.IGNORECASE)

# Mapping from ComponentRef class names to RefType enum
COMPONENT_REF_TO_REF_TYPE: dict[str, RefType] = {
    "LLMRef": RefType.LLM,
    "EmbedderRef": RefType.EMBEDDER,
    "FunctionRef": RefType.FUNCTION,
    "FunctionGroupRef": RefType.FUNCTION_GROUP,
    "RetrieverRef": RefType.RETRIEVER,
    "MemoryRef": RefType.MEMORY,
    "ObjectStoreRef": RefType.OBJECT_STORE,
    "AuthenticationRef": RefType.AUTHENTICATION,
    "MiddlewareRef": RefType.MIDDLEWARE,
}

# Mapping from SDK class names to RefType enum
# These are the Nat* SDK classes that represent component dependencies
SDK_CLASS_TO_REF_TYPE: dict[str, RefType] = {
    "NatLLM": RefType.LLM,
    "NatEmbedder": RefType.EMBEDDER,
    "NatFunction": RefType.FUNCTION,
    "NatFunctionGroup": RefType.FUNCTION_GROUP,
    "NatAgent": RefType.FUNCTION,  # Agents can be used as functions/tools
    "NatRetriever": RefType.RETRIEVER,
    "NatMemory": RefType.MEMORY,
    "NatObjectStore": RefType.OBJECT_STORE,
    "NatAuthProvider": RefType.AUTHENTICATION,
    "NatMiddleware": RefType.MIDDLEWARE,  # Front-end and observability
    "NatFrontEnd": RefType.FRONT_END,
    "NatLogger": RefType.LOGGER,
    "NatTelemetryExporter": RefType.TELEMETRY_EXPORTER,  # Evaluation
    "NatEvaluator": RefType.EVALUATOR,  # Workflow configuration containers (for NatWorkflow connections)
    "NatGeneralConfiguration": RefType.GENERAL_CONFIG,
    "NatEvaluation": RefType.EVALUATION_CONFIG,
    "NatOptimizer": RefType.OPTIMIZER_CONFIG,
    "NatFinetuner": RefType.FINETUNER_CONFIG,  # Finetuning components
    "NatTrainer": RefType.TRAINER,
    "NatTrajectoryBuilder": RefType.TRAJECTORY_BUILDER,
    "NatTrainerAdapter": RefType.TRAINER_ADAPTER,
}


def _get_type_name(annotation: Any) -> str:
    """Get the type name from an annotation, handling various type forms."""
    if annotation is None:
        return ""
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _get_all_component_ref_types(annotation: Any) -> tuple[bool, list[RefType], bool]:
    """
    Check if an annotation is a ComponentRef type and return ALL ref types it accepts.

    This handles union types like `FunctionRef | FunctionGroupRef` and lists of unions
    like `list[FunctionRef | FunctionGroupRef]`.

    Args:
        annotation: The type annotation to check.

    Returns:
        Tuple of (is_ref, ref_types, is_list) where ref_types is a list of all accepted types
    """
    if annotation is None:
        return False, [], False

    type_name = _get_type_name(annotation)

    # Direct ComponentRef type (e.g., LLMRef)
    if type_name in COMPONENT_REF_TO_REF_TYPE:
        return True, [COMPONENT_REF_TO_REF_TYPE[type_name]], False

    # Direct SDK type (e.g., NatLLM)
    if type_name in SDK_CLASS_TO_REF_TYPE:
        return True, [SDK_CLASS_TO_REF_TYPE[type_name]], False

    # Check for list/sequence of ComponentRef or SDK types
    origin = get_origin(annotation)
    if origin in (list, list, typing.Sequence):
        args = get_args(annotation)
        if args:
            # Check for Union types inside the list (e.g., list[NatFunction | NatAgent])
            # Python 3.10+ uses types.UnionType for X | Y syntax
            if len(args) == 1:
                inner_origin = get_origin(args[0])
                # Check for both typing.Union and types.UnionType (Python 3.10+ | syntax)
                is_union = inner_origin is typing.Union or isinstance(args[0], types.UnionType)
                if is_union:
                    inner_args = get_args(args[0])
                    ref_types = []
                    for inner_arg in inner_args:
                        inner_type_name = _get_type_name(inner_arg)
                        if inner_type_name in SDK_CLASS_TO_REF_TYPE:
                            ref_types.append(SDK_CLASS_TO_REF_TYPE[inner_type_name])
                        elif inner_type_name in COMPONENT_REF_TO_REF_TYPE:
                            ref_types.append(COMPONENT_REF_TO_REF_TYPE[inner_type_name])
                    if ref_types:
                        return True, ref_types, True

            # Single type in list
            for arg in args:
                inner_type_name = _get_type_name(arg)
                if inner_type_name in COMPONENT_REF_TO_REF_TYPE:
                    return True, [COMPONENT_REF_TO_REF_TYPE[inner_type_name]], True
                if inner_type_name in SDK_CLASS_TO_REF_TYPE:
                    return True, [SDK_CLASS_TO_REF_TYPE[inner_type_name]], True

    # Check for Optional/Union types (not inside list)
    # Python 3.10+ uses types.UnionType for X | Y syntax
    is_union = origin is typing.Union or isinstance(annotation, types.UnionType)
    if is_union:
        args = get_args(annotation)
        ref_types = []
        for arg in args:
            if arg is type(None):
                continue
            is_ref, inner_ref_types, is_list = _get_all_component_ref_types(arg)
            if is_ref:
                ref_types.extend(inner_ref_types)
        if ref_types:
            return True, ref_types, False

    return False, [], False


def _is_component_ref_type(annotation: Any) -> tuple[bool, RefType | None, bool]:
    """
    Check if an annotation is a ComponentRef type, SDK type, or a list of them.

    Returns:
        Tuple of (is_ref, ref_type, is_list) where ref_type is the primary type
    """
    is_ref, ref_types, is_list = _get_all_component_ref_types(annotation)
    primary_ref_type = ref_types[0] if ref_types else None
    return is_ref, primary_ref_type, is_list


def extract_connection_ports(config_type: type[BaseModel]) -> list[ConnectionPort]:
    """
    Extract connection port information from a Pydantic model.

    This inspects the model's fields to find any that are ComponentRef types
    (LLMRef, EmbedderRef, FunctionRef, etc.) and creates ConnectionPort objects
    for each one.

    Args:
        config_type: The Pydantic model class.

    Returns:
        List of ConnectionPort objects.
    """
    ports = []

    for field_name, field_info in config_type.model_fields.items():
        if field_name.startswith("_"):
            continue

        annotation = field_info.annotation
        is_ref, all_ref_types, is_list = _get_all_component_ref_types(annotation)

        if is_ref and all_ref_types:
            primary_ref_type = all_ref_types[0]

            # Determine if required
            is_required = field_info.is_required()

            # Get description
            description = field_info.description

            # Create human-readable title
            title = field_name.replace("_", " ").title()

            ports.append(
                ConnectionPort(
                    field_name=field_name,
                    ref_type=primary_ref_type,
                    accepts_ref_types=all_ref_types,
                    required=is_required,
                    is_list=is_list,
                    description=description,
                    title=title,
                ))

    return ports


def find_sdk_class_for_config(config_type: type[BaseModel]) -> type | None:
    """
    Find the SDK class that corresponds to a config type.

    For example, given ReActAgentWorkflowConfig, find NatReActAgent.
    SDK classes inherit from both the config type and a NatBase subclass.

    Args:
        config_type: The config type to find SDK class for.

    Returns:
        The SDK class if found, None otherwise.
    """
    # Get all subclasses of the config type
    try:
        for subclass in config_type.__subclasses__():
            # SDK classes typically have "Nat" prefix and inherit from NatBase
            class_name = subclass.__name__
            if class_name.startswith("Nat"):
                return subclass
            # Also check if class name contains "Nat" somewhere
            # (for classes like NatReActAgent which might not start with Nat)
            for parent in subclass.__mro__:
                parent_name = parent.__name__
                if parent_name.startswith("Nat") and parent_name not in ("NatBase", "NatFunction", "NatAgent"):
                    return subclass
    except Exception:
        pass
    return None


def extract_sdk_connection_ports(sdk_class: type) -> list[ConnectionPort]:
    """
    Extract connection ports from an SDK class.

    SDK classes have fields like `llm: NatLLM` that represent component dependencies.
    These are different from config fields - they expect actual component instances.

    Args:
        sdk_class: The SDK class (e.g., NatReActAgent).

    Returns:
        List of ConnectionPort objects for SDK-specific dependencies.
    """
    ports = []

    if not hasattr(sdk_class, "model_fields"):
        return ports

    for field_name, field_info in sdk_class.model_fields.items():
        # Skip private fields and ref fields (those are handled by config)
        if field_name.startswith("_"):
            continue
        if field_name.endswith("_name") or field_name.endswith("_names"):
            continue

        annotation = field_info.annotation
        is_ref, all_ref_types, is_list = _get_all_component_ref_types(annotation)

        if is_ref and all_ref_types:
            primary_ref_type = all_ref_types[0]
            # Check if there's already a port for this ref type from the config
            # SDK fields like `llm: NatLLM` map to config fields like `llm_name: LLMRef`

            # Determine if required - SDK fields are typically required if no default
            is_required = field_info.is_required()

            # Get description
            description = field_info.description
            if not description:
                # Generate a description based on the field type
                type_name = _get_type_name(annotation)
                if len(all_ref_types) > 1:
                    type_names = " or ".join(rt.value.replace("_", " ").title() for rt in all_ref_types)
                    description = f"{type_names} component to use"
                else:
                    description = f"{type_name.replace('Nat', '')} component to use"

            # Create human-readable title
            title = field_name.replace("_", " ").title()

            ports.append(
                ConnectionPort(
                    field_name=field_name,
                    ref_type=primary_ref_type,
                    accepts_ref_types=all_ref_types,
                    required=is_required,
                    is_list=is_list,
                    description=description,
                    title=title,
                ))

    return ports


def extract_all_connection_ports(config_type: type[BaseModel]) -> list[ConnectionPort]:
    """
    Extract all connection ports from both config type and its SDK class.

    This combines ports from:
    1. Config fields with ComponentRef types (LLMRef, FunctionRef, etc.)
    2. SDK class fields with Nat* types (NatLLM, NatFunction, etc.)

    Args:
        config_type: The config type to extract ports from.

    Returns:
        Deduplicated list of ConnectionPort objects.
    """
    # Get ports from config (ComponentRef types)
    config_ports = extract_connection_ports(config_type)

    # Try to find and get ports from SDK class
    sdk_class = find_sdk_class_for_config(config_type)
    if sdk_class:
        sdk_ports = extract_sdk_connection_ports(sdk_class)

        # Merge SDK ports with config ports, preferring SDK field names
        # since they're more user-friendly (e.g., "llm" vs "llm_name")

        for sdk_port in sdk_ports:
            # Only add SDK port if we don't already have a port for this ref type
            # from the config, OR if the SDK port provides more info
            existing = next((p for p in config_ports if p.ref_type == sdk_port.ref_type), None)
            if existing:
                # Replace with SDK port if it has a better field name
                if not existing.field_name.endswith("_name") and not existing.field_name.endswith("_names"):
                    continue
                config_ports.remove(existing)
            config_ports.append(sdk_port)

    return config_ports


class NATJsonSchemaGenerator(GenerateJsonSchema):
    """
    Custom JSON schema generator that handles NAT-specific types.

    ComponentRef types (LLMRef, FunctionRef, EmbedderRef, etc.) use
    plain validator functions that can't be converted to JSON schema.
    We handle these by generating a simple string schema with a description.
    """

    def no_info_plain_validator_function_schema(self, schema: CoreSchema) -> JsonSchemaValue:
        """
        Handle plain validator functions (used by ComponentRef types).

        These are string references to other components in the workflow,
        so we represent them as strings with a descriptive title.
        """
        # Get the validator function/class
        validator = schema.get("function")

        if validator is not None:
            # Get the class name for a better description
            validator_name = getattr(validator, "__name__", str(validator))

            # Check if this is a ComponentRef type
            if "Ref" in validator_name:
                # Extract the component type (e.g., "LLM" from "LLMRef")
                component_type = validator_name.replace("Ref", "")
                ref_type = COMPONENT_REF_TO_REF_TYPE.get(validator_name)

                return {
                    "type": "string",
                    "title": f"{component_type} Reference",
                    "description": f"Reference name of a {component_type} component defined in the workflow",
                    # Custom extension to mark this as a component ref
                    "x-component-ref": ref_type.value if ref_type else validator_name.lower().replace("ref", ""),
                }

        # Default to a generic string schema
        return {
            "type": "string",
            "title": "Component Reference",
            "description": "Reference to another component in the workflow",
        }

    def handle_invalid_for_json_schema(self, schema: CoreSchema, error_info: str) -> JsonSchemaValue:
        """
        Handle schemas that are invalid for JSON schema generation.

        Instead of raising an error, we return a permissive schema.
        """
        logger.debug(f"Handling invalid schema: {error_info}")
        return {
            "type": "object",
            "description": "Complex type - configure via YAML",
            "additionalProperties": True,
        }


def extract_json_schema(config_type: type[BaseModel]) -> dict[str, Any]:
    """
    Extract the JSON Schema from a Pydantic model.

    Uses a custom schema generator to handle NAT-specific types like
    ComponentRef that can't be directly converted to JSON schema.

    Args:
        config_type: The Pydantic model class to extract schema from.

    Returns:
        The JSON Schema as a dictionary.
    """
    try:
        # Use our custom schema generator
        schema = config_type.model_json_schema(schema_generator=NATJsonSchemaGenerator)
        return schema
    except Exception as e:
        logger.warning(f"Failed to extract schema for {config_type}: {e}")

        # Try to build a minimal schema from model fields
        try:
            return _build_fallback_schema(config_type)
        except Exception:
            return {"type": "object", "properties": {}, "title": config_type.__name__}


def _build_fallback_schema(config_type: type[BaseModel]) -> dict[str, Any]:
    """
    Build a minimal fallback schema by inspecting model fields directly.

    This is used when the standard schema generation fails.
    """
    properties = {}
    required = []

    for field_name, field_info in config_type.model_fields.items():
        if field_name.startswith("_"):
            continue

        # Determine basic field type
        annotation = field_info.annotation
        field_schema: dict[str, Any] = {"title": field_name.replace("_", " ").title()}

        if field_info.description:
            field_schema["description"] = field_info.description

        if field_info.default is not None:
            field_schema["default"] = field_info.default

        # Try to determine type
        type_name = getattr(annotation, "__name__", str(annotation))

        # Check for ComponentRef types
        is_ref, ref_type, is_list = _is_component_ref_type(annotation)
        if is_ref and ref_type:
            component_type = ref_type.value.upper()
            if is_list:
                field_schema.update({
                    "type": "array",
                    "items": {
                        "type": "string",
                        "x-component-ref": ref_type.value,
                    },
                    "description": f"List of references to {component_type} components",
                    "x-component-ref": ref_type.value,
                    "x-is-ref-list": True,
                })
            else:
                field_schema.update({
                    "type": "string",
                    "description": f"Reference to a {component_type} component",
                    "x-component-ref": ref_type.value,
                })
        elif type_name in ("str", "string"):
            field_schema["type"] = "string"
        elif type_name in ("int", "integer"):
            field_schema["type"] = "integer"
        elif type_name in ("float", "number"):
            field_schema["type"] = "number"
        elif type_name in ("bool", "boolean"):
            field_schema["type"] = "boolean"
        elif type_name in ("list", "List"):
            field_schema["type"] = "array"
        elif type_name in ("dict", "Dict"):
            field_schema["type"] = "object"
        else:
            # Unknown type - make it a string for simplicity
            field_schema["type"] = "string"

        properties[field_name] = field_schema

        if field_info.is_required():
            required.append(field_name)

    return {
        "type": "object",
        "title": config_type.__name__,
        "properties": properties,
        "required": required if required else None,
    }


def parse_field_info(field_name: str,
                     field_schema: dict[str, Any],
                     required_fields: list[str],
                     definitions: dict[str, Any] | None = None) -> FieldInfo:
    """
    Parse a field schema into a FieldInfo object.

    Args:
        field_name: Name of the field.
        field_schema: The JSON Schema for this field.
        required_fields: List of required field names.
        definitions: Schema definitions for resolving $ref.

    Returns:
        Parsed FieldInfo object.
    """
    # Resolve $ref if present
    if "$ref" in field_schema and definitions:
        ref_path = field_schema["$ref"].split("/")[-1]
        if ref_path in definitions:
            field_schema = {**definitions[ref_path], **{k: v for k, v in field_schema.items() if k != "$ref"}}

    # Handle anyOf/oneOf (common for Optional types)
    if "anyOf" in field_schema:
        # Find the non-null type
        for option in field_schema["anyOf"]:
            if option.get("type") != "null":
                field_schema = {**option, **{k: v for k, v in field_schema.items() if k not in ("anyOf", )}}
                break

    field_type = field_schema.get("type", "string")
    if isinstance(field_type, list):
        # Handle union types like ["string", "null"]
        field_type = next((t for t in field_type if t != "null"), "string")

    # Extract component ref information
    x_component_ref = field_schema.get("x-component-ref")
    is_component_ref = x_component_ref is not None
    ref_type = RefType(x_component_ref) if x_component_ref and x_component_ref in [r.value for r in RefType] else None
    is_ref_list = field_schema.get("x-is-ref-list", False)

    return FieldInfo(
        name=field_name,
        type=field_type,
        title=field_schema.get("title"),
        description=field_schema.get("description"),
        required=field_name in required_fields,
        default=field_schema.get("default"),
        enum=field_schema.get("enum"),
        minimum=field_schema.get("minimum"),
        maximum=field_schema.get("maximum"),
        pattern=field_schema.get("pattern"),
        items=field_schema.get("items"),
        properties=field_schema.get("properties"),
        is_component_ref=is_component_ref,
        ref_type=ref_type,
        is_ref_list=is_ref_list,
    )


# Fields from OptimizableMixin that should be excluded from the UI
OPTIMIZABLE_MIXIN_FIELDS = {
    "optimizable_params",
    "search_space",
}

# Additional fields to exclude from the UI (internal implementation details)
EXCLUDED_FIELDS = {
    "type",  # Type discriminator field
    "model_config",  # Pydantic config
}


def get_sdk_excluded_fields(config_type: type[BaseModel]) -> set[str]:
    """
    Get field names that should be excluded from the UI based on SDK class configuration.

    Fields marked with `init=False` in the SDK class are meant to be set via connections
    (like llm_name, tool_names) rather than direct user input, so they should be excluded
    from the configuration form.

    Args:
        config_type: The config type to check for SDK exclusions.

    Returns:
        Set of field names to exclude.
    """
    excluded = set()

    # Find the SDK class for this config
    sdk_class = find_sdk_class_for_config(config_type)
    if sdk_class is None:
        return excluded

    if not hasattr(sdk_class, "model_fields"):
        return excluded

    for field_name, field_info in sdk_class.model_fields.items():
        # Check if field has init=False (set via model_fields metadata)
        # In Pydantic v2, init=False is stored in the field's metadata or as a frozen attribute
        if hasattr(field_info, "init") and field_info.init is False:
            excluded.add(field_name)
            continue

        # Check for exclude=True (internal SDK fields)
        if getattr(field_info, "exclude", False):
            excluded.add(field_name)
            continue

        # Also check if there's init=False in the extra/json_schema_extra
        if hasattr(field_info, "json_schema_extra"):
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and extra.get("init") is False:
                excluded.add(field_name)

    return excluded


def extract_fields(schema: dict[str, Any], sdk_excluded_fields: set[str] | None = None) -> list[FieldInfo]:
    """
    Extract field information from a JSON Schema.

    Args:
        schema: The JSON Schema dictionary.
        sdk_excluded_fields: Optional set of field names to exclude (typically from SDK class).

    Returns:
        List of FieldInfo objects for each field.
    """
    fields = []
    properties = schema.get("properties", {})
    required_fields = schema.get("required", []) or []
    definitions = schema.get("$defs", {}) or schema.get("definitions", {})
    excluded = sdk_excluded_fields or set()

    for field_name, field_schema in properties.items():
        # Skip internal fields
        if field_name.startswith("_"):
            continue

        # Skip OptimizableMixin fields
        if field_name in OPTIMIZABLE_MIXIN_FIELDS:
            continue

        # Skip other excluded fields
        if field_name in EXCLUDED_FIELDS:
            continue

        # Skip SDK-excluded fields (init=False fields that are set via connections)
        if field_name in excluded:
            continue

        try:
            field_info = parse_field_info(field_name, field_schema, required_fields, definitions)
            fields.append(field_info)
        except Exception as e:
            logger.warning(f"Failed to parse field {field_name}: {e}")

    return fields


def get_model_description(config_type: type[BaseModel]) -> str | None:
    """
    Get the description from a Pydantic model's docstring or schema.

    Strips the icon markdown syntax (![Icon](url)) from the description if present.

    Args:
        config_type: The Pydantic model class.

    Returns:
        Description string or None.
    """
    # Try docstring first
    if config_type.__doc__:
        # Get first non-empty line that isn't the icon marker
        lines = config_type.__doc__.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped and not _ICON_PATTERN.match(stripped):
                return stripped
        return None

    # Try schema description (but don't fail if schema can't be generated)
    try:
        schema = config_type.model_json_schema(schema_generator=NATJsonSchemaGenerator)
        return schema.get("description")
    except Exception:
        return None


def get_icon_url_from_docstring(config_type: type[BaseModel]) -> str | None:
    """
    Extract icon URL from a Pydantic model's docstring.

    Looks for markdown image syntax: ![Icon](https://example.com/icon.svg)

    Args:
        config_type: The Pydantic model class.

    Returns:
        Icon URL string or None if not found.
    """
    if not config_type.__doc__:
        return None

    match = _ICON_PATTERN.search(config_type.__doc__)
    if match:
        return match.group(1).strip()

    return None
