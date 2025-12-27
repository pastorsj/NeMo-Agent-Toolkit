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
JSON Schema extraction utilities.

Handles extraction of JSON schemas from NAT Pydantic models,
field parsing, and model description extraction.
"""

import logging
import re
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from nat.workflow_builder_api.constants.ref_types import COMPONENT_REF_TO_REF_TYPE
from nat.workflow_builder_api.constants.ref_types import is_component_ref_type
from nat.workflow_builder_api.models import FieldInfo
from nat.workflow_builder_api.models import RefType
from nat.workflow_builder_api.utils.connections import find_sdk_class_for_config

logger = logging.getLogger(__name__)

_ICON_PATTERN = re.compile(r'!\[Icon\]\(([^)]+)\)', re.IGNORECASE)
_NAME_PATTERN = re.compile(r'^Name:\s*(.+?)$', re.MULTILINE)

# =============================================================================
# JSON SCHEMA GENERATION
# =============================================================================


class NATJsonSchemaGenerator(GenerateJsonSchema):
    """Custom JSON schema generator for NAT-specific types."""

    def no_info_plain_validator_function_schema(self, schema: CoreSchema) -> JsonSchemaValue:
        """Handle plain validator functions (used by ComponentRef types)."""
        validator = schema.get("function")

        if validator is not None:
            validator_name = getattr(validator, "__name__", str(validator))

            if "Ref" in validator_name:
                component_type = validator_name.replace("Ref", "")
                ref_type = COMPONENT_REF_TO_REF_TYPE.get(validator_name)

                return {
                    "type": "string",
                    "title": f"{component_type} Reference",
                    "description": f"Reference name of a {component_type} component defined in the workflow",
                    "x-component-ref": ref_type.value if ref_type else validator_name.lower().replace("ref", ""),
                }

        return {
            "type": "string",
            "title": "Component Reference",
            "description": "Reference to another component in the workflow",
        }

    def handle_invalid_for_json_schema(self, schema: CoreSchema, error_info: str) -> JsonSchemaValue:
        """Handle schemas that are invalid for JSON schema generation."""
        logger.debug("Handling invalid schema: %s", error_info)
        return {
            "type": "object",
            "description": "Complex type - configure via YAML",
            "additionalProperties": True,
        }


def extract_json_schema(config_type: type[BaseModel]) -> dict[str, Any]:
    """Extract the JSON Schema from a Pydantic model."""
    try:
        return config_type.model_json_schema(schema_generator=NATJsonSchemaGenerator)
    except Exception as e:
        logger.warning("Failed to extract schema for %s: %s", config_type, e)
        try:
            return _build_fallback_schema(config_type)
        except Exception:
            return {"type": "object", "properties": {}, "title": config_type.__name__}


def _build_fallback_schema(config_type: type[BaseModel]) -> dict[str, Any]:
    """Build a minimal fallback schema by inspecting model fields directly."""
    properties = {}
    required = []

    for field_name, field_info in config_type.model_fields.items():
        if field_name.startswith("_"):
            continue

        annotation = field_info.annotation
        field_schema: dict[str, Any] = {"title": field_name.replace("_", " ").title()}

        if field_info.description:
            field_schema["description"] = field_info.description

        if field_info.default is not None:
            field_schema["default"] = field_info.default

        type_name = getattr(annotation, "__name__", str(annotation))

        is_ref, ref_type, is_list = is_component_ref_type(annotation)
        if is_ref and ref_type:
            component_type = ref_type.value.upper()
            if is_list:
                field_schema.update({
                    "type": "array",
                    "items": {
                        "type": "string", "x-component-ref": ref_type.value
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


# =============================================================================
# FIELD EXTRACTION
# =============================================================================

OPTIMIZABLE_MIXIN_FIELDS = {"optimizable_params", "search_space"}
EXCLUDED_FIELDS = {"type", "model_config"}


def parse_field_info(field_name: str,
                     field_schema: dict[str, Any],
                     required_fields: list[str],
                     definitions: dict[str, Any] | None = None) -> FieldInfo:
    """Parse a field schema into a FieldInfo object."""
    if "$ref" in field_schema and definitions:
        ref_path = field_schema["$ref"].split("/")[-1]
        if ref_path in definitions:
            field_schema = {**definitions[ref_path], **{k: v for k, v in field_schema.items() if k != "$ref"}}

    if "anyOf" in field_schema:
        for option in field_schema["anyOf"]:
            if option.get("type") != "null":
                field_schema = {**option, **{k: v for k, v in field_schema.items() if k not in ("anyOf", )}}
                break

    field_type = field_schema.get("type", "string")
    if isinstance(field_type, list):
        field_type = next((t for t in field_type if t != "null"), "string")

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


def get_sdk_excluded_fields(config_type: type[BaseModel]) -> set[str]:
    """Get field names that should be excluded from the UI based on SDK class."""
    excluded = set()

    sdk_class = find_sdk_class_for_config(config_type)
    if sdk_class is None or not hasattr(sdk_class, "model_fields"):
        return excluded

    for field_name, field_info in sdk_class.model_fields.items():
        if hasattr(field_info, "init") and field_info.init is False:
            excluded.add(field_name)
            continue

        if getattr(field_info, "exclude", False):
            excluded.add(field_name)
            continue

        if hasattr(field_info, "json_schema_extra"):
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and extra.get("init") is False:
                excluded.add(field_name)

    return excluded


def extract_fields(
    schema: dict[str, Any],
    sdk_excluded_fields: set[str] | None = None,
    config_type: type[BaseModel] | None = None,
) -> list[FieldInfo]:
    """
    Extract field information from a JSON Schema.

    Args:
        schema: The JSON schema to extract fields from.
        sdk_excluded_fields: Set of field names to exclude.
        config_type: Optional config class to check for field options.
    """
    fields = []
    properties = schema.get("properties", {})
    required_fields = schema.get("required", []) or []
    definitions = schema.get("$defs", {}) or schema.get("definitions", {})
    excluded = sdk_excluded_fields or set()

    # Get field options from config class if available
    field_options_map = _get_field_options_from_config(config_type) if config_type else {}

    for field_name, field_schema in properties.items():
        if field_name.startswith("_"):
            continue
        if field_name in OPTIMIZABLE_MIXIN_FIELDS:
            continue
        if field_name in EXCLUDED_FIELDS:
            continue
        if field_name in excluded:
            continue

        try:
            field_info = parse_field_info(field_name, field_schema, required_fields, definitions)

            # Add field options if available
            if field_name in field_options_map:
                field_info.options = field_options_map[field_name]

            fields.append(field_info)
        except Exception as e:
            logger.warning("Failed to parse field %s: %s", field_name, e)

    return fields


def _get_field_options_from_config(config_type: type[BaseModel]) -> dict[str, list[str]]:
    """
    Get field options from a config class if it implements get_field_options.

    The config class can implement a classmethod `get_field_options(field_name: str)`
    that returns a list of valid options for that field.

    Args:
        config_type: The config class to check.

    Returns:
        Dictionary mapping field names to their available options.
    """
    options_map: dict[str, list[str]] = {}

    # Check if config class has get_field_options method
    if not hasattr(config_type, "get_field_options"):
        return options_map

    get_options = getattr(config_type, "get_field_options")
    if not callable(get_options):
        return options_map

    # Get options for each field
    for field_name in config_type.model_fields.keys():
        try:
            options = get_options(field_name)
            if options and isinstance(options, list):
                options_map[field_name] = options
        except Exception as e:
            logger.debug("Error getting options for field %s: %s", field_name, e)

    return options_map


# =============================================================================
# MODEL DESCRIPTION UTILITIES
# =============================================================================


def get_model_description(config_type: type[BaseModel]) -> str | None:
    """Get the description from a Pydantic model's docstring or schema."""
    if config_type.__doc__:
        lines = config_type.__doc__.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped and not _ICON_PATTERN.match(stripped):
                return stripped
        return None

    try:
        schema = config_type.model_json_schema(schema_generator=NATJsonSchemaGenerator)
        return schema.get("description")
    except Exception:
        return None


def get_icon_url_from_docstring(config_type: type[BaseModel]) -> str | None:
    """Extract icon URL from a Pydantic model's docstring."""
    if not config_type.__doc__:
        return None

    match = _ICON_PATTERN.search(config_type.__doc__)
    if match:
        return match.group(1).strip()

    return None


def get_display_name_from_docstring(config_type: type[BaseModel]) -> str | None:
    """
    Extract display name from a Pydantic model's docstring.

    Looks for a line like "Name: My Display Name" in the docstring,
    typically under a "## Details" section.

    Example docstring format:
        '''
        Description of the component.

        ## Details
        Name: My Component
        Icon: ![Icon](https://example.com/icon.svg)
        '''

    Returns:
        The display name if found, None otherwise.
    """
    if not config_type.__doc__:
        return None

    match = _NAME_PATTERN.search(config_type.__doc__)
    if match:
        return match.group(1).strip()

    return None
