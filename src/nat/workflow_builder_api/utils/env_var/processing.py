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
Config processing for environment variables.

Provides functions to process configs, detect env vars, replace with placeholders,
and restore/apply values during export.
"""

import copy
import re
from typing import Any

from nat.workflow_builder_api.utils.env_var.detection import extract_env_var_names
from nat.workflow_builder_api.utils.env_var.detection import is_env_var_reference
from nat.workflow_builder_api.utils.env_var.detection import is_sensitive_field
from nat.workflow_builder_api.utils.env_var.models import EnvironmentVariable
from nat.workflow_builder_api.utils.env_var.models import EnvVarLocation
from nat.workflow_builder_api.utils.env_var.models import EnvVarProcessingResult


def generate_placeholder_value(var_name: str, field_path: str) -> str:
    """Generate a placeholder value for an environment variable.

    The placeholder needs to be a valid value that will pass Pydantic validation.
    We use a format that's clearly identifiable as a placeholder.

    Args:
        var_name: The environment variable name
        field_path: Path to the field (used for context)

    Returns:
        A placeholder string value
    """
    # Use a format that's obviously a placeholder but valid for most string fields
    # The __ENV_VAR__ prefix makes it easy to identify and restore later
    return f"__ENV_VAR__{var_name}__"


def _process_value(
    value: Any,
    path: str,
    env_vars: dict[str, EnvironmentVariable],
    env_var_locations: dict[str, str],
) -> Any:
    """Process a single value, detecting and replacing env var references.

    Args:
        value: The value to process
        path: Current path in the config (dot-separated)
        env_vars: Dict to collect discovered env vars (name -> EnvironmentVariable)
        env_var_locations: Dict to track path -> original reference

    Returns:
        Processed value with env vars replaced by placeholders
    """
    if isinstance(value, str):
        if is_env_var_reference(value):
            var_names = extract_env_var_names(value)

            for var_name in var_names:
                # Parse the path to extract component info
                path_parts = path.split(".")
                component_type = path_parts[0] if path_parts else None
                component_id = path_parts[1] if len(path_parts) > 1 else None
                field_name = path_parts[-1] if path_parts else ""

                location = EnvVarLocation(
                    path=path,
                    component_type=component_type,
                    component_id=component_id,
                    field_name=field_name,
                )

                if var_name not in env_vars:
                    env_vars[var_name] = EnvironmentVariable(
                        name=var_name,
                        locations=[location],
                        value=None,
                        is_sensitive=is_sensitive_field(field_name, path),
                        export_as_variable=True,
                        original_reference=f"${{{var_name}}}",
                    )
                else:
                    # Add this location if not already present
                    existing_paths = [loc.path for loc in env_vars[var_name].locations]
                    if path not in existing_paths:
                        env_vars[var_name].locations.append(location)

            # Store original reference for this path
            env_var_locations[path] = value

            # Replace with placeholder
            # If the entire value is a single env var, replace it completely
            # If it's embedded in a string, replace each occurrence
            if value == f"${{{var_names[0]}}}" and len(var_names) == 1:
                return generate_placeholder_value(var_names[0], path)
            else:
                # Multiple env vars or embedded - replace each
                result = value
                for var_name in var_names:
                    result = result.replace(f"${{{var_name}}}", generate_placeholder_value(var_name, path))
                return result

        return value

    elif isinstance(value, dict):
        return {
            k: _process_value(v, f"{path}.{k}" if path else k, env_vars, env_var_locations)
            for k, v in value.items()
        }

    elif isinstance(value, list):
        return [_process_value(item, f"{path}[{i}]", env_vars, env_var_locations) for i, item in enumerate(value)]

    else:
        # Numbers, booleans, None - return as-is
        return value


def process_config_env_vars(config_dict: dict[str, Any]) -> EnvVarProcessingResult:
    """Process a config dictionary, detecting and replacing environment variables.

    This is the main entry point for env var processing. It:
    1. Scans the entire config for ${VAR_NAME} patterns
    2. Replaces them with placeholder values that pass validation
    3. Returns the processed config and metadata about discovered env vars

    Args:
        config_dict: Raw config dictionary from YAML parsing

    Returns:
        EnvVarProcessingResult with processed config and env var metadata
    """
    # Make a deep copy to avoid mutating the original
    config_copy = copy.deepcopy(config_dict)

    env_vars: dict[str, EnvironmentVariable] = {}
    env_var_locations: dict[str, str] = {}

    # Process each top-level section
    processed = {}
    for key, value in config_copy.items():
        processed[key] = _process_value(value, key, env_vars, env_var_locations)

    return EnvVarProcessingResult(
        processed_config=processed,
        environment_variables=list(env_vars.values()),
        env_var_locations=env_var_locations,
    )


def _parse_path(path: str) -> list[str | int]:
    """Parse a dot-separated path with array indices.

    Args:
        path: Path like "authentication.my_auth.scopes[0]"

    Returns:
        List of path parts, with integers for array indices
    """
    parts: list[str | int] = []
    # Split by dots first
    segments = path.split(".")

    for segment in segments:
        # Check for array index
        if "[" in segment:
            # Split "field[0]" into "field" and "0"
            field_name = segment[:segment.index("[")]
            if field_name:
                parts.append(field_name)

            # Extract all indices (handles nested like field[0][1])
            indices = re.findall(r"\[(\d+)\]", segment)
            for idx in indices:
                parts.append(int(idx))
        else:
            parts.append(segment)

    return parts


def _set_value_at_path(obj: dict | list, path_parts: list[str | int], value: Any) -> None:
    """Set a value at a nested path in a dict/list structure.

    Args:
        obj: Root object to modify
        path_parts: List of path parts (strings for dict keys, ints for list indices)
        value: Value to set
    """
    if not path_parts:
        return

    current = obj
    for part in path_parts[:-1]:
        if isinstance(part, int):
            current = current[part]
        else:
            current = current[part]

    final_key = path_parts[-1]
    if isinstance(final_key, int):
        current[final_key] = value
    else:
        current[final_key] = value


def restore_env_var_references(
    config_dict: dict[str, Any],
    env_vars: list[EnvironmentVariable],
    use_resolved_values: bool = False,
) -> dict[str, Any]:
    """Restore environment variable references in a config for export.

    When exporting, this function can either:
    - Restore the original ${VAR_NAME} references (for portable configs)
    - Use the resolved values (for configs that don't need env vars)

    Args:
        config_dict: Config dictionary to process
        env_vars: List of environment variables with their locations and values
        use_resolved_values: If True, use resolved values; if False, restore ${} references

    Returns:
        Config dict with env var references restored or values applied
    """
    result = copy.deepcopy(config_dict)

    for env_var in env_vars:
        for location in env_var.locations:
            path = location.path
            path_parts = _parse_path(path)

            if env_var.export_as_variable and not use_resolved_values:
                # Restore original ${VAR_NAME} reference
                new_value = env_var.original_reference
            elif env_var.value is not None:
                # Use the resolved value
                new_value = env_var.value
            else:
                # No value and not exporting as variable - keep placeholder
                continue

            _set_value_at_path(result, path_parts, new_value)

    return result


def _replace_placeholder(obj: Any, placeholder: str, value: str) -> Any:
    """Recursively replace placeholder values in a config.

    Args:
        obj: Object to process (dict, list, or scalar)
        placeholder: Placeholder string to find
        value: Value to replace it with

    Returns:
        Processed object with replacements made
    """
    if isinstance(obj, str):
        if obj == placeholder:
            return value
        elif placeholder in obj:
            return obj.replace(placeholder, value)
        return obj

    elif isinstance(obj, dict):
        return {k: _replace_placeholder(v, placeholder, value) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [_replace_placeholder(item, placeholder, value) for item in obj]

    else:
        return obj


def apply_env_var_values(
    config_dict: dict[str, Any],
    env_vars: list[EnvironmentVariable],
) -> dict[str, Any]:
    """Apply resolved environment variable values to a config.

    This replaces placeholder values with actual resolved values.
    Used when running a workflow with user-provided env var values.

    Args:
        config_dict: Config dictionary with placeholders
        env_vars: List of environment variables with resolved values

    Returns:
        Config dict with placeholders replaced by actual values
    """
    result = copy.deepcopy(config_dict)

    for env_var in env_vars:
        if env_var.value is None:
            continue

        placeholder = generate_placeholder_value(env_var.name, "")

        # Recursively replace placeholders
        result = _replace_placeholder(result, placeholder, env_var.value)

    return result
