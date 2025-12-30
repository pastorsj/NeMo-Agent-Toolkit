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
YAML preprocessing for environment variables.

Replaces ${VAR_NAME} patterns with placeholders before YAML parsing
to prevent NAT's yaml_loads from resolving them to None.
"""

import re

from nat.workflow_builder_api.utils.env_var.detection import ENV_VAR_PATTERN
from nat.workflow_builder_api.utils.env_var.detection import URL_FIELD_PATTERNS
from nat.workflow_builder_api.utils.env_var.detection import YAML_ENV_VAR_PATTERN
from nat.workflow_builder_api.utils.env_var.models import EnvVarOccurrence
from nat.workflow_builder_api.utils.env_var.models import YamlEnvVarPreprocessResult


def _generate_placeholder_for_var(var_name: str) -> str:
    """
    Generate a placeholder value for an environment variable.

    The placeholder needs to be valid for the expected field type.
    We use naming heuristics to determine the appropriate format.

    Args:
        var_name: The environment variable name (e.g., 'SERVICE_ACCOUNT_TOKEN_URL')

    Returns:
        A placeholder string that passes validation for the expected type
    """
    # Check if this looks like a URL field
    for pattern in URL_FIELD_PATTERNS:
        if pattern.match(var_name):
            # Return a valid placeholder URL
            return f"https://placeholder.example.com/env/{var_name.lower()}"

    # Default: return a simple placeholder string
    return f"__ENV_VAR__{var_name}__"


def _parse_yaml_context(lines: list[str], line_index: int) -> tuple[list[str], str]:
    """
    Parse the YAML context to determine the key path for a given line.

    Args:
        lines: All lines of the YAML file
        line_index: 0-based index of the line containing the env var

    Returns:
        Tuple of (key_path, field_name) where key_path is like
        ['authentication', 'my_auth'] and field_name is 'client_id'
    """
    key_path: list[str] = []
    current_line = lines[line_index]

    # Extract the field name from the current line (key before the colon)
    field_match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", current_line)
    field_name = field_match.group(1) if field_match else ""

    # Calculate the indentation of the current line
    current_indent = len(current_line) - len(current_line.lstrip())

    # Walk backwards to find parent keys based on indentation
    parent_indent = current_indent
    for i in range(line_index - 1, -1, -1):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Calculate this line's indentation
        indent = len(line) - len(line.lstrip())

        # If this line has less indentation, it's a parent
        if indent < parent_indent:
            key_match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", line)
            if key_match:
                key_path.insert(0, key_match.group(1))
                parent_indent = indent

        # Stop at root level
        if indent == 0 and line.strip():
            break

    return key_path, field_name


def preprocess_yaml_env_vars(yaml_content: str) -> YamlEnvVarPreprocessResult:
    """
    Preprocess raw YAML content to replace ${VAR_NAME} patterns with placeholders.

    This must be called BEFORE yaml.load() or yaml_loads() because NAT's
    yaml_loads resolves env vars and returns None for missing ones.

    The placeholders are designed to:
    1. Be valid string values for most fields
    2. Be clearly identifiable for later restoration
    3. Pass basic validation (URL patterns, etc.)

    This function also parses the YAML structure to determine the path/location
    of each environment variable occurrence.

    Args:
        yaml_content: Raw YAML string content

    Returns:
        YamlEnvVarPreprocessResult with processed YAML and detected variables
    """
    detected_vars: list[str] = []
    var_to_placeholder: dict[str, str] = {}
    occurrences: list[EnvVarOccurrence] = []

    # First pass: find all env vars and their line numbers
    lines = yaml_content.split("\n")
    for line_num, line in enumerate(lines, start=1):
        matches = YAML_ENV_VAR_PATTERN.findall(line)
        for match in matches:
            var_match = ENV_VAR_PATTERN.search(match)
            if var_match:
                var_name = var_match.group(1)
                if var_name not in var_to_placeholder:
                    detected_vars.append(var_name)
                    var_to_placeholder[var_name] = _generate_placeholder_for_var(var_name)

                # Try to parse the key path from the YAML structure
                key_path, field_name = _parse_yaml_context(lines, line_num - 1)

                occurrences.append(
                    EnvVarOccurrence(
                        var_name=var_name,
                        line_number=line_num,
                        key_path=key_path,
                        field_name=field_name,
                    ))

    # Replace all ${VAR_NAME} patterns in the YAML content
    def replace_env_var(match: re.Match) -> str:
        """Replace a single ${VAR_NAME} with a placeholder."""
        full_match = match.group(0)  # ${VAR_NAME}
        var_name_match = ENV_VAR_PATTERN.search(full_match)
        if var_name_match:
            name = var_name_match.group(1)
            return var_to_placeholder.get(name, full_match)
        return full_match

    processed_yaml = YAML_ENV_VAR_PATTERN.sub(replace_env_var, yaml_content)

    return YamlEnvVarPreprocessResult(
        processed_yaml=processed_yaml,
        detected_vars=detected_vars,
        var_to_placeholder=var_to_placeholder,
        occurrences=occurrences,
    )
