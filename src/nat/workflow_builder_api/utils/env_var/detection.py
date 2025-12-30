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
Pattern matching and detection for environment variables.

Provides regex patterns and functions to detect ${VAR_NAME} patterns
and identify sensitive fields.
"""

import re

# Pattern to match environment variable references: ${VAR_NAME}
# Matches: ${VAR}, ${VAR_NAME}, ${MY_VAR_123}
# Does not match: $VAR, ${}, ${123VAR}
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Pattern for YAML string replacement (handles quoted and unquoted values)
# Matches: "${VAR_NAME}" or ${VAR_NAME} in YAML context
YAML_ENV_VAR_PATTERN = re.compile(r'(\$\{[A-Za-z_][A-Za-z0-9_]*\})')

# Sensitive field name patterns (for auto-detection of secrets)
SENSITIVE_FIELD_PATTERNS = [
    re.compile(r".*password.*", re.IGNORECASE),
    re.compile(r".*secret.*", re.IGNORECASE),
    re.compile(r".*token.*", re.IGNORECASE),
    re.compile(r".*api_key.*", re.IGNORECASE),
    re.compile(r".*apikey.*", re.IGNORECASE),
    re.compile(r".*private.*", re.IGNORECASE),
    re.compile(r".*credential.*", re.IGNORECASE),
    re.compile(r".*auth.*", re.IGNORECASE),
]

# Patterns that suggest a field expects a URL value
URL_FIELD_PATTERNS = [
    re.compile(r".*_url$", re.IGNORECASE),
    re.compile(r".*url$", re.IGNORECASE),
    re.compile(r".*endpoint.*", re.IGNORECASE),
    re.compile(r".*host.*", re.IGNORECASE),
    re.compile(r".*uri.*", re.IGNORECASE),
]


def is_env_var_reference(value: str) -> bool:
    """Check if a string value contains an environment variable reference.

    Args:
        value: The string value to check

    Returns:
        True if the value contains ${VAR_NAME} pattern
    """
    if not isinstance(value, str):
        return False
    return bool(ENV_VAR_PATTERN.search(value))


def extract_env_var_names(value: str) -> list[str]:
    """Extract all environment variable names from a string.

    Args:
        value: String that may contain ${VAR_NAME} patterns

    Returns:
        List of variable names (without ${} wrapper)
    """
    if not isinstance(value, str):
        return []
    return ENV_VAR_PATTERN.findall(value)


def is_sensitive_field(field_name: str, path: str) -> bool:
    """Determine if a field is likely to contain sensitive data.

    Args:
        field_name: The field name (e.g., 'client_secret')
        path: Full path to the field (e.g., 'authentication.my_auth.client_secret')

    Returns:
        True if the field appears to be sensitive
    """
    # Check field name
    for pattern in SENSITIVE_FIELD_PATTERNS:
        if pattern.match(field_name):
            return True

    # Check full path (e.g., 'authentication' section is often sensitive)
    if "authentication" in path.lower():
        return True

    return False
