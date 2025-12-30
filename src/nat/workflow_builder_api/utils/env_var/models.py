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
Data models for environment variable processing.

These are the internal models used during env var detection and processing.
The API models in nat.workflow_builder_api.models are the public-facing versions.
"""

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class EnvVarLocation(BaseModel):
    """Location where an environment variable is used in the config."""

    path: str = Field(description="Dot-separated path to the field (e.g., 'authentication.my_auth.client_id')")
    component_type: str | None = Field(default=None, description="Type of component (e.g., 'authentication', 'llm')")
    component_id: str | None = Field(default=None, description="Component identifier (e.g., 'my_auth', 'nim_llm')")
    field_name: str = Field(description="Name of the field containing the env var")
    is_secret_field: bool = Field(
        default=False, description="Whether this field is a secret type (SerializableSecretStr or OptionalSecretStr)")


class EnvironmentVariable(BaseModel):
    """Represents an environment variable detected in the config."""

    name: str = Field(description="Variable name without ${} (e.g., 'API_KEY')")
    locations: list[EnvVarLocation] = Field(default_factory=list, description="All locations where this var is used")
    value: str | None = Field(default=None, description="User-provided value (None = unresolved)")
    is_sensitive: bool = Field(default=False, description="Whether this appears to be a sensitive value")
    export_as_variable: bool = Field(default=True, description="Whether to export with ${} notation")
    original_reference: str = Field(description="Original ${VAR_NAME} reference string")
    # Type-based secret detection
    has_secret_field_usage: bool = Field(
        default=False, description="Whether any location is a SerializableSecretStr/OptionalSecretStr field")
    has_non_secret_field_usage: bool = Field(default=False,
                                             description="Whether any location is a non-secret field (str, list, etc.)")
    warning: str | None = Field(default=None, description="Warning message if env var is used in non-secret fields")


class EnvVarProcessingResult(BaseModel):
    """Result of processing environment variables in a config."""

    processed_config: dict[str, Any] = Field(description="Config with env vars replaced by placeholders")
    environment_variables: list[EnvironmentVariable] = Field(default_factory=list,
                                                             description="All detected environment variables")
    env_var_locations: dict[str, str] = Field(
        default_factory=dict,
        description="Map of path -> original env var reference (for restoration)",
    )


class EnvVarOccurrence:
    """Represents a single occurrence of an env var in YAML."""

    def __init__(self, var_name: str, line_number: int, key_path: list[str], field_name: str):
        self.var_name = var_name
        self.line_number = line_number
        self.key_path = key_path  # e.g., ['authentication', 'my_auth', 'client_id']
        self.field_name = field_name


class YamlEnvVarPreprocessResult:
    """Result of preprocessing YAML content for environment variables."""

    def __init__(
        self,
        processed_yaml: str,
        detected_vars: list[str],
        var_to_placeholder: dict[str, str],
        occurrences: list[EnvVarOccurrence] | None = None,
    ):
        self.processed_yaml = processed_yaml
        self.detected_vars = detected_vars
        self.var_to_placeholder = var_to_placeholder  # VAR_NAME -> placeholder value
        self.occurrences = occurrences or []  # Detailed location info
