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
Environment variable detection and processing for config imports.

This module re-exports all functionality from the env_var submodule for
backward compatibility. New code should import directly from:
- nat.workflow_builder_api.utils.env_var.models
- nat.workflow_builder_api.utils.env_var.detection
- nat.workflow_builder_api.utils.env_var.preprocessing
- nat.workflow_builder_api.utils.env_var.processing
- nat.workflow_builder_api.utils.env_var.enrichment
"""

# Re-export everything for backward compatibility
from nat.workflow_builder_api.utils.env_var import ENV_VAR_PATTERN
from nat.workflow_builder_api.utils.env_var import SENSITIVE_FIELD_PATTERNS
from nat.workflow_builder_api.utils.env_var import YAML_ENV_VAR_PATTERN
from nat.workflow_builder_api.utils.env_var import EnvironmentVariable
from nat.workflow_builder_api.utils.env_var import EnvVarLocation
from nat.workflow_builder_api.utils.env_var import EnvVarOccurrence
from nat.workflow_builder_api.utils.env_var import EnvVarProcessingResult
from nat.workflow_builder_api.utils.env_var import YamlEnvVarPreprocessResult
from nat.workflow_builder_api.utils.env_var import apply_env_var_values
from nat.workflow_builder_api.utils.env_var import enrich_env_vars_with_field_types
from nat.workflow_builder_api.utils.env_var import extract_env_var_names
from nat.workflow_builder_api.utils.env_var import generate_placeholder_value
from nat.workflow_builder_api.utils.env_var import get_config_class_for_component
from nat.workflow_builder_api.utils.env_var import is_env_var_reference
from nat.workflow_builder_api.utils.env_var import is_sensitive_field
from nat.workflow_builder_api.utils.env_var import preprocess_yaml_env_vars
from nat.workflow_builder_api.utils.env_var import process_config_env_vars
from nat.workflow_builder_api.utils.env_var import restore_env_var_references

__all__ = [
    # Models
    "EnvVarLocation",
    "EnvVarOccurrence",
    "EnvVarProcessingResult",
    "EnvironmentVariable",
    "YamlEnvVarPreprocessResult",  # Detection
    "ENV_VAR_PATTERN",
    "SENSITIVE_FIELD_PATTERNS",
    "YAML_ENV_VAR_PATTERN",
    "extract_env_var_names",
    "is_env_var_reference",
    "is_sensitive_field",  # Preprocessing
    "preprocess_yaml_env_vars",  # Processing
    "apply_env_var_values",
    "generate_placeholder_value",
    "process_config_env_vars",
    "restore_env_var_references",  # Enrichment
    "enrich_env_vars_with_field_types",
    "get_config_class_for_component",
]
