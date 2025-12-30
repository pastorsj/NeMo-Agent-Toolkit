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
Environment variable detection and processing module.

This module provides utilities for handling environment variables in YAML configurations:
- Detection of ${VAR_NAME} patterns
- Extraction and tracking of variable locations
- Placeholder generation for validation
- Secret field type detection
- Warning generation for non-secret field usage
"""

# Core data models
# Pattern matching and detection
from nat.workflow_builder_api.utils.env_var.detection import ENV_VAR_PATTERN
from nat.workflow_builder_api.utils.env_var.detection import SENSITIVE_FIELD_PATTERNS
from nat.workflow_builder_api.utils.env_var.detection import YAML_ENV_VAR_PATTERN
from nat.workflow_builder_api.utils.env_var.detection import extract_env_var_names
from nat.workflow_builder_api.utils.env_var.detection import is_env_var_reference
from nat.workflow_builder_api.utils.env_var.detection import is_sensitive_field

# Secret field enrichment
from nat.workflow_builder_api.utils.env_var.enrichment import enrich_env_vars_with_field_types
from nat.workflow_builder_api.utils.env_var.enrichment import get_config_class_for_component
from nat.workflow_builder_api.utils.env_var.models import EnvironmentVariable
from nat.workflow_builder_api.utils.env_var.models import EnvVarLocation
from nat.workflow_builder_api.utils.env_var.models import EnvVarOccurrence
from nat.workflow_builder_api.utils.env_var.models import EnvVarProcessingResult
from nat.workflow_builder_api.utils.env_var.models import YamlEnvVarPreprocessResult

# YAML preprocessing
from nat.workflow_builder_api.utils.env_var.preprocessing import preprocess_yaml_env_vars

# Config processing
from nat.workflow_builder_api.utils.env_var.processing import apply_env_var_values
from nat.workflow_builder_api.utils.env_var.processing import generate_placeholder_value
from nat.workflow_builder_api.utils.env_var.processing import process_config_env_vars
from nat.workflow_builder_api.utils.env_var.processing import restore_env_var_references

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
