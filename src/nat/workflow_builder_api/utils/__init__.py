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
Utility functions for the Workflow Builder API.

This package contains:
- type_builder: Build RegisteredTypeInfo from registry/SDK classes
- validation: YAML config validation
- schema: JSON schema extraction from Pydantic models
- connections: Connection port extraction
- config_parser: Parse validated configs into workflow state
- config_exporter: Export workflow state to YAML configuration
"""

from nat.workflow_builder_api.utils.config_exporter import export_workflow_to_yaml
from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state
from nat.workflow_builder_api.utils.connections import extract_all_connection_ports
from nat.workflow_builder_api.utils.connections import extract_connection_ports
from nat.workflow_builder_api.utils.schema import extract_fields
from nat.workflow_builder_api.utils.schema import extract_json_schema
from nat.workflow_builder_api.utils.schema import get_icon_url_from_docstring
from nat.workflow_builder_api.utils.schema import get_model_description
from nat.workflow_builder_api.utils.schema import get_sdk_excluded_fields
from nat.workflow_builder_api.utils.type_builder import get_category_types
from nat.workflow_builder_api.utils.validation import validate_yaml_config

__all__ = [
    # Type builder
    "get_category_types",  # Validation
    "validate_yaml_config",  # Config parser
    "parse_config_to_workflow_state",  # Config exporter
    "export_workflow_to_yaml",  # Schema
    "extract_json_schema",
    "extract_fields",
    "get_model_description",
    "get_icon_url_from_docstring",
    "get_sdk_excluded_fields",  # Connections
    "extract_connection_ports",
    "extract_all_connection_ports",
]
