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
Response models for the Workflow Builder API.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ComponentCategory(str, Enum):
    """Categories of NAT components supported by the workflow builder."""

    LLM = "llm"
    EMBEDDER = "embedder"
    FUNCTION = "function"
    FUNCTION_GROUP = "function_group"
    AGENT = "agent"
    RETRIEVER = "retriever"
    MEMORY = "memory"
    OBJECT_STORE = "object_store"
    AUTHENTICATION = "authentication"
    MIDDLEWARE = "middleware"
    # Front-end and observability
    FRONT_END = "front_end"
    LOGGER = "logger"
    TELEMETRY_EXPORTER = "telemetry_exporter"
    # Evaluation and optimization
    EVALUATOR = "evaluator"
    # Finetuning components
    TRAINER = "trainer"
    TRAJECTORY_BUILDER = "trajectory_builder"
    TRAINER_ADAPTER = "trainer_adapter"
    # Test-Time Compute strategies
    TTC_STRATEGY = "ttc_strategy"
    # Workflow-level configuration containers (each is a single type, no variants)
    NAT_WORKFLOW = "nat_workflow"
    GENERAL_CONFIG = "general_config"
    EVALUATION_CONFIG = "evaluation_config"
    OPTIMIZER_CONFIG = "optimizer_config"
    FINETUNER_CONFIG = "finetuner_config"


class RefType(str, Enum):
    """Types of component references that can be connected."""

    LLM = "llm"
    EMBEDDER = "embedder"
    FUNCTION = "function"
    FUNCTION_GROUP = "function_group"
    RETRIEVER = "retriever"
    MEMORY = "memory"
    OBJECT_STORE = "object_store"
    AUTHENTICATION = "authentication"
    MIDDLEWARE = "middleware"
    # Front-end and observability
    FRONT_END = "front_end"
    LOGGER = "logger"
    TELEMETRY_EXPORTER = "telemetry_exporter"
    # Evaluation
    EVALUATOR = "evaluator"
    # Finetuning components
    TRAINER = "trainer"
    TRAJECTORY_BUILDER = "trajectory_builder"
    TRAINER_ADAPTER = "trainer_adapter"
    # Test-Time Compute strategies
    TTC_STRATEGY = "ttc_strategy"
    # Workflow-level configuration containers
    NAT_WORKFLOW = "nat_workflow"
    GENERAL_CONFIG = "general_config"
    EVALUATION_CONFIG = "evaluation_config"
    OPTIMIZER_CONFIG = "optimizer_config"
    FINETUNER_CONFIG = "finetuner_config"


class ConnectionPort(BaseModel):
    """
    A connection port on a component that accepts references to other components.

    This represents fields like `llm_name: LLMRef` which indicate that this
    component can be connected to an LLM component.

    For union types like `list[FunctionRef | FunctionGroupRef]`, the port will
    have multiple `accepts_ref_types` to indicate it can connect to either type.
    """

    field_name: str = Field(description="Name of the field that accepts the reference")
    ref_type: RefType = Field(description="Primary type of component this port connects to")
    accepts_ref_types: list[RefType] = Field(default_factory=list,
                                             description="All ref types this port can accept (for union types)")
    required: bool = Field(description="Whether this connection is required")
    is_list: bool = Field(default=False, description="Whether this accepts multiple connections")
    description: str | None = Field(default=None, description="Description of what this connection is for")
    title: str | None = Field(default=None, description="Human-readable title")


class FieldInfo(BaseModel):
    """Information about a configuration field."""

    name: str = Field(description="Field name")
    type: str = Field(description="Field type (string, integer, number, boolean, array, object)")
    title: str | None = Field(default=None, description="Human-readable title")
    description: str | None = Field(default=None, description="Field description")
    required: bool = Field(default=False, description="Whether the field is required")
    default: Any = Field(default=None, description="Default value if any")
    enum: list[Any] | None = Field(default=None, description="Allowed values for enum fields")
    minimum: float | None = Field(default=None, description="Minimum value for numeric fields")
    maximum: float | None = Field(default=None, description="Maximum value for numeric fields")
    pattern: str | None = Field(default=None, description="Regex pattern for string fields")
    items: dict[str, Any] | None = Field(default=None, description="Schema for array items")
    properties: dict[str, Any] | None = Field(default=None, description="Nested object properties")
    # ComponentRef information
    is_component_ref: bool = Field(default=False, description="Whether this field is a component reference")
    ref_type: RefType | None = Field(default=None, description="Type of component reference if is_component_ref")
    is_ref_list: bool = Field(default=False, description="Whether this is a list of component references")
    # Suggested options for dropdown (not strict like enum - allows free text)
    options: list[str] | None = Field(default=None, description="Suggested values for dropdown selection")


class RegisteredTypeInfo(BaseModel):
    """Information about a registered component type."""

    full_type: str = Field(description="Full type identifier (module/name)")
    module_name: str = Field(description="Module name")
    local_name: str = Field(description="Local type name")
    # Display name parsed from docstring (e.g., "NIM LLM" from "Name: NIM LLM")
    display_name: str | None = Field(default=None, description="Human-readable display name from docstring")
    description: str | None = Field(default=None, description="Type description")
    json_schema: dict[str, Any] = Field(description="Full JSON Schema for configuration")
    fields: list[FieldInfo] = Field(default_factory=list, description="Parsed field information")
    is_per_user: bool = Field(default=False, description="Whether this is a per-user component")
    # Connection ports
    input_ports: list[ConnectionPort] = Field(
        default_factory=list, description="Input connection ports - fields that accept references to other components")
    # Custom icon URL for the component (e.g., provider logo)
    icon_url: str | None = Field(default=None, description="URL to custom SVG icon for this component type")


class ComponentTypeInfo(BaseModel):
    """Information about a component category and its registered types."""

    category: ComponentCategory = Field(description="Component category")
    display_name: str = Field(description="Human-readable display name")
    description: str = Field(description="Category description")
    registered_types: list[RegisteredTypeInfo] = Field(default_factory=list,
                                                       description="All registered types in this category")
    provides_ref_type: RefType | None = Field(
        default=None, description="The reference type that components in this category provide (for connections)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    registry_loaded: bool = Field(description="Whether the type registry is loaded")


# =============================================================================
# CONFIG VALIDATION MODELS
# =============================================================================


class ConfigValidationRequest(BaseModel):
    """Request to validate a YAML configuration."""

    yaml_content: str = Field(description="The YAML content to validate")


class ConfigValidationResponse(BaseModel):
    """Response from config validation."""

    valid: bool = Field(description="Whether the configuration is valid")
    error_message: str | None = Field(default=None, description="Error message if validation failed")
    error_details: list[str] | None = Field(default=None, description="Detailed error messages (e.g., per field)")
    config_dict: dict | None = Field(default=None, description="Parsed config dictionary if valid")


# =============================================================================
# CONFIG IMPORT MODELS
# =============================================================================


class Position(BaseModel):
    """2D position for a component on the canvas."""

    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")


class ImportedComponent(BaseModel):
    """A component parsed from a config file for placement on the canvas."""

    id: str = Field(description="Unique identifier (usually the config key name)")
    component_type: str = Field(description="UI component type (e.g., 'llm', 'agent', 'function')")
    name: str = Field(description="Display name for the component")
    position: Position = Field(description="Canvas position for layout")
    full_type: str = Field(description="Full type string from config (e.g., 'nim/NIMModelConfig')")
    config: dict[str, Any] = Field(default_factory=dict, description="Component configuration values")
    # Input ports for connections (same as RegisteredTypeInfo.input_ports)
    input_ports: list[ConnectionPort] = Field(default_factory=list, description="Input connection ports")
    # Fields for the config form (same as RegisteredTypeInfo.fields)
    fields: list[FieldInfo] = Field(default_factory=list, description="Configuration fields")
    # Icon URL for display
    icon_url: str | None = Field(default=None, description="Icon URL for the component")
    # Human-readable display name from docstring
    display_name: str | None = Field(default=None, description="Display name parsed from docstring")


class ImportedConnection(BaseModel):
    """A connection between two components parsed from the config."""

    id: str = Field(description="Unique connection identifier")
    source_id: str = Field(description="ID of the source component (provides the value)")
    target_id: str = Field(description="ID of the target component (receives the value)")
    target_field: str = Field(description="Field on target that receives the connection")
    ref_type: RefType = Field(description="Type of reference for this connection")


class ImportedWorkflowState(BaseModel):
    """Complete workflow state imported from a config file."""

    components: list[ImportedComponent] = Field(default_factory=list, description="All components to place")
    connections: list[ImportedConnection] = Field(default_factory=list, description="All connections to create")


# =============================================================================
# CONFIG EXPORT MODELS
# =============================================================================


class ExportComponent(BaseModel):
    """A component from the UI canvas to export."""

    id: str = Field(description="Unique component identifier (will be used as config key)")
    component_type: str = Field(description="UI component type (e.g., 'llm', 'agent', 'function')")
    full_type: str = Field(description="Full type string (e.g., 'nim/NIMModelConfig')")
    config: dict[str, Any] = Field(default_factory=dict, description="Component configuration values")


class ExportConnection(BaseModel):
    """A connection between components to export."""

    source_id: str = Field(description="ID of the source component")
    target_id: str = Field(description="ID of the target component")
    target_field: str = Field(description="Field on target that receives the connection")


class ExportWorkflowRequest(BaseModel):
    """Request to export a workflow to YAML configuration."""

    components: list[ExportComponent] = Field(description="All components on the canvas")
    connections: list[ExportConnection] = Field(description="All connections between components")
    workflow_name: str = Field(default="my_workflow", description="Name for the workflow")


class ExportConfigResponse(BaseModel):
    """Response from config export."""

    success: bool = Field(description="Whether the export was successful")
    yaml_content: str | None = Field(default=None, description="Generated YAML configuration")
    error_message: str | None = Field(default=None, description="Error message if export failed")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings during export")
