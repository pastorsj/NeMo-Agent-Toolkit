# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Config builder utility for converting UI workflow state to a validated Config object.

This module extends the export functionality to produce a validated Config object
that can be used to create a SessionManager for running workflows.
"""

from __future__ import annotations

import logging

from nat.data_models.config import Config
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils.data_models.schema_validator import validate_schema
from nat.workflow_builder_api.models import ExportComponent
from nat.workflow_builder_api.models import ExportConnection
from nat.workflow_builder_api.models import ExportWorkflowRequest
from nat.workflow_builder_api.utils.config_exporter import _build_config_dict

logger = logging.getLogger(__name__)


class ConfigBuildError(Exception):
    """Raised when config building fails."""

    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or []


def build_config_from_workflow(
    components: list[ExportComponent],
    connections: list[ExportConnection],
) -> Config:
    """
    Build a validated Config object from UI workflow components and connections.

    Args:
        components: List of components from the UI canvas.
        connections: List of connections between components.

    Returns:
        A validated Config object ready for use with SessionManager.

    Raises:
        ConfigBuildError: If the config cannot be built or validated.
    """
    # Ensure all plugins are loaded
    discover_and_register_plugins(PluginTypes.ALL)

    # Create request object
    request = ExportWorkflowRequest(
        components=components,
        connections=connections,
        workflow_name="runtime_workflow",
    )

    # Build the config dictionary
    warnings: list[str] = []
    try:
        config_dict = _build_config_dict(request, warnings)
    except Exception as e:
        logger.exception("Error building config dict: %s", e)
        raise ConfigBuildError(
            message=f"Failed to build configuration: {e!s}",
            details=[str(e)],
        ) from e

    # Validate against Config schema
    try:
        # Rebuild annotations to include all registered types
        Config.rebuild_annotations()
        validated_config = validate_schema(config_dict, Config)
        return validated_config
    except Exception as e:
        logger.exception("Config validation failed: %s", e)
        raise ConfigBuildError(
            message=f"Configuration validation failed: {e!s}",
            details=warnings + [str(e)],
        ) from e


def build_config_from_request(request: ExportWorkflowRequest) -> Config:
    """
    Build a validated Config object from an ExportWorkflowRequest.

    Args:
        request: The export workflow request containing components and connections.

    Returns:
        A validated Config object ready for use with SessionManager.

    Raises:
        ConfigBuildError: If the config cannot be built or validated.
    """
    return build_config_from_workflow(
        components=request.components,
        connections=request.connections,
    )


def validate_workflow_for_execution(
    components: list[ExportComponent],
    connections: list[ExportConnection],
) -> list[str]:
    """
    Validate that a workflow is ready for execution.

    Args:
        components: List of components from the UI canvas.
        connections: List of connections between components.

    Returns:
        List of validation errors. Empty list means the workflow is valid.
    """
    errors: list[str] = []

    # Check for workflow component
    workflow_component = next(
        (c for c in components if c.component_type == "nat_workflow"),
        None,
    )
    if not workflow_component:
        errors.append("Missing Workflow component. Add a Workflow component to define the entrypoint.")
        return errors

    # Check for entrypoint connection
    entrypoint_connection = next(
        (conn for conn in connections if conn.target_id == workflow_component.id and conn.target_field == "entrypoint"),
        None,
    )
    if not entrypoint_connection:
        errors.append("Workflow has no entrypoint. Connect a function or agent to the Workflow's entrypoint.")
        return errors

    # Check that entrypoint component exists
    entrypoint_component = next(
        (c for c in components if c.id == entrypoint_connection.source_id),
        None,
    )
    if not entrypoint_component:
        errors.append(f"Entrypoint component '{entrypoint_connection.source_id}' not found.")
        return errors

    # Check that entrypoint has a valid type
    if not entrypoint_component.full_type:
        errors.append("Entrypoint component is not configured. Click on it to select a type.")

    return errors
