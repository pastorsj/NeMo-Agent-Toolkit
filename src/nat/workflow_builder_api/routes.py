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
FastAPI routes for the Workflow Builder API.

Provides endpoints for:
- Health checks
- Component category listing
- Component type information
- Configuration validation
"""

import logging

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from pydantic import BaseModel
from pydantic import Field

from nat.cli.type_registry import GlobalTypeRegistry
from nat.workflow_builder_api.constants import UI_CATEGORIES
from nat.workflow_builder_api.models import ComponentCategory
from nat.workflow_builder_api.models import ComponentTypeInfo
from nat.workflow_builder_api.models import ConfigValidationRequest
from nat.workflow_builder_api.models import ConfigValidationResponse
from nat.workflow_builder_api.models import ExportConfigResponse
from nat.workflow_builder_api.models import ExportWorkflowRequest
from nat.workflow_builder_api.models import HealthResponse
from nat.workflow_builder_api.models import ImportedWorkflowState
from nat.workflow_builder_api.session_registry import WorkflowSessionRegistry
from nat.workflow_builder_api.utils import export_workflow_to_yaml
from nat.workflow_builder_api.utils import get_category_types
from nat.workflow_builder_api.utils import parse_config_to_workflow_state
from nat.workflow_builder_api.utils import validate_yaml_config
from nat.workflow_builder_api.utils.config_builder import ConfigBuildError
from nat.workflow_builder_api.utils.config_builder import build_config_from_request
from nat.workflow_builder_api.utils.config_builder import validate_workflow_for_execution
from nat.workflow_builder_api.websocket_handler import WorkflowBuilderWebSocketHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["registry"])

# =============================================================================
# HEALTH CHECK
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health of the API and registry."""
    try:
        registry = GlobalTypeRegistry.get()
        _ = registry.get_registered_llm_providers()
        return HealthResponse(status="healthy", registry_loaded=True)
    except Exception:
        return HealthResponse(status="healthy", registry_loaded=False)


# =============================================================================
# COMPONENT CATEGORIES
# =============================================================================


@router.get("/categories", response_model=list[ComponentCategory])
async def get_available_categories() -> list[ComponentCategory]:
    """Get list of available component categories."""
    return list(UI_CATEGORIES)


@router.get("/components/{category}", response_model=ComponentTypeInfo)
async def get_category_components(category: ComponentCategory) -> ComponentTypeInfo:
    """Get all registered component types for a specific category."""
    if category not in UI_CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' is not available. Use /categories to see available categories.",
        )

    try:
        return get_category_types(category)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error loading category %s: %s", category, e)
        raise HTTPException(status_code=500, detail=f"Failed to load category: {str(e)}") from e


# =============================================================================
# CONFIG VALIDATION
# =============================================================================


@router.post("/config/validate", response_model=ConfigValidationResponse)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """Validate a YAML configuration file against the NAT Config schema."""
    return validate_yaml_config(request.yaml_content)


# =============================================================================
# CONFIG IMPORT
# =============================================================================


@router.post("/config/import", response_model=ImportedWorkflowState)
async def import_config(request: ConfigValidationRequest) -> ImportedWorkflowState:
    """
    Import a YAML configuration file and return a workflow state for the UI.

    This endpoint:
    1. Validates the YAML configuration
    2. Parses it into components and connections
    3. Calculates layout positions for each component
    4. Returns the complete workflow state

    The returned state can be loaded directly into the UI canvas.
    """
    # First validate the config
    validation_result = validate_yaml_config(request.yaml_content)

    if not validation_result.valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": validation_result.error_message,
                "errors": validation_result.error_details,
            },
        )

    if not validation_result.config_dict:
        raise HTTPException(
            status_code=400,
            detail={"message": "Config validation succeeded but config_dict is empty"},
        )

    # Parse the config into workflow state
    try:
        workflow_state = parse_config_to_workflow_state(validation_result.config_dict)
        return workflow_state
    except Exception as e:
        logger.error("Error parsing config: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"message": f"Failed to parse configuration: {str(e)}"},
        ) from e


# =============================================================================
# CONFIG EXPORT
# =============================================================================


@router.post("/config/export", response_model=ExportConfigResponse)
async def export_config(request: ExportWorkflowRequest) -> ExportConfigResponse:
    """
    Export a workflow from the UI to a YAML configuration file.

    This endpoint:
    1. Takes the current workflow state (components and connections)
    2. Converts it to a valid NAT configuration YAML format
    3. Returns the YAML string for download

    The exported config can be used with `nat run` or `nat serve` commands.
    """
    try:
        result = export_workflow_to_yaml(request)

        if not result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": result.error_message, "warnings": result.warnings
                },
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting config: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"message": f"Failed to export configuration: {str(e)}"},
        ) from e


# =============================================================================
# WORKFLOW SESSION MANAGEMENT
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a workflow session."""

    components: list = Field(default_factory=list, description="Components from the UI canvas")
    connections: list = Field(default_factory=list, description="Connections between components")


class CreateSessionResponse(BaseModel):
    """Response from creating a workflow session."""

    session_id: str = Field(description="Unique session identifier")
    websocket_path: str = Field(description="Path to WebSocket endpoint for this session")


class ValidateWorkflowRequest(BaseModel):
    """Request to validate a workflow for execution."""

    components: list = Field(default_factory=list, description="Components from the UI canvas")
    connections: list = Field(default_factory=list, description="Connections between components")


class ValidateWorkflowResponse(BaseModel):
    """Response from workflow validation."""

    valid: bool = Field(description="Whether the workflow is valid for execution")
    errors: list[str] = Field(default_factory=list, description="List of validation errors")


@router.post("/workflow/validate", response_model=ValidateWorkflowResponse)
async def validate_workflow(request: ValidateWorkflowRequest) -> ValidateWorkflowResponse:
    """
    Validate that a workflow is ready for execution.

    Checks that:
    - A Workflow component exists
    - The Workflow has an entrypoint connected
    - The entrypoint is properly configured
    """
    from nat.workflow_builder_api.models import ExportComponent
    from nat.workflow_builder_api.models import ExportConnection

    try:
        components = [ExportComponent(**c) for c in request.components]
        connections = [ExportConnection(**c) for c in request.connections]

        errors = validate_workflow_for_execution(components, connections)

        return ValidateWorkflowResponse(
            valid=len(errors) == 0,
            errors=errors,
        )
    except Exception as e:
        logger.exception("Error validating workflow: %s", e)
        return ValidateWorkflowResponse(
            valid=False,
            errors=[f"Validation error: {e!s}"],
        )


@router.post("/workflow/session/create", response_model=CreateSessionResponse)
async def create_workflow_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """
    Create a new workflow session for the chat interface.

    This endpoint:
    1. Validates the workflow components and connections
    2. Builds a Config object from the workflow
    3. Creates a SessionManager for running the workflow
    4. Returns a session ID for WebSocket connection
    """
    from nat.workflow_builder_api.models import ExportComponent
    from nat.workflow_builder_api.models import ExportConnection

    try:
        # Parse components and connections
        components = [ExportComponent(**c) for c in request.components]
        connections = [ExportConnection(**c) for c in request.connections]

        # Validate the workflow
        errors = validate_workflow_for_execution(components, connections)
        if errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Workflow validation failed", "errors": errors
                },
            )

        # Build the Export request
        export_request = ExportWorkflowRequest(
            components=components,
            connections=connections,
            workflow_name="runtime_workflow",
        )

        # Build and validate the config
        config = build_config_from_request(export_request)

        # Log the config for debugging
        logger.info("Built config for session:")
        logger.info("  LLMs: %s", list(config.llms.keys()) if config.llms else [])
        logger.info("  Functions: %s", list(config.functions.keys()) if config.functions else [])
        logger.info("  Workflow type: %s", type(config.workflow).__name__ if config.workflow else None)

        # Create the session
        registry = WorkflowSessionRegistry.get()
        session = await registry.create_session(config)

        return CreateSessionResponse(
            session_id=session.session_id,
            websocket_path=f"/api/v1/workflow/session/{session.session_id}/ws",
        )

    except ConfigBuildError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": e.message, "errors": e.details
            },
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating workflow session: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"message": f"Failed to create session: {e!s}"},
        ) from e


@router.delete("/workflow/session/{session_id}")
async def destroy_workflow_session(session_id: str) -> dict:
    """
    Destroy a workflow session and clean up resources.
    """
    registry = WorkflowSessionRegistry.get()
    destroyed = await registry.destroy_session(session_id)

    if not destroyed:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "destroyed", "session_id": session_id}


@router.websocket("/workflow/session/{session_id}/ws")
async def workflow_session_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for chatting with a workflow session.

    Connect to this endpoint after creating a session to send messages
    and receive streaming responses.
    """
    registry = WorkflowSessionRegistry.get()
    session = await registry.get_session(session_id)

    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    try:
        async with WorkflowBuilderWebSocketHandler(websocket, session) as handler:
            await handler.run()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as e:
        logger.exception("WebSocket error for session %s: %s", session_id, e)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
