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

import asyncio
import logging
import os
import subprocess
import tempfile
import uuid
from typing import ClassVar

import httpx
from fastapi import APIRouter
from fastapi import HTTPException
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
from nat.workflow_builder_api.utils import export_workflow_to_yaml
from nat.workflow_builder_api.utils import get_category_types
from nat.workflow_builder_api.utils import parse_config_to_workflow_state
from nat.workflow_builder_api.utils import validate_yaml_config

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
# WORKFLOW START/STOP - Run workflows using nat serve
# =============================================================================


class StartWorkflowResponse(BaseModel):
    """Response from starting a workflow."""

    success: bool = Field(description="Whether the workflow started successfully")
    url: str = Field(default="", description="URL where the workflow UI is running")
    process_id: str = Field(default="", description="Process ID for tracking/stopping")
    config_path: str = Field(default="", description="Path to the temporary config file")
    error_message: str | None = Field(default=None, description="Error message if failed")


class WorkflowProcessManager:
    """Manages running nat serve processes and their associated UIs."""

    _instance: ClassVar["WorkflowProcessManager | None"] = None
    _processes: ClassVar[dict[str, dict]] = {}
    _next_backend_port: ClassVar[int] = 8101
    _next_ui_port: ClassVar[int] = 3101

    @classmethod
    def get(cls) -> "WorkflowProcessManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_next_ports(self) -> tuple[int, int]:
        """Get the next available backend and UI ports."""
        backend_port = WorkflowProcessManager._next_backend_port
        ui_port = WorkflowProcessManager._next_ui_port
        WorkflowProcessManager._next_backend_port += 1
        WorkflowProcessManager._next_ui_port += 1
        return backend_port, ui_port

    async def start_workflow(self, yaml_content: str) -> StartWorkflowResponse:
        """Start nat serve and nat-ui processes with the given config."""
        process_id = str(uuid.uuid4())

        # Write config to temp file
        temp_dir = tempfile.mkdtemp(prefix="nat_workflow_")
        config_path = os.path.join(temp_dir, "config.yml")
        with open(config_path, "w") as f:
            f.write(yaml_content)

        # Find available ports for backend and UI
        backend_port, ui_port = self.get_next_ports()
        backend_url = f"http://localhost:{backend_port}"
        ui_url = f"http://localhost:{ui_port}"

        # Get the nat-ui directory path
        # __file__ is src/nat/workflow_builder_api/routes.py
        # We need to go up 4 levels to get to the project root
        project_root = os.path.dirname(  # nat-fork/
            os.path.dirname(  # src/
                os.path.dirname(  # nat/
                    os.path.dirname(__file__)  # workflow_builder_api/
                )))
        nat_ui_dir = os.path.join(project_root, "external", "nat-ui")

        try:
            # Start nat serve process (backend)
            backend_process = subprocess.Popen(
                [
                    "nat",
                    "serve",
                    "--config_file",
                    config_path,
                    "--port",
                    str(backend_port),
                    "--host",
                    "0.0.0.0",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            logger.info("Started backend process %s on port %d", process_id, backend_port)

            # Wait for the backend to be ready (FastAPI - check /openapi.json)
            ready = await self._wait_for_server_ready(backend_url, timeout=60, check_path="/openapi.json")

            if not ready:
                # Check if process is still running
                if backend_process.poll() is not None:
                    # Process exited - get error output
                    _, stderr = backend_process.communicate()
                    return StartWorkflowResponse(
                        success=False,
                        error_message=f"Backend failed to start: {stderr[:500] if stderr else 'Unknown error'}",
                    )
                else:
                    # Still starting - continue anyway but warn
                    logger.warning("Backend not ready after timeout, continuing anyway")

            # Start the nat-ui (proxy + Next.js dev server)
            ui_env = os.environ.copy()
            ui_env["NAT_BACKEND_URL"] = backend_url
            ui_env["PORT"] = str(ui_port)
            ui_env["NEXT_TELEMETRY_DISABLED"] = "1"

            ui_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=nat_ui_dir,
                env=ui_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            logger.info("Started UI (npm run dev) on port %d pointing to %s", ui_port, backend_url)

            # Wait for the UI to be ready (Next.js compilation can take a while)
            # Check root path "/" for Next.js apps
            ui_ready = await self._wait_for_server_ready(ui_url, timeout=120, check_path="/")
            if not ui_ready:
                logger.warning("UI not ready after timeout, returning URL anyway - it may still be compiling")

            # Store process info
            WorkflowProcessManager._processes[process_id] = {
                "backend_process": backend_process,
                "ui_process": ui_process,
                "config_path": config_path,
                "temp_dir": temp_dir,
                "backend_port": backend_port,
                "ui_port": ui_port,
                "backend_url": backend_url,
                "url": ui_url,
            }

            return StartWorkflowResponse(
                success=True,
                url=ui_url,
                process_id=process_id,
                config_path=config_path,
            )

        except Exception as e:
            logger.exception("Failed to start workflow: %s", e)
            # Clean up temp file on error
            try:
                os.remove(config_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

            return StartWorkflowResponse(
                success=False,
                error_message=str(e),
            )

    async def _wait_for_server_ready(self, url: str, timeout: int = 60, check_path: str = "/openapi.json") -> bool:
        """Poll the server until it's ready or timeout.

        Args:
            url: Base URL to check
            timeout: Maximum seconds to wait
            check_path: Path to check for readiness (default: /openapi.json for FastAPI)
        """
        check_url = f"{url}{check_path}"
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient() as client:
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    response = await client.get(check_url, timeout=2.0)
                    if response.status_code == 200:
                        logger.info("Server is ready at %s", url)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                except Exception as e:
                    logger.debug("Server check error: %s", e)

                # Wait before retrying
                await asyncio.sleep(1.0)

        logger.warning("Server not ready after %d seconds", timeout)
        return False

    def stop_workflow(self, process_id: str) -> bool:
        """Stop running workflow processes (backend and UI)."""
        if process_id not in WorkflowProcessManager._processes:
            return False

        info = WorkflowProcessManager._processes.pop(process_id)

        # Stop backend process
        backend_process = info.get("backend_process") or info.get("process")
        if backend_process:
            try:
                backend_process.terminate()
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
            except Exception as e:
                logger.error("Error stopping backend process %s: %s", process_id, e)

        # Stop UI process
        ui_process = info.get("ui_process")
        if ui_process:
            try:
                ui_process.terminate()
                ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_process.kill()
            except Exception as e:
                logger.error("Error stopping UI process %s: %s", process_id, e)

        # Clean up temp files
        try:
            os.remove(info["config_path"])
            os.rmdir(info["temp_dir"])
        except Exception:
            pass

        logger.info("Stopped workflow processes %s", process_id)
        return True

    def cleanup_all(self) -> None:
        """Stop all running workflows."""
        for process_id in list(WorkflowProcessManager._processes.keys()):
            self.stop_workflow(process_id)


@router.post("/workflow/start", response_model=StartWorkflowResponse)
async def start_workflow_endpoint(request: ExportWorkflowRequest) -> StartWorkflowResponse:
    """
    Start a workflow using nat serve.

    This endpoint:
    1. Exports the workflow to a temporary YAML config file
    2. Starts `nat serve` with that config on an available port
    3. Waits for the server to be ready (health check)
    4. Returns the URL where the workflow UI is running

    The UI can then open this URL in a new browser tab.
    """
    try:
        # Export workflow to YAML
        export_result = export_workflow_to_yaml(request)

        if not export_result.success or not export_result.yaml_content:
            return StartWorkflowResponse(
                success=False,
                error_message=export_result.error_message or "Failed to export workflow",
            )

        # Start the workflow and wait for it to be ready
        manager = WorkflowProcessManager.get()
        return await manager.start_workflow(export_result.yaml_content)

    except Exception as e:
        logger.exception("Error starting workflow: %s", e)
        return StartWorkflowResponse(
            success=False,
            error_message=str(e),
        )


@router.delete("/workflow/stop/{process_id}")
async def stop_workflow(process_id: str) -> dict:
    """
    Stop a running workflow process.
    """
    manager = WorkflowProcessManager.get()
    stopped = manager.stop_workflow(process_id)

    if not stopped:
        raise HTTPException(status_code=404, detail="Process not found")

    return {"status": "stopped", "process_id": process_id}
