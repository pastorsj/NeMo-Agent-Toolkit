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
FastAPI application for the Workflow Builder API.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nat.workflow_builder_api.routes import router

logger = logging.getLogger(__name__)


def create_app(
    title: str = "NAT Workflow Builder API",
    description: str = "API for discovering NAT component types and their configuration schemas",
    version: str = "1.0.0",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create the FastAPI application for the Workflow Builder API.

    Args:
        title: API title.
        description: API description.
        version: API version.
        cors_origins: List of allowed CORS origins. Defaults to allowing localhost.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        """Application lifespan handler."""
        logger.info("Starting NAT Workflow Builder API")

        # Discover and load all NAT plugins to populate the registry
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins

        discover_and_register_plugins(PluginTypes.ALL)

        logger.info("NAT plugins loaded, TypeRegistry populated")

        yield

        logger.info("Shutting down NAT Workflow Builder API")

    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
    )

    # Configure CORS
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:3099",
            "http://localhost:3100",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3099",
            "http://127.0.0.1:3100",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    return app


# Create default app instance for uvicorn
def get_app() -> FastAPI:
    """Get the default FastAPI app instance."""
    return create_app()
