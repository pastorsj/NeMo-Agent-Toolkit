# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SDK classes for Front Ends.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.utils.sdk.nat_auth_provider import NatAuthProvider
from nat.utils.sdk.nat_front_end import NatFrontEnd

from .console.console_front_end_config import ConsoleFrontEndConfig
from .fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from .mcp.mcp_front_end_config import MCPFrontEndConfig


class MCPFrontEnd(MCPFrontEndConfig, NatFrontEnd):
    """MCP Front End"""

    tool_names: list[str] = Field(default_factory=list,
                                  description="The list of tools MCP server will expose (default: all tools)."
                                  "Tool names can be functions or function groups",
                                  init=False)

    tools: list[FunctionBaseConfig] | None = Field(default=None, exclude=True)

    auth_provider_obj: NatAuthProvider | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set tool names and auth provider from objects if provided."""
        if self.tools:
            self.tool_names = [tool.computed_name for tool in self.tools]
        if self.auth_provider_obj:
            self.auth_provider = AuthenticationRef(value=self.auth_provider_obj.computed_name)
        return self


class ConsoleFrontEnd(ConsoleFrontEndConfig, NatFrontEnd):
    """Console Front End"""
    user_id: str | None = Field(default=None, description="User ID to use for the workflow session.")


class FastApiFrontEnd(FastApiFrontEndConfig, NatFrontEnd):
    """FastAPI Front End"""

    object_store: ObjectStoreRef | None = Field(
        default=None,
        description=(
            "Object store reference for the FastAPI app. If present, static files can be uploaded via a POST "
            "request to '/static' and files will be served from the object store. The files will be served from the "
            "object store at '/static/{file_name}'."),
        init=False)

    object_store_obj: ObjectStoreBaseConfig | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set object store reference from object store object if provided."""
        if self.object_store_obj:
            self.object_store = ObjectStoreRef(value=self.object_store_obj.computed_name)
        return self
