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
SDK classes for MCP plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.utils.sdk.nat_auth_provider import NatAuthProvider
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_function_group import NatFunctionGroup

from .auth.auth_provider_config import MCPOAuth2ProviderConfig
from .auth.service_account.provider_config import MCPServiceAccountProviderConfig
from .client_config import MCPClientConfig
from .tool import MCPToolConfig


class MCPClient(MCPClientConfig, NatFunctionGroup):
    """MCP Client Function Group"""

    auth_provider_obj: NatAuthProvider | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set auth provider reference from object if provided."""
        if self.auth_provider_obj:
            # We need to modify the nested server config
            self.server.auth_provider = AuthenticationRef(value=self.auth_provider_obj.computed_name)
        return self


class MCPTool(MCPToolConfig, NatFunction):
    """MCP Tool Wrapper"""


class MCPServiceAccountProvider(MCPServiceAccountProviderConfig, NatAuthProvider):
    """MCP Service Account Authentication Provider"""


class MCPOAuth2Provider(MCPOAuth2ProviderConfig, NatAuthProvider):
    """MCP OAuth2 Authentication Provider"""

    token_storage_object_store: str | None = Field(
        default=None,
        description="Reference to object store for secure token storage. If None, uses in-memory storage.",
        init=False,
    )

    token_storage_object_store_obj: ObjectStoreBaseConfig | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set object store reference from object store object if provided."""
        if self.token_storage_object_store_obj:
            self.token_storage_object_store = ObjectStoreRef(value=self.token_storage_object_store_obj.computed_name)
        return self
