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
SDK classes for Authentication providers.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from nat.utils.sdk.nat_auth_provider import NatAuthProvider

from .api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from .http_basic_auth.register import HTTPBasicAuthProviderConfig
from .oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from .oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig


class HTTPBasicAuth(HTTPBasicAuthProviderConfig, NatAuthProvider):
    """HTTP Basic Authentication Provider"""


class OAuth2AuthCodeFlow(OAuth2AuthCodeFlowProviderConfig, NatAuthProvider):
    """OAuth2 Authorization Code Flow Provider"""


class APIKeyAuth(APIKeyAuthProviderConfig, NatAuthProvider):
    """API Key Authentication Provider"""


class OAuth2ResourceServer(OAuth2ResourceServerConfig, NatAuthProvider):
    """OAuth2 Resource Server Authentication Provider"""
