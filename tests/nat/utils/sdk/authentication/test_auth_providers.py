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
"""SDK tests for Authentication Provider configuration.

This module tests creating and configuring auth providers using the SDK.
"""

import pytest

from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from nat.authentication.api_key.api_key_auth_provider_config import HeaderAuthScheme
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins


@pytest.fixture(scope="module", autouse=True)
def discover_plugins():
    """Discover and register all plugins before running tests."""
    discover_and_register_plugins(PluginTypes.ALL)


# ============================================================================
# API Key Auth Provider SDK Initialization Tests
# ============================================================================


class TestAPIKeyAuthProviderInitialization:
    """Tests for APIKeyAuthProviderConfig initialization via SDK."""

    def test_basic_api_key_auth_creation(self):
        """Test creating a basic API key auth provider."""
        auth = APIKeyAuthProviderConfig(raw_key="test_api_key")

        assert auth.raw_key.get_secret_value() == "test_api_key"

    def test_api_key_auth_with_bearer_scheme(self):
        """Test API key auth with Bearer scheme."""
        auth = APIKeyAuthProviderConfig(
            raw_key="test_api_key",
            auth_scheme=HeaderAuthScheme.BEARER,
        )

        assert auth.auth_scheme == HeaderAuthScheme.BEARER

    def test_api_key_auth_with_x_api_key_scheme(self):
        """Test API key auth with X-API-Key scheme."""
        auth = APIKeyAuthProviderConfig(
            raw_key="test_api_key",
            auth_scheme=HeaderAuthScheme.X_API_KEY,
        )

        assert auth.auth_scheme == HeaderAuthScheme.X_API_KEY

    def test_api_key_auth_with_custom_scheme(self):
        """Test API key auth with custom scheme."""
        auth = APIKeyAuthProviderConfig(
            raw_key="test_api_key",
            auth_scheme=HeaderAuthScheme.CUSTOM,
            custom_header_name="X-Custom-Auth",
            custom_header_prefix="Token",
        )

        assert auth.auth_scheme == HeaderAuthScheme.CUSTOM
        assert auth.custom_header_name == "X-Custom-Auth"
        assert auth.custom_header_prefix == "Token"


# ============================================================================
# Auth Provider Secret Value Tests
# ============================================================================


class TestAuthProviderSecretValue:
    """Tests for auth provider secret value handling."""

    def test_raw_key_is_secret(self):
        """Test that raw_key is stored as a secret."""
        auth = APIKeyAuthProviderConfig(raw_key="super_secret_key")

        # The raw key should not expose the value directly
        assert str(auth.raw_key) != "super_secret_key"
        # But we can get it with get_secret_value()
        assert auth.raw_key.get_secret_value() == "super_secret_key"


# ============================================================================
# Auth Provider Default Values Tests
# ============================================================================


class TestAuthProviderDefaults:
    """Tests for auth provider default values."""

    def test_default_scheme_is_bearer(self):
        """Test that default scheme is Bearer."""
        auth = APIKeyAuthProviderConfig(raw_key="long_enough_key")

        assert auth.auth_scheme == HeaderAuthScheme.BEARER

    def test_custom_header_fields_are_optional(self):
        """Test that custom header fields are optional for non-custom schemes."""
        auth = APIKeyAuthProviderConfig(
            raw_key="long_enough_key",
            auth_scheme=HeaderAuthScheme.BEARER,
        )

        assert auth.custom_header_name is None or auth.custom_header_name == "Authorization"
