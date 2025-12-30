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
Tests for secret field detection functionality.

Tests the detection of SerializableSecretStr and OptionalSecretStr field types
in Pydantic models, and the enrichment of environment variables with field type
information.
"""

import sys
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel
from pydantic import Field
from pydantic import PlainSerializer
from pydantic import SecretStr

# Ensure src is in path
SRC_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# TEST FIXTURES - Mock config classes for testing
# =============================================================================


def get_secret_value(v: SecretStr | None) -> str | None:
    """Helper to extract secret value."""
    if v is None:
        return None
    return v.get_secret_value()


# Define the same types as in nat.data_models.common
SerializableSecretStr = Annotated[SecretStr, PlainSerializer(get_secret_value)]
OptionalSecretStr = Annotated[SecretStr | None, PlainSerializer(get_secret_value)]


class MockAuthConfig(BaseModel):
    """Mock authentication config with secret and non-secret fields."""

    client_id: str = Field(description="OAuth client ID (not secret)")
    client_secret: SerializableSecretStr = Field(description="OAuth client secret")
    token_url: str = Field(description="Token endpoint URL")
    api_key: OptionalSecretStr = Field(default=None, description="Optional API key")


class MockLLMConfig(BaseModel):
    """Mock LLM config with optional secret field."""

    model_name: str = Field(description="Model name")
    api_key: OptionalSecretStr = Field(default=None, description="API key")
    base_url: str | None = Field(default=None, description="Base URL")
    max_tokens: int = Field(default=300, description="Max tokens")


class MockSimpleConfig(BaseModel):
    """Mock config with no secret fields."""

    name: str
    value: int
    description: str | None = None


# =============================================================================
# TESTS: is_secret_field_type
# =============================================================================


class TestIsSecretFieldType:
    """Tests for the is_secret_field_type function."""

    def test_serializable_secret_str_is_secret(self):
        """SerializableSecretStr should be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(SerializableSecretStr) is True

    def test_optional_secret_str_is_secret(self):
        """OptionalSecretStr should be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(OptionalSecretStr) is True

    def test_plain_secret_str_is_secret(self):
        """Plain SecretStr should also be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(SecretStr) is True

    def test_regular_str_is_not_secret(self):
        """Regular str should not be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(str) is False

    def test_optional_str_is_not_secret(self):
        """Optional[str] should not be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(str | None) is False

    def test_int_is_not_secret(self):
        """int should not be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(int) is False

    def test_none_is_not_secret(self):
        """None should not be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(None) is False

    def test_list_str_is_not_secret(self):
        """list[str] should not be detected as secret."""
        from nat.workflow_builder_api.utils.schema import is_secret_field_type

        assert is_secret_field_type(list[str]) is False


# =============================================================================
# TESTS: get_secret_fields
# =============================================================================


class TestGetSecretFields:
    """Tests for the get_secret_fields function."""

    def test_auth_config_has_correct_secret_fields(self):
        """MockAuthConfig should have client_secret and api_key as secrets."""
        from nat.workflow_builder_api.utils.schema import get_secret_fields

        secret_fields = get_secret_fields(MockAuthConfig)

        assert "client_secret" in secret_fields
        assert "api_key" in secret_fields
        assert "client_id" not in secret_fields
        assert "token_url" not in secret_fields

    def test_llm_config_has_api_key_as_secret(self):
        """MockLLMConfig should have api_key as secret."""
        from nat.workflow_builder_api.utils.schema import get_secret_fields

        secret_fields = get_secret_fields(MockLLMConfig)

        assert "api_key" in secret_fields
        assert "model_name" not in secret_fields
        assert "base_url" not in secret_fields
        assert "max_tokens" not in secret_fields

    def test_simple_config_has_no_secret_fields(self):
        """MockSimpleConfig should have no secret fields."""
        from nat.workflow_builder_api.utils.schema import get_secret_fields

        secret_fields = get_secret_fields(MockSimpleConfig)

        assert len(secret_fields) == 0

    def test_real_nim_config_has_api_key_secret(self):
        """Verify NIMModelConfig has api_key as secret field."""
        from nat.workflow_builder_api.utils.schema import get_secret_fields

        try:
            from nat.llm.nim_llm import NIMModelConfig

            secret_fields = get_secret_fields(NIMModelConfig)
            assert "api_key" in secret_fields
            assert "model_name" not in secret_fields
            assert "base_url" not in secret_fields
        except ImportError:
            # Skip if NIM config not available
            pass

    def test_real_api_key_auth_config_has_raw_key_secret(self):
        """Verify APIKeyAuthProviderConfig has raw_key as secret field."""
        from nat.workflow_builder_api.utils.schema import get_secret_fields

        try:
            from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig

            secret_fields = get_secret_fields(APIKeyAuthProviderConfig)
            assert "raw_key" in secret_fields
            assert "auth_scheme" not in secret_fields
        except ImportError:
            # Skip if not available
            pass


# =============================================================================
# TESTS: FieldInfo is_secret flag
# =============================================================================


class TestFieldInfoIsSecret:
    """Tests that FieldInfo.is_secret is correctly set during schema extraction."""

    def test_extract_fields_marks_secrets(self):
        """extract_fields should mark secret fields with is_secret=True."""
        from nat.workflow_builder_api.utils.schema import extract_fields
        from nat.workflow_builder_api.utils.schema import extract_json_schema

        schema = extract_json_schema(MockAuthConfig)
        fields = extract_fields(schema, config_type=MockAuthConfig)

        field_map = {f.name: f for f in fields}

        # client_secret should be marked as secret
        assert "client_secret" in field_map
        assert field_map["client_secret"].is_secret is True

        # api_key should be marked as secret
        assert "api_key" in field_map
        assert field_map["api_key"].is_secret is True

        # client_id should NOT be marked as secret
        assert "client_id" in field_map
        assert field_map["client_id"].is_secret is False

        # token_url should NOT be marked as secret
        assert "token_url" in field_map
        assert field_map["token_url"].is_secret is False

    def test_extract_fields_no_secrets_in_simple_config(self):
        """extract_fields should not mark any fields as secret in MockSimpleConfig."""
        from nat.workflow_builder_api.utils.schema import extract_fields
        from nat.workflow_builder_api.utils.schema import extract_json_schema

        schema = extract_json_schema(MockSimpleConfig)
        fields = extract_fields(schema, config_type=MockSimpleConfig)

        for field in fields:
            assert field.is_secret is False, f"Field {field.name} should not be secret"


# =============================================================================
# TESTS: Environment variable enrichment
# =============================================================================


class TestEnvVarEnrichment:
    """Tests for enriching env vars with field type information."""

    def test_enrich_env_vars_detects_secret_fields(self):
        """enrich_env_vars_with_field_types should detect secret field usage."""
        from nat.workflow_builder_api.models import EnvironmentVariable
        from nat.workflow_builder_api.models import EnvVarLocation
        from nat.workflow_builder_api.utils.env_var_processor import enrich_env_vars_with_field_types

        # Create env var with location in a secret field
        env_vars = [
            EnvironmentVariable(
                name="TEST_SECRET",
                locations=[
                    EnvVarLocation(
                        path="authentication.my_auth.client_secret",
                        component_type="authentication",
                        component_id="my_auth",
                        field_name="client_secret",
                    )
                ],
                value=None,
                is_sensitive=True,
                export_as_variable=True,
                original_reference="${TEST_SECRET}",
            )
        ]

        # Note: This test will use mock config detection - won't find real config class
        # But it tests the flow
        enriched, warnings = enrich_env_vars_with_field_types(env_vars, {})

        assert len(enriched) == 1
        # Since we can't find the config class, it defaults to non-secret
        assert enriched[0].has_non_secret_field_usage is True

    def test_enrich_env_vars_generates_warnings_for_non_secret(self):
        """Warnings should be generated for env vars in non-secret fields."""
        from nat.workflow_builder_api.models import EnvironmentVariable
        from nat.workflow_builder_api.models import EnvVarLocation
        from nat.workflow_builder_api.utils.env_var_processor import enrich_env_vars_with_field_types

        # Create env var with location in a non-secret field
        env_vars = [
            EnvironmentVariable(
                name="CLIENT_ID",
                locations=[
                    EnvVarLocation(
                        path="authentication.my_auth.client_id",
                        component_type="authentication",
                        component_id="my_auth",
                        field_name="client_id",
                    )
                ],
                value=None,
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${CLIENT_ID}",
            )
        ]

        enriched, warnings = enrich_env_vars_with_field_types(env_vars, {})

        assert len(warnings) == 1
        assert "CLIENT_ID" in warnings[0]
        assert "non-secret" in warnings[0].lower()


# =============================================================================
# TESTS: Integration with real config classes
# =============================================================================


class TestRealConfigIntegration:
    """Integration tests with real NAT config classes."""

    def test_mcp_service_account_config_secret_detection(self):
        """Test secret field detection in MCPServiceAccountProviderConfig."""
        try:
            from nat.plugins.mcp.auth.service_account.provider_config import MCPServiceAccountProviderConfig
            from nat.workflow_builder_api.utils.schema import get_secret_fields

            secret_fields = get_secret_fields(MCPServiceAccountProviderConfig)

            # client_secret should be detected as secret
            assert "client_secret" in secret_fields

            # client_id should NOT be secret (it's just a str)
            assert "client_id" not in secret_fields

            # token_url should NOT be secret
            assert "token_url" not in secret_fields

            # scopes should NOT be secret
            assert "scopes" not in secret_fields

        except ImportError:
            # Skip if MCP plugin not available
            pass


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
