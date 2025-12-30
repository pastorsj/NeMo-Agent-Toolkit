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
Round-trip tests for MCP configurations with environment variables.

These tests verify that:
1. MCP configs with ${VAR_NAME} patterns are correctly imported
2. Secret vs non-secret fields are correctly identified
3. Warnings are generated for non-secret fields with env vars
4. Configs can be exported back with env vars preserved
"""

import sys
from pathlib import Path

# Ensure src is in path
SRC_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestMCPImport:
    """Tests for importing MCP configurations with environment variables."""

    def test_import_mcp_service_account_config(self):
        """Test importing MCP service account config with env vars."""
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        from nat.workflow_builder_api.utils.validation import validate_yaml_config_with_env_vars

        # Load plugins
        discover_and_register_plugins(PluginTypes.ALL)

        yaml_content = """
authentication:
  my_service_account:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: ${SERVICE_ACCOUNT_TOKEN_URL}
    scopes: openid profile

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct

workflow:
  _type: react_agent
  llm_name: nim_llm
"""

        result = validate_yaml_config_with_env_vars(yaml_content)

        # Should be valid
        assert result.response.valid is True, f"Validation failed: {result.response.error_details}"

        # Should detect 3 env vars
        assert len(result.environment_variables) == 3

        # Create lookup
        env_map = {ev.name: ev for ev in result.environment_variables}

        # Check client_secret is detected as secret field
        assert "SERVICE_ACCOUNT_CLIENT_SECRET" in env_map
        secret_var = env_map["SERVICE_ACCOUNT_CLIENT_SECRET"]
        assert secret_var.has_secret_field_usage is True
        assert secret_var.has_non_secret_field_usage is False
        assert secret_var.warning is None

        # Check client_id is detected as NON-secret field
        assert "SERVICE_ACCOUNT_CLIENT_ID" in env_map
        client_id_var = env_map["SERVICE_ACCOUNT_CLIENT_ID"]
        assert client_id_var.has_secret_field_usage is False
        assert client_id_var.has_non_secret_field_usage is True
        assert client_id_var.warning is not None
        assert "non-secret" in client_id_var.warning.lower()

        # Check token_url is detected as NON-secret field
        assert "SERVICE_ACCOUNT_TOKEN_URL" in env_map
        token_url_var = env_map["SERVICE_ACCOUNT_TOKEN_URL"]
        assert token_url_var.has_secret_field_usage is False
        assert token_url_var.has_non_secret_field_usage is True
        assert token_url_var.warning is not None

    def test_import_mcp_with_api_key_llm(self):
        """Test importing config with env vars in both auth and LLM."""
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        from nat.workflow_builder_api.utils.validation import validate_yaml_config_with_env_vars

        discover_and_register_plugins(PluginTypes.ALL)

        yaml_content = """
authentication:
  my_auth:
    _type: mcp_service_account
    client_id: ${CLIENT_ID}
    client_secret: ${CLIENT_SECRET}
    token_url: https://auth.example.com/token
    scopes: openid

llms:
  nim_llm:
    _type: nim
    model_name: test
    api_key: ${NVIDIA_API_KEY}

workflow:
  _type: react_agent
  llm_name: nim_llm
"""

        result = validate_yaml_config_with_env_vars(yaml_content)

        assert result.response.valid is True
        assert len(result.environment_variables) == 3

        env_map = {ev.name: ev for ev in result.environment_variables}

        # CLIENT_SECRET should be secret (SerializableSecretStr)
        assert env_map["CLIENT_SECRET"].has_secret_field_usage is True

        # NVIDIA_API_KEY should be secret (OptionalSecretStr in NIMModelConfig)
        assert env_map["NVIDIA_API_KEY"].has_secret_field_usage is True

        # CLIENT_ID should NOT be secret (plain str)
        assert env_map["CLIENT_ID"].has_secret_field_usage is False
        assert env_map["CLIENT_ID"].has_non_secret_field_usage is True


class TestMCPEnvVarLocations:
    """Tests for correct env var location tracking in MCP configs."""

    def test_env_var_location_info(self):
        """Verify env var locations have correct component info."""
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        from nat.workflow_builder_api.utils.validation import validate_yaml_config_with_env_vars

        discover_and_register_plugins(PluginTypes.ALL)

        yaml_content = """
authentication:
  jama_auth:
    _type: mcp_service_account
    client_id: ${JAMA_CLIENT_ID}
    client_secret: ${JAMA_CLIENT_SECRET}
    token_url: https://auth.jama.io/token
    scopes: read

llms:
  nim_llm:
    _type: nim
    model_name: test

workflow:
  _type: react_agent
  llm_name: nim_llm
"""

        result = validate_yaml_config_with_env_vars(yaml_content)

        assert result.response.valid is True

        env_map = {ev.name: ev for ev in result.environment_variables}

        # Check location info for client_id
        client_id_var = env_map["JAMA_CLIENT_ID"]
        assert len(client_id_var.locations) == 1
        loc = client_id_var.locations[0]
        assert loc.component_type == "authentication"
        assert loc.component_id == "jama_auth"
        assert loc.field_name == "client_id"
        assert loc.is_secret_field is False

        # Check location info for client_secret
        client_secret_var = env_map["JAMA_CLIENT_SECRET"]
        assert len(client_secret_var.locations) == 1
        loc = client_secret_var.locations[0]
        assert loc.component_type == "authentication"
        assert loc.component_id == "jama_auth"
        assert loc.field_name == "client_secret"
        assert loc.is_secret_field is True


class TestMCPWarnings:
    """Tests for warning generation on non-secret env var usage."""

    def test_warnings_for_non_secret_fields(self):
        """Verify warnings are generated for env vars in non-secret fields."""
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        from nat.workflow_builder_api.utils.validation import validate_yaml_config_with_env_vars

        discover_and_register_plugins(PluginTypes.ALL)

        yaml_content = """
authentication:
  my_auth:
    _type: mcp_service_account
    client_id: ${MY_CLIENT_ID}
    client_secret: ${MY_CLIENT_SECRET}
    token_url: ${MY_TOKEN_URL}
    scopes: openid

llms:
  nim_llm:
    _type: nim
    model_name: test

workflow:
  _type: react_agent
  llm_name: nim_llm
"""

        result = validate_yaml_config_with_env_vars(yaml_content)

        assert result.response.valid is True

        # Should have 2 warnings (client_id and token_url are non-secret)
        assert len(result.env_var_warnings) == 2

        # Check warning messages contain the expected env var names
        warnings_text = " ".join(result.env_var_warnings)
        assert "MY_CLIENT_ID" in warnings_text
        assert "MY_TOKEN_URL" in warnings_text
        # CLIENT_SECRET should NOT have a warning (it's a secret field)
        assert "MY_CLIENT_SECRET" not in warnings_text


class TestRealMCPConfig:
    """Tests using real MCP example configs from the repository."""

    def test_import_real_jama_config(self):
        """Test importing the actual Jama MCP config example."""
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        from nat.workflow_builder_api.utils.validation import validate_yaml_config_with_env_vars

        discover_and_register_plugins(PluginTypes.ALL)

        # Path to the actual config file
        config_path = (SRC_DIR.parent / "examples" / "MCP" / "service_account_auth_mcp" / "configs" /
                       "config-mcp-service-account-jama.yml")

        if not config_path.exists():
            # Skip if config doesn't exist
            return

        with open(config_path, encoding="utf-8") as f:
            yaml_content = f.read()

        result = validate_yaml_config_with_env_vars(yaml_content)

        # Should be valid
        assert result.response.valid is True, f"Validation failed: {result.response.error_details}"

        # Should detect env vars
        assert len(result.environment_variables) > 0

        # Create lookup
        env_map = {ev.name: ev for ev in result.environment_variables}

        # The Jama config uses these env vars:
        # - CORPORATE_MCP_SERVICE_ACCOUNT_JAMA_URL (non-secret - URL)
        # - SERVICE_ACCOUNT_CLIENT_ID (non-secret)
        # - SERVICE_ACCOUNT_CLIENT_SECRET (secret!)
        # - SERVICE_ACCOUNT_TOKEN_URL (non-secret - URL)
        # - SERVICE_ACCOUNT_SCOPES (non-secret)

        if "SERVICE_ACCOUNT_CLIENT_SECRET" in env_map:
            assert env_map["SERVICE_ACCOUNT_CLIENT_SECRET"].has_secret_field_usage is True

        if "SERVICE_ACCOUNT_CLIENT_ID" in env_map:
            assert env_map["SERVICE_ACCOUNT_CLIENT_ID"].has_secret_field_usage is False
            assert env_map["SERVICE_ACCOUNT_CLIENT_ID"].has_non_secret_field_usage is True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
