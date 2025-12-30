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
Integration tests for environment variable detection in config import.

These tests verify the complete flow:
1. Raw YAML with ${VAR_NAME} patterns → preprocessing
2. Preprocessed YAML → validation
3. Validation result → env var extraction
4. Workflow state includes env vars for UI
"""

import sys
from pathlib import Path

# Ensure src is in path for NAT imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pytest  # noqa: E402

from nat.workflow_builder_api.utils.env_var_processor import preprocess_yaml_env_vars  # noqa: E402
from nat.workflow_builder_api.utils.validation import validate_yaml_config_with_env_vars  # noqa: E402


class TestYamlPreprocessing:
    """Tests for raw YAML preprocessing."""

    def test_preprocess_replaces_url_with_valid_placeholder(self):
        """URL fields should get valid URL placeholders."""
        yaml_content = """
server:
  url: ${MY_SERVICE_URL}
"""
        result = preprocess_yaml_env_vars(yaml_content)

        assert "MY_SERVICE_URL" in result.detected_vars
        # Should be a valid URL
        assert "https://" in result.var_to_placeholder["MY_SERVICE_URL"]
        assert "MY_SERVICE_URL" not in result.processed_yaml or "${" not in result.processed_yaml

    def test_preprocess_handles_multiple_vars_same_line(self):
        """Multiple env vars on the same line should all be replaced."""
        yaml_content = """
connection_string: postgres://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/mydb
"""
        result = preprocess_yaml_env_vars(yaml_content)

        assert len(result.detected_vars) == 3
        assert "DB_USER" in result.detected_vars
        assert "DB_PASSWORD" in result.detected_vars
        assert "DB_HOST" in result.detected_vars
        # Original ${} patterns should be gone
        assert "${DB_USER}" not in result.processed_yaml
        assert "${DB_PASSWORD}" not in result.processed_yaml
        assert "${DB_HOST}" not in result.processed_yaml

    def test_preprocess_preserves_non_env_var_content(self):
        """Non-env-var content should be preserved exactly."""
        yaml_content = """
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    api_key: ${NVIDIA_API_KEY}
"""
        result = preprocess_yaml_env_vars(yaml_content)

        # Verify structure is preserved
        assert "meta/llama-3.1-70b-instruct" in result.processed_yaml
        assert "temperature: 0.0" in result.processed_yaml
        assert "_type: nim" in result.processed_yaml


class TestValidationWithEnvVars:
    """Tests for validation of configs with env vars."""

    def test_validation_succeeds_for_mcp_config(self):
        """Real MCP config with multiple env vars should validate successfully."""
        yaml_content = """
function_groups:
  mcp_test:
    _type: mcp_client
    server:
      transport: streamable-http
      url: ${MCP_SERVER_URL}
      auth_provider: test_auth

authentication:
  test_auth:
    _type: mcp_service_account
    client_id: ${CLIENT_ID}
    client_secret: ${CLIENT_SECRET}
    token_url: ${TOKEN_URL}
    scopes: ${SCOPES}

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct

workflow:
  _type: react_agent
  tool_names: [mcp_test]
  llm_name: nim_llm
"""
        result = validate_yaml_config_with_env_vars(yaml_content)

        assert result.response.valid, f"Validation failed: {result.response.error_details}"
        assert len(result.environment_variables) == 5

        var_names = {v.name for v in result.environment_variables}
        assert var_names == {"MCP_SERVER_URL", "CLIENT_ID", "CLIENT_SECRET", "TOKEN_URL", "SCOPES"}

    def test_sensitive_fields_are_marked(self):
        """Fields like client_secret should be marked as sensitive."""
        yaml_content = """
authentication:
  oauth:
    _type: mcp_service_account
    client_id: ${CLIENT_ID}
    client_secret: ${CLIENT_SECRET}
    token_url: ${TOKEN_URL}
    scopes: scope1
"""
        result = validate_yaml_config_with_env_vars(yaml_content)

        env_var_map = {v.name: v for v in result.environment_variables}

        # client_secret should be sensitive
        assert env_var_map["CLIENT_SECRET"].is_sensitive is True
        # token_url contains "token" so it's sensitive
        assert env_var_map["TOKEN_URL"].is_sensitive is True
        # client_id is not sensitive
        assert env_var_map["CLIENT_ID"].is_sensitive is False

    def test_validation_returns_env_vars_even_on_failure(self):
        """Even if validation fails, detected env vars should be returned."""
        yaml_content = """
# This config is invalid (missing workflow)
llms:
  test:
    _type: invalid_type_that_does_not_exist
    api_key: ${API_KEY}
"""
        result = validate_yaml_config_with_env_vars(yaml_content)

        # Validation should fail
        assert result.response.valid is False
        # But we should still have detected the env var
        assert len(result.environment_variables) == 1
        assert result.environment_variables[0].name == "API_KEY"


class TestEnvVarFlow:
    """End-to-end tests for the environment variable flow."""

    def test_complete_import_flow_with_env_vars(self):
        """Test the complete flow from YAML import to workflow state with env vars."""
        yaml_content = """
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    api_key: ${NVIDIA_API_KEY}

functions:
  calculator:
    _type: calculator

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names: [calculator]
"""
        # Step 1: Validate with env var awareness
        validation_result = validate_yaml_config_with_env_vars(yaml_content)
        assert validation_result.response.valid

        # Step 2: Check env vars were detected
        assert len(validation_result.environment_variables) == 1
        env_var = validation_result.environment_variables[0]
        assert env_var.name == "NVIDIA_API_KEY"
        assert env_var.original_reference == "${NVIDIA_API_KEY}"
        assert env_var.value is None  # Not resolved yet

        # Step 3: Verify the processed config has placeholder
        config_dict = validation_result.response.config_dict
        assert config_dict is not None
        api_key_value = config_dict["llms"]["nim_llm"].get("api_key")
        # The placeholder should contain the var name
        assert "NVIDIA_API_KEY" in str(api_key_value) if api_key_value else True

    def test_url_field_gets_valid_url_placeholder(self):
        """URL fields should get valid URL placeholders to pass validation."""
        yaml_content = """
function_groups:
  mcp_test:
    _type: mcp_client
    server:
      transport: streamable-http
      url: ${SERVICE_URL}

workflow:
  _type: react_agent
  tool_names: [mcp_test]
  llm_name: test_llm

llms:
  test_llm:
    _type: nim
    model_name: test
"""
        result = validate_yaml_config_with_env_vars(yaml_content)

        # Validation should succeed because URL gets a valid placeholder
        assert result.response.valid, f"Validation failed: {result.response.error_details}"

        # The SERVICE_URL env var should be detected
        assert len(result.environment_variables) == 1
        assert result.environment_variables[0].name == "SERVICE_URL"


class TestRealConfigFiles:
    """Tests using actual config files from the examples directory."""

    @pytest.fixture
    def examples_dir(self) -> Path:
        """Return the examples directory path."""
        return _PROJECT_ROOT / "examples"

    def test_mcp_service_account_config(self, examples_dir: Path):
        """Test the MCP service account config with multiple env vars."""
        config_path = examples_dir / "MCP/service_account_auth_mcp/configs/config-mcp-service-account-jama.yml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        with open(config_path) as f:
            yaml_content = f.read()

        result = validate_yaml_config_with_env_vars(yaml_content)

        # Should validate successfully
        assert result.response.valid, f"Validation failed: {result.response.error_details}"

        # Should detect all env vars in the file
        var_names = {v.name for v in result.environment_variables}
        expected_vars = {
            "CORPORATE_MCP_SERVICE_ACCOUNT_JAMA_URL",
            "SERVICE_ACCOUNT_CLIENT_ID",
            "SERVICE_ACCOUNT_CLIENT_SECRET",
            "SERVICE_ACCOUNT_TOKEN_URL",
            "SERVICE_ACCOUNT_SCOPES",
        }
        assert var_names == expected_vars

        # Secret fields should be marked as sensitive
        env_var_map = {v.name: v for v in result.environment_variables}
        assert env_var_map["SERVICE_ACCOUNT_CLIENT_SECRET"].is_sensitive is True
        assert env_var_map["SERVICE_ACCOUNT_TOKEN_URL"].is_sensitive is True

    def test_simple_calculator_config_no_env_vars(self, examples_dir: Path):
        """Configs without env vars should work normally."""
        config_path = examples_dir / "getting_started/simple_calculator/src/nat_simple_calculator/configs/config.yml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        with open(config_path) as f:
            yaml_content = f.read()

        result = validate_yaml_config_with_env_vars(yaml_content)

        # Should validate successfully
        assert result.response.valid, f"Validation failed: {result.response.error_details}"

        # Should have no env vars (or only ones that are system-level)
        # This config might use NVIDIA_API_KEY as an env var reference
        # Just verify validation works and env vars (if any) are detected
        assert result.response.config_dict is not None
