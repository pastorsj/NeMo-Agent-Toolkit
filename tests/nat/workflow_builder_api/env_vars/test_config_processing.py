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
Tests for config processing with environment variables.

Tests process_config_env_vars, placeholder generation, and location tracking.
"""

import sys
from pathlib import Path

# Ensure src is in path for NAT imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nat.workflow_builder_api.utils.env_var.processing import generate_placeholder_value  # noqa: E402
from nat.workflow_builder_api.utils.env_var.processing import process_config_env_vars  # noqa: E402


class TestGeneratePlaceholderValue:
    """Tests for placeholder value generation."""

    def test_generates_identifiable_placeholder(self):
        """Placeholder should be clearly identifiable."""
        placeholder = generate_placeholder_value("API_KEY", "auth.api_key")
        assert "__ENV_VAR__" in placeholder
        assert "API_KEY" in placeholder

    def test_placeholder_format_consistent(self):
        """Placeholders should have consistent format."""
        placeholder = generate_placeholder_value("MY_VAR", "some.path")
        assert placeholder == "__ENV_VAR__MY_VAR__"


class TestProcessConfigEnvVars:
    """Tests for the main config processing function."""

    def test_processes_simple_config(self):
        """Should process config with single env var."""
        config = {"authentication": {"my_auth": {"client_id": "${CLIENT_ID}", }}}

        result = process_config_env_vars(config)

        assert len(result.environment_variables) == 1
        assert result.environment_variables[0].name == "CLIENT_ID"
        assert result.processed_config["authentication"]["my_auth"]["client_id"] == "__ENV_VAR__CLIENT_ID__"

    def test_processes_multiple_env_vars(self):
        """Should detect multiple distinct env vars."""
        config = {
            "authentication": {
                "my_auth": {
                    "client_id": "${CLIENT_ID}",
                    "client_secret": "${CLIENT_SECRET}",
                    "token_url": "${TOKEN_URL}",
                }
            }
        }

        result = process_config_env_vars(config)

        assert len(result.environment_variables) == 3
        var_names = {v.name for v in result.environment_variables}
        assert var_names == {"CLIENT_ID", "CLIENT_SECRET", "TOKEN_URL"}

    def test_tracks_multiple_locations_same_var(self):
        """Should track when same var is used in multiple places."""
        config = {
            "llms": {
                "llm1": {
                    "api_key": "${API_KEY}"
                },
                "llm2": {
                    "api_key": "${API_KEY}"
                },
            }
        }

        result = process_config_env_vars(config)

        assert len(result.environment_variables) == 1
        api_key_var = result.environment_variables[0]
        assert api_key_var.name == "API_KEY"
        assert len(api_key_var.locations) == 2

    def test_preserves_non_env_var_values(self):
        """Should leave non-env-var values unchanged."""
        config = {
            "llms": {
                "nim_llm": {
                    "model_name": "meta/llama-3.1-70b-instruct",
                    "temperature": 0.0,
                    "max_tokens": 1024,
                }
            }
        }

        result = process_config_env_vars(config)

        assert len(result.environment_variables) == 0
        assert result.processed_config["llms"]["nim_llm"]["model_name"] == "meta/llama-3.1-70b-instruct"
        assert result.processed_config["llms"]["nim_llm"]["temperature"] == 0.0

    def test_handles_nested_structures(self):
        """Should process deeply nested configs."""
        config = {
            "function_groups": {
                "mcp_jama": {
                    "server": {
                        "transport": "streamable-http",
                        "url": "${MCP_URL}",
                        "auth_provider": "jama_auth",
                    }
                }
            }
        }

        result = process_config_env_vars(config)

        assert len(result.environment_variables) == 1
        assert result.environment_variables[0].name == "MCP_URL"
        location = result.environment_variables[0].locations[0]
        assert location.path == "function_groups.mcp_jama.server.url"

    def test_handles_lists(self):
        """Should process env vars inside lists."""
        config = {"authentication": {"my_auth": {"scopes": ["${SCOPE_1}", "${SCOPE_2}"], }}}

        result = process_config_env_vars(config)

        var_names = {v.name for v in result.environment_variables}
        assert var_names == {"SCOPE_1", "SCOPE_2"}

    def test_marks_sensitive_fields(self):
        """Should mark sensitive fields appropriately."""
        config = {"authentication": {"oauth": {"client_secret": "${CLIENT_SECRET}", }}}

        result = process_config_env_vars(config)

        assert len(result.environment_variables) == 1
        assert result.environment_variables[0].is_sensitive is True

    def test_does_not_mutate_original_config(self):
        """Should not modify the original config dict."""
        config = {"auth": {"key": "${API_KEY}", }}
        original_value = config["auth"]["key"]

        process_config_env_vars(config)

        assert config["auth"]["key"] == original_value


class TestEnvVarLocationParsing:
    """Tests for path parsing in env var locations."""

    def test_parses_simple_path(self):
        """Should correctly identify component info from path."""
        config = {"authentication": {"my_auth": {"client_id": "${CLIENT_ID}", }}}

        result = process_config_env_vars(config)

        location = result.environment_variables[0].locations[0]
        assert location.component_type == "authentication"
        assert location.component_id == "my_auth"
        assert location.field_name == "client_id"
        assert location.path == "authentication.my_auth.client_id"


class TestRealWorldConfig:
    """Tests with realistic MCP configurations."""

    def test_processes_mcp_config(self):
        """Should handle a realistic MCP config with multiple env vars."""
        config = {
            "function_groups": {
                "mcp_jama": {
                    "_type": "mcp_client",
                    "server": {
                        "transport": "streamable-http",
                        "url": "${CORPORATE_MCP_SERVICE_ACCOUNT_JAMA_URL}",
                        "auth_provider": "jama_service_account",
                    },
                }
            },
            "authentication": {
                "jama_service_account": {
                    "_type": "mcp_service_account",
                    "client_id": "${SERVICE_ACCOUNT_CLIENT_ID}",
                    "client_secret": "${SERVICE_ACCOUNT_CLIENT_SECRET}",
                    "token_url": "${SERVICE_ACCOUNT_TOKEN_URL}",
                    "scopes": "${SERVICE_ACCOUNT_SCOPES}",
                    "token_cache_buffer_seconds": 300,
                }
            },
            "llms": {
                "nim_llm": {
                    "_type": "nim",
                    "model_name": "meta/llama-3.1-70b-instruct",
                    "temperature": 0.0,
                    "max_tokens": 1024,
                }
            },
            "workflow": {
                "_type": "react_agent",
                "tool_names": ["mcp_jama"],
                "llm_name": "nim_llm",
                "verbose": True,
            },
        }

        result = process_config_env_vars(config)

        # Should detect 5 env vars
        var_names = {v.name for v in result.environment_variables}
        assert var_names == {
            "CORPORATE_MCP_SERVICE_ACCOUNT_JAMA_URL",
            "SERVICE_ACCOUNT_CLIENT_ID",
            "SERVICE_ACCOUNT_CLIENT_SECRET",
            "SERVICE_ACCOUNT_TOKEN_URL",
            "SERVICE_ACCOUNT_SCOPES",
        }

        # Non-env-var values should be preserved
        assert result.processed_config["llms"]["nim_llm"]["model_name"] == "meta/llama-3.1-70b-instruct"
        assert result.processed_config["authentication"]["jama_service_account"]["token_cache_buffer_seconds"] == 300


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
