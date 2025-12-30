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
Tests for restoring and applying environment variable values.

Tests restore_env_var_references and apply_env_var_values functions.
"""

import sys
from pathlib import Path

# Ensure src is in path for NAT imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nat.workflow_builder_api.utils.env_var.models import EnvironmentVariable  # noqa: E402
from nat.workflow_builder_api.utils.env_var.models import EnvVarLocation  # noqa: E402
from nat.workflow_builder_api.utils.env_var.processing import apply_env_var_values  # noqa: E402
from nat.workflow_builder_api.utils.env_var.processing import restore_env_var_references  # noqa: E402


class TestRestoreEnvVarReferences:
    """Tests for restoring environment variable references during export."""

    def test_restores_original_references(self):
        """Should restore ${VAR} notation for export."""
        config = {"auth": {"key": "__ENV_VAR__API_KEY__", }}
        env_vars = [
            EnvironmentVariable(
                name="API_KEY",
                locations=[
                    EnvVarLocation(
                        path="auth.key",
                        component_type="auth",
                        component_id=None,
                        field_name="key",
                    )
                ],
                value="actual_value",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${API_KEY}",
            )
        ]

        result = restore_env_var_references(config, env_vars, use_resolved_values=False)

        assert result["auth"]["key"] == "${API_KEY}"

    def test_uses_resolved_values_when_requested(self):
        """Should use resolved values when flag is set."""
        config = {"auth": {"key": "__ENV_VAR__API_KEY__", }}
        env_vars = [
            EnvironmentVariable(
                name="API_KEY",
                locations=[
                    EnvVarLocation(
                        path="auth.key",
                        component_type="auth",
                        component_id=None,
                        field_name="key",
                    )
                ],
                value="my_actual_api_key_123",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${API_KEY}",
            )
        ]

        result = restore_env_var_references(config, env_vars, use_resolved_values=True)

        assert result["auth"]["key"] == "my_actual_api_key_123"

    def test_handles_multiple_env_vars(self):
        """Should restore multiple env vars correctly."""
        config = {
            "auth": {
                "user": "__ENV_VAR__USER__",
                "password": "__ENV_VAR__PASSWORD__",
            }
        }
        env_vars = [
            EnvironmentVariable(
                name="USER",
                locations=[
                    EnvVarLocation(
                        path="auth.user",
                        component_type="auth",
                        component_id=None,
                        field_name="user",
                    )
                ],
                value="admin",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${USER}",
            ),
            EnvironmentVariable(
                name="PASSWORD",
                locations=[
                    EnvVarLocation(
                        path="auth.password",
                        component_type="auth",
                        component_id=None,
                        field_name="password",
                    )
                ],
                value="secret123",
                is_sensitive=True,
                export_as_variable=True,
                original_reference="${PASSWORD}",
            ),
        ]

        result = restore_env_var_references(config, env_vars, use_resolved_values=False)

        assert result["auth"]["user"] == "${USER}"
        assert result["auth"]["password"] == "${PASSWORD}"


class TestApplyEnvVarValues:
    """Tests for applying resolved values to a config."""

    def test_applies_values_to_placeholders(self):
        """Should replace placeholders with actual values."""
        config = {
            "auth": {
                "key": "__ENV_VAR__API_KEY__",
                "secret": "__ENV_VAR__API_SECRET__",
            }
        }
        env_vars = [
            EnvironmentVariable(
                name="API_KEY",
                locations=[],
                value="key123",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${API_KEY}",
            ),
            EnvironmentVariable(
                name="API_SECRET",
                locations=[],
                value="secret456",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${API_SECRET}",
            ),
        ]

        result = apply_env_var_values(config, env_vars)

        assert result["auth"]["key"] == "key123"
        assert result["auth"]["secret"] == "secret456"

    def test_skips_unresolved_env_vars(self):
        """Should leave placeholders for unresolved env vars."""
        config = {"auth": {"key": "__ENV_VAR__API_KEY__", }}
        env_vars = [
            EnvironmentVariable(
                name="API_KEY",
                locations=[],
                value=None,  # Unresolved
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${API_KEY}",
            )
        ]

        result = apply_env_var_values(config, env_vars)

        # Placeholder should remain
        assert result["auth"]["key"] == "__ENV_VAR__API_KEY__"

    def test_applies_values_in_nested_structures(self):
        """Should apply values in deeply nested configs."""
        config = {"level1": {"level2": {"level3": {"value": "__ENV_VAR__DEEP_VALUE__", }}}}
        env_vars = [
            EnvironmentVariable(
                name="DEEP_VALUE",
                locations=[],
                value="nested_secret",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${DEEP_VALUE}",
            ),
        ]

        result = apply_env_var_values(config, env_vars)

        assert result["level1"]["level2"]["level3"]["value"] == "nested_secret"

    def test_applies_values_in_lists(self):
        """Should apply values inside list elements."""
        config = {
            "items": [
                "__ENV_VAR__ITEM1__",
                "static_value",
                "__ENV_VAR__ITEM3__",
            ]
        }
        env_vars = [
            EnvironmentVariable(
                name="ITEM1",
                locations=[],
                value="first",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${ITEM1}",
            ),
            EnvironmentVariable(
                name="ITEM3",
                locations=[],
                value="third",
                is_sensitive=False,
                export_as_variable=True,
                original_reference="${ITEM3}",
            ),
        ]

        result = apply_env_var_values(config, env_vars)

        assert result["items"] == ["first", "static_value", "third"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
