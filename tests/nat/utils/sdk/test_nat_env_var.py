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
"""Tests for NatEnvironmentVariable class.

This module tests the environment variable reference functionality used
for serializing API keys and secrets in config files.
"""

import os

import pytest
from pydantic import SecretStr

from nat.utils.sdk.nat_env_var import NatEnvironmentVariable

# ============================================================================
# Initialization Tests
# ============================================================================


class TestNatEnvironmentVariableInit:
    """Tests for NatEnvironmentVariable initialization."""

    def test_init_with_name_only(self):
        """Test initialization with just the env var name."""
        env_var = NatEnvironmentVariable("MY_API_KEY")

        assert env_var.name == "MY_API_KEY"
        assert env_var.required is True  # Default

    def test_init_with_required_true(self):
        """Test initialization with required=True."""
        env_var = NatEnvironmentVariable("MY_API_KEY", required=True)

        assert env_var.name == "MY_API_KEY"
        assert env_var.required is True

    def test_init_with_required_false(self):
        """Test initialization with required=False."""
        env_var = NatEnvironmentVariable("OPTIONAL_KEY", required=False)

        assert env_var.name == "OPTIONAL_KEY"
        assert env_var.required is False

    def test_init_with_keyword_args(self):
        """Test initialization using keyword arguments."""
        env_var = NatEnvironmentVariable(name="MY_KEY", required=False)

        assert env_var.name == "MY_KEY"
        assert env_var.required is False


# ============================================================================
# resolve() Tests
# ============================================================================


class TestResolve:
    """Tests for the resolve() method."""

    def test_resolve_existing_env_var(self, restore_environ):
        """Test resolving an existing environment variable."""
        os.environ["TEST_VAR"] = "test_value"

        env_var = NatEnvironmentVariable("TEST_VAR")
        result = env_var.resolve()

        assert result == "test_value"

    def test_resolve_missing_required_env_var_raises(self):
        """Test that resolving a missing required env var raises ValueError."""
        # Ensure the var doesn't exist
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]

        env_var = NatEnvironmentVariable("NONEXISTENT_VAR", required=True)

        with pytest.raises(ValueError, match="required but not set"):
            env_var.resolve()

    def test_resolve_missing_optional_env_var_returns_none(self):
        """Test that resolving a missing optional env var returns None."""
        # Ensure the var doesn't exist
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]

        env_var = NatEnvironmentVariable("NONEXISTENT_VAR", required=False)
        result = env_var.resolve()

        assert result is None

    def test_resolve_empty_value(self, restore_environ):
        """Test resolving an env var with empty string value."""
        os.environ["EMPTY_VAR"] = ""

        env_var = NatEnvironmentVariable("EMPTY_VAR")
        result = env_var.resolve()

        assert result == ""


# ============================================================================
# to_secret_str() Tests
# ============================================================================


class TestToSecretStr:
    """Tests for the to_secret_str() method."""

    def test_to_secret_str_existing_var(self, restore_environ):
        """Test converting an existing env var to SecretStr."""
        os.environ["SECRET_VAR"] = "super_secret"

        env_var = NatEnvironmentVariable("SECRET_VAR")
        result = env_var.to_secret_str()

        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == "super_secret"

    def test_to_secret_str_optional_missing_returns_none(self):
        """Test that optional missing env var returns None as SecretStr."""
        if "NONEXISTENT" in os.environ:
            del os.environ["NONEXISTENT"]

        env_var = NatEnvironmentVariable("NONEXISTENT", required=False)
        result = env_var.to_secret_str()

        assert result is None

    def test_to_secret_str_required_missing_raises(self):
        """Test that required missing env var raises error."""
        if "NONEXISTENT" in os.environ:
            del os.environ["NONEXISTENT"]

        env_var = NatEnvironmentVariable("NONEXISTENT", required=True)

        with pytest.raises(ValueError):
            env_var.to_secret_str()


# ============================================================================
# to_yaml_reference() Tests
# ============================================================================


class TestToYamlReference:
    """Tests for the to_yaml_reference() method."""

    def test_yaml_reference_format(self):
        """Test that YAML reference has correct format."""
        env_var = NatEnvironmentVariable("MY_API_KEY")
        result = env_var.to_yaml_reference()

        assert result == "${MY_API_KEY}"

    def test_yaml_reference_with_underscores(self):
        """Test YAML reference with underscores in name."""
        env_var = NatEnvironmentVariable("NVIDIA_API_KEY")
        result = env_var.to_yaml_reference()

        assert result == "${NVIDIA_API_KEY}"

    def test_yaml_reference_with_numbers(self):
        """Test YAML reference with numbers in name."""
        env_var = NatEnvironmentVariable("API_KEY_V2")
        result = env_var.to_yaml_reference()

        assert result == "${API_KEY_V2}"


# ============================================================================
# String Representation Tests
# ============================================================================


class TestStringRepresentation:
    """Tests for __str__ and __repr__ methods."""

    def test_str_returns_yaml_reference(self):
        """Test that __str__ returns YAML reference format."""
        env_var = NatEnvironmentVariable("MY_KEY")
        result = str(env_var)

        assert result == "${MY_KEY}"

    def test_repr_includes_details(self):
        """Test that __repr__ includes name and required."""
        env_var = NatEnvironmentVariable("MY_KEY", required=False)
        result = repr(env_var)

        assert "NatEnvironmentVariable" in result
        assert "MY_KEY" in result
        assert "required=False" in result

    def test_repr_required_true(self):
        """Test __repr__ with required=True."""
        env_var = NatEnvironmentVariable("API_KEY", required=True)
        result = repr(env_var)

        assert "required=True" in result


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_special_characters_in_name(self):
        """Test env var name with allowed special characters."""
        # Typical env var names use uppercase and underscores
        env_var = NatEnvironmentVariable("MY_APP_API_KEY_123")

        assert env_var.name == "MY_APP_API_KEY_123"
        assert env_var.to_yaml_reference() == "${MY_APP_API_KEY_123}"

    def test_lowercase_name(self):
        """Test env var with lowercase name."""
        # Some env vars are lowercase
        env_var = NatEnvironmentVariable("my_lowercase_key")

        assert env_var.name == "my_lowercase_key"
        assert env_var.to_yaml_reference() == "${my_lowercase_key}"

    def test_empty_name(self):
        """Test env var with empty name (should still work, though unusual)."""
        env_var = NatEnvironmentVariable("")

        assert env_var.name == ""
        assert env_var.to_yaml_reference() == "${}"

    def test_value_with_special_characters(self, restore_environ):
        """Test resolving env var with special characters in value."""
        os.environ["SPECIAL_VAR"] = "p@ssw0rd!#$%"

        env_var = NatEnvironmentVariable("SPECIAL_VAR")
        result = env_var.resolve()

        assert result == "p@ssw0rd!#$%"

    def test_value_with_spaces(self, restore_environ):
        """Test resolving env var with spaces in value."""
        os.environ["SPACED_VAR"] = "value with spaces"

        env_var = NatEnvironmentVariable("SPACED_VAR")
        result = env_var.resolve()

        assert result == "value with spaces"

    def test_value_with_newlines(self, restore_environ):
        """Test resolving env var with newlines in value."""
        os.environ["MULTILINE_VAR"] = "line1\nline2\nline3"

        env_var = NatEnvironmentVariable("MULTILINE_VAR")
        result = env_var.resolve()

        assert result == "line1\nline2\nline3"


# ============================================================================
# Pydantic Model Behavior Tests
# ============================================================================


class TestPydanticModelBehavior:
    """Tests for Pydantic model behavior."""

    def test_model_dump(self):
        """Test Pydantic model_dump()."""
        env_var = NatEnvironmentVariable("MY_KEY", required=False)
        data = env_var.model_dump()

        assert data == {"name": "MY_KEY", "required": False}

    def test_model_validate(self):
        """Test Pydantic model_validate()."""
        data = {"name": "RESTORED_KEY", "required": True}
        env_var = NatEnvironmentVariable.model_validate(data)

        assert env_var.name == "RESTORED_KEY"
        assert env_var.required is True

    def test_model_copy(self):
        """Test copying the model."""
        original = NatEnvironmentVariable("ORIG_KEY", required=True)
        copied = original.model_copy(update={"required": False})

        assert original.required is True
        assert copied.required is False
        assert original.name == copied.name


# ============================================================================
# Integration with Config Serialization Tests
# ============================================================================


class TestConfigSerializationIntegration:
    """Tests for integration with config file serialization.

    These tests verify that NatEnvironmentVariable works correctly
    when used with the workflow config system.
    """

    def test_env_var_reference_preserved_in_yaml(self, tmp_path, restore_environ):
        """Test that env var references are preserved when saving to YAML.

        Note: This is an integration test that requires the full SDK.
        The actual preservation happens in NatWorkflow._apply_env_var_refs_to_file.
        """
        os.environ["TEST_API_KEY"] = "secret_value"

        env_var = NatEnvironmentVariable("TEST_API_KEY")

        # The env var should resolve to the actual value
        assert env_var.resolve() == "secret_value"

        # But to_yaml_reference should give us the reference format
        assert env_var.to_yaml_reference() == "${TEST_API_KEY}"

    def test_multiple_env_vars(self, restore_environ):
        """Test multiple env vars can coexist."""
        os.environ["KEY_1"] = "value1"
        os.environ["KEY_2"] = "value2"

        var1 = NatEnvironmentVariable("KEY_1")
        var2 = NatEnvironmentVariable("KEY_2")

        assert var1.resolve() == "value1"
        assert var2.resolve() == "value2"
        assert var1.to_yaml_reference() == "${KEY_1}"
        assert var2.to_yaml_reference() == "${KEY_2}"
