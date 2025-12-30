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
Tests for environment variable pattern matching and detection.

Tests the regex patterns, is_env_var_reference, extract_env_var_names,
and is_sensitive_field functions.
"""

import sys
from pathlib import Path

# Ensure src is in path for NAT imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nat.workflow_builder_api.utils.env_var.detection import ENV_VAR_PATTERN  # noqa: E402
from nat.workflow_builder_api.utils.env_var.detection import extract_env_var_names  # noqa: E402
from nat.workflow_builder_api.utils.env_var.detection import is_env_var_reference  # noqa: E402
from nat.workflow_builder_api.utils.env_var.detection import is_sensitive_field  # noqa: E402


class TestEnvVarPatternMatching:
    """Tests for environment variable pattern detection."""

    def test_env_var_pattern_matches_simple_var(self):
        """Pattern should match simple variable names."""
        match = ENV_VAR_PATTERN.search("${API_KEY}")
        assert match is not None
        assert match.group(1) == "API_KEY"

    def test_env_var_pattern_matches_underscore_var(self):
        """Pattern should match variables with underscores."""
        match = ENV_VAR_PATTERN.search("${SERVICE_ACCOUNT_CLIENT_ID}")
        assert match is not None
        assert match.group(1) == "SERVICE_ACCOUNT_CLIENT_ID"

    def test_env_var_pattern_matches_mixed_case(self):
        """Pattern should match mixed case variables."""
        match = ENV_VAR_PATTERN.search("${MyVariable123}")
        assert match is not None
        assert match.group(1) == "MyVariable123"

    def test_env_var_pattern_does_not_match_bare_dollar(self):
        """Pattern should not match $VAR without braces."""
        match = ENV_VAR_PATTERN.search("$API_KEY")
        assert match is None

    def test_env_var_pattern_does_not_match_empty_braces(self):
        """Pattern should not match empty ${}."""
        match = ENV_VAR_PATTERN.search("${}")
        assert match is None

    def test_env_var_pattern_does_not_match_number_prefix(self):
        """Pattern should not match variables starting with numbers."""
        match = ENV_VAR_PATTERN.search("${123VAR}")
        assert match is None

    def test_env_var_pattern_findall_multiple(self):
        """Pattern should find all variables in a string."""
        text = "prefix_${VAR1}_middle_${VAR2}_suffix"
        matches = ENV_VAR_PATTERN.findall(text)
        assert matches == ["VAR1", "VAR2"]


class TestIsEnvVarReference:
    """Tests for the is_env_var_reference function."""

    def test_detects_simple_env_var(self):
        """Should detect simple env var reference."""
        assert is_env_var_reference("${API_KEY}") is True

    def test_detects_embedded_env_var(self):
        """Should detect env var embedded in string."""
        assert is_env_var_reference("https://api.example.com?key=${API_KEY}") is True

    def test_detects_multiple_env_vars(self):
        """Should detect multiple env vars in one string."""
        assert is_env_var_reference("${USER}:${PASSWORD}") is True

    def test_rejects_non_env_var_string(self):
        """Should reject strings without env vars."""
        assert is_env_var_reference("just a normal string") is False

    def test_rejects_non_string_types(self):
        """Should return False for non-string types."""
        assert is_env_var_reference(123) is False
        assert is_env_var_reference(None) is False
        assert is_env_var_reference(["list"]) is False
        assert is_env_var_reference({"dict": "value"}) is False


class TestExtractEnvVarNames:
    """Tests for the extract_env_var_names function."""

    def test_extracts_single_var(self):
        """Should extract single variable name."""
        names = extract_env_var_names("${API_KEY}")
        assert names == ["API_KEY"]

    def test_extracts_multiple_vars(self):
        """Should extract multiple variable names."""
        names = extract_env_var_names("${USER}:${PASSWORD}@${HOST}")
        assert names == ["USER", "PASSWORD", "HOST"]

    def test_returns_empty_for_no_vars(self):
        """Should return empty list when no vars present."""
        names = extract_env_var_names("no variables here")
        assert names == []

    def test_returns_empty_for_non_string(self):
        """Should return empty list for non-string input."""
        assert extract_env_var_names(123) == []
        assert extract_env_var_names(None) == []


class TestIsSensitiveField:
    """Tests for sensitive field detection."""

    def test_detects_password_field(self):
        """Should detect password-related fields."""
        assert is_sensitive_field("password", "auth.password") is True
        assert is_sensitive_field("db_password", "database.db_password") is True
        assert is_sensitive_field("PASSWORD", "auth.PASSWORD") is True

    def test_detects_secret_field(self):
        """Should detect secret-related fields."""
        assert is_sensitive_field("client_secret", "oauth.client_secret") is True
        assert is_sensitive_field("SECRET_KEY", "app.SECRET_KEY") is True

    def test_detects_token_field(self):
        """Should detect token-related fields."""
        assert is_sensitive_field("access_token", "oauth.access_token") is True
        assert is_sensitive_field("api_token", "api.api_token") is True

    def test_detects_api_key_field(self):
        """Should detect API key fields."""
        assert is_sensitive_field("api_key", "service.api_key") is True
        assert is_sensitive_field("apikey", "service.apikey") is True

    def test_detects_auth_section(self):
        """Should detect fields in authentication sections."""
        assert is_sensitive_field("client_id", "authentication.my_auth.client_id") is True

    def test_rejects_non_sensitive_field(self):
        """Should not flag non-sensitive fields."""
        assert is_sensitive_field("model_name", "llm.model_name") is False
        assert is_sensitive_field("temperature", "llm.temperature") is False
        assert is_sensitive_field("url", "service.url") is False


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
