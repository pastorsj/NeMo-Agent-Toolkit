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
"""Tests for importing authentication configurations from YAML to workflow state.

These tests verify that authentication and front-end configurations are correctly
parsed and all field values are accurately preserved during import.
"""

from pathlib import Path

import yaml


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportAuthenticationFromConfig:
    """Tests for importing authentication configurations."""

    def test_authentication_components_are_created(self, auth_config: Path, set_test_env_vars):
        """Verify that authentication components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        auth = [c for c in workflow_state.components if c.component_type == "authentication"]

        # Count auth providers in original
        auth_providers = config_dict.get("general", {}).get("authentication", {})
        original_count = len(auth_providers) if auth_providers else 0

        assert len(auth) == original_count, (
            f"Authentication count mismatch: expected {original_count}, got {len(auth)}")

    def test_authentication_type_is_preserved(self, auth_config: Path, set_test_env_vars):
        """Verify authentication _type is preserved in full_type."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        auth_providers = config_dict.get("general", {}).get("authentication", {})
        for name, orig_config in auth_providers.items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Authentication '{name}' not found"

            imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Authentication type mismatch for '{name}': expected '{orig_type_short}', got '{imported_type_short}'")


class TestImportFrontEndFromConfig:
    """Tests for importing front-end configurations."""

    def test_front_end_component_is_created(self, auth_config: Path, set_test_env_vars):
        """Verify that front_end components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        front_ends = [c for c in workflow_state.components if c.component_type == "front_end"]

        # Should have at least one front_end if config has general.front_end
        if config_dict.get("general", {}).get("front_end"):
            assert len(front_ends) >= 1, "Expected at least one front_end component"

    def test_front_end_type_is_preserved(self, auth_config: Path, set_test_env_vars):
        """Verify front_end _type is preserved in full_type."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        front_end_config = config_dict.get("general", {}).get("front_end", {})
        if not front_end_config:
            return

        orig_type = front_end_config.get("_type", "")
        orig_type_short = orig_type.split("/")[-1]

        front_ends = [c for c in workflow_state.components if c.component_type == "front_end"]
        if not front_ends:
            return

        imported = front_ends[0]
        imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type

        assert imported_type_short == orig_type_short, (
            f"Front-end type mismatch: expected '{orig_type_short}', got '{imported_type_short}'")

    def test_front_end_auth_connection_is_created(self, auth_config: Path, set_test_env_vars):
        """Verify that front_end's auth reference creates a connection."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        front_end_config = config_dict.get("general", {}).get("front_end", {})
        auth_ref = front_end_config.get("authentication")
        if auth_ref is None:
            return

        # Should have a connection from front_end to auth
        auth_connections = [c for c in workflow_state.connections if "auth" in c.target_field.lower()]
        assert len(auth_connections) >= 1, (f"Expected connection to authentication '{auth_ref}'")
