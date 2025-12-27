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
"""Tests for importing function configurations from YAML to workflow state.

These tests verify that function (tool) configurations are correctly parsed
and all field values are accurately preserved during import.
"""

from pathlib import Path

import yaml

from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportFunctionsFromReactConfig:
    """Tests for importing functions from react agent config.

    The react agent config contains multiple functions/tools.
    These tests verify each function is correctly imported.
    """

    def test_functions_are_created(self, react_agent_config: Path, set_test_env_vars):
        """Verify that function components are created during import."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        functions = [c for c in workflow_state.components if c.component_type == "function"]
        assert len(functions) >= 1, "Expected at least one function component"

    def test_function_count_matches_non_agent_functions(self, react_agent_config: Path, set_test_env_vars):
        """Verify the count of imported functions matches non-agent functions in config."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        # Count non-agent functions in original config
        original_functions = config_dict.get("functions", {})
        non_agent_count = sum(1 for cfg in original_functions.values() if "agent" not in cfg.get("_type", "").lower())

        imported_functions = [c for c in workflow_state.components if c.component_type == "function"]
        assert len(imported_functions) == non_agent_count, (
            f"Function count mismatch: expected {non_agent_count}, got {len(imported_functions)}")

    def test_function_ids_match_original_names(self, react_agent_config: Path, set_test_env_vars):
        """Verify function IDs match the key names from the original config."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        # Get non-agent function names from original
        original_functions = config_dict.get("functions", {})
        function_names = {
            name
            for name, cfg in original_functions.items() if "agent" not in cfg.get("_type", "").lower()
        }

        imported_function_ids = {c.id for c in workflow_state.components if c.component_type == "function"}

        assert function_names == imported_function_ids, (
            f"Function ID mismatch: original={function_names}, imported={imported_function_ids}")

    def test_function_types_are_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify function _type is preserved in full_type."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_functions = config_dict.get("functions", {})
        for name, func_config in original_functions.items():
            orig_type = func_config.get("_type", "")
            if "agent" in orig_type.lower():
                continue  # Skip agents

            orig_type_short = orig_type.split("/")[-1]
            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Function '{name}' not found"

            imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Function type mismatch for '{name}': expected '{orig_type_short}', got '{imported_type_short}'")


class TestImportFunctionGroups:
    """Tests for importing function group configurations."""

    def test_function_groups_are_created(self, function_groups_config: Path, set_test_env_vars):
        """Verify that function_group components are created during import."""
        config_dict = load_config_dict(function_groups_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        # Check if config has function_groups
        if "function_groups" not in config_dict:
            return  # Skip if no function groups in this config

        function_groups = [c for c in workflow_state.components if c.component_type == "function_group"]
        original_count = len(config_dict.get("function_groups", {}))

        assert len(function_groups) == original_count, (
            f"Function group count mismatch: expected {original_count}, got {len(function_groups)}")
