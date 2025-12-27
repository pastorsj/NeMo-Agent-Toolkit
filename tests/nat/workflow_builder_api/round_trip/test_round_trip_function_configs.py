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
"""Round-trip tests for function and agent configurations.

These tests verify that function/agent configurations survive the import -> export
cycle with all field values preserved exactly as in the original config.
"""

from pathlib import Path

import yaml


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def do_round_trip(config_path: Path) -> tuple[dict, dict]:
    """
    Perform a full round-trip: load config -> import -> export.

    Returns (original_config, exported_config).
    """
    from nat.workflow_builder_api.models import ExportComponent
    from nat.workflow_builder_api.models import ExportConnection
    from nat.workflow_builder_api.models import ExportWorkflowRequest
    from nat.workflow_builder_api.utils.config_exporter import export_workflow_to_yaml
    from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

    original_config = load_config_dict(config_path)
    workflow_state = parse_config_to_workflow_state(original_config)

    components = [
        ExportComponent(id=c.id, component_type=c.component_type, full_type=c.full_type, config=c.config or {})
        for c in workflow_state.components
    ]
    connections = [
        ExportConnection(source_id=c.source_id, target_id=c.target_id, target_field=c.target_field)
        for c in workflow_state.connections
    ]

    export_request = ExportWorkflowRequest(components=components, connections=connections)
    export_result = export_workflow_to_yaml(export_request)

    assert export_result.success, f"Export failed: {export_result.error_message}"
    exported_config = yaml.safe_load(export_result.yaml_content)

    return original_config, exported_config


class TestFunctionRoundTripPreservesValues:
    """Tests that verify function field values are preserved through round-trip."""

    def test_functions_section_exists_after_round_trip(self, react_agent_config: Path, set_test_env_vars):
        """Verify that functions section exists in exported config."""
        original, exported = do_round_trip(react_agent_config)

        assert "functions" in original, "Test config should have functions section"
        assert "functions" in exported, "Exported config should have functions section"

    def test_function_names_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that all function names are preserved through round-trip."""
        original, exported = do_round_trip(react_agent_config)

        original_names = set(original.get("functions", {}).keys())
        exported_names = set(exported.get("functions", {}).keys())

        assert original_names == exported_names, (
            f"Function names mismatch: original={original_names}, exported={exported_names}")

    def test_function_types_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that function _type fields are preserved (short name match)."""
        original, exported = do_round_trip(react_agent_config)

        for name, orig_config in original.get("functions", {}).items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            exp_config = exported["functions"].get(name, {})
            exp_type = exp_config.get("_type", "")
            exp_type_short = exp_type.split("/")[-1]

            assert exp_type_short == orig_type_short, (
                f"_type mismatch for function '{name}': expected '{orig_type_short}', got '{exp_type_short}'")

    def test_agent_llm_name_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that agent's llm_name reference is preserved."""
        original, exported = do_round_trip(react_agent_config)

        for name, orig_config in original.get("functions", {}).items():
            if "agent" not in orig_config.get("_type", "").lower():
                continue

            orig_llm = orig_config.get("llm_name") or orig_config.get("llm")
            if orig_llm is None:
                continue

            exp_config = exported["functions"].get(name, {})
            exp_llm = exp_config.get("llm_name") or exp_config.get("llm")

            assert exp_llm == orig_llm, (
                f"llm_name mismatch for agent '{name}': expected '{orig_llm}', got '{exp_llm}'")


class TestWorkflowSectionRoundTrip:
    """Tests for workflow section round-trip."""

    def test_workflow_entry_function_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that workflow.entry_function is preserved."""
        original, exported = do_round_trip(simple_calculator_config)

        if "workflow" not in original:
            return  # Skip if no workflow section

        orig_entry = original["workflow"].get("entry_function")
        if orig_entry is None:
            return

        assert "workflow" in exported, "Exported config should have workflow section"
        exp_entry = exported["workflow"].get("entry_function")

        assert exp_entry == orig_entry, (f"entry_function mismatch: expected '{orig_entry}', got '{exp_entry}'")

    def test_workflow_tool_names_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that workflow.tool_names list is preserved."""
        original, exported = do_round_trip(react_agent_config)

        if "workflow" not in original:
            return

        orig_tools = original["workflow"].get("tool_names", [])
        if not orig_tools:
            return

        exp_tools = exported.get("workflow", {}).get("tool_names", [])

        assert set(orig_tools) == set(exp_tools), (f"tool_names mismatch: expected {orig_tools}, got {exp_tools}")


class TestAgentRoundTripAcrossConfigs:
    """Tests for agent round-trip across different configurations."""

    def test_react_agent_round_trip(self, react_agent_config: Path, set_test_env_vars):
        """Verify react agent config survives round-trip."""
        original, exported = do_round_trip(react_agent_config)

        # Find react agents in original
        orig_agents = {
            name: cfg
            for name, cfg in original.get("functions", {}).items()
            if "react" in cfg.get("_type", "").lower() and "agent" in cfg.get("_type", "").lower()
        }

        for name, orig_config in orig_agents.items():
            assert name in exported.get("functions", {}), f"Missing agent '{name}'"

            # Verify llm_name
            orig_llm = orig_config.get("llm_name") or orig_config.get("llm")
            if orig_llm:
                exp_llm = exported["functions"][name].get("llm_name") or exported["functions"][name].get("llm")
                assert exp_llm == orig_llm, f"llm_name mismatch for '{name}'"

    def test_rewoo_agent_round_trip(self, rewoo_agent_config: Path, set_test_env_vars):
        """Verify rewoo agent config survives round-trip."""
        original, exported = do_round_trip(rewoo_agent_config)

        # Find rewoo agents
        orig_agents = {
            name: cfg
            for name, cfg in original.get("functions", {}).items() if "rewoo" in cfg.get("_type", "").lower()
        }

        for name in orig_agents:
            assert name in exported.get("functions", {}), f"Missing agent '{name}'"

    def test_tool_calling_agent_round_trip(self, tool_calling_agent_config: Path, set_test_env_vars):
        """Verify tool-calling agent config survives round-trip."""
        original, exported = do_round_trip(tool_calling_agent_config)

        # Find tool-calling agents
        orig_agents = {
            name: cfg
            for name, cfg in original.get("functions", {}).items() if "tool_calling" in cfg.get("_type", "").lower()
        }

        for name in orig_agents:
            assert name in exported.get("functions", {}), f"Missing agent '{name}'"
