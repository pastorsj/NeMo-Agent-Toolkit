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
"""Tests for importing agent configurations from YAML to workflow state.

These tests verify that agent configurations are correctly parsed and all
field values are accurately preserved during import.
"""

from pathlib import Path

import yaml

from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportReactAgent:
    """Tests for importing ReAct agent configuration.

    The react agent config contains a ReAct agent with specific settings.
    These tests verify each field is correctly imported.
    """

    def test_react_agent_is_identified_correctly(self, react_agent_config: Path, set_test_env_vars):
        """Verify ReAct agent is identified with component_type='agent'."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        agents = [c for c in workflow_state.components if c.component_type == "agent"]
        assert len(agents) >= 1, "Expected at least one agent component"

        # Check that we have a react agent
        react_agents = [a for a in agents if "react" in a.full_type.lower()]
        assert len(react_agents) >= 1, "Expected at least one ReAct agent"

    def test_agent_full_type_matches_original(self, react_agent_config: Path, set_test_env_vars):
        """Verify agent full_type matches the original _type from config."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        # Find agent in functions section
        original_functions = config_dict.get("functions", {})
        for name, func_config in original_functions.items():
            orig_type = func_config.get("_type", "")
            if "agent" not in orig_type.lower():
                continue

            orig_type_short = orig_type.split("/")[-1]
            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Agent '{name}' not found in imported components"

            imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Agent type mismatch for '{name}': expected '{orig_type_short}', got '{imported_type_short}'")

    def test_agent_id_matches_original_name(self, react_agent_config: Path, set_test_env_vars):
        """Verify agent ID matches the key name from the original config."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_functions = config_dict.get("functions", {})
        agent_names = {name for name, cfg in original_functions.items() if "agent" in cfg.get("_type", "").lower()}
        imported_agent_ids = {c.id for c in workflow_state.components if c.component_type == "agent"}

        assert agent_names == imported_agent_ids, (
            f"Agent ID mismatch: original={agent_names}, imported={imported_agent_ids}")

    def test_agent_llm_reference_creates_connection(self, react_agent_config: Path, set_test_env_vars):
        """Verify that agent's llm_name creates a connection to the LLM."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        # Find agent and its LLM reference
        original_functions = config_dict.get("functions", {})
        for name, func_config in original_functions.items():
            if "agent" not in func_config.get("_type", "").lower():
                continue

            llm_ref = func_config.get("llm_name") or func_config.get("llm")
            if llm_ref is None:
                continue

            # Should have a connection from agent to LLM
            llm_connections = [
                c for c in workflow_state.connections if c.source_id == name and "llm" in c.target_field.lower()
            ]
            assert len(llm_connections) >= 1, (f"Expected connection from agent '{name}' to LLM '{llm_ref}'")

            # Connection target should be the referenced LLM
            assert llm_connections[0].target_id == llm_ref, (
                f"Connection target mismatch: expected '{llm_ref}', got '{llm_connections[0].target_id}'")


class TestImportReWOOAgent:
    """Tests for importing ReWOO agent configuration."""

    def test_rewoo_agent_is_identified_correctly(self, rewoo_agent_config: Path, set_test_env_vars):
        """Verify ReWOO agent is identified with component_type='agent'."""
        config_dict = load_config_dict(rewoo_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        agents = [c for c in workflow_state.components if c.component_type == "agent"]
        assert len(agents) >= 1, "Expected at least one agent component"

        # Check for rewoo agent
        rewoo_agents = [a for a in agents if "rewoo" in a.full_type.lower()]
        assert len(rewoo_agents) >= 1, "Expected at least one ReWOO agent"


class TestImportToolCallingAgent:
    """Tests for importing tool-calling agent configuration."""

    def test_tool_calling_agent_is_identified(self, tool_calling_agent_config: Path, set_test_env_vars):
        """Verify tool-calling agent is identified with component_type='agent'."""
        config_dict = load_config_dict(tool_calling_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        agents = [c for c in workflow_state.components if c.component_type == "agent"]
        assert len(agents) >= 1, "Expected at least one agent component"

        # Check for tool_calling agent
        tc_agents = [a for a in agents if "tool_calling" in a.full_type.lower()]
        assert len(tc_agents) >= 1, "Expected at least one tool-calling agent"
