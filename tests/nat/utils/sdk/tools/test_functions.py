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
"""SDK tests for Function/Tool configuration and serialization.

This module tests creating, configuring, and serializing Functions using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.config import Config
from nat.llm.nim_llm import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_workflow import NatWorkflow


@pytest.fixture(scope="module", autouse=True)
def discover_plugins():
    """Discover and register all plugins before running tests."""
    discover_and_register_plugins(PluginTypes.ALL)


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# Function/Tool Initialization Tests
# ============================================================================


class TestFunctionInitialization:
    """Tests for function/tool initialization via SDK."""

    def test_basic_tool_creation(self):
        """Test creating a basic tool."""
        tool = CurrentTimeTool(name="time_tool")

        assert tool.name == "time_tool"

    def test_tool_with_name(self):
        """Test tool with explicit name."""
        tool = CurrentTimeTool(name="my_time_tool")

        assert tool.name == "my_time_tool"

    def test_tool_auto_generated_name(self):
        """Test that tool without explicit name gets auto-generated name."""
        tool = CurrentTimeTool()

        # Should have some form of name (computed or default)
        computed_name = tool.computed_name
        assert computed_name is not None
        assert len(computed_name) > 0


# ============================================================================
# Function/Tool Serialization Tests
# ============================================================================


class TestFunctionSerialization:
    """Tests for serializing functions to YAML config."""

    def test_tool_saves_in_functions_section(self, tmp_path: Path):
        """Test that a tool saves in the functions section."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "functions" in content
        assert "datetime_tool" in content["functions"]

    def test_tool_type_saved_correctly(self, tmp_path: Path):
        """Test that tool _type is saved correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        tool_config = content["functions"]["datetime_tool"]
        assert "_type" in tool_config
        assert tool_config["_type"] == "current_datetime"

    def test_multiple_tools_save_correctly(self, tmp_path: Path):
        """Test that multiple tools save correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool1 = CurrentTimeTool(name="time_tool_1")
        tool2 = CurrentTimeTool(name="time_tool_2")
        agent = NatReActAgent(llm=llm, tools=[tool1, tool2], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "time_tool_1" in content["functions"]
        assert "time_tool_2" in content["functions"]

    def test_tool_name_serializes(self, tmp_path: Path):
        """Test that tool name is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # Tool should exist with correct name
        assert "datetime_tool" in content["functions"]


# ============================================================================
# Function/Tool Round-Trip Tests
# ============================================================================


class TestFunctionRoundTrip:
    """Tests for round-trip serialization of functions."""

    def test_tool_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert "datetime_tool" in loaded.functions


# ============================================================================
# Agent with Tool Reference Tests
# ============================================================================


class TestAgentToolReferences:
    """Tests for agent tool references in YAML output."""

    def test_agent_references_tool_by_name(self, tmp_path: Path):
        """Test that agent references tool by name in workflow config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # Agent should reference tool by name
        assert "tool_names" in content["workflow"]
        assert "datetime_tool" in content["workflow"]["tool_names"]


# ============================================================================
# Tool Edge Cases
# ============================================================================


class TestToolEdgeCases:
    """Edge case tests for tool SDK usage."""

    def test_duplicate_tool_names_use_same_config(self, tmp_path: Path):
        """Test that same tool added twice uses same config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        # Add same tool twice
        agent = NatReActAgent(llm=llm, tools=[tool, tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # Should only have one entry in functions
        assert len([k for k in content["functions"] if k.startswith("datetime")]) == 1

    def test_empty_tools_list(self, tmp_path: Path):
        """Test agent with empty tools list."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # functions section may not exist or be empty
        if "functions" in content:
            # If present, should be empty or None
            assert content["functions"] is None or len(content["functions"]) == 0
