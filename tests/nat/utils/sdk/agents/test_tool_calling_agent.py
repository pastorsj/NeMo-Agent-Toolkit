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
"""SDK tests for Tool Calling Agent configuration and serialization.

This module tests creating, configuring, and serializing Tool Calling Agents using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.sdk import ToolCallingAgent
from nat.data_models.config import Config
from nat.llm.sdk import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.tool.sdk import CurrentTimeTool
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
# Tool Calling Agent SDK Initialization Tests
# ============================================================================


class TestToolCallingAgentInitialization:
    """Tests for ToolCallingAgent initialization via SDK."""

    def test_basic_tool_calling_agent_creation(self):
        """Test creating a basic Tool Calling agent with just LLM."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = ToolCallingAgent(llm=llm, tools=[], verbose=True)

        assert agent.llm is llm
        assert agent.tools == []
        assert agent.verbose is True

    def test_tool_calling_agent_with_system_prompt(self):
        """Test Tool Calling agent with custom system prompt."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        system_prompt = "You are a helpful assistant."
        agent = ToolCallingAgent(
            llm=llm,
            tools=[],
            verbose=True,
            system_prompt=system_prompt,
        )

        assert agent.system_prompt == system_prompt

    def test_tool_calling_agent_with_tools(self):
        """Test Tool Calling agent with tools."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = ToolCallingAgent(llm=llm, tools=[tool], verbose=True)

        assert len(agent.tools) == 1
        assert agent.tools[0] is tool

    def test_tool_calling_agent_with_additional_instructions(self):
        """Test Tool Calling agent with additional instructions."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        instructions = "Always use tools when available."
        agent = ToolCallingAgent(
            llm=llm,
            tools=[],
            verbose=True,
            additional_instructions=instructions,
        )

        assert agent.additional_instructions == instructions


# ============================================================================
# Tool Calling Agent Serialization Tests
# ============================================================================


class TestToolCallingAgentSerialization:
    """Tests for serializing Tool Calling agents to YAML config."""

    def test_basic_tool_calling_agent_saves_correctly(self, tmp_path: Path):
        """Test that a basic Tool Calling agent saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = ToolCallingAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "workflow" in content
        assert content["workflow"]["_type"] == "tool_calling_agent"

    def test_tool_calling_agent_llm_is_discovered(self, tmp_path: Path):
        """Test that LLM is discovered and saved separately."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="my_llm")
        agent = ToolCallingAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "llms" in content
        assert "my_llm" in content["llms"]
        assert content["llms"]["my_llm"]["model"] == "meta/llama-3.1-70b-instruct"

    def test_tool_calling_agent_tools_are_discovered(self, tmp_path: Path):
        """Test that tools are discovered and saved separately."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = ToolCallingAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "functions" in content
        assert "datetime_tool" in content["functions"]

    def test_tool_calling_agent_system_prompt_serializes(self, tmp_path: Path):
        """Test that system prompt is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = ToolCallingAgent(
            llm=llm,
            tools=[],
            verbose=True,
            system_prompt="Be helpful and concise.",
        )
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["workflow"]["system_prompt"] == "Be helpful and concise."


# ============================================================================
# Tool Calling Agent Config Round-Trip Tests
# ============================================================================


class TestToolCallingAgentRoundTrip:
    """Tests for round-trip serialization of Tool Calling agents."""

    def test_tool_calling_agent_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = ToolCallingAgent(
            llm=llm,
            tools=[],
            verbose=True,
        )
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.workflow is not None
        assert loaded.workflow.verbose is True

    def test_tool_calling_agent_with_tool_round_trip(self, tmp_path: Path):
        """Test round-trip with tools included."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = ToolCallingAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.workflow is not None
        assert "time_tool" in loaded.functions


# ============================================================================
# Tool Calling Agent Edge Cases
# ============================================================================


class TestToolCallingAgentEdgeCases:
    """Edge case tests for Tool Calling agent SDK usage."""

    def test_tool_calling_agent_with_multiple_tools(self, tmp_path: Path):
        """Test Tool Calling agent with multiple tools."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool1 = CurrentTimeTool(name="time_tool_1")
        tool2 = CurrentTimeTool(name="time_tool_2")
        agent = ToolCallingAgent(llm=llm, tools=[tool1, tool2], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "time_tool_1" in content["functions"]
        assert "time_tool_2" in content["functions"]

    def test_tool_calling_agent_auto_named(self, tmp_path: Path):
        """Test that agent without explicit name gets auto-generated name."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")
        agent = ToolCallingAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # LLM should have an auto-generated name
        assert "llms" in content
        # Should have exactly one LLM
        assert len(content["llms"]) == 1
