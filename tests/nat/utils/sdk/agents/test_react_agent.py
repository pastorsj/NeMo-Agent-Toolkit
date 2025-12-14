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
"""SDK tests for ReAct Agent configuration and serialization.

This module tests creating, configuring, and serializing ReAct Agents using the SDK.
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
# ReAct Agent SDK Initialization Tests
# ============================================================================


class TestReActAgentInitialization:
    """Tests for NatReActAgent initialization via SDK."""

    def test_basic_react_agent_creation(self):
        """Test creating a basic ReAct agent with just LLM."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        assert agent.llm is llm
        assert agent.tools == []
        assert agent.verbose is True

    def test_react_agent_with_system_prompt(self):
        """Test ReAct agent with custom system prompt."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        system_prompt = "You are a helpful assistant."
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            system_prompt=system_prompt,
        )

        assert agent.system_prompt == system_prompt

    def test_react_agent_with_additional_instructions(self):
        """Test ReAct agent with additional instructions."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        instructions = "Always be concise."
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            additional_instructions=instructions,
        )

        assert agent.additional_instructions == instructions

    def test_react_agent_with_tools(self):
        """Test ReAct agent with tools."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)

        assert len(agent.tools) == 1
        assert agent.tools[0] is tool

    def test_react_agent_with_max_iterations(self):
        """Test ReAct agent with max tool calls configuration."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True, max_tool_calls=25)

        assert agent.max_tool_calls == 25

    def test_react_agent_with_retry_settings(self):
        """Test ReAct agent with retry configuration."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            retry_agent_response_parsing_errors=False,
            parse_agent_response_max_retries=3,
        )

        assert agent.retry_agent_response_parsing_errors is False
        assert agent.parse_agent_response_max_retries == 3

    def test_react_agent_with_max_history(self):
        """Test ReAct agent with max history configuration."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True, max_history=50)

        assert agent.max_history == 50


# ============================================================================
# ReAct Agent Serialization Tests
# ============================================================================


class TestReActAgentSerialization:
    """Tests for serializing ReAct agents to YAML config."""

    def test_basic_react_agent_saves_correctly(self, tmp_path: Path):
        """Test that a basic ReAct agent saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "workflow" in content
        assert content["workflow"]["_type"] == "react_agent"

    def test_react_agent_llm_is_discovered(self, tmp_path: Path):
        """Test that LLM is discovered and saved separately."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="my_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "llms" in content
        assert "my_llm" in content["llms"]
        assert content["llms"]["my_llm"]["model"] == "meta/llama-3.1-70b-instruct"

    def test_react_agent_tools_are_discovered(self, tmp_path: Path):
        """Test that tools are discovered and saved separately."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "functions" in content
        assert "datetime_tool" in content["functions"]

    def test_react_agent_system_prompt_serializes(self, tmp_path: Path):
        """Test that system prompt is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(
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

    def test_react_agent_additional_instructions_serializes(self, tmp_path: Path):
        """Test that additional instructions are serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            additional_instructions="Always explain your reasoning.",
        )
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        wf = content["workflow"]
        assert wf["additional_instructions"] == "Always explain your reasoning."

    def test_react_agent_max_tool_calls_serializes(self, tmp_path: Path):
        """Test that max tool calls setting is serialized."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True, max_tool_calls=30)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["workflow"]["max_tool_calls"] == 30


# ============================================================================
# ReAct Agent Config Round-Trip Tests
# ============================================================================


class TestReActAgentRoundTrip:
    """Tests for round-trip serialization of ReAct agents."""

    def test_react_agent_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            max_tool_calls=20,
            max_history=25,
        )
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.workflow is not None
        assert loaded.workflow.verbose is True
        assert loaded.workflow.max_tool_calls == 20
        assert loaded.workflow.max_history == 25

    def test_react_agent_with_tool_round_trip(self, tmp_path: Path):
        """Test round-trip with tools included."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.workflow is not None
        assert "time_tool" in loaded.functions


# ============================================================================
# ReAct Agent Edge Cases
# ============================================================================


class TestReActAgentEdgeCases:
    """Edge case tests for ReAct agent SDK usage."""

    def test_react_agent_with_empty_system_prompt(self, tmp_path: Path):
        """Test that empty string system prompt is handled."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True, system_prompt="")
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # Empty system prompt should be serialized
        assert content["workflow"]["system_prompt"] == ""

    def test_react_agent_with_multiple_tools(self, tmp_path: Path):
        """Test ReAct agent with multiple tools."""
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

    def test_react_agent_auto_named(self, tmp_path: Path):
        """Test that agent without explicit name gets auto-generated name."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # LLM should have an auto-generated name
        assert "llms" in content
        # Should have exactly one LLM
        assert len(content["llms"]) == 1

    def test_react_agent_verbose_false(self, tmp_path: Path):
        """Test ReAct agent with verbose=False."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=False)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["workflow"]["verbose"] is False
