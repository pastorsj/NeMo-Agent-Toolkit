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
"""SDK tests for ReWOO Agent configuration and serialization.

This module tests creating, configuring, and serializing ReWOO Agents using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.rewoo_agent.register import ReWOOAgentWorkflow
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
# ReWOO Agent SDK Initialization Tests
# ============================================================================


class TestReWOOAgentInitialization:
    """Tests for ReWOOAgentWorkflow initialization via SDK."""

    def test_basic_rewoo_agent_creation(self):
        """Test creating a basic ReWOO agent with just LLM."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = ReWOOAgentWorkflow(llm=llm, tools=[], verbose=True)

        assert agent.llm is llm
        assert agent.verbose is True

    def test_rewoo_agent_with_tools(self):
        """Test ReWOO agent with tools."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = ReWOOAgentWorkflow(llm=llm, tools=[tool], verbose=True)

        assert len(agent.tools) == 1
        assert agent.tools[0] is tool


# ============================================================================
# ReWOO Agent Serialization Tests
# ============================================================================


class TestReWOOAgentSerialization:
    """Tests for serializing ReWOO agents to YAML config."""

    def test_basic_rewoo_agent_saves_correctly(self, tmp_path: Path):
        """Test that a basic ReWOO agent saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = ReWOOAgentWorkflow(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "workflow" in content
        assert content["workflow"]["_type"] == "rewoo_agent"

    def test_rewoo_agent_llm_is_discovered(self, tmp_path: Path):
        """Test that LLM is discovered and saved separately."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="my_llm")
        agent = ReWOOAgentWorkflow(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "llms" in content
        assert "my_llm" in content["llms"]
        assert content["llms"]["my_llm"]["model"] == "meta/llama-3.1-70b-instruct"

    def test_rewoo_agent_tools_are_discovered(self, tmp_path: Path):
        """Test that tools are discovered and saved separately."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="datetime_tool")
        agent = ReWOOAgentWorkflow(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "functions" in content
        assert "datetime_tool" in content["functions"]


# ============================================================================
# ReWOO Agent Config Round-Trip Tests
# ============================================================================


class TestReWOOAgentRoundTrip:
    """Tests for round-trip serialization of ReWOO agents."""

    def test_rewoo_agent_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = ReWOOAgentWorkflow(
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

    def test_rewoo_agent_with_tool_round_trip(self, tmp_path: Path):
        """Test round-trip with tools included."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = ReWOOAgentWorkflow(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.workflow is not None
        assert "time_tool" in loaded.functions
