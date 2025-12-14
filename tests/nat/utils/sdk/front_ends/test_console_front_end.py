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
"""SDK tests for Console Front End configuration and serialization.

This module tests creating, configuring, and serializing Console Front Ends using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.config import Config
from nat.front_ends.console.console_front_end_config import ConsoleFrontEnd
from nat.llm.nim_llm import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils.sdk.nat_general_configuraton import NatGeneralConfiguration
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
# Console Front End SDK Initialization Tests
# ============================================================================


class TestConsoleFrontEndInitialization:
    """Tests for ConsoleFrontEnd initialization via SDK."""

    def test_basic_console_front_end_creation(self):
        """Test creating a basic console front end."""
        front_end = ConsoleFrontEnd(name="console")

        assert front_end.name == "console"


# ============================================================================
# Console Front End Serialization Tests
# ============================================================================


class TestConsoleFrontEndSerialization:
    """Tests for serializing console front ends to YAML config."""

    def test_console_front_end_saves_correctly(self, tmp_path: Path):
        """Test that console front end saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = ConsoleFrontEnd(name="console")
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "general" in content
        assert "front_end" in content["general"]

    def test_console_front_end_type_saved_correctly(self, tmp_path: Path):
        """Test that _type is saved correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = ConsoleFrontEnd(name="console")
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        front_end_config = content["general"]["front_end"]
        assert front_end_config["_type"] == "console"


# ============================================================================
# Console Front End Round-Trip Tests
# ============================================================================


class TestConsoleFrontEndRoundTrip:
    """Tests for round-trip serialization of console front ends."""

    def test_console_front_end_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = ConsoleFrontEnd(name="console")
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.general.front_end is not None
