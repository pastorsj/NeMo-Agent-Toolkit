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
"""SDK tests for FastAPI Front End configuration and serialization.

This module tests creating, configuring, and serializing FastAPI Front Ends using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEnd
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
# FastAPI Front End SDK Initialization Tests
# ============================================================================


class TestFastApiFrontEndInitialization:
    """Tests for FastApiFrontEnd initialization via SDK."""

    def test_basic_fastapi_front_end_creation(self):
        """Test creating a basic FastAPI front end."""
        front_end = FastApiFrontEnd(name="api_server")

        assert front_end.name == "api_server"

    def test_fastapi_front_end_with_port(self):
        """Test FastAPI front end with custom port."""
        front_end = FastApiFrontEnd(name="api_server", port=8888)

        assert front_end.port == 8888

    def test_fastapi_front_end_with_host(self):
        """Test FastAPI front end with custom host."""
        front_end = FastApiFrontEnd(name="api_server", host="0.0.0.0")

        assert front_end.host == "0.0.0.0"

    def test_fastapi_front_end_with_all_params(self):
        """Test FastAPI front end with multiple params."""
        front_end = FastApiFrontEnd(
            name="api_server",
            port=9000,
            host="0.0.0.0",
        )

        assert front_end.port == 9000
        assert front_end.host == "0.0.0.0"


# ============================================================================
# FastAPI Front End Serialization Tests
# ============================================================================


class TestFastApiFrontEndSerialization:
    """Tests for serializing FastAPI front ends to YAML config."""

    def test_fastapi_front_end_saves_correctly(self, tmp_path: Path):
        """Test that FastAPI front end saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = FastApiFrontEnd(name="api_server")
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "general" in content
        assert "front_end" in content["general"]

    def test_fastapi_front_end_type_saved_correctly(self, tmp_path: Path):
        """Test that _type is saved correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = FastApiFrontEnd(name="api_server")
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        front_end_config = content["general"]["front_end"]
        assert front_end_config["_type"] == "fastapi"

    def test_fastapi_front_end_port_serializes(self, tmp_path: Path):
        """Test that port is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = FastApiFrontEnd(name="api_server", port=9000)
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        front_end_config = content["general"]["front_end"]
        assert front_end_config["port"] == 9000

    def test_fastapi_front_end_host_serializes(self, tmp_path: Path):
        """Test that host is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = FastApiFrontEnd(name="api_server", host="127.0.0.1")
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        front_end_config = content["general"]["front_end"]
        assert front_end_config["host"] == "127.0.0.1"


# ============================================================================
# FastAPI Front End Round-Trip Tests
# ============================================================================


class TestFastApiFrontEndRoundTrip:
    """Tests for round-trip serialization of FastAPI front ends."""

    def test_fastapi_front_end_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = FastApiFrontEnd(name="api_server", port=8080, host="0.0.0.0")
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.general.front_end is not None
        loaded_fe = loaded.general.front_end
        assert loaded_fe.port == 8080
        assert loaded_fe.host == "0.0.0.0"


# ============================================================================
# FastAPI Front End Edge Cases
# ============================================================================


class TestFastApiFrontEndEdgeCases:
    """Edge case tests for FastAPI front end SDK usage."""

    def test_fastapi_front_end_multiple_params_save(self, tmp_path: Path):
        """Test FastAPI front end with multiple params saves."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        front_end = FastApiFrontEnd(
            name="api_server",
            port=8888,
            host="127.0.0.1",
        )
        config = NatGeneralConfiguration(front_end_configuration=front_end)

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        front_end_config = content["general"]["front_end"]
        assert front_end_config["port"] == 8888
        assert front_end_config["host"] == "127.0.0.1"

    def test_workflow_without_front_end(self, tmp_path: Path):
        """Test workflow without front end configuration."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Should save without errors
        assert config_path.exists()
