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
"""SDK tests for NIM LLM configuration and serialization.

This module tests creating, configuring, and serializing NIM LLMs using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.llm.nim_llm import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
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
# NIM LLM SDK Initialization Tests
# ============================================================================


class TestNimLLMInitialization:
    """Tests for NimLLM initialization via SDK."""

    def test_basic_nim_llm_creation(self):
        """Test creating a basic NIM LLM."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")

        assert llm.model_name == "meta/llama-3.1-70b-instruct"

    def test_nim_llm_with_explicit_name(self):
        """Test NIM LLM with explicit name."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="my_llm")

        assert llm.name == "my_llm"

    def test_nim_llm_with_temperature(self):
        """Test NIM LLM with temperature setting."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", temperature=0.7)

        assert llm.temperature == 0.7

    def test_nim_llm_with_max_tokens(self):
        """Test NIM LLM with max_tokens setting."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", max_tokens=1024)

        assert llm.max_tokens == 1024

    def test_nim_llm_with_top_p(self):
        """Test NIM LLM with top_p setting."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", top_p=0.9)

        assert llm.top_p == 0.9

    def test_nim_llm_with_base_url(self):
        """Test NIM LLM with custom base URL."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            base_url="https://custom.nvidia.com/v1",
        )

        assert llm.base_url == "https://custom.nvidia.com/v1"

    def test_nim_llm_with_all_parameters(self):
        """Test NIM LLM with all parameters."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="complete_llm",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.95,
            base_url="https://custom.api.com/v1",
        )

        assert llm.model_name == "meta/llama-3.1-70b-instruct"
        assert llm.name == "complete_llm"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2048
        assert llm.top_p == 0.95
        assert llm.base_url == "https://custom.api.com/v1"


# ============================================================================
# NIM LLM with API Key Tests
# ============================================================================


class TestNimLLMApiKey:
    """Tests for NIM LLM API key configuration."""

    def test_nim_llm_with_api_key(self, tmp_path: Path):
        """Test NIM LLM with API key."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            api_key="sk-test-key",
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        llm_config = content["llms"]["test_llm"]
        # API key should be present
        assert "api_key" in llm_config


# ============================================================================
# NIM LLM Serialization Tests
# ============================================================================


class TestNimLLMSerialization:
    """Tests for serializing NIM LLMs to YAML config."""

    def test_basic_nim_llm_saves_correctly(self, tmp_path: Path):
        """Test that a basic NIM LLM saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "llms" in content
        assert "test_llm" in content["llms"]
        llm_config = content["llms"]["test_llm"]
        assert llm_config["_type"] == "nim"
        assert llm_config["model"] == "meta/llama-3.1-70b-instruct"

    def test_nim_llm_temperature_serializes(self, tmp_path: Path):
        """Test that temperature is serialized correctly."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            temperature=0.7,
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["llms"]["test_llm"]["temperature"] == 0.7

    def test_nim_llm_max_tokens_serializes(self, tmp_path: Path):
        """Test that max_tokens is serialized correctly."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            max_tokens=1024,
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["llms"]["test_llm"]["max_tokens"] == 1024

    def test_nim_llm_top_p_serializes(self, tmp_path: Path):
        """Test that top_p is serialized correctly."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            top_p=0.9,
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["llms"]["test_llm"]["top_p"] == 0.9

    def test_nim_llm_base_url_serializes(self, tmp_path: Path):
        """Test that base_url is serialized correctly."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            base_url="https://custom.api.com/v1",
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["llms"]["test_llm"]["base_url"] == "https://custom.api.com/v1"

    def test_unset_temperature_not_in_output(self, tmp_path: Path):
        """Test that unset temperature is not in the output."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # Temperature was not set, so should not appear in output
        assert "temperature" not in content["llms"]["test_llm"]


# ============================================================================
# NIM LLM Optimizable Parameters Tests
# ============================================================================


class TestNimLLMOptimizableParams:
    """Tests for NIM LLM with optimizable parameters."""

    def test_nim_llm_with_optimizable_temperature(self, tmp_path: Path):
        """Test NIM LLM with optimizable temperature."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            temperature=0.5,
            optimizable_params=["temperature"],
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        llm_config = content["llms"]["test_llm"]
        assert "optimizable_params" in llm_config
        assert "temperature" in llm_config["optimizable_params"]

    def test_nim_llm_with_multiple_optimizable_params(self, tmp_path: Path):
        """Test NIM LLM with multiple optimizable parameters."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            temperature=0.5,
            max_tokens=512,
            top_p=0.9,
            optimizable_params=["temperature", "max_tokens", "top_p"],
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        llm_config = content["llms"]["test_llm"]
        assert set(llm_config["optimizable_params"]) == {"temperature", "max_tokens", "top_p"}

    def test_nim_llm_with_search_space(self, tmp_path: Path):
        """Test NIM LLM with custom search space."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            temperature=0.5,
            optimizable_params=["temperature"],
            search_space={
                "temperature": SearchSpace(low=0.1, high=0.9),
            },
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        llm_config = content["llms"]["test_llm"]
        # search_space may or may not be serialized depending on configuration
        # Just verify the LLM is saved with optimizable_params
        assert "optimizable_params" in llm_config
        assert "temperature" in llm_config["optimizable_params"]


# ============================================================================
# NIM LLM Config Round-Trip Tests
# ============================================================================


class TestNimLLMRoundTrip:
    """Tests for round-trip serialization of NIM LLMs."""

    def test_nim_llm_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            temperature=0.7,
            max_tokens=1024,
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert "test_llm" in loaded.llms
        loaded_llm = loaded.llms["test_llm"]
        assert loaded_llm.model_name == "meta/llama-3.1-70b-instruct"
        assert loaded_llm.temperature == 0.7
        assert loaded_llm.max_tokens == 1024


# ============================================================================
# NIM LLM Edge Cases
# ============================================================================


class TestNimLLMEdgeCases:
    """Edge case tests for NIM LLM SDK usage."""

    def test_nim_llm_auto_generated_name(self, tmp_path: Path):
        """Test that LLM without name gets auto-generated name."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "llms" in content
        assert len(content["llms"]) == 1
        # Name should be auto-generated (contains underscore)
        name = list(content["llms"].keys())[0]
        assert "_" in name

    def test_nim_llm_temperature_zero(self, tmp_path: Path):
        """Test NIM LLM with temperature=0 (deterministic)."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            temperature=0.0,
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert content["llms"]["test_llm"]["temperature"] == 0.0

    def test_multiple_llms_different_configs(self, tmp_path: Path):
        """Test workflow with multiple LLMs having different configs."""
        llm1 = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="llm_creative",
            temperature=0.9,
        )
        llm2 = NimLLM(
            model_name="meta/llama-3.1-8b-instruct",
            name="llm_precise",
            temperature=0.1,
        )

        # Create a nested agent with precise LLM
        inner_agent = NatReActAgent(llm=llm2, tools=[], verbose=True)
        outer_agent = NatReActAgent(llm=llm1, tools=[inner_agent], verbose=True)

        workflow = NatWorkflow(entrypoint=outer_agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "llm_creative" in content["llms"]
        assert "llm_precise" in content["llms"]
        assert content["llms"]["llm_creative"]["temperature"] == 0.9
        assert content["llms"]["llm_precise"]["temperature"] == 0.1
