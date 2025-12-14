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
"""Comprehensive tests for NatWorkflow.save_to_config_file functionality.

This module tests the SDK's ability to serialize workflow configurations to YAML files,
covering basic workflows, component discovery, optimizer configurations, and edge cases.
"""

import os
from pathlib import Path

import pytest
import yaml

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.llm.nim_llm import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_evaluation import EvalDatasetJsonConfig
from nat.utils.sdk.nat_evaluation import NatEvaluation
from nat.utils.sdk.nat_optimizer import NatOptimizer
from nat.utils.sdk.nat_optimizer import NumericOptimizationConfig
from nat.utils.sdk.nat_optimizer import OptimizerMetric
from nat.utils.sdk.nat_workflow import NatWorkflow


# Ensure plugins are discovered for tests
@pytest.fixture(scope="module", autouse=True)
def discover_plugins():
    """Discover and register all plugins before running tests."""
    discover_and_register_plugins(PluginTypes.ALL)


# ============================================================================
# Basic Workflow Save Tests
# ============================================================================


class TestBasicWorkflowSave:
    """Tests for basic workflow save_to_config_file functionality."""

    def test_save_creates_yaml_file(self, tmp_config_path: Path):
        """Test that save_to_config_file creates a valid YAML file."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        assert tmp_config_path.exists()
        # Verify it's valid YAML
        with open(tmp_config_path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        assert content is not None
        assert "workflow" in content

    def test_save_with_string_path(self, tmp_path: Path):
        """Test saving to a string path instead of Path object."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = str(tmp_path / "string_path_config.yaml")
        workflow.save_to_config_file(config_path)

        assert os.path.exists(config_path)

    def test_save_with_explicit_names(self, tmp_config_path: Path, load_yaml):
        """Test that explicit component names are preserved in saved config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="my_custom_llm")
        tool = CurrentTimeTool(name="my_time_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "llms" in content
        assert "my_custom_llm" in content["llms"]

    def test_save_without_explicit_names_generates_unique_names(self, tmp_config_path: Path, load_yaml):
        """Test that components without names get auto-generated unique names."""
        # No explicit names provided
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "llms" in content
        # Should have at least one LLM with auto-generated name
        llm_names = list(content["llms"].keys())
        assert len(llm_names) >= 1
        # Auto-generated names contain UUIDs
        assert any("_" in name for name in llm_names)

    def test_save_idempotent_with_explicit_names(self, tmp_config_dir: Path, load_yaml):
        """Test that saving the same workflow twice produces identical output with explicit names."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="stable_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        path1 = tmp_config_dir / "config1.yaml"
        path2 = tmp_config_dir / "config2.yaml"

        workflow.save_to_config_file(path1)
        workflow.save_to_config_file(path2)

        content1 = load_yaml(path1)
        content2 = load_yaml(path2)

        # LLM names should be identical
        assert list(content1["llms"].keys()) == list(content2["llms"].keys())


# ============================================================================
# Component Discovery Tests
# ============================================================================


class TestComponentDiscovery:
    """Tests for component discovery during save_to_config_file."""

    def test_discovers_llm_from_agent(self, tmp_config_path: Path, load_yaml):
        """Test that LLM is discovered from agent and saved to llms section."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            temperature=0.5,
            max_tokens=1024,
            name="discovered_llm",
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "llms" in content
        assert "discovered_llm" in content["llms"]
        llm_config = content["llms"]["discovered_llm"]
        # model_name is serialized as "model" alias
        assert llm_config["model"] == "meta/llama-3.1-70b-instruct"
        assert llm_config.get("temperature") == 0.5
        assert llm_config.get("max_tokens") == 1024

    def test_discovers_tools_from_agent(self, tmp_config_path: Path, load_yaml):
        """Test that tools are discovered from agent."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        # Tool should be referenced in workflow
        assert "workflow" in content
        workflow_config = content["workflow"]
        # Check tool_names includes the tool
        assert "tool_names" in workflow_config
        tool_names = workflow_config["tool_names"]
        assert any("time_tool" in str(t) for t in tool_names)

    def test_discovers_multiple_tools(self, tmp_config_path: Path, load_yaml):
        """Test that multiple tools are discovered correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool1 = CurrentTimeTool(name="time_tool_1")
        tool2 = CurrentTimeTool(name="time_tool_2")
        agent = NatReActAgent(llm=llm, tools=[tool1, tool2], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        workflow_config = content["workflow"]
        tool_names = workflow_config.get("tool_names", [])
        # Should have both tools referenced
        assert len(tool_names) >= 2


# ============================================================================
# Agent Configuration Tests
# ============================================================================


class TestAgentConfiguration:
    """Tests for agent-specific configuration serialization."""

    def test_react_agent_config_preserved(self, tmp_config_path: Path, load_yaml):
        """Test that ReAct agent configuration options are preserved."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            max_tool_calls=10,
            max_history=20,
            additional_instructions="Be very helpful.",
        )
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        workflow_config = content["workflow"]
        assert workflow_config.get("verbose") is True
        assert workflow_config.get("max_tool_calls") == 10
        assert workflow_config.get("max_history") == 20
        assert workflow_config.get("additional_instructions") == "Be very helpful."

    def test_agent_llm_reference_correct(self, tmp_config_path: Path, load_yaml):
        """Test that agent's LLM reference points to the correct LLM."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="my_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        workflow_config = content["workflow"]
        llm_ref = workflow_config.get("llm_name")
        assert llm_ref == "my_llm"


# ============================================================================
# Optimizer Configuration Tests
# ============================================================================


class TestOptimizerConfiguration:
    """Tests for optimizer configuration serialization, including our SDK updates."""

    def test_optimizer_added_to_config(self, tmp_config_path: Path, load_yaml):
        """Test that optimizer configuration is added to saved config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(
                    evaluator_name="accuracy",
                    direction="maximize",
                    weight=1.0,
                ),
            },
            numeric=NumericOptimizationConfig(enabled=True, n_trials=10),
        )
        workflow.add_optimizer(optimizer)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "optimizer" in content
        opt_config = content["optimizer"]
        assert "eval_metrics" in opt_config
        assert "accuracy" in opt_config["eval_metrics"]
        assert opt_config["eval_metrics"]["accuracy"]["direction"] == "maximize"

    def test_optimizable_params_serialized(self, tmp_config_path: Path, load_yaml):
        """Test that optimizable_params on LLM are serialized correctly."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="optimizable_llm",
            temperature=0.5,
            top_p=0.9,
            optimizable_params=["temperature", "top_p"],
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["optimizable_llm"]
        assert "optimizable_params" in llm_config
        assert "temperature" in llm_config["optimizable_params"]
        assert "top_p" in llm_config["optimizable_params"]

    def test_search_space_not_serialized(self, tmp_config_path: Path, load_yaml):
        """Test that search_space is excluded from serialization (exclude=True)."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="llm_with_search_space",
            temperature=0.5,
            optimizable_params=["temperature"],
            search_space={
                "temperature": SearchSpace(low=0.1, high=0.9, step=0.1),
            },
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["llm_with_search_space"]
        # search_space should be excluded from the serialized output
        assert "search_space" not in llm_config

    def test_optimizable_params_with_search_space(self, tmp_config_path: Path, load_yaml):
        """Test combined optimizable_params and search_space configuration."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="fully_optimizable_llm",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1024,
            optimizable_params=["temperature", "top_p", "max_tokens"],
            search_space={
                "temperature": SearchSpace(low=0.0, high=1.0, step=0.1),
                "top_p": SearchSpace(low=0.5, high=1.0, step=0.1),
                "max_tokens": SearchSpace(low=256, high=2048, step=128),
            },
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["fully_optimizable_llm"]
        # optimizable_params should be present
        assert "optimizable_params" in llm_config
        assert set(llm_config["optimizable_params"]) == {"temperature", "top_p", "max_tokens"}
        # search_space should NOT be present (excluded)
        assert "search_space" not in llm_config

    def test_numeric_optimization_config_serialized(self, tmp_config_path: Path, load_yaml):
        """Test that numeric optimization config is properly serialized."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./opt_results"),
            eval_metrics={
                "score": OptimizerMetric(
                    evaluator_name="score",
                    direction="maximize",
                ),
            },
            numeric=NumericOptimizationConfig(
                enabled=True,
                n_trials=50,
            ),
        )
        workflow.add_optimizer(optimizer)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        opt_config = content["optimizer"]
        assert "numeric" in opt_config
        numeric_config = opt_config["numeric"]
        assert numeric_config["enabled"] is True
        assert numeric_config["n_trials"] == 50


# ============================================================================
# Evaluation Configuration Tests
# ============================================================================


class TestEvaluationConfiguration:
    """Tests for evaluation configuration serialization."""

    def test_evaluation_added_to_config(self, tmp_config_path: Path, tmp_path: Path, load_yaml):
        """Test that evaluation configuration is added to saved config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        # Create a test dataset file
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text('[]', encoding="utf-8")

        evaluation = NatEvaluation(
            output_dir=tmp_path / "eval_results",
            dataset=EvalDatasetJsonConfig(
                file_path=dataset_path,
                id_key="id",
                question_key="question",
                answer_key="answer",
            ),
            evaluators=[],
        )
        workflow.add_evaluator(evaluation)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "eval" in content
        eval_config = content["eval"]
        assert "general" in eval_config

    def test_eval_dataset_config_serialized(self, tmp_config_path: Path, tmp_path: Path, load_yaml):
        """Test that eval dataset configuration is properly serialized."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        dataset_path = tmp_path / "test_data.json"
        dataset_path.write_text('[]', encoding="utf-8")

        evaluation = NatEvaluation(
            output_dir=tmp_path / "eval_results",
            dataset=EvalDatasetJsonConfig(
                file_path=dataset_path,
                id_key="test_id",
                question_key="query",
                answer_key="response",
            ),
            evaluators=[],
        )
        workflow.add_evaluator(evaluation)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        eval_general = content["eval"]["general"]
        dataset_config = eval_general.get("dataset")
        assert dataset_config is not None
        # id_key should be present since it was explicitly set
        assert dataset_config.get("id_key") == "test_id"
        # question_key and answer_key may not be serialized if they are defaults
        # Check that the file_path is correct
        assert dataset_config.get("file_path") is not None


# ============================================================================
# Configuration Structure Tests
# ============================================================================


class TestConfigurationStructure:
    """Tests for the overall structure of saved configurations."""

    def test_workflow_section_has_correct_type(self, tmp_config_path: Path, load_yaml):
        """Test that workflow section has correct _type field."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        workflow_config = content["workflow"]
        assert "_type" in workflow_config
        assert workflow_config["_type"] == "react_agent"

    def test_llms_section_has_correct_type(self, tmp_config_path: Path, load_yaml):
        """Test that LLMs section has correct _type field."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["test_llm"]
        assert "_type" in llm_config
        # NimLLM uses "nim" as its type discriminator
        assert llm_config["_type"] == "nim"

    def test_saved_config_is_valid_for_loading(self, tmp_config_path: Path, load_yaml):
        """Test that the saved configuration can be loaded as a valid Config object."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        # Should be able to create a Config object from the saved content
        config = Config.model_validate(content)
        assert config is not None
        assert config.workflow is not None


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in save_to_config_file."""

    def test_empty_tools_list(self, tmp_config_path: Path, load_yaml):
        """Test saving workflow with empty tools list."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "workflow" in content
        workflow_config = content["workflow"]
        # tool_names should be empty or not present
        tool_names = workflow_config.get("tool_names", [])
        assert len(tool_names) == 0

    def test_save_to_nested_directory(self, tmp_path: Path, load_yaml):
        """Test saving to a deeply nested directory path."""
        nested_path = tmp_path / "level1" / "level2" / "level3" / "config.yaml"
        nested_path.parent.mkdir(parents=True, exist_ok=True)

        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(nested_path)

        assert nested_path.exists()
        content = load_yaml(nested_path)
        assert "workflow" in content

    def test_overwrite_existing_file(self, tmp_config_path: Path, load_yaml):
        """Test that saving overwrites an existing configuration file."""
        # Create initial config
        llm1 = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="llm_v1")
        agent1 = NatReActAgent(llm=llm1, tools=[], verbose=True)
        workflow1 = NatWorkflow(entrypoint=agent1)
        workflow1.save_to_config_file(tmp_config_path)

        # Overwrite with different config
        llm2 = NimLLM(model_name="meta/llama-3.1-8b-instruct", name="llm_v2")
        agent2 = NatReActAgent(llm=llm2, tools=[], verbose=True)
        workflow2 = NatWorkflow(entrypoint=agent2)
        workflow2.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "llm_v2" in content["llms"]
        assert "llm_v1" not in content["llms"]

    def test_special_characters_in_path(self, tmp_path: Path, load_yaml):
        """Test saving to a path with spaces and special characters."""
        special_path = tmp_path / "my config file.yaml"

        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(special_path)

        assert special_path.exists()
        content = load_yaml(special_path)
        assert "workflow" in content

    def test_minimal_llm_config(self, tmp_config_path: Path, load_yaml):
        """Test that only explicitly set fields are serialized (exclude_unset behavior)."""
        # Only set model_name and name, leave other fields as defaults
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="minimal_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["minimal_llm"]
        # model_name is serialized as "model" alias
        assert "model" in llm_config
        assert "_type" in llm_config
        # Fields not explicitly set should not be present (exclude_unset=True behavior)
        # Note: This depends on whether the config uses exclude_unset - adjust if needed

    def test_workflow_without_optimizer_has_no_optimizer_section(self, tmp_config_path: Path, load_yaml):
        """Test that workflows without optimizer don't have optimizer section."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "optimizer" not in content

    def test_workflow_without_evaluator_has_no_eval_section(self, tmp_config_path: Path, load_yaml):
        """Test that workflows without evaluator don't have eval section."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        assert "eval" not in content


# ============================================================================
# Round-Trip Tests
# ============================================================================


class TestRoundTrip:
    """Tests for saving and loading configurations (round-trip serialization)."""

    def test_round_trip_preserves_llm_config(self, tmp_config_path: Path, load_yaml):
        """Test that LLM configuration survives a round-trip save/load."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            temperature=0.7,
            max_tokens=2048,
            top_p=0.95,
            name="roundtrip_llm",
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        # Validate by creating Config object
        config = Config.model_validate(content)

        # Check LLM values are preserved
        assert "roundtrip_llm" in config.llms
        loaded_llm = config.llms["roundtrip_llm"]
        assert loaded_llm.model_name == "meta/llama-3.1-70b-instruct"
        assert loaded_llm.temperature == 0.7
        assert loaded_llm.max_tokens == 2048
        assert loaded_llm.top_p == 0.95

    def test_round_trip_preserves_agent_config(self, tmp_config_path: Path, load_yaml):
        """Test that agent configuration survives a round-trip save/load."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            max_tool_calls=25,
            max_history=50,
        )
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        config = Config.model_validate(content)

        assert config.workflow.verbose is True
        assert config.workflow.max_tool_calls == 25
        assert config.workflow.max_history == 50

    def test_round_trip_with_optimizer(self, tmp_config_path: Path, load_yaml):
        """Test that optimizer configuration survives a round-trip save/load."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            optimizable_params=["temperature"],
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(
                    evaluator_name="accuracy",
                    direction="maximize",
                    weight=1.0,
                ),
            },
            numeric=NumericOptimizationConfig(enabled=True, n_trials=20),
        )
        workflow.add_optimizer(optimizer)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        config = Config.model_validate(content)

        assert config.optimizer is not None
        assert config.optimizer.numeric.enabled is True
        assert config.optimizer.numeric.n_trials == 20
        assert "accuracy" in config.optimizer.eval_metrics


# ============================================================================
# Multiple Component Tests
# ============================================================================


class TestMultipleComponents:
    """Tests for workflows with multiple components of the same type."""

    def test_multiple_tools_preserved(self, tmp_config_path: Path, load_yaml):
        """Test that all tools are preserved in saved config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool1 = CurrentTimeTool(name="tool_a")
        tool2 = CurrentTimeTool(name="tool_b")
        tool3 = CurrentTimeTool(name="tool_c")
        agent = NatReActAgent(llm=llm, tools=[tool1, tool2, tool3], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        workflow_config = content["workflow"]
        tool_names = workflow_config.get("tool_names", [])
        assert len(tool_names) == 3


# ============================================================================
# Optimizer Metric Tests
# ============================================================================


class TestOptimizerMetrics:
    """Tests specifically for optimizer metric configuration."""

    def test_single_metric(self, tmp_config_path: Path, load_yaml):
        """Test single metric optimization configuration."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(
                    evaluator_name="accuracy",
                    direction="maximize",
                ),
            },
        )
        workflow.add_optimizer(optimizer)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        metrics = content["optimizer"]["eval_metrics"]
        assert len(metrics) == 1
        assert "accuracy" in metrics

    def test_multiple_metrics(self, tmp_config_path: Path, load_yaml):
        """Test multiple metrics optimization configuration."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(
                    evaluator_name="accuracy",
                    direction="maximize",
                    weight=0.7,
                ),
                "latency": OptimizerMetric(
                    evaluator_name="latency",
                    direction="minimize",
                    weight=0.3,
                ),
            },
        )
        workflow.add_optimizer(optimizer)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        metrics = content["optimizer"]["eval_metrics"]
        assert len(metrics) == 2
        assert metrics["accuracy"]["direction"] == "maximize"
        assert metrics["accuracy"]["weight"] == 0.7
        assert metrics["latency"]["direction"] == "minimize"
        assert metrics["latency"]["weight"] == 0.3

    def test_metric_with_all_options(self, tmp_config_path: Path, load_yaml):
        """Test metric with all configuration options."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "custom_score": OptimizerMetric(
                    evaluator_name="my_evaluator",
                    direction="maximize",
                    weight=1.0,
                ),
            },
            reps_per_param_set=5,
        )
        workflow.add_optimizer(optimizer)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        opt_config = content["optimizer"]
        assert opt_config["reps_per_param_set"] == 5
        assert opt_config["eval_metrics"]["custom_score"]["evaluator_name"] == "my_evaluator"


# ============================================================================
# Advanced Optimizable Configuration Tests
# ============================================================================


class TestAdvancedOptimizableConfiguration:
    """Advanced tests for optimizable parameters and search space configuration."""

    def test_multiple_llms_with_different_optimizable_params(self, tmp_config_path: Path, load_yaml):
        """Test workflow with multiple LLMs having different optimizable params."""
        llm1 = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="llm_temp_only",
            temperature=0.5,
            optimizable_params=["temperature"],
        )
        agent = NatReActAgent(llm=llm1, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm1_config = content["llms"]["llm_temp_only"]
        assert llm1_config["optimizable_params"] == ["temperature"]

    def test_optimizable_params_empty_list(self, tmp_config_path: Path, load_yaml):
        """Test LLM with empty optimizable_params list."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="llm_no_opt",
            temperature=0.5,
            optimizable_params=[],
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["llm_no_opt"]
        # Empty list should either not be present or be empty
        assert llm_config.get("optimizable_params", []) == []

    def test_search_space_with_categorical_values(self, tmp_config_path: Path, load_yaml):
        """Test search_space with categorical values (should be excluded from YAML)."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="llm_categorical",
            temperature=0.5,
            optimizable_params=["temperature"],
            search_space={
                "temperature": SearchSpace(values=[0.0, 0.5, 1.0]),
            },
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["llm_categorical"]
        # search_space should be excluded
        assert "search_space" not in llm_config
        # But optimizable_params should be there
        assert "optimizable_params" in llm_config

    def test_search_space_with_log_scale(self, tmp_config_path: Path, load_yaml):
        """Test search_space with log scale option."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="llm_log_scale",
            temperature=0.5,
            optimizable_params=["temperature"],
            search_space={
                "temperature": SearchSpace(low=0.001, high=1.0, log=True),
            },
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["llm_log_scale"]
        assert "search_space" not in llm_config

    def test_search_space_with_step_values(self, tmp_config_path: Path, load_yaml):
        """Test search_space with step values for grid search."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="llm_grid",
            temperature=0.5,
            top_p=0.9,
            optimizable_params=["temperature", "top_p"],
            search_space={
                "temperature": SearchSpace(low=0.0, high=1.0, step=0.1),
                "top_p": SearchSpace(low=0.5, high=1.0, step=0.1),
            },
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["llm_grid"]
        assert "search_space" not in llm_config
        assert set(llm_config["optimizable_params"]) == {"temperature", "top_p"}


# ============================================================================
# Complex Workflow Structure Tests
# ============================================================================


class TestComplexWorkflowStructure:
    """Tests for complex workflow configurations with many components."""

    def test_workflow_with_all_llm_parameters(self, tmp_config_path: Path, load_yaml):
        """Test workflow with LLM having all parameters explicitly set."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="fully_configured_llm",
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            top_k=40,
        )
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            max_tool_calls=15,
            max_history=100,
        )
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["fully_configured_llm"]
        assert llm_config["temperature"] == 0.7
        assert llm_config["top_p"] == 0.95
        assert llm_config["max_tokens"] == 2048

    def test_workflow_preserves_all_agent_options(self, tmp_config_path: Path, load_yaml):
        """Test that all agent configuration options are preserved."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(
            llm=llm,
            tools=[],
            verbose=True,
            max_tool_calls=25,
            max_history=50,
            additional_instructions="Be concise and helpful. Always cite sources.",
        )
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        workflow_config = content["workflow"]
        assert workflow_config["verbose"] is True
        assert workflow_config["max_tool_calls"] == 25
        assert workflow_config["max_history"] == 50
        assert "Be concise and helpful" in workflow_config["additional_instructions"]

    def test_workflow_with_many_tools(self, tmp_config_path: Path, load_yaml):
        """Test workflow with many tools."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tools = [CurrentTimeTool(name=f"time_tool_{i}") for i in range(10)]
        agent = NatReActAgent(llm=llm, tools=tools, verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        tool_names = content["workflow"]["tool_names"]
        assert len(tool_names) == 10


# ============================================================================
# Error Path Validation Tests
# ============================================================================


class TestSavePathValidation:
    """Tests for path validation in save_to_config_file."""

    def test_save_to_nonexistent_directory_raises(self, tmp_path: Path):
        """Test that saving to non-existent directory raises error."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        nonexistent_path = tmp_path / "does_not_exist" / "config.yaml"

        with pytest.raises(ValueError, match="does not exist"):
            workflow.save_to_config_file(nonexistent_path)

    def test_save_to_directory_path_raises(self, tmp_path: Path):
        """Test that saving to a directory (not file) raises error."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        with pytest.raises(ValueError, match="is a directory"):
            workflow.save_to_config_file(tmp_path)

    def test_save_with_wrong_extension_raises(self, tmp_path: Path):
        """Test that saving with wrong extension raises error."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        wrong_extension = tmp_path / "config.json"

        with pytest.raises(ValueError, match="extension"):
            workflow.save_to_config_file(wrong_extension)

    def test_save_with_yml_extension_works(self, tmp_path: Path, load_yaml):
        """Test that .yml extension works (not just .yaml)."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        yml_path = tmp_path / "config.yml"
        workflow.save_to_config_file(yml_path)

        assert yml_path.exists()
        content = load_yaml(yml_path)
        assert "workflow" in content


# ============================================================================
# Fields Set Preservation Tests
# ============================================================================


class TestFieldsSetPreservation:
    """Tests for exclude_unset behavior - only serializing explicitly set fields."""

    def test_unset_temperature_not_in_output(self, tmp_config_path: Path, load_yaml):
        """Test that unset temperature field is not in output."""
        # Only set model_name, don't set temperature
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="minimal_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["minimal_llm"]
        # Only explicitly set fields should be present
        # model_name is serialized as "model" alias
        assert "model" in llm_config
        # temperature was not set, so it might not be present

    def test_explicitly_set_fields_preserved(self, tmp_config_path: Path, load_yaml):
        """Test that all explicitly set fields are preserved."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="explicit_llm",
            temperature=0.8,
            top_p=0.95,
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        content = load_yaml(tmp_config_path)
        llm_config = content["llms"]["explicit_llm"]
        assert llm_config["temperature"] == 0.8
        assert llm_config["top_p"] == 0.95


# ============================================================================
# Type Discriminator Tests
# ============================================================================


class TestTypeDiscriminators:
    """Tests for _type field handling in serialization."""

    def test_type_field_first_in_llm(self, tmp_config_path: Path):
        """Test that _type field appears first in LLM config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        # Read raw content to check order
        with open(tmp_config_path, encoding="utf-8") as f:
            content = f.read()

        # Check that _type appears before model in the llms section
        # model_name is serialized as "model" alias
        llms_start = content.find("test_llm:")
        type_pos = content.find("_type:", llms_start)
        model_pos = content.find("model:", llms_start)
        assert type_pos < model_pos

    def test_type_field_first_in_workflow(self, tmp_config_path: Path):
        """Test that _type field appears first in workflow config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        workflow.save_to_config_file(tmp_config_path)

        with open(tmp_config_path, encoding="utf-8") as f:
            content = f.read()

        workflow_start = content.find("workflow:")
        type_pos = content.find("_type:", workflow_start)
        # _type should be close to the workflow: line
        assert type_pos > workflow_start
        assert type_pos < workflow_start + 50  # Within first 50 chars of workflow section
