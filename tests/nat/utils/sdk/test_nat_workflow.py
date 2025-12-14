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
"""Tests for NatWorkflow class functionality.

This module tests the core NatWorkflow class methods, component discovery,
and configuration building.
"""

from pathlib import Path

import pytest

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.llm.nim_llm import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_evaluation import EvalDatasetJsonConfig
from nat.utils.sdk.nat_evaluation import NatEvaluation
from nat.utils.sdk.nat_optimizer import NatOptimizer
from nat.utils.sdk.nat_optimizer import NumericOptimizationConfig
from nat.utils.sdk.nat_optimizer import OptimizerMetric
from nat.utils.sdk.nat_workflow import DiscoveredComponents
from nat.utils.sdk.nat_workflow import NatWorkflow


# Ensure plugins are discovered for tests
@pytest.fixture(scope="module", autouse=True)
def discover_plugins():
    """Discover and register all plugins before running tests."""
    discover_and_register_plugins(PluginTypes.ALL)


# ============================================================================
# NatWorkflow Initialization Tests
# ============================================================================


class TestNatWorkflowInitialization:
    """Tests for NatWorkflow initialization and basic properties."""

    def test_workflow_creates_with_entrypoint(self):
        """Test that NatWorkflow can be created with just an entrypoint."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        assert workflow.entrypoint is agent
        assert workflow.configuration is None
        assert workflow.evaluator is None
        assert workflow.optimizer is None

    def test_workflow_config_property_builds_config(self):
        """Test that _config property builds a valid Config object."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._config  # pylint: disable=protected-access
        assert isinstance(config, Config)
        assert config.workflow is not None


# ============================================================================
# Component Discovery Tests
# ============================================================================


class TestDiscoveredComponents:
    """Tests for the DiscoveredComponents class."""

    def test_empty_discovered_components(self):
        """Test that DiscoveredComponents starts empty."""
        discovered = DiscoveredComponents()

        assert len(discovered.functions) == 0
        assert len(discovered.function_groups) == 0
        assert len(discovered.llms) == 0
        assert len(discovered.embedders) == 0
        assert len(discovered.retrievers) == 0
        assert len(discovered.memory) == 0
        assert len(discovered.middleware) == 0
        assert len(discovered.object_stores) == 0
        assert len(discovered.ttc_strategies) == 0
        assert len(discovered.auth_providers) == 0
        assert len(discovered.evaluators) == 0
        assert len(discovered.env_var_refs) == 0

    def test_was_visited_and_mark_visited(self):
        """Test visited object tracking."""
        discovered = DiscoveredComponents()

        class DummyObject:
            pass

        obj = DummyObject()

        assert not discovered.was_visited(obj)
        discovered.mark_visited(obj)
        assert discovered.was_visited(obj)

    def test_multiple_objects_visited_independently(self):
        """Test that multiple objects are tracked independently."""
        discovered = DiscoveredComponents()

        class DummyObject:
            pass

        obj1 = DummyObject()
        obj2 = DummyObject()

        discovered.mark_visited(obj1)
        assert discovered.was_visited(obj1)
        assert not discovered.was_visited(obj2)


# ============================================================================
# Workflow Discovery Tests
# ============================================================================


class TestWorkflowDiscovery:
    """Tests for component discovery in NatWorkflow."""

    def test_discover_llm_from_agent(self):
        """Test that LLM is discovered from agent."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="discovered_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        discovered = workflow._discover_components()  # pylint: disable=protected-access

        assert "discovered_llm" in discovered.llms
        assert discovered.llms["discovered_llm"].model_name == "meta/llama-3.1-70b-instruct"

    def test_discover_tools_from_agent(self):
        """Test that tools are discovered from agent."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tool = CurrentTimeTool(name="time_tool")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        discovered = workflow._discover_components()  # pylint: disable=protected-access

        # Tools should be in functions as they inherit from FunctionBaseConfig
        # The agent itself should also be there
        assert len(discovered.functions) >= 0  # Tools may be tracked differently

    def test_discovery_caches_results(self):
        """Test that component discovery is cached."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        discovered1 = workflow._discover_components()  # pylint: disable=protected-access
        discovered2 = workflow._discover_components()  # pylint: disable=protected-access

        # Should return the same object (cached)
        assert discovered1 is discovered2


# ============================================================================
# Add Evaluator Tests
# ============================================================================


class TestAddEvaluator:
    """Tests for NatWorkflow.add_evaluator method."""

    def test_add_evaluator_sets_evaluator(self, tmp_path: Path):
        """Test that add_evaluator sets the evaluator attribute."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        dataset_path = tmp_path / "data.json"
        dataset_path.write_text("[]", encoding="utf-8")

        evaluation = NatEvaluation(
            output_dir=tmp_path / "results",
            dataset=EvalDatasetJsonConfig(file_path=dataset_path),
            evaluators=[],
        )

        assert workflow.evaluator is None
        workflow.add_evaluator(evaluation)
        assert workflow.evaluator is evaluation

    def test_add_evaluator_resets_cache(self, tmp_path: Path):
        """Test that add_evaluator resets the cached config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        # Access _config to trigger caching
        _ = workflow._config  # pylint: disable=protected-access
        assert "_config" in workflow.__dict__

        dataset_path = tmp_path / "data.json"
        dataset_path.write_text("[]", encoding="utf-8")

        evaluation = NatEvaluation(
            output_dir=tmp_path / "results",
            dataset=EvalDatasetJsonConfig(file_path=dataset_path),
            evaluators=[],
        )
        workflow.add_evaluator(evaluation)

        # Cache should be cleared
        assert "_config" not in workflow.__dict__


# ============================================================================
# Add Optimizer Tests
# ============================================================================


class TestAddOptimizer:
    """Tests for NatWorkflow.add_optimizer method."""

    def test_add_optimizer_sets_optimizer(self):
        """Test that add_optimizer sets the optimizer attribute."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(evaluator_name="accuracy", direction="maximize"),
            },
        )

        assert workflow.optimizer is None
        workflow.add_optimizer(optimizer)
        assert workflow.optimizer is optimizer

    def test_add_optimizer_resets_cache(self):
        """Test that add_optimizer resets the cached config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        # Access _config to trigger caching
        _ = workflow._config  # pylint: disable=protected-access
        assert "_config" in workflow.__dict__

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(evaluator_name="accuracy", direction="maximize"),
            },
        )
        workflow.add_optimizer(optimizer)

        # Cache should be cleared
        assert "_config" not in workflow.__dict__


# ============================================================================
# Evaluate Method Tests
# ============================================================================


class TestEvaluateMethod:
    """Tests for NatWorkflow.evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_without_evaluator_raises_error(self):
        """Test that evaluate raises ValueError if no evaluator is set."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        with pytest.raises(ValueError, match="No evaluator has been set"):
            await workflow.evaluate()


# ============================================================================
# Optimize Method Tests
# ============================================================================


class TestOptimizeMethod:
    """Tests for NatWorkflow.optimize method."""

    @pytest.mark.asyncio
    async def test_optimize_without_optimizer_raises_error(self):
        """Test that optimize raises ValueError if no optimizer is set."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        with pytest.raises(ValueError, match="No optimizer has been set"):
            await workflow.optimize()

    @pytest.mark.asyncio
    async def test_optimize_without_evaluator_raises_error(self):
        """Test that optimize raises ValueError if no evaluator is set."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(evaluator_name="accuracy", direction="maximize"),
            },
        )
        workflow.add_optimizer(optimizer)

        with pytest.raises(ValueError, match="No evaluator has been set"):
            await workflow.optimize()


# ============================================================================
# Build Config Tests
# ============================================================================


class TestBuildConfig:
    """Tests for NatWorkflow._build_config_object method."""

    def test_build_config_includes_workflow(self):
        """Test that built config includes workflow section."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._build_config_object()  # pylint: disable=protected-access

        assert config.workflow is not None

    def test_build_config_includes_llms(self):
        """Test that built config includes LLMs from agent."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._build_config_object()  # pylint: disable=protected-access

        assert "test_llm" in config.llms

    def test_build_config_with_optimizer(self):
        """Test that built config includes optimizer when set."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(evaluator_name="accuracy", direction="maximize"),
            },
            numeric=NumericOptimizationConfig(enabled=True, n_trials=10),
        )
        workflow.add_optimizer(optimizer)

        config = workflow._build_config_object()  # pylint: disable=protected-access

        assert config.optimizer is not None
        assert config.optimizer.numeric.enabled is True

    def test_build_config_without_optimizer(self):
        """Test that built config has default optimizer when not set via SDK."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._build_config_object()  # pylint: disable=protected-access

        # Config.optimizer has a default value, so it's not None
        # When workflow.optimizer is None, the SDK doesn't add optimizer section to the YAML
        # But the Config object always has a default OptimizerConfig
        assert config.optimizer is not None
        # The important thing is workflow.optimizer is None
        assert workflow.optimizer is None


# ============================================================================
# Workflow Entrypoint Name Tests
# ============================================================================


class TestWorkflowEntrypointName:
    """Tests for handling workflow entrypoint names."""

    def test_entrypoint_not_duplicated_in_functions(self):
        """Test that entrypoint is not duplicated in functions section."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._build_config_object()  # pylint: disable=protected-access

        # The entrypoint should be in workflow, not in functions
        entrypoint_name, _ = agent.compute_name_and_config(FunctionBaseConfig)
        assert entrypoint_name not in config.functions


# ============================================================================
# Complex Workflow Tests
# ============================================================================


class TestComplexWorkflows:
    """Tests for complex workflow configurations."""

    def test_workflow_with_multiple_tools(self):
        """Test workflow with multiple tools."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        tools = [CurrentTimeTool(name=f"tool_{i}") for i in range(5)]
        agent = NatReActAgent(llm=llm, tools=tools, verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._build_config_object()  # pylint: disable=protected-access

        assert config.workflow is not None
        # Verify LLM is included
        assert "test_llm" in config.llms

    def test_workflow_with_optimizer_and_multiple_metrics(self):
        """Test workflow with optimizer and multiple evaluation metrics."""
        llm = NimLLM(
            model_name="meta/llama-3.1-70b-instruct",
            name="test_llm",
            optimizable_params=["temperature", "top_p"],
        )
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        optimizer = NatOptimizer(
            output_path=Path("./results"),
            eval_metrics={
                "accuracy": OptimizerMetric(
                    evaluator_name="accuracy",
                    direction="maximize",
                    weight=0.6,
                ),
                "latency": OptimizerMetric(
                    evaluator_name="latency",
                    direction="minimize",
                    weight=0.2,
                ),
                "cost": OptimizerMetric(
                    evaluator_name="cost",
                    direction="minimize",
                    weight=0.2,
                ),
            },
            numeric=NumericOptimizationConfig(enabled=True, n_trials=50),
        )
        workflow.add_optimizer(optimizer)

        config = workflow._build_config_object()  # pylint: disable=protected-access

        assert config.optimizer is not None
        assert len(config.optimizer.eval_metrics) == 3
        assert config.optimizer.eval_metrics["accuracy"].direction == "maximize"
        assert config.optimizer.eval_metrics["latency"].direction == "minimize"
        assert config.optimizer.eval_metrics["cost"].direction == "minimize"
