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
"""SDK tests for evaluation configuration and serialization.

This module tests creating, configuring, and serializing evaluations using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.sdk import NatReActAgent
from nat.data_models.config import Config
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.dataset_handler import EvalDatasetStructureConfig
from nat.llm.sdk import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils.sdk.nat_evaluation import NatEvaluation
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
# NatEvaluation Initialization Tests
# ============================================================================


class TestNatEvaluationInitialization:
    """Tests for NatEvaluation initialization via SDK."""

    def test_basic_evaluation_creation(self):
        """Test creating a basic evaluation config."""
        structure = EvalDatasetStructureConfig(
            question_key="query",
            answer_key="response",
        )
        dataset = EvalDatasetJsonConfig(
            file_path="/path/to/data.json",
            structure=structure,
        )
        evaluation = NatEvaluation(dataset=dataset)

        assert evaluation.dataset is dataset

    def test_evaluation_with_output_dir(self):
        """Test evaluation with output directory."""
        dataset = EvalDatasetJsonConfig(file_path="/path/to/data.json")
        evaluation = NatEvaluation(
            dataset=dataset,
            output_dir=Path("./results"),
        )

        assert evaluation.output_dir == Path("./results")

    def test_evaluation_with_max_concurrency(self):
        """Test evaluation with max concurrency."""
        dataset = EvalDatasetJsonConfig(file_path="/path/to/data.json")
        evaluation = NatEvaluation(dataset=dataset, max_concurrency=4)

        assert evaluation.max_concurrency == 4


# ============================================================================
# EvalDatasetJsonConfig Tests
# ============================================================================


class TestEvalDatasetJsonConfig:
    """Tests for EvalDatasetJsonConfig initialization."""

    def test_basic_json_dataset_creation(self):
        """Test creating a basic JSON dataset config."""
        structure = EvalDatasetStructureConfig(
            question_key="query",
            answer_key="response",
        )
        dataset = EvalDatasetJsonConfig(
            file_path="/path/to/data.json",
            structure=structure,
        )

        assert str(dataset.file_path) == "/path/to/data.json"
        assert dataset.structure.question_key == "query"
        assert dataset.structure.answer_key == "response"

    def test_json_dataset_with_defaults(self):
        """Test JSON dataset with default structure."""
        dataset = EvalDatasetJsonConfig(file_path="/path/to/data.json")

        assert str(dataset.file_path) == "/path/to/data.json"
        assert dataset.structure.question_key == "question"
        assert dataset.structure.answer_key == "answer"


# ============================================================================
# Evaluation Serialization Tests
# ============================================================================


class TestEvaluationSerialization:
    """Tests for serializing evaluation config to YAML."""

    def test_evaluation_saves_in_eval_section(self, tmp_path: Path):
        """Test that evaluation config saves in eval section."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        dataset = EvalDatasetJsonConfig(file_path="/path/to/data.json")
        evaluation = NatEvaluation(dataset=dataset)

        workflow = NatWorkflow(entrypoint=agent, evaluator=evaluation)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "eval" in content

    def test_evaluation_dataset_serializes(self, tmp_path: Path):
        """Test that dataset is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        dataset = EvalDatasetJsonConfig(file_path="/path/to/data.json")
        evaluation = NatEvaluation(dataset=dataset)

        workflow = NatWorkflow(entrypoint=agent, evaluator=evaluation)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        # Dataset is under eval.general.dataset
        assert "dataset" in content["eval"]["general"]


# ============================================================================
# Evaluation Config Round-Trip Tests
# ============================================================================


class TestEvaluationRoundTrip:
    """Tests for round-trip serialization of evaluations."""

    def test_evaluation_round_trip(self, tmp_path: Path):
        """Test that config can be loaded back correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        dataset = EvalDatasetJsonConfig(file_path="/path/to/data.json")
        evaluation = NatEvaluation(dataset=dataset, max_concurrency=4)

        workflow = NatWorkflow(entrypoint=agent, evaluator=evaluation)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.eval is not None


# ============================================================================
# Evaluation Edge Cases
# ============================================================================


class TestEvaluationEdgeCases:
    """Edge case tests for evaluation SDK usage."""

    def test_workflow_without_evaluation(self, tmp_path: Path):
        """Test workflow without evaluation config."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Should save without errors
        assert config_path.exists()
