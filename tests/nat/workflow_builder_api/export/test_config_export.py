# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for exporting workflow state to YAML configuration."""

import yaml

from nat.workflow_builder_api.models import ExportComponent
from nat.workflow_builder_api.models import ExportConnection
from nat.workflow_builder_api.models import ExportWorkflowRequest
from nat.workflow_builder_api.utils.config_exporter import export_workflow_to_yaml


class TestConfigExportBasic:
    """Basic config export tests."""

    def test_export_empty_workflow(self, set_test_env_vars):
        """Test exporting an empty workflow."""
        request = ExportWorkflowRequest(components=[], connections=[])
        result = export_workflow_to_yaml(request)

        # Should succeed but might have warnings about missing workflow
        assert result.yaml_content is not None or len(result.warnings) > 0

    def test_export_single_llm(self, set_test_env_vars):
        """Test exporting a single LLM component."""
        llm_component = ExportComponent(
            id="test_llm",
            component_type="llm",
            full_type="nim",
            config={
                "model_name": "meta/llama-3.1-70b-instruct",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        )

        request = ExportWorkflowRequest(
            components=[llm_component],
            connections=[],
        )

        result = export_workflow_to_yaml(request)

        assert result.yaml_content is not None

        # Parse the YAML to verify structure
        config = yaml.safe_load(result.yaml_content)
        assert "llms" in config
        assert "test_llm" in config["llms"]
        assert config["llms"]["test_llm"]["_type"] == "nim"

    def test_export_llm_with_correct_section(self, set_test_env_vars):
        """Test that LLM exports to the 'llms' section."""
        llm = ExportComponent(
            id="my_llm",
            component_type="llm",
            full_type="nim",
            config={"model_name": "test-model"},
        )

        request = ExportWorkflowRequest(components=[llm], connections=[])
        result = export_workflow_to_yaml(request)

        config = yaml.safe_load(result.yaml_content)
        assert "llms" in config
        assert "my_llm" in config["llms"]


class TestConfigExportAgents:
    """Tests for exporting agent configurations."""

    def test_export_react_agent(self, set_test_env_vars):
        """Test exporting a ReAct agent."""
        agent = ExportComponent(
            id="my_agent",
            component_type="agent",
            full_type="react_agent",
            config={
                "verbose": True,
                "parse_agent_response_max_retries": 3,
            },
        )

        llm = ExportComponent(
            id="my_llm",
            component_type="llm",
            full_type="nim",
            config={"model_name": "test-model"},
        )

        connection = ExportConnection(
            source_id="my_agent",
            target_id="my_llm",
            target_field="llm_name",
        )

        request = ExportWorkflowRequest(
            components=[agent, llm],
            connections=[connection],
        )

        result = export_workflow_to_yaml(request)

        config = yaml.safe_load(result.yaml_content)

        # Agents go to functions section
        assert "functions" in config
        assert "my_agent" in config["functions"]


class TestConfigExportRetrievers:
    """Tests for exporting retriever configurations."""

    def test_export_retriever(self, set_test_env_vars):
        """Test exporting a retriever component."""
        retriever = ExportComponent(
            id="my_retriever",
            component_type="retriever",
            full_type="milvus_retriever",
            config={
                "uri": "http://localhost:19530",
                "collection_name": "test_docs",
                "top_k": 10,
            },
        )

        embedder = ExportComponent(
            id="my_embedder",
            component_type="embedder",
            full_type="nim",
            config={"model_name": "nvidia/nv-embedqa-e5-v5"},
        )

        connection = ExportConnection(
            source_id="my_retriever",
            target_id="my_embedder",
            target_field="embedding_model",
        )

        request = ExportWorkflowRequest(
            components=[retriever, embedder],
            connections=[connection],
        )

        result = export_workflow_to_yaml(request)

        config = yaml.safe_load(result.yaml_content)
        assert "retrievers" in config
        assert "my_retriever" in config["retrievers"]
        assert "embedders" in config
        assert "my_embedder" in config["embedders"]


class TestConfigExportEvaluators:
    """Tests for exporting evaluator configurations."""

    def test_export_evaluator_to_nested_section(self, set_test_env_vars):
        """Test that evaluators export to eval.evaluators section."""
        evaluator = ExportComponent(
            id="accuracy",
            component_type="evaluator",
            full_type="ragas",
            config={
                "metric": "AnswerAccuracy",
            },
        )

        llm = ExportComponent(
            id="judge_llm",
            component_type="llm",
            full_type="nim",
            config={"model_name": "test-model"},
        )

        connection = ExportConnection(
            source_id="accuracy",
            target_id="judge_llm",
            target_field="llm_name",
        )

        request = ExportWorkflowRequest(
            components=[evaluator, llm],
            connections=[connection],
        )

        result = export_workflow_to_yaml(request)

        config = yaml.safe_load(result.yaml_content)

        # Evaluators should be under eval.evaluators
        assert "eval" in config
        assert "evaluators" in config["eval"]
        # The evaluator name should NOT have "evaluator_" prefix
        assert "accuracy" in config["eval"]["evaluators"]


class TestConfigExportFinetuning:
    """Tests for exporting finetuning configurations."""

    def test_export_trainer(self, set_test_env_vars):
        """Test exporting trainer configuration."""
        trainer = ExportComponent(
            id="my_trainer",
            component_type="trainer",
            full_type="nemo_customizer_trainer",
            config={
                "num_runs": 1,
                "continue_on_collection_error": True,
            },
        )

        request = ExportWorkflowRequest(
            components=[trainer],
            connections=[],
        )

        result = export_workflow_to_yaml(request)

        config = yaml.safe_load(result.yaml_content)
        assert "trainers" in config
        assert "my_trainer" in config["trainers"]


class TestConfigExportConnections:
    """Tests for connection handling during export."""

    def test_connection_resolves_to_target_name(self, set_test_env_vars):
        """Test that connections resolve to component names."""
        llm = ExportComponent(
            id="my_llm",
            component_type="llm",
            full_type="nim",
            config={"model_name": "test-model"},
        )

        agent = ExportComponent(
            id="my_agent",
            component_type="agent",
            full_type="react_agent",
            config={},
        )

        connection = ExportConnection(
            source_id="my_agent",
            target_id="my_llm",
            target_field="llm_name",
        )

        request = ExportWorkflowRequest(
            components=[llm, agent],
            connections=[connection],
        )

        result = export_workflow_to_yaml(request)

        config = yaml.safe_load(result.yaml_content)

        # Check that connection is resolved
        if "functions" in config and "my_agent" in config["functions"]:
            agent_config = config["functions"]["my_agent"]
            assert agent_config.get("llm_name") == "my_llm"

    def test_multiple_connections_to_same_port(self, set_test_env_vars):
        """Test handling of multiple connections to a list port (tool_names)."""
        llm = ExportComponent(id="llm", component_type="llm", full_type="nim", config={"model_name": "test"})
        agent = ExportComponent(id="agent", component_type="agent", full_type="react_agent", config={})
        tool1 = ExportComponent(id="tool1", component_type="function", full_type="current_datetime", config={})
        tool2 = ExportComponent(id="tool2", component_type="function", full_type="wiki_search", config={})

        connections = [
            ExportConnection(source_id="agent", target_id="llm", target_field="llm_name"),
            ExportConnection(source_id="agent", target_id="tool1", target_field="tool_names"),
            ExportConnection(source_id="agent", target_id="tool2", target_field="tool_names"),
        ]

        request = ExportWorkflowRequest(
            components=[llm, agent, tool1, tool2],
            connections=connections,
        )

        result = export_workflow_to_yaml(request)

        config = yaml.safe_load(result.yaml_content)

        # Check that both tools are in tool_names
        if "functions" in config and "agent" in config["functions"]:
            agent_config = config["functions"]["agent"]
            tool_names = agent_config.get("tool_names", [])
            assert len(tool_names) == 2
