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
"""Tests for exporting connections to YAML configuration.

These tests verify that connection objects are correctly exported as field
references (llm_name, tool_names, etc.) in the output YAML.
"""

import yaml

from nat.workflow_builder_api.models import ExportComponent
from nat.workflow_builder_api.models import ExportConnection
from nat.workflow_builder_api.models import ExportWorkflowRequest
from nat.workflow_builder_api.utils.config_exporter import export_workflow_to_yaml


class TestExportSingleReferenceConnections:
    """Tests for exporting single-reference connections (e.g., llm_name)."""

    def test_llm_connection_exports_as_llm_name(self, set_test_env_vars):
        """Verify that LLM connection is exported as llm_name field."""
        llm = ExportComponent(
            id="my_llm",
            component_type="llm",
            full_type="nim",
            config={"model_name": "meta/llama-3.1-8b-instruct"},
        )

        agent = ExportComponent(
            id="my_agent",
            component_type="agent",
            full_type="nat.agent.react_agent/react_agent",
            config={},
        )

        connection = ExportConnection(
            source_id="my_agent",
            target_id="my_llm",
            target_field="llm_name",
        )

        request = ExportWorkflowRequest(components=[llm, agent], connections=[connection])
        result = export_workflow_to_yaml(request)

        assert result.success, f"Export failed: {result.error_message}"
        config = yaml.safe_load(result.yaml_content)

        # Verify LLM is in llms section
        assert "llms" in config, "Expected 'llms' section in exported config"
        assert "my_llm" in config["llms"], "Expected 'my_llm' in llms section"

        # Verify agent has llm_name field set to LLM ID
        assert "functions" in config, "Expected 'functions' section in exported config"
        assert "my_agent" in config["functions"], "Expected 'my_agent' in functions section"
        assert config["functions"]["my_agent"].get("llm_name") == "my_llm", (
            f"Expected llm_name='my_llm', got '{config['functions']['my_agent'].get('llm_name')}'")

    def test_embedder_connection_exports_as_embedding_model(self, set_test_env_vars):
        """Verify that embedder connection is exported as embedding_model field."""
        embedder = ExportComponent(
            id="my_embedder",
            component_type="embedder",
            full_type="nim",
            config={"model_name": "nvidia/nv-embedqa-e5-v5"},
        )

        retriever = ExportComponent(
            id="my_retriever",
            component_type="retriever",
            full_type="milvus",
            config={
                "uri": "http://localhost:19530", "collection_name": "test_collection"
            },
        )

        connection = ExportConnection(
            source_id="my_retriever",
            target_id="my_embedder",
            target_field="embedding_model",
        )

        request = ExportWorkflowRequest(components=[embedder, retriever], connections=[connection])
        result = export_workflow_to_yaml(request)

        assert result.success, f"Export failed: {result.error_message}"
        config = yaml.safe_load(result.yaml_content)

        # Verify retriever has embedding_model field set to embedder ID
        assert "retrievers" in config, "Expected 'retrievers' section in exported config"
        assert "my_retriever" in config["retrievers"], "Expected 'my_retriever' in retrievers section"
        actual = config["retrievers"]["my_retriever"].get("embedding_model")
        assert actual == "my_embedder", f"Expected embedding_model='my_embedder', got '{actual}'"


class TestExportListReferenceConnections:
    """Tests for exporting list-reference connections (e.g., tool_names)."""

    def test_tool_connections_export_as_tool_names_list(self, set_test_env_vars):
        """Verify that multiple tool connections are exported as tool_names list."""
        llm = ExportComponent(
            id="llm",
            component_type="llm",
            full_type="nim",
            config={"model_name": "meta/llama-3.1-8b-instruct"},
        )

        agent = ExportComponent(
            id="my_agent",
            component_type="agent",
            full_type="nat.agent.react_agent/react_agent",
            config={},
        )

        tool_a = ExportComponent(
            id="datetime_tool",
            component_type="function",
            full_type="nat.tool/current_datetime",
            config={},
        )

        tool_b = ExportComponent(
            id="search_tool",
            component_type="function",
            full_type="nat.plugins.langchain.tools/wiki_search",
            config={},
        )

        connections = [
            ExportConnection(source_id="my_agent", target_id="llm", target_field="llm_name"),
            ExportConnection(source_id="my_agent", target_id="datetime_tool", target_field="tool_names"),
            ExportConnection(source_id="my_agent", target_id="search_tool", target_field="tool_names"),
        ]

        request = ExportWorkflowRequest(components=[llm, agent, tool_a, tool_b], connections=connections)
        result = export_workflow_to_yaml(request)

        assert result.success, f"Export failed: {result.error_message}"
        config = yaml.safe_load(result.yaml_content)

        # Verify agent has tool_names list with both tools
        assert "functions" in config, "Expected 'functions' section"
        agent_config = config["functions"].get("my_agent", {})
        tool_names = agent_config.get("tool_names", [])

        assert isinstance(tool_names, list), f"Expected tool_names to be a list, got {type(tool_names)}"
        assert "datetime_tool" in tool_names, "Expected 'datetime_tool' in tool_names"
        assert "search_tool" in tool_names, "Expected 'search_tool' in tool_names"

    def test_tool_names_list_preserves_all_connections(self, set_test_env_vars):
        """Verify that all tool connections are preserved in the exported list."""
        llm = ExportComponent(id="llm", component_type="llm", full_type="nim", config={"model_name": "test"})
        agent = ExportComponent(id="agent",
                                component_type="agent",
                                full_type="nat.agent.react_agent/react_agent",
                                config={})

        # Create 5 tools
        tools = []
        connections = [ExportConnection(source_id="agent", target_id="llm", target_field="llm_name")]
        for i in range(5):
            tool = ExportComponent(
                id=f"tool_{i}",
                component_type="function",
                full_type="nat.tool/current_datetime",
                config={},
            )
            tools.append(tool)
            connections.append(ExportConnection(source_id="agent", target_id=f"tool_{i}", target_field="tool_names"))

        request = ExportWorkflowRequest(components=[llm, agent, *tools], connections=connections)
        result = export_workflow_to_yaml(request)

        assert result.success, f"Export failed: {result.error_message}"
        config = yaml.safe_load(result.yaml_content)

        agent_config = config["functions"].get("agent", {})
        tool_names = agent_config.get("tool_names", [])

        assert len(tool_names) == 5, f"Expected 5 tools in tool_names, got {len(tool_names)}"
        for i in range(5):
            assert f"tool_{i}" in tool_names, f"Expected 'tool_{i}' in tool_names"


class TestExportConnectionPreservesValues:
    """Tests for verifying connection export preserves component field values."""

    def test_llm_model_name_preserved_with_connection(self, set_test_env_vars):
        """Verify that LLM model_name is preserved when exported with connections."""
        model_name = "meta/llama-3.1-405b-instruct"
        llm = ExportComponent(
            id="big_llm",
            component_type="llm",
            full_type="nim",
            config={
                "model_name": model_name, "temperature": 0.7, "max_tokens": 2048
            },
        )

        agent = ExportComponent(
            id="agent",
            component_type="agent",
            full_type="nat.agent.react_agent/react_agent",
            config={},
        )

        connection = ExportConnection(source_id="agent", target_id="big_llm", target_field="llm_name")
        request = ExportWorkflowRequest(components=[llm, agent], connections=[connection])
        result = export_workflow_to_yaml(request)

        assert result.success
        config = yaml.safe_load(result.yaml_content)

        # Verify LLM config is preserved
        llm_config = config["llms"]["big_llm"]
        assert llm_config.get("model_name") == model_name, (
            f"model_name mismatch: expected '{model_name}', got '{llm_config.get('model_name')}'")
        assert llm_config.get("temperature") == 0.7, (
            f"temperature mismatch: expected 0.7, got {llm_config.get('temperature')}")
        assert llm_config.get("max_tokens") == 2048, (
            f"max_tokens mismatch: expected 2048, got {llm_config.get('max_tokens')}")
