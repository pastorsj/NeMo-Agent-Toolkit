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
"""Tests for parsing connections from YAML configurations.

These tests verify that component references (like llm_name, embedding_model)
are correctly parsed into connection objects with accurate source, target,
and port information.
"""

from pathlib import Path

import yaml

from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestParseLLMConnections:
    """Tests for parsing LLM connections from llm_name fields."""

    def test_agent_llm_connection_is_parsed(self, react_agent_config: Path, set_test_env_vars):
        """Verify that agent's llm_name field creates a connection."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        # Find all LLM connections
        llm_connections = [c for c in workflow_state.connections if "llm" in c.target_field.lower()]
        assert len(llm_connections) >= 1, "Expected at least one LLM connection"

    def test_llm_connection_source_is_agent(self, react_agent_config: Path, set_test_env_vars):
        """Verify that LLM connection source is a valid agent/function component."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        component_ids = {c.id for c in workflow_state.components}
        llm_connections = [c for c in workflow_state.connections if "llm" in c.target_field.lower()]

        for conn in llm_connections:
            assert conn.source_id in component_ids, (f"Connection source '{conn.source_id}' not found in components")

    def test_llm_connection_target_is_llm(self, react_agent_config: Path, set_test_env_vars):
        """Verify that LLM connection target is a valid LLM component."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        llm_ids = {c.id for c in workflow_state.components if c.component_type == "llm"}
        llm_connections = [c for c in workflow_state.connections if "llm" in c.target_field.lower()]

        for conn in llm_connections:
            assert conn.target_id in llm_ids, (f"Connection target '{conn.target_id}' is not an LLM component")

    def test_llm_connection_matches_config_reference(self, react_agent_config: Path, set_test_env_vars):
        """Verify that connection target matches the llm_name value in config."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_functions = config_dict.get("functions", {})
        for name, func_config in original_functions.items():
            llm_ref = func_config.get("llm_name") or func_config.get("llm")
            if llm_ref is None:
                continue

            # Find the connection from this function to LLM
            matching_conns = [
                c for c in workflow_state.connections if c.source_id == name and "llm" in c.target_field.lower()
            ]
            assert len(matching_conns) >= 1, (f"Expected connection from '{name}' to LLM '{llm_ref}'")
            assert matching_conns[0].target_id == llm_ref, (
                f"Connection target mismatch: expected '{llm_ref}', got '{matching_conns[0].target_id}'")


class TestParseEmbedderConnections:
    """Tests for parsing embedder connections from embedding_model fields."""

    def test_retriever_embedder_connection_is_parsed(self, rag_config: Path, set_test_env_vars):
        """Verify that retriever's embedding_model field creates a connection."""
        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        embedder_connections = [c for c in workflow_state.connections if "embed" in c.target_field.lower()]
        assert len(embedder_connections) >= 1, "Expected at least one embedder connection"

    def test_embedder_connection_target_is_embedder(self, rag_config: Path, set_test_env_vars):
        """Verify that embedder connection target is a valid embedder component."""
        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        embedder_ids = {c.id for c in workflow_state.components if c.component_type == "embedder"}
        embedder_connections = [c for c in workflow_state.connections if "embed" in c.target_field.lower()]

        for conn in embedder_connections:
            assert conn.target_id in embedder_ids, (
                f"Connection target '{conn.target_id}' is not an embedder component")

    def test_embedder_connection_matches_config_reference(self, rag_config: Path, set_test_env_vars):
        """Verify that connection target matches the embedding_model value in config."""
        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_retrievers = config_dict.get("retrievers", {})
        for name, ret_config in original_retrievers.items():
            embedder_ref = ret_config.get("embedding_model")
            if embedder_ref is None:
                continue

            matching_conns = [
                c for c in workflow_state.connections if c.source_id == name and "embed" in c.target_field.lower()
            ]
            assert len(matching_conns) >= 1, (f"Expected connection from '{name}' to embedder '{embedder_ref}'")
            assert matching_conns[0].target_id == embedder_ref, (
                f"Connection target mismatch: expected '{embedder_ref}', got '{matching_conns[0].target_id}'")


class TestParseMemoryConnections:
    """Tests for parsing memory connections."""

    def test_memory_connection_is_parsed(self, memory_config: Path, set_test_env_vars):
        """Verify that memory references create connections."""
        config_dict = load_config_dict(memory_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        memory_connections = [c for c in workflow_state.connections if "memory" in c.target_field.lower()]
        assert len(memory_connections) >= 1, "Expected at least one memory connection"

    def test_memory_connection_target_is_memory(self, memory_config: Path, set_test_env_vars):
        """Verify that memory connection target is a valid memory component."""
        config_dict = load_config_dict(memory_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        memory_ids = {c.id for c in workflow_state.components if c.component_type == "memory"}
        memory_connections = [c for c in workflow_state.connections if "memory" in c.target_field.lower()]

        for conn in memory_connections:
            assert conn.target_id in memory_ids, (f"Connection target '{conn.target_id}' is not a memory component")


class TestParseAuthConnections:
    """Tests for parsing authentication connections."""

    def test_auth_connection_is_parsed(self, auth_config: Path, set_test_env_vars):
        """Verify that authentication references create connections."""
        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        auth_connections = [c for c in workflow_state.connections if "auth" in c.target_field.lower()]
        assert len(auth_connections) >= 1, "Expected at least one auth connection"

    def test_auth_connection_target_is_auth(self, auth_config: Path, set_test_env_vars):
        """Verify that auth connection target is a valid authentication component."""
        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        auth_ids = {c.id for c in workflow_state.components if c.component_type == "authentication"}
        auth_connections = [c for c in workflow_state.connections if "auth" in c.target_field.lower()]

        for conn in auth_connections:
            assert conn.target_id in auth_ids, (
                f"Connection target '{conn.target_id}' is not an authentication component")
