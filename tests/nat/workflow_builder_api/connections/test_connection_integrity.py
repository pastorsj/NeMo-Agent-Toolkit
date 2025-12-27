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
"""Tests for connection data integrity.

These tests verify that connections maintain referential integrity:
- Source and target components exist
- Connections have valid port names
- No orphan or dangling connections
"""

from pathlib import Path

import yaml

from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestConnectionReferentialIntegrity:
    """Tests for referential integrity of connections."""

    def test_all_connection_sources_exist(self, react_agent_config: Path, set_test_env_vars):
        """Verify that all connection source IDs refer to existing components."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        component_ids = {c.id for c in workflow_state.components}

        for conn in workflow_state.connections:
            assert conn.source_id in component_ids, (
                f"Connection source '{conn.source_id}' does not exist in components. "
                f"Available: {component_ids}")

    def test_all_connection_targets_exist(self, react_agent_config: Path, set_test_env_vars):
        """Verify that all connection target IDs refer to existing components."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        component_ids = {c.id for c in workflow_state.components}

        for conn in workflow_state.connections:
            assert conn.target_id in component_ids, (
                f"Connection target '{conn.target_id}' does not exist in components. "
                f"Available: {component_ids}")

    def test_all_connections_have_non_empty_target_field(self, react_agent_config: Path, set_test_env_vars):
        """Verify that all connections have a non-empty target_field."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        for conn in workflow_state.connections:
            assert conn.target_field, (
                f"Connection from '{conn.source_id}' to '{conn.target_id}' has empty target_field")
            assert len(conn.target_field) > 0, (
                f"Connection from '{conn.source_id}' to '{conn.target_id}' has empty target_field")


class TestConnectionIntegrityAcrossConfigs:
    """Tests for connection integrity across different config types."""

    def test_rag_config_connections_are_valid(self, rag_config: Path, set_test_env_vars):
        """Verify all connections in RAG config are valid."""
        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        component_ids = {c.id for c in workflow_state.components}

        for conn in workflow_state.connections:
            assert conn.source_id in component_ids, f"Invalid source: {conn.source_id}"
            assert conn.target_id in component_ids, f"Invalid target: {conn.target_id}"
            assert conn.target_field, "Empty target_field for connection"

    def test_memory_config_connections_are_valid(self, memory_config: Path, set_test_env_vars):
        """Verify all connections in memory config are valid."""
        config_dict = load_config_dict(memory_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        component_ids = {c.id for c in workflow_state.components}

        for conn in workflow_state.connections:
            assert conn.source_id in component_ids, f"Invalid source: {conn.source_id}"
            assert conn.target_id in component_ids, f"Invalid target: {conn.target_id}"

    def test_auth_config_connections_are_valid(self, auth_config: Path, set_test_env_vars):
        """Verify all connections in auth config are valid."""
        config_dict = load_config_dict(auth_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        component_ids = {c.id for c in workflow_state.components}

        for conn in workflow_state.connections:
            assert conn.source_id in component_ids, f"Invalid source: {conn.source_id}"
            assert conn.target_id in component_ids, f"Invalid target: {conn.target_id}"


class TestConnectionTypeConsistency:
    """Tests for type consistency of connections."""

    def test_llm_connections_target_llm_components(self, react_agent_config: Path, set_test_env_vars):
        """Verify that llm_name connections always target LLM components."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        llm_ids = {c.id for c in workflow_state.components if c.component_type == "llm"}

        for conn in workflow_state.connections:
            if "llm" in conn.target_field.lower():
                assert conn.target_id in llm_ids, (f"LLM connection target '{conn.target_id}' is not an LLM. "
                                                   f"LLMs: {llm_ids}")

    def test_embedder_connections_target_embedder_components(self, rag_config: Path, set_test_env_vars):
        """Verify that embedding_model connections always target embedder components."""
        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        embedder_ids = {c.id for c in workflow_state.components if c.component_type == "embedder"}

        for conn in workflow_state.connections:
            if "embed" in conn.target_field.lower():
                assert conn.target_id in embedder_ids, (
                    f"Embedder connection target '{conn.target_id}' is not an embedder. "
                    f"Embedders: {embedder_ids}")

    def test_memory_connections_target_memory_components(self, memory_config: Path, set_test_env_vars):
        """Verify that memory connections always target memory components."""
        config_dict = load_config_dict(memory_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        memory_ids = {c.id for c in workflow_state.components if c.component_type == "memory"}

        for conn in workflow_state.connections:
            if "memory" in conn.target_field.lower():
                assert conn.target_id in memory_ids, (
                    f"Memory connection target '{conn.target_id}' is not a memory component. "
                    f"Memory components: {memory_ids}"
                )
