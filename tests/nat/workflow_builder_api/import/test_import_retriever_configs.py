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
"""Tests for importing retriever configurations from YAML to workflow state.

These tests verify that retriever configurations are correctly parsed and all
field values are accurately preserved during import.
"""

from pathlib import Path

import yaml


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportRetrieversFromRAGConfig:
    """Tests for importing retrievers from RAG config.

    The RAG config contains Milvus retrievers with specific settings.
    """

    def test_retriever_components_are_created(self, rag_config: Path, set_test_env_vars):
        """Verify that retriever components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        retrievers = [c for c in workflow_state.components if c.component_type == "retriever"]
        original_count = len(config_dict.get("retrievers", {}))

        assert len(retrievers) == original_count, (
            f"Retriever count mismatch: expected {original_count}, got {len(retrievers)}")

    def test_retriever_ids_match_original_names(self, rag_config: Path, set_test_env_vars):
        """Verify retriever IDs match the key names from the original config."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_names = set(config_dict.get("retrievers", {}).keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "retriever"}

        assert original_names == imported_ids, (
            f"Retriever ID mismatch: original={original_names}, imported={imported_ids}")

    def test_retriever_type_is_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify retriever _type is preserved in full_type."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_retrievers = config_dict.get("retrievers", {})
        for name, orig_config in original_retrievers.items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Retriever '{name}' not found"

            imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Retriever type mismatch for '{name}': expected '{orig_type_short}', got '{imported_type_short}'")

    def test_retriever_uri_is_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify retriever URI field is exactly preserved."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_retrievers = config_dict.get("retrievers", {})
        for name, orig_config in original_retrievers.items():
            if "uri" not in orig_config:
                continue

            orig_uri = str(orig_config["uri"]).rstrip("/")
            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None

            imported_uri = str(imported.config.get("uri", "")).rstrip("/")
            assert imported_uri == orig_uri, (
                f"URI mismatch for retriever '{name}': expected '{orig_uri}', got '{imported_uri}'")

    def test_retriever_collection_name_is_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify retriever collection_name field is exactly preserved."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_retrievers = config_dict.get("retrievers", {})
        for name, orig_config in original_retrievers.items():
            if "collection_name" not in orig_config:
                continue

            orig_collection = orig_config["collection_name"]
            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None

            imported_collection = imported.config.get("collection_name")
            assert imported_collection == orig_collection, (
                f"collection_name mismatch for '{name}': expected '{orig_collection}', got '{imported_collection}'")

    def test_retriever_top_k_is_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify retriever top_k field is exactly preserved."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_retrievers = config_dict.get("retrievers", {})
        for name, orig_config in original_retrievers.items():
            if "top_k" not in orig_config:
                continue

            orig_top_k = orig_config["top_k"]
            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None

            imported_top_k = imported.config.get("top_k")
            assert imported_top_k == orig_top_k, (
                f"top_k mismatch for '{name}': expected {orig_top_k}, got {imported_top_k}")

    def test_retriever_embedder_connection_is_created(self, rag_config: Path, set_test_env_vars):
        """Verify that retriever's embedding_model creates a connection."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(rag_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_retrievers = config_dict.get("retrievers", {})
        for name, orig_config in original_retrievers.items():
            embedder_ref = orig_config.get("embedding_model")
            if embedder_ref is None:
                continue

            # Should have a connection from retriever to embedder
            embedder_connections = [
                c for c in workflow_state.connections if c.source_id == name and "embed" in c.target_field.lower()
            ]
            assert len(embedder_connections) >= 1, (
                f"Expected connection from retriever '{name}' to embedder '{embedder_ref}'")

            assert embedder_connections[0].target_id == embedder_ref, (
                f"Connection target mismatch: expected '{embedder_ref}', got '{embedder_connections[0].target_id}'")
