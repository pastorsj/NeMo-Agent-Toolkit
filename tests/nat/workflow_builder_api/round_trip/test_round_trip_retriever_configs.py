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
"""Round-trip tests for retriever and embedder configurations.

These tests verify that retriever/embedder configurations survive the import -> export
cycle with all field values preserved exactly as in the original config.
"""

from pathlib import Path

import yaml

from nat.workflow_builder_api.models import ExportComponent
from nat.workflow_builder_api.models import ExportConnection
from nat.workflow_builder_api.models import ExportWorkflowRequest
from nat.workflow_builder_api.utils.config_exporter import export_workflow_to_yaml
from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def do_round_trip(config_path: Path) -> tuple[dict, dict]:
    """Perform a full round-trip: load config -> import -> export."""
    original_config = load_config_dict(config_path)
    workflow_state = parse_config_to_workflow_state(original_config)

    components = [
        ExportComponent(id=c.id, component_type=c.component_type, full_type=c.full_type, config=c.config or {})
        for c in workflow_state.components
    ]
    connections = [
        ExportConnection(source_id=c.source_id, target_id=c.target_id, target_field=c.target_field)
        for c in workflow_state.connections
    ]

    export_request = ExportWorkflowRequest(components=components, connections=connections)
    export_result = export_workflow_to_yaml(export_request)

    assert export_result.success, f"Export failed: {export_result.error_message}"
    exported_config = yaml.safe_load(export_result.yaml_content)

    return original_config, exported_config


class TestRetrieverRoundTripPreservesValues:
    """Tests that verify retriever field values are preserved through round-trip."""

    def test_retrievers_section_exists_after_round_trip(self, rag_config: Path, set_test_env_vars):
        """Verify that retrievers section exists in exported config."""
        original, exported = do_round_trip(rag_config)

        assert "retrievers" in original, "Test config should have retrievers section"
        assert "retrievers" in exported, "Exported config should have retrievers section"

    def test_retriever_names_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that all retriever names are preserved through round-trip."""
        original, exported = do_round_trip(rag_config)

        original_names = set(original.get("retrievers", {}).keys())
        exported_names = set(exported.get("retrievers", {}).keys())

        assert original_names == exported_names, (
            f"Retriever names mismatch: original={original_names}, exported={exported_names}")

    def test_retriever_uri_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that retriever URI field is preserved."""
        original, exported = do_round_trip(rag_config)

        for name, orig_config in original.get("retrievers", {}).items():
            if "uri" not in orig_config:
                continue

            orig_uri = str(orig_config["uri"]).rstrip("/")
            exp_config = exported["retrievers"].get(name, {})
            exp_uri = str(exp_config.get("uri", "")).rstrip("/")

            assert exp_uri == orig_uri, (f"URI mismatch for retriever '{name}': expected '{orig_uri}', got '{exp_uri}'")

    def test_retriever_collection_name_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that retriever collection_name field is preserved."""
        original, exported = do_round_trip(rag_config)

        for name, orig_config in original.get("retrievers", {}).items():
            if "collection_name" not in orig_config:
                continue

            orig_collection = orig_config["collection_name"]
            exp_config = exported["retrievers"].get(name, {})
            exp_collection = exp_config.get("collection_name")

            assert exp_collection == orig_collection, (
                f"collection_name mismatch for '{name}': expected '{orig_collection}', got '{exp_collection}'")

    def test_retriever_top_k_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that retriever top_k field is preserved."""
        original, exported = do_round_trip(rag_config)

        for name, orig_config in original.get("retrievers", {}).items():
            if "top_k" not in orig_config:
                continue

            orig_top_k = orig_config["top_k"]
            exp_config = exported["retrievers"].get(name, {})
            exp_top_k = exp_config.get("top_k")

            assert exp_top_k == orig_top_k, (f"top_k mismatch for '{name}': expected {orig_top_k}, got {exp_top_k}")

    def test_retriever_embedding_model_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that retriever embedding_model reference is preserved."""
        original, exported = do_round_trip(rag_config)

        for name, orig_config in original.get("retrievers", {}).items():
            if "embedding_model" not in orig_config:
                continue

            orig_embedder = orig_config["embedding_model"]
            exp_config = exported["retrievers"].get(name, {})
            exp_embedder = exp_config.get("embedding_model")

            assert exp_embedder == orig_embedder, (
                f"embedding_model mismatch for '{name}': expected '{orig_embedder}', got '{exp_embedder}'")


class TestEmbedderRoundTripPreservesValues:
    """Tests that verify embedder field values are preserved through round-trip."""

    def test_embedders_section_exists_after_round_trip(self, rag_config: Path, set_test_env_vars):
        """Verify that embedders section exists in exported config."""
        original, exported = do_round_trip(rag_config)

        assert "embedders" in original, "Test config should have embedders section"
        assert "embedders" in exported, "Exported config should have embedders section"

    def test_embedder_names_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that all embedder names are preserved through round-trip."""
        original, exported = do_round_trip(rag_config)

        original_names = set(original.get("embedders", {}).keys())
        exported_names = set(exported.get("embedders", {}).keys())

        assert original_names == exported_names, (
            f"Embedder names mismatch: original={original_names}, exported={exported_names}")

    def test_embedder_model_name_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that embedder model_name field is preserved."""
        original, exported = do_round_trip(rag_config)

        for name, orig_config in original.get("embedders", {}).items():
            orig_model = orig_config.get("model_name") or orig_config.get("model")
            if orig_model is None:
                continue

            exp_config = exported["embedders"].get(name, {})
            exp_model = exp_config.get("model_name") or exp_config.get("model")

            assert exp_model == orig_model, (
                f"model_name mismatch for embedder '{name}': expected '{orig_model}', got '{exp_model}'")

    def test_embedder_type_preserved(self, rag_config: Path, set_test_env_vars):
        """Verify that embedder _type field is preserved (short name match)."""
        original, exported = do_round_trip(rag_config)

        for name, orig_config in original.get("embedders", {}).items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            exp_config = exported["embedders"].get(name, {})
            exp_type = exp_config.get("_type", "")
            exp_type_short = exp_type.split("/")[-1]

            assert exp_type_short == orig_type_short, (
                f"_type mismatch for embedder '{name}': expected '{orig_type_short}', got '{exp_type_short}'")
