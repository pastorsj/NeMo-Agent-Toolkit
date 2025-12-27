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
"""Round-trip tests for LLM configurations.

These tests verify that LLM configurations survive the import -> export cycle
with all field values preserved exactly as in the original config.
"""

from pathlib import Path
from typing import Any

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
    """
    Perform a full round-trip: load config -> import -> export.

    Returns (original_config, exported_config).
    """
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


def normalize_value(value: Any) -> Any:
    """Normalize a value for comparison (handles int/float equivalence)."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return value


class TestLLMRoundTripPreservesValues:
    """Tests that verify LLM field values are preserved through round-trip."""

    def test_llm_section_exists_after_round_trip(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that llms section exists in exported config."""
        original, exported = do_round_trip(simple_calculator_config)

        assert "llms" in original, "Test config should have llms section"
        assert "llms" in exported, "Exported config should have llms section"

    def test_llm_names_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that all LLM names are preserved through round-trip."""
        original, exported = do_round_trip(simple_calculator_config)

        original_names = set(original.get("llms", {}).keys())
        exported_names = set(exported.get("llms", {}).keys())

        assert original_names == exported_names, (
            f"LLM names mismatch: original={original_names}, exported={exported_names}")

    def test_llm_model_name_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that LLM model_name field is exactly preserved."""
        original, exported = do_round_trip(simple_calculator_config)

        for name, orig_config in original.get("llms", {}).items():
            orig_model = orig_config.get("model_name") or orig_config.get("model")
            if orig_model is None:
                continue

            exp_config = exported["llms"].get(name, {})
            exp_model = exp_config.get("model_name") or exp_config.get("model")

            assert exp_model == orig_model, (
                f"model_name mismatch for LLM '{name}': expected '{orig_model}', got '{exp_model}'")

    def test_llm_temperature_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that LLM temperature field is exactly preserved."""
        original, exported = do_round_trip(simple_calculator_config)

        for name, orig_config in original.get("llms", {}).items():
            if "temperature" not in orig_config:
                continue

            orig_temp = normalize_value(orig_config["temperature"])
            exp_config = exported["llms"].get(name, {})
            exp_temp = normalize_value(exp_config.get("temperature"))

            assert exp_temp == orig_temp, (
                f"temperature mismatch for LLM '{name}': expected {orig_temp}, got {exp_temp}")

    def test_llm_max_tokens_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that LLM max_tokens field is exactly preserved."""
        original, exported = do_round_trip(simple_calculator_config)

        for name, orig_config in original.get("llms", {}).items():
            if "max_tokens" not in orig_config:
                continue

            orig_tokens = orig_config["max_tokens"]
            exp_config = exported["llms"].get(name, {})
            exp_tokens = exp_config.get("max_tokens")

            assert exp_tokens == orig_tokens, (
                f"max_tokens mismatch for LLM '{name}': expected {orig_tokens}, got {exp_tokens}")

    def test_llm_type_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that LLM _type field is preserved (short name match)."""
        original, exported = do_round_trip(simple_calculator_config)

        for name, orig_config in original.get("llms", {}).items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            exp_config = exported["llms"].get(name, {})
            exp_type = exp_config.get("_type", "")
            exp_type_short = exp_type.split("/")[-1]

            assert exp_type_short == orig_type_short, (
                f"_type mismatch for LLM '{name}': expected '{orig_type_short}', got '{exp_type_short}'")


class TestLLMRoundTripWithDifferentConfigs:
    """Tests for LLM round-trip across different example configurations."""

    def test_react_agent_llm_round_trip(self, react_agent_config: Path, set_test_env_vars):
        """Verify LLMs in react agent config survive round-trip."""
        original, exported = do_round_trip(react_agent_config)

        original_llm_count = len(original.get("llms", {}))
        exported_llm_count = len(exported.get("llms", {}))

        assert exported_llm_count == original_llm_count, (
            f"LLM count mismatch: expected {original_llm_count}, got {exported_llm_count}")

        # Verify each LLM's model_name
        for name, orig_config in original.get("llms", {}).items():
            orig_model = orig_config.get("model_name") or orig_config.get("model")
            if orig_model is None:
                continue

            exp_config = exported["llms"].get(name, {})
            exp_model = exp_config.get("model_name") or exp_config.get("model")
            assert exp_model == orig_model, f"model_name mismatch for '{name}'"

    def test_rag_config_llm_round_trip(self, rag_config: Path, set_test_env_vars):
        """Verify LLMs in RAG config survive round-trip."""
        original, exported = do_round_trip(rag_config)

        for name, orig_config in original.get("llms", {}).items():
            assert name in exported.get("llms", {}), f"Missing LLM '{name}' in exported config"

            orig_model = orig_config.get("model_name") or orig_config.get("model")
            exp_model = exported["llms"][name].get("model_name") or exported["llms"][name].get("model")
            assert exp_model == orig_model, f"model_name mismatch for '{name}'"
