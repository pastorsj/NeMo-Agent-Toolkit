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
"""Tests for importing LLM configurations from YAML to workflow state.

These tests verify that LLM configurations are correctly parsed and all
field values are accurately preserved during import.
"""

from pathlib import Path

import yaml

from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportLLMFromSimpleCalculator:
    """Tests for importing LLM from simple calculator config.

    The simple calculator config contains a NIM LLM with specific settings.
    These tests verify each field is correctly imported.
    """

    def test_llm_component_is_created(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify that LLM components are created during import."""
        config_dict = load_config_dict(simple_calculator_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        llms = [c for c in workflow_state.components if c.component_type == "llm"]
        assert len(llms) >= 1, "Expected at least one LLM component"

    def test_llm_has_correct_full_type(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify LLM full_type matches the original _type from config."""
        config_dict = load_config_dict(simple_calculator_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        llms = [c for c in workflow_state.components if c.component_type == "llm"]
        assert len(llms) >= 1

        # Get original LLM config
        original_llms = config_dict.get("llms", {})
        for name, orig_config in original_llms.items():
            orig_type = orig_config.get("_type", "").split("/")[-1]

            imported_llm = next((c for c in llms if c.id == name), None)
            assert imported_llm is not None, f"LLM '{name}' not found in imported components"

            imported_type = imported_llm.full_type.split(
                "/")[-1] if "/" in imported_llm.full_type else imported_llm.full_type
            assert imported_type == orig_type, f"LLM type mismatch: expected '{orig_type}', got '{imported_type}'"

    def test_llm_model_name_is_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify LLM model_name field is exactly preserved."""
        config_dict = load_config_dict(simple_calculator_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_llms = config_dict.get("llms", {})
        for name, orig_config in original_llms.items():
            orig_model = orig_config.get("model_name") or orig_config.get("model")
            if orig_model is None:
                continue

            imported_llm = next((c for c in workflow_state.components if c.id == name), None)
            assert imported_llm is not None, f"LLM '{name}' not found"

            imported_model = imported_llm.config.get("model_name") or imported_llm.config.get("model")
            assert imported_model == orig_model, (
                f"model_name mismatch for '{name}': expected '{orig_model}', got '{imported_model}'")

    def test_llm_temperature_is_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify LLM temperature field is exactly preserved."""
        config_dict = load_config_dict(simple_calculator_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_llms = config_dict.get("llms", {})
        for name, orig_config in original_llms.items():
            if "temperature" not in orig_config:
                continue

            orig_temp = orig_config["temperature"]
            imported_llm = next((c for c in workflow_state.components if c.id == name), None)
            assert imported_llm is not None

            imported_temp = imported_llm.config.get("temperature")
            assert imported_temp == orig_temp, (
                f"temperature mismatch for '{name}': expected {orig_temp}, got {imported_temp}")

    def test_llm_max_tokens_is_preserved(self, simple_calculator_config: Path, set_test_env_vars):
        """Verify LLM max_tokens field is exactly preserved."""
        config_dict = load_config_dict(simple_calculator_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_llms = config_dict.get("llms", {})
        for name, orig_config in original_llms.items():
            if "max_tokens" not in orig_config:
                continue

            orig_tokens = orig_config["max_tokens"]
            imported_llm = next((c for c in workflow_state.components if c.id == name), None)
            assert imported_llm is not None

            imported_tokens = imported_llm.config.get("max_tokens")
            assert imported_tokens == orig_tokens, (
                f"max_tokens mismatch for '{name}': expected {orig_tokens}, got {imported_tokens}")


class TestImportLLMFromReactAgent:
    """Tests for importing LLM from react agent config.

    The react agent config may have different LLM settings.
    """

    def test_llm_count_matches_original(self, react_agent_config: Path, set_test_env_vars):
        """Verify the number of imported LLMs matches the original config."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_llm_count = len(config_dict.get("llms", {}))
        imported_llms = [c for c in workflow_state.components if c.component_type == "llm"]

        assert len(imported_llms) == original_llm_count, (
            f"LLM count mismatch: expected {original_llm_count}, got {len(imported_llms)}")

    def test_all_llm_names_are_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify all LLM names from config are preserved as component IDs."""
        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_llm_names = set(config_dict.get("llms", {}).keys())
        imported_llm_ids = {c.id for c in workflow_state.components if c.component_type == "llm"}

        assert original_llm_names == imported_llm_ids, (
            f"LLM name mismatch: original={original_llm_names}, imported={imported_llm_ids}")
