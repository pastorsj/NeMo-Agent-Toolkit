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
"""Tests for importing evaluator configurations from YAML to workflow state.

These tests verify that evaluator configurations are correctly parsed and all
field values are accurately preserved during import.
"""

from pathlib import Path

import yaml


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportEvaluatorsFromReactConfig:
    """Tests for importing evaluators from react agent config.

    The react agent config contains RAGAS evaluators in the eval section.
    """

    def test_evaluator_components_are_created(self, react_agent_config: Path, set_test_env_vars):
        """Verify that evaluator components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        evaluators = [c for c in workflow_state.components if c.component_type == "evaluator"]
        original_count = len(config_dict.get("eval", {}).get("evaluators", {}))

        assert len(evaluators) == original_count, (
            f"Evaluator count mismatch: expected {original_count}, got {len(evaluators)}")

    def test_evaluator_ids_match_original_names(self, react_agent_config: Path, set_test_env_vars):
        """Verify evaluator IDs match the key names from the original config."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_names = set(config_dict.get("eval", {}).get("evaluators", {}).keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "evaluator"}

        assert original_names == imported_ids, (
            f"Evaluator ID mismatch: original={original_names}, imported={imported_ids}")

    def test_evaluator_metric_is_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify evaluator metric field is exactly preserved."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_evaluators = config_dict.get("eval", {}).get("evaluators", {})
        for name, orig_config in original_evaluators.items():
            if "metric" not in orig_config:
                continue

            orig_metric = orig_config["metric"]
            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Evaluator '{name}' not found"

            imported_metric = imported.config.get("metric")
            assert imported_metric == orig_metric, (
                f"metric mismatch for evaluator '{name}': expected '{orig_metric}', got '{imported_metric}'")

    def test_evaluator_type_is_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify evaluator _type is preserved in full_type."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_evaluators = config_dict.get("eval", {}).get("evaluators", {})
        for name, orig_config in original_evaluators.items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Evaluator '{name}' not found"

            imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Evaluator type mismatch for '{name}': expected '{orig_type_short}', got '{imported_type_short}'")

    def test_evaluator_llm_connection_is_created(self, react_agent_config: Path, set_test_env_vars):
        """Verify that evaluator's llm_name creates a connection."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(react_agent_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_evaluators = config_dict.get("eval", {}).get("evaluators", {})
        for name, orig_config in original_evaluators.items():
            llm_ref = orig_config.get("llm_name") or orig_config.get("llm")
            if llm_ref is None:
                continue

            # Should have a connection from evaluator to LLM
            llm_connections = [
                c for c in workflow_state.connections if c.source_id == name and "llm" in c.target_field.lower()
            ]
            assert len(llm_connections) >= 1, (f"Expected connection from evaluator '{name}' to LLM '{llm_ref}'")
