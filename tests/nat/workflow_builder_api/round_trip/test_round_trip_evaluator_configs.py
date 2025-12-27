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
"""Round-trip tests for evaluator configurations.

These tests verify that evaluator configurations survive the import -> export
cycle with all field values preserved exactly as in the original config.
"""

from pathlib import Path

import yaml


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def do_round_trip(config_path: Path) -> tuple[dict, dict]:
    """Perform a full round-trip: load config -> import -> export."""
    from nat.workflow_builder_api.models import ExportComponent
    from nat.workflow_builder_api.models import ExportConnection
    from nat.workflow_builder_api.models import ExportWorkflowRequest
    from nat.workflow_builder_api.utils.config_exporter import export_workflow_to_yaml
    from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

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


class TestEvaluatorRoundTripPreservesValues:
    """Tests that verify evaluator field values are preserved through round-trip."""

    def test_eval_section_exists_after_round_trip(self, react_agent_config: Path, set_test_env_vars):
        """Verify that eval section exists in exported config."""
        original, exported = do_round_trip(react_agent_config)

        if "eval" not in original:
            return  # Skip if no eval section

        assert "eval" in exported, "Exported config should have eval section"

    def test_evaluator_names_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that all evaluator names are preserved through round-trip."""
        original, exported = do_round_trip(react_agent_config)

        if "eval" not in original or "evaluators" not in original.get("eval", {}):
            return

        original_names = set(original["eval"]["evaluators"].keys())
        exported_names = set(exported.get("eval", {}).get("evaluators", {}).keys())

        assert original_names == exported_names, (
            f"Evaluator names mismatch: original={original_names}, exported={exported_names}")

    def test_evaluator_metric_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that evaluator metric field is preserved."""
        original, exported = do_round_trip(react_agent_config)

        orig_evaluators = original.get("eval", {}).get("evaluators", {})
        exp_evaluators = exported.get("eval", {}).get("evaluators", {})

        for name, orig_config in orig_evaluators.items():
            if "metric" not in orig_config:
                continue

            orig_metric = orig_config["metric"]
            exp_config = exp_evaluators.get(name, {})
            exp_metric = exp_config.get("metric")

            assert exp_metric == orig_metric, (
                f"metric mismatch for evaluator '{name}': expected '{orig_metric}', got '{exp_metric}'")

    def test_evaluator_type_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that evaluator _type field is preserved (short name match)."""
        original, exported = do_round_trip(react_agent_config)

        orig_evaluators = original.get("eval", {}).get("evaluators", {})
        exp_evaluators = exported.get("eval", {}).get("evaluators", {})

        for name, orig_config in orig_evaluators.items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            exp_config = exp_evaluators.get(name, {})
            exp_type = exp_config.get("_type", "")
            exp_type_short = exp_type.split("/")[-1]

            assert exp_type_short == orig_type_short, (
                f"_type mismatch for evaluator '{name}': expected '{orig_type_short}', got '{exp_type_short}'")

    def test_evaluator_llm_name_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that evaluator's llm_name reference is preserved."""
        original, exported = do_round_trip(react_agent_config)

        orig_evaluators = original.get("eval", {}).get("evaluators", {})
        exp_evaluators = exported.get("eval", {}).get("evaluators", {})

        for name, orig_config in orig_evaluators.items():
            orig_llm = orig_config.get("llm_name") or orig_config.get("llm")
            if orig_llm is None:
                continue

            exp_config = exp_evaluators.get(name, {})
            exp_llm = exp_config.get("llm_name") or exp_config.get("llm")

            assert exp_llm == orig_llm, (
                f"llm_name mismatch for evaluator '{name}': expected '{orig_llm}', got '{exp_llm}'")


class TestEvalGeneralRoundTrip:
    """Tests for eval.general section round-trip."""

    def test_eval_general_output_dir_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that eval.general.output_dir is preserved."""
        original, exported = do_round_trip(react_agent_config)

        orig_general = original.get("eval", {}).get("general", {})
        if "output_dir" not in orig_general:
            return

        orig_output_dir = str(orig_general["output_dir"]).rstrip("/")
        exp_general = exported.get("eval", {}).get("general", {})
        exp_output_dir = str(exp_general.get("output_dir", "")).rstrip("/")

        # Normalize leading ./ for comparison
        if orig_output_dir.startswith("./"):
            orig_output_dir = orig_output_dir[2:]
        if exp_output_dir.startswith("./"):
            exp_output_dir = exp_output_dir[2:]

        assert exp_output_dir == orig_output_dir, (
            f"output_dir mismatch: expected '{orig_output_dir}', got '{exp_output_dir}'")

    def test_eval_general_parallel_preserved(self, react_agent_config: Path, set_test_env_vars):
        """Verify that eval.general.parallel is preserved."""
        original, exported = do_round_trip(react_agent_config)

        orig_general = original.get("eval", {}).get("general", {})
        if "parallel" not in orig_general:
            return

        orig_parallel = orig_general["parallel"]
        exp_general = exported.get("eval", {}).get("general", {})
        exp_parallel = exp_general.get("parallel")

        assert exp_parallel == orig_parallel, (f"parallel mismatch: expected {orig_parallel}, got {exp_parallel}")
