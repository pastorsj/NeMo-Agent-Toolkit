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
"""Tests for importing finetuning configurations from YAML to workflow state.

These tests verify that trainer, trajectory builder, and trainer adapter
configurations are correctly parsed and all field values are accurately
preserved during import.
"""

from pathlib import Path

import yaml


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportTrainersFromConfig:
    """Tests for importing trainer configurations."""

    def test_trainer_components_are_created(self, finetuning_config: Path, set_test_env_vars):
        """Verify that trainer components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(finetuning_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        trainers = [c for c in workflow_state.components if c.component_type == "trainer"]
        original_count = len(config_dict.get("trainers", {}))

        assert len(trainers) == original_count, (
            f"Trainer count mismatch: expected {original_count}, got {len(trainers)}")

    def test_trainer_ids_match_original_names(self, finetuning_config: Path, set_test_env_vars):
        """Verify trainer IDs match the key names from the original config."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(finetuning_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_names = set(config_dict.get("trainers", {}).keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "trainer"}

        assert original_names == imported_ids, (
            f"Trainer ID mismatch: original={original_names}, imported={imported_ids}")

    def test_trainer_type_is_preserved(self, finetuning_config: Path, set_test_env_vars):
        """Verify trainer _type is preserved in full_type."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(finetuning_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_trainers = config_dict.get("trainers", {})
        for name, orig_config in original_trainers.items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Trainer '{name}' not found"

            imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Trainer type mismatch for '{name}': expected '{orig_type_short}', got '{imported_type_short}'")


class TestImportTrajectoryBuildersFromConfig:
    """Tests for importing trajectory builder configurations."""

    def test_trajectory_builder_components_are_created(self, finetuning_config: Path, set_test_env_vars):
        """Verify that trajectory_builder components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(finetuning_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        traj_builders = [c for c in workflow_state.components if c.component_type == "trajectory_builder"]
        original_count = len(config_dict.get("trajectory_builders", {}))

        assert len(traj_builders) == original_count, (
            f"Trajectory builder count mismatch: expected {original_count}, got {len(traj_builders)}")

    def test_trajectory_builder_ids_match_original_names(self, finetuning_config: Path, set_test_env_vars):
        """Verify trajectory builder IDs match the key names from the original config."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(finetuning_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_names = set(config_dict.get("trajectory_builders", {}).keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "trajectory_builder"}

        assert original_names == imported_ids, (
            f"Trajectory builder ID mismatch: original={original_names}, imported={imported_ids}")


class TestImportTrainerAdaptersFromConfig:
    """Tests for importing trainer adapter configurations."""

    def test_trainer_adapter_components_are_created(self, finetuning_config: Path, set_test_env_vars):
        """Verify that trainer_adapter components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(finetuning_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        adapters = [c for c in workflow_state.components if c.component_type == "trainer_adapter"]
        original_count = len(config_dict.get("trainer_adapters", {}))

        assert len(adapters) == original_count, (
            f"Trainer adapter count mismatch: expected {original_count}, got {len(adapters)}")

    def test_trainer_adapter_ids_match_original_names(self, finetuning_config: Path, set_test_env_vars):
        """Verify trainer adapter IDs match the key names from the original config."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(finetuning_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_names = set(config_dict.get("trainer_adapters", {}).keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "trainer_adapter"}

        assert original_names == imported_ids, (
            f"Trainer adapter ID mismatch: original={original_names}, imported={imported_ids}")


class TestImportTTCStrategiesFromConfig:
    """Tests for importing TTC strategy configurations."""

    def test_ttc_strategy_components_are_created(self, ttc_strategy_config: Path, set_test_env_vars):
        """Verify that ttc_strategy components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(ttc_strategy_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        ttc = [c for c in workflow_state.components if c.component_type == "ttc_strategy"]
        original_count = len(config_dict.get("ttc_strategies", {}))

        assert len(ttc) == original_count, (f"TTC strategy count mismatch: expected {original_count}, got {len(ttc)}")

    def test_ttc_strategy_ids_match_original_names(self, ttc_strategy_config: Path, set_test_env_vars):
        """Verify TTC strategy IDs match the key names from the original config."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(ttc_strategy_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_names = set(config_dict.get("ttc_strategies", {}).keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "ttc_strategy"}

        assert original_names == imported_ids, (
            f"TTC strategy ID mismatch: original={original_names}, imported={imported_ids}")
