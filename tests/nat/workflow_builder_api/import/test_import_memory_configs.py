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
"""Tests for importing memory configurations from YAML to workflow state.

These tests verify that memory configurations are correctly parsed and all
field values are accurately preserved during import.
"""

from pathlib import Path

import yaml


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportMemoryFromRedisConfig:
    """Tests for importing memory from Redis memory config.

    The Redis memory config contains memory components with specific settings.
    """

    def test_memory_components_are_created(self, memory_config: Path, set_test_env_vars):
        """Verify that memory components are created during import."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(memory_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        memory = [c for c in workflow_state.components if c.component_type == "memory"]
        original_count = len(config_dict.get("memory", {}))

        assert len(memory) == original_count, (f"Memory count mismatch: expected {original_count}, got {len(memory)}")

    def test_memory_ids_match_original_names(self, memory_config: Path, set_test_env_vars):
        """Verify memory IDs match the key names from the original config."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(memory_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_names = set(config_dict.get("memory", {}).keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "memory"}

        assert original_names == imported_ids, (
            f"Memory ID mismatch: original={original_names}, imported={imported_ids}")

    def test_memory_type_is_preserved(self, memory_config: Path, set_test_env_vars):
        """Verify memory _type is preserved in full_type."""
        from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

        config_dict = load_config_dict(memory_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        original_memory = config_dict.get("memory", {})
        for name, orig_config in original_memory.items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Memory '{name}' not found"

            imported_type_short = imported.full_type.split("/")[-1] if "/" in imported.full_type else imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Memory type mismatch for '{name}': expected '{orig_type_short}', got '{imported_type_short}'")
