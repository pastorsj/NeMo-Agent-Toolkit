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
"""Tests for importing observability configurations from YAML to workflow state.

These tests verify that telemetry exporter and logger configurations are
correctly parsed and all field values are accurately preserved during import.
"""

from pathlib import Path

import yaml

from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state


def load_config_dict(config_path: Path) -> dict:
    """Load YAML config from a file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestImportTelemetryExporters:
    """Tests for importing telemetry exporter configurations.

    The observability config contains telemetry exporters in general.telemetry.tracing.
    """

    def test_telemetry_exporter_components_are_created(self, observability_config: Path, set_test_env_vars):
        """Verify that telemetry_exporter components are created during import."""
        config_dict = load_config_dict(observability_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        telemetry = [c for c in workflow_state.components if c.component_type == "telemetry_exporter"]

        # Count telemetry exporters in original config
        tracing = config_dict.get("general", {}).get("telemetry", {}).get("tracing", {})
        original_count = len(tracing)

        assert len(telemetry) == original_count, (
            f"Telemetry exporter count mismatch: expected {original_count}, got {len(telemetry)}")

    def test_telemetry_exporter_ids_match_original_names(self, observability_config: Path, set_test_env_vars):
        """Verify telemetry exporter IDs match the key names from config."""
        config_dict = load_config_dict(observability_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        tracing = config_dict.get("general", {}).get("telemetry", {}).get("tracing", {})
        original_names = set(tracing.keys())
        imported_ids = {c.id for c in workflow_state.components if c.component_type == "telemetry_exporter"}

        assert original_names == imported_ids, (
            f"Telemetry exporter ID mismatch: original={original_names}, imported={imported_ids}")

    def test_telemetry_exporter_type_is_preserved(self, observability_config: Path, set_test_env_vars):
        """Verify telemetry exporter _type is preserved in full_type."""
        config_dict = load_config_dict(observability_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        tracing = config_dict.get("general", {}).get("telemetry", {}).get("tracing", {})
        for name, orig_config in tracing.items():
            orig_type = orig_config.get("_type", "")
            orig_type_short = orig_type.split("/")[-1]

            imported = next((c for c in workflow_state.components if c.id == name), None)
            assert imported is not None, f"Telemetry exporter '{name}' not found"

            if "/" in imported.full_type:
                imported_type_short = imported.full_type.split("/")[-1]
            else:
                imported_type_short = imported.full_type
            assert imported_type_short == orig_type_short, (
                f"Telemetry exporter type mismatch for '{name}': "
                f"expected '{orig_type_short}', got '{imported_type_short}'"
            )


class TestImportLoggers:
    """Tests for importing logger configurations.

    Loggers are defined in general.telemetry.logging.
    """

    def test_logger_components_are_created(self, observability_config: Path, set_test_env_vars):
        """Verify that logger components are created during import."""
        config_dict = load_config_dict(observability_config)
        workflow_state = parse_config_to_workflow_state(config_dict)

        loggers = [c for c in workflow_state.components if c.component_type == "logger"]

        # Count loggers in original config
        logging_config = config_dict.get("general", {}).get("telemetry", {}).get("logging", {})
        original_count = len(logging_config) if logging_config else 0

        assert len(loggers) == original_count, (f"Logger count mismatch: expected {original_count}, got {len(loggers)}")
