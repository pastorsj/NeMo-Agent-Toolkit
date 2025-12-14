# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SDK tests for Telemetry Exporter configuration and serialization.

This module tests creating, configuring, and serializing telemetry exporters using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.config import Config
from nat.llm.nim_llm import NimLLM
from nat.observability.register import FileTelemetryExporter
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.utils.sdk.nat_general_configuraton import NatGeneralConfiguration
from nat.utils.sdk.nat_workflow import NatWorkflow


@pytest.fixture(scope="module", autouse=True)
def discover_plugins():
    """Discover and register all plugins before running tests."""
    discover_and_register_plugins(PluginTypes.ALL)


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# File Telemetry Exporter SDK Initialization Tests
# ============================================================================


class TestFileTelemetryExporterInitialization:
    """Tests for FileTelemetryExporter initialization via SDK."""

    def test_basic_file_exporter_creation(self):
        """Test creating a basic file telemetry exporter."""
        exporter = FileTelemetryExporter(
            output_path="/tmp/traces",
            project="test_project",
            name="file_traces",
        )

        assert exporter.output_path == "/tmp/traces"
        assert exporter.project == "test_project"

    def test_file_exporter_with_rolling(self):
        """Test file exporter with rolling enabled."""
        exporter = FileTelemetryExporter(
            output_path="/tmp/traces",
            project="test_project",
            enable_rolling=True,
            max_file_size=1024 * 1024,  # 1MB
            max_files=5,
            name="rolling_traces",
        )

        assert exporter.enable_rolling is True
        assert exporter.max_file_size == 1024 * 1024
        assert exporter.max_files == 5


# ============================================================================
# Telemetry Exporter Serialization Tests
# ============================================================================


class TestTelemetryExporterSerialization:
    """Tests for serializing telemetry exporters to YAML config."""

    def test_file_exporter_saves_correctly(self, tmp_path: Path):
        """Test that file exporter saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        exporter = FileTelemetryExporter(
            output_path="/tmp/traces",
            project="test_project",
            name="file_traces",
        )
        config = NatGeneralConfiguration(telemetry_exporters=[exporter])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "general" in content
        assert "telemetry" in content["general"]
        assert "tracing" in content["general"]["telemetry"]
        assert "file_traces" in content["general"]["telemetry"]["tracing"]

    def test_file_exporter_output_path_serializes(self, tmp_path: Path):
        """Test that output_path is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        exporter = FileTelemetryExporter(
            output_path="/var/log/traces",
            project="prod",
            name="prod_traces",
        )
        config = NatGeneralConfiguration(telemetry_exporters=[exporter])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        exporter_config = content["general"]["telemetry"]["tracing"]["prod_traces"]
        assert exporter_config["output_path"] == "/var/log/traces"
        assert exporter_config["project"] == "prod"

    def test_file_exporter_with_rolling_serializes(self, tmp_path: Path):
        """Test that rolling config is serialized correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        exporter = FileTelemetryExporter(
            output_path="/tmp/traces",
            project="test",
            enable_rolling=True,
            max_file_size=2 * 1024 * 1024,  # 2MB
            max_files=10,
            name="rolling_exporter",
        )
        config = NatGeneralConfiguration(telemetry_exporters=[exporter])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        exporter_config = content["general"]["telemetry"]["tracing"]["rolling_exporter"]
        assert exporter_config["enable_rolling"] is True
        assert exporter_config["max_file_size"] == 2 * 1024 * 1024
        assert exporter_config["max_files"] == 10


# ============================================================================
# Telemetry Exporter Round-Trip Tests
# ============================================================================


class TestTelemetryExporterRoundTrip:
    """Tests for round-trip serialization of telemetry exporters."""

    def test_file_exporter_round_trip(self, tmp_path: Path):
        """Test that file exporter config survives round-trip."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        exporter = FileTelemetryExporter(
            output_path="/tmp/my_traces",
            project="my_project",
            enable_rolling=True,
            max_files=10,
            name="rolling_exporter",
        )
        config = NatGeneralConfiguration(telemetry_exporters=[exporter])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.general.telemetry is not None
        assert loaded.general.telemetry.tracing is not None
        tracing_config = loaded.general.telemetry.tracing["rolling_exporter"]
        assert tracing_config.output_path == "/tmp/my_traces"
        assert tracing_config.project == "my_project"
        assert tracing_config.enable_rolling is True
        assert tracing_config.max_files == 10


# ============================================================================
# Combined Logging and Telemetry Tests
# ============================================================================


class TestCombinedLoggingAndTelemetry:
    """Tests for workflows with both logging and telemetry."""

    def test_loggers_and_exporters_save_correctly(self, tmp_path: Path):
        """Test workflow with both loggers and exporters."""
        from nat.observability.register import ConsoleLogger

        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        console_logger = ConsoleLogger(level="INFO", name="console")
        file_exporter = FileTelemetryExporter(
            output_path="/tmp/traces",
            project="test",
            name="traces",
        )
        config = NatGeneralConfiguration(
            loggers=[console_logger],
            telemetry_exporters=[file_exporter],
        )

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        telemetry = content["general"]["telemetry"]
        assert "logging" in telemetry
        assert "tracing" in telemetry
        assert "console" in telemetry["logging"]
        assert "traces" in telemetry["tracing"]


# ============================================================================
# Telemetry Exporter Edge Cases
# ============================================================================


class TestTelemetryExporterEdgeCases:
    """Edge case tests for telemetry exporter SDK usage."""

    def test_workflow_without_telemetry(self, tmp_path: Path):
        """Test workflow without telemetry configuration."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Should save without errors
        assert config_path.exists()

    def test_file_exporter_with_special_path(self, tmp_path: Path):
        """Test file exporter with path containing spaces."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        exporter = FileTelemetryExporter(
            output_path="/tmp/my traces/project",
            project="test project",
            name="special_traces",
        )
        config = NatGeneralConfiguration(telemetry_exporters=[exporter])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        exporter_config = content["general"]["telemetry"]["tracing"]["special_traces"]
        assert exporter_config["output_path"] == "/tmp/my traces/project"
        assert exporter_config["project"] == "test project"
