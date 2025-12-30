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
"""SDK tests for Logging configuration and serialization.

This module tests creating, configuring, and serializing loggers using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.sdk import NatReActAgent
from nat.data_models.config import Config
from nat.llm.sdk import NimLLM
from nat.observability.sdk import ConsoleLogger
from nat.observability.sdk import FileLogger
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
# Console Logger SDK Initialization Tests
# ============================================================================


class TestConsoleLoggerInitialization:
    """Tests for ConsoleLogger initialization via SDK."""

    def test_basic_console_logger_creation(self):
        """Test creating a basic console logger."""
        logger = ConsoleLogger(level="INFO", name="console")

        assert logger.level == "INFO"
        assert logger.name == "console"

    def test_console_logger_with_debug_level(self):
        """Test console logger with DEBUG level."""
        logger = ConsoleLogger(level="DEBUG", name="debug_console")

        assert logger.level == "DEBUG"

    def test_console_logger_with_warning_level(self):
        """Test console logger with WARNING level."""
        logger = ConsoleLogger(level="WARNING", name="warn_console")

        assert logger.level == "WARNING"


# ============================================================================
# File Logger SDK Initialization Tests
# ============================================================================


class TestFileLoggerInitialization:
    """Tests for FileLogger initialization via SDK."""

    def test_basic_file_logger_creation(self):
        """Test creating a basic file logger."""
        logger = FileLogger(path="/tmp/app.log", level="INFO", name="file")

        assert logger.path == "/tmp/app.log"
        assert logger.level == "INFO"

    def test_file_logger_with_create_option(self):
        """Test file logger with create_if_not_exists option."""
        logger = FileLogger(
            path="/tmp/app.log",
            level="DEBUG",
            create_if_not_exists=True,
            name="file_create",
        )

        assert logger.create_if_not_exists is True


# ============================================================================
# Logger Serialization Tests
# ============================================================================


class TestLoggerSerialization:
    """Tests for serializing loggers to YAML config."""

    def test_console_logger_saves_correctly(self, tmp_path: Path):
        """Test that console logger saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        logger = ConsoleLogger(level="INFO", name="console")
        config = NatGeneralConfiguration(loggers=[logger])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        assert "general" in content
        assert "telemetry" in content["general"]
        assert "logging" in content["general"]["telemetry"]
        assert "console" in content["general"]["telemetry"]["logging"]

    def test_file_logger_saves_correctly(self, tmp_path: Path):
        """Test that file logger saves to YAML correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        logger = FileLogger(path="/tmp/app.log", level="DEBUG", name="file_log")
        config = NatGeneralConfiguration(loggers=[logger])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        logging_config = content["general"]["telemetry"]["logging"]
        assert "file_log" in logging_config
        assert logging_config["file_log"]["path"] == "/tmp/app.log"
        assert logging_config["file_log"]["level"] == "DEBUG"

    def test_multiple_loggers_save_correctly(self, tmp_path: Path):
        """Test that multiple loggers save correctly."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        console = ConsoleLogger(level="INFO", name="console")
        file_log = FileLogger(path="/tmp/app.log", level="DEBUG", name="file")
        config = NatGeneralConfiguration(loggers=[console, file_log])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        logging_config = content["general"]["telemetry"]["logging"]
        assert "console" in logging_config
        assert "file" in logging_config


# ============================================================================
# Logger Round-Trip Tests
# ============================================================================


class TestLoggerRoundTrip:
    """Tests for round-trip serialization of loggers."""

    def test_console_logger_round_trip(self, tmp_path: Path):
        """Test that console logger config survives round-trip."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        logger = ConsoleLogger(level="WARNING", name="warn_console")
        config = NatGeneralConfiguration(loggers=[logger])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        assert loaded.general.telemetry is not None
        assert loaded.general.telemetry.logging is not None
        assert "warn_console" in loaded.general.telemetry.logging
        assert loaded.general.telemetry.logging["warn_console"].level == "WARNING"

    def test_file_logger_round_trip(self, tmp_path: Path):
        """Test that file logger config survives round-trip."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        logger = FileLogger(
            path="/tmp/detailed.log",
            level="DEBUG",
            create_if_not_exists=True,
            name="detailed_file",
        )
        config = NatGeneralConfiguration(loggers=[logger])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Load and validate
        content = load_yaml(config_path)
        loaded = Config.model_validate(content)

        loaded_logger = loaded.general.telemetry.logging["detailed_file"]
        assert loaded_logger.path == "/tmp/detailed.log"
        assert loaded_logger.level == "DEBUG"


# ============================================================================
# Logger Edge Cases
# ============================================================================


class TestLoggerEdgeCases:
    """Edge case tests for logger SDK usage."""

    def test_workflow_without_loggers(self, tmp_path: Path):
        """Test workflow without any loggers."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        # Should save without errors
        assert config_path.exists()

    def test_file_logger_with_special_path(self, tmp_path: Path):
        """Test file logger with path containing spaces."""
        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)

        logger = FileLogger(
            path="/tmp/my app/logs/test file.log",
            level="INFO",
            name="special_path",
        )
        config = NatGeneralConfiguration(loggers=[logger])

        workflow = NatWorkflow(entrypoint=agent, configuration=config)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        logger_config = content["general"]["telemetry"]["logging"]["special_path"]
        assert logger_config["path"] == "/tmp/my app/logs/test file.log"
