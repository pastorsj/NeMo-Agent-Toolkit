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

import logging
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_logging_method
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.logging import LoggingBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.file_mode import FileMode

logger = logging.getLogger(__name__)


class FileTelemetryExporterConfig(TelemetryExporterBaseConfig, name="file"):
    """
    A telemetry exporter that writes runtime traces to local files with optional rolling.

    ## Details
    Name: File Telemetry Exporter
    Icon: ![Icon](https://cdn.simpleicons.org/opentelemetry/F5A800)
    """

    output_path: str = Field(description="Output path for logs. When rolling is disabled: exact file path. "
                             "When rolling is enabled: directory path or file path (directory + base name).")
    project: str = Field(description="Name to affiliate with this application.")
    mode: FileMode = Field(
        default=FileMode.APPEND,
        description="File write mode: 'append' to add to existing file or 'overwrite' to start fresh.")
    enable_rolling: bool = Field(default=False, description="Enable rolling log files based on size limits.")
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file size in bytes before rolling to a new file.")
    max_files: int = Field(default=5, description="Maximum number of rolled files to keep.")
    cleanup_on_init: bool = Field(default=False, description="Clean up old files during initialization.")


@register_telemetry_exporter(config_type=FileTelemetryExporterConfig)
async def file_telemetry_exporter(config: FileTelemetryExporterConfig, builder: Builder):
    """
    Build and return a FileExporter for file-based telemetry export with optional rolling.
    """

    from nat.observability.exporter.file_exporter import FileExporter

    yield FileExporter(output_path=config.output_path,
                       project=config.project,
                       mode=config.mode,
                       enable_rolling=config.enable_rolling,
                       max_file_size=config.max_file_size,
                       max_files=config.max_files,
                       cleanup_on_init=config.cleanup_on_init)


class ConsoleLoggingMethodConfig(LoggingBaseConfig, name="console"):
    """
    A logger to write runtime logs to the console.

    ## Details
    Name: Console Logger
    Icon: ![Icon](https://cdn.simpleicons.org/gnubash/4EAA25)
    """

    level: str = Field(description="The logging level of console logger.")


@register_logging_method(config_type=ConsoleLoggingMethodConfig)
async def console_logging_method(config: ConsoleLoggingMethodConfig, builder: Builder):
    """
    Build and return a StreamHandler for console-based logging.
    """
    import sys

    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)

    # Set formatter to match the default CLI format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    yield handler


class FileLoggingMethod(LoggingBaseConfig, name="file"):
    """
    A logger to write runtime logs to a file.

    ## Details
    Name: File Logger
    Icon: ![Icon](https://cdn.simpleicons.org/files/1976D2)
    """

    path: str = Field(description="The file path to save the logging output.")
    level: str = Field(description="The logging level of file logger.")
    create_if_not_exists: bool = Field(description="Create the file to write to if it does not exist", default=False)


@register_logging_method(config_type=FileLoggingMethod)
async def file_logging_method(config: FileLoggingMethod, builder: Builder):
    """
    Build and return a FileHandler for file-based logging.
    """

    path_to_log_file = Path(config.path).resolve()

    if config.create_if_not_exists and not path_to_log_file.exists():
        path_to_log_file.parent.mkdir(parents=True, exist_ok=True)
        path_to_log_file.touch()
    elif not path_to_log_file.exists():
        raise FileNotFoundError("Path is not found. Please create it.")

    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.FileHandler(filename=config.path, mode="a", encoding="utf-8")
    handler.setLevel(level)

    # Set formatter to match the default CLI format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    yield handler
