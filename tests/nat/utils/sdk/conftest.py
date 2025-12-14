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
"""Fixtures for SDK tests."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture(name="tmp_config_path")
def tmp_config_path_fixture(tmp_path: Path) -> Path:
    """Provide a temporary path for config file output."""
    return tmp_path / "test_config.yaml"


@pytest.fixture(name="tmp_config_dir")
def tmp_config_dir_fixture(tmp_path: Path) -> Path:
    """Provide a temporary directory for config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def load_yaml_file(file_path: Path) -> dict:
    """Helper to load a YAML file and return its contents as a dict."""
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(name="load_yaml")
def load_yaml_fixture():
    """Fixture that returns the load_yaml_file helper."""
    return load_yaml_file
