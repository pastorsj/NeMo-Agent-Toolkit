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
"""Shared fixtures for workflow builder API tests."""

import os
import sys
from pathlib import Path

import pytest

# Get project directories
TESTS_DIR = Path(__file__).parent.parent.parent.parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
EXAMPLES_DIR = PROJECT_DIR / "examples"

# Ensure src is in path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session", autouse=True)
def discover_plugins():
    """Discover and register all NAT plugins before tests."""
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    discover_and_register_plugins(PluginTypes.ALL)


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Return the examples directory path."""
    return EXAMPLES_DIR


@pytest.fixture
def set_test_env_vars(restore_environ):
    """Set test environment variables for configs that need them."""
    test_vars = {
        "NVIDIA_API_KEY": "test-api-key",
        "NGC_API_KEY": "test-ngc-key",
        "OPENAI_API_KEY": "test-openai-key",
        "NAT_OAUTH_CLIENT_ID": "test-client-id",
        "NAT_OAUTH_CLIENT_SECRET": "test-client-secret",
        "CUSTOMIZER_HOST": "http://localhost:8080",
        "DATASTORE_HOST": "http://localhost:8081",
    }
    for key, value in test_vars.items():
        os.environ[key] = value


# =============================================================================
# CONFIG FILE FIXTURES - Grouped by component types
# =============================================================================


@pytest.fixture
def simple_calculator_config(examples_dir: Path) -> Path:
    """Simple workflow with LLM, functions, and react agent."""
    return examples_dir / "getting_started/simple_calculator/src/nat_simple_calculator/configs/config.yml"


@pytest.fixture
def react_agent_config(examples_dir: Path) -> Path:
    """React agent with evaluators (LLM, functions, eval section)."""
    return examples_dir / "agents/react/configs/config.yml"


@pytest.fixture
def rag_config(examples_dir: Path) -> Path:
    """RAG workflow with retrievers and embedders."""
    return examples_dir / "RAG/simple_rag/configs/milvus_rag_config.yml"


@pytest.fixture
def rag_tools_config(examples_dir: Path) -> Path:
    """RAG workflow with retriever functions as tools."""
    return examples_dir / "RAG/simple_rag/configs/milvus_rag_tools_config.yml"


@pytest.fixture
def observability_config(examples_dir: Path) -> Path:
    """Workflow with telemetry/logging (general section)."""
    return examples_dir / "observability/simple_calculator_observability/configs/config-langfuse.yml"


@pytest.fixture
def memory_config(examples_dir: Path) -> Path:
    """Workflow with memory components."""
    return examples_dir / "memory/redis/configs/config.yml"


@pytest.fixture
def auth_config(examples_dir: Path) -> Path:
    """Workflow with authentication and front-end."""
    return examples_dir / "front_ends/simple_auth/src/nat_simple_auth/configs/config.yml"


@pytest.fixture
def finetuning_config(examples_dir: Path) -> Path:
    """Workflow with finetuning components (trainers, trajectory builders)."""
    return examples_dir / "finetuning/dpo_tic_tac_toe/src/dpo_tic_tac_toe/configs/config.yml"


@pytest.fixture
def ttc_strategy_config(examples_dir: Path) -> Path:
    """Workflow with TTC strategies."""
    return examples_dir / "finetuning/dpo_tic_tac_toe/src/dpo_tic_tac_toe/configs/config.yml"


@pytest.fixture
def rewoo_agent_config(examples_dir: Path) -> Path:
    """ReWOO agent workflow."""
    return examples_dir / "agents/rewoo/configs/config.yml"


@pytest.fixture
def tool_calling_agent_config(examples_dir: Path) -> Path:
    """Tool-calling agent workflow."""
    return examples_dir / "agents/tool_calling/configs/config.yml"


@pytest.fixture
def function_groups_config(examples_dir: Path) -> Path:
    """Workflow with function groups."""
    return examples_dir / "observability/simple_calculator_observability/configs/config-langfuse.yml"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def load_yaml_content(config_path: Path) -> str:
    """Load YAML content from a config file."""
    with open(config_path) as f:
        return f.read()


def validate_config_dict(config_dict: dict) -> tuple[bool, list[str]]:
    """Validate a config dict using the Config class."""
    from nat.data_models.config import Config
    from nat.utils.data_models.schema_validator import validate_schema

    success, errors = validate_schema(Config, config_dict)
    return success, errors
