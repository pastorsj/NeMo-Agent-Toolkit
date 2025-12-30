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
"""SDK tests for Middleware configuration and serialization.

This module tests creating, configuring, and serializing middleware using the SDK.
"""

from pathlib import Path

import pytest
import yaml

from nat.agent.sdk import NatReActAgent
from nat.llm.sdk import NimLLM
from nat.middleware.sdk import CacheMiddlewareConfig
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.tool.sdk import CurrentTimeTool
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
# Cache Middleware SDK Initialization Tests
# ============================================================================


class TestCacheMiddlewareInitialization:
    """Tests for CacheMiddlewareConfig initialization via SDK."""

    def test_basic_cache_middleware_creation(self):
        """Test creating a basic cache middleware."""
        middleware = CacheMiddlewareConfig()

        # Should create with defaults
        assert middleware is not None

    def test_cache_middleware_with_threshold(self):
        """Test cache middleware with similarity threshold."""
        middleware = CacheMiddlewareConfig(similarity_threshold=0.8)

        assert middleware.similarity_threshold == 0.8


# ============================================================================
# Middleware Edge Cases
# ============================================================================


class TestMiddlewareEdgeCases:
    """Edge case tests for middleware SDK usage."""

    def test_tool_without_middleware(self, tmp_path: Path):
        """Test tool without middleware."""
        tool = CurrentTimeTool(name="datetime_tool")

        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[tool], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config_path = tmp_path / "config.yaml"
        workflow.save_to_config_file(config_path)

        content = load_yaml(config_path)
        tool_config = content["functions"]["datetime_tool"]
        # Middleware should be absent or empty
        middleware = tool_config.get("middleware", [])
        assert middleware == [] or middleware is None
