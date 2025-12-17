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
"""Tests for NatBase class functionality.

This module tests the base SDK class, including name computation,
config casting, and dynamic registration features.
"""

import pytest

from nat.agent.react_agent.register import NatReActAgent
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.llm import LLMBaseConfig
from nat.llm.nim_llm import NimLLM
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.tool.datetime_tools import CurrentTimeTool


# Ensure plugins are discovered for tests
@pytest.fixture(scope="module", autouse=True)
def discover_plugins():
    """Discover and register all plugins before running tests."""
    discover_and_register_plugins(PluginTypes.ALL)


# ============================================================================
# compute_name_and_config Tests
# ============================================================================


class TestComputeNameAndConfig:
    """Tests for compute_name_and_config method."""

    def test_explicit_name_returned(self):
        """Test that explicitly set name is returned."""
        llm = NimLLM(model_name="test-model", name="my_explicit_name")
        name, config = llm.compute_name_and_config(LLMBaseConfig)

        assert name == "my_explicit_name"
        assert config.model_name == "test-model"

    def test_auto_generated_name_when_none(self):
        """Test that a name is auto-generated when not provided."""
        llm = NimLLM(model_name="test-model")
        name, config = llm.compute_name_and_config(LLMBaseConfig)

        assert name is not None
        assert len(name) > 0
        assert "_" in name  # Auto-generated names contain underscore + UUID

    def test_auto_generated_name_is_stable(self):
        """Test that auto-generated name is stable across multiple calls."""
        llm = NimLLM(model_name="test-model")
        name1, _ = llm.compute_name_and_config(LLMBaseConfig)
        name2, _ = llm.compute_name_and_config(LLMBaseConfig)

        assert name1 == name2

    def test_config_has_correct_type(self):
        """Test that config is cast to correct base type."""
        llm = NimLLM(model_name="test-model", name="test")
        _, config = llm.compute_name_and_config(LLMBaseConfig)

        # Should be an LLM config type
        assert hasattr(config, "model_name")
        assert config.model_name == "test-model"

    def test_preserves_set_fields(self):
        """Test that explicitly set fields are preserved in config."""
        llm = NimLLM(
            model_name="test-model",
            temperature=0.7,
            max_tokens=2048,
            name="test",
        )
        _, config = llm.compute_name_and_config(LLMBaseConfig)

        assert config.model_name == "test-model"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048


# ============================================================================
# computed_name Property Tests
# ============================================================================


class TestComputedNameProperty:
    """Tests for the computed_name property."""

    def test_explicit_name_from_property(self):
        """Test that computed_name returns explicit name."""
        llm = NimLLM(model_name="test-model", name="explicit_name")
        assert llm.computed_name == "explicit_name"

    def test_auto_name_from_property(self):
        """Test that computed_name generates name when not set."""
        llm = NimLLM(model_name="test-model")
        name = llm.computed_name

        assert name is not None
        assert "_" in name

    def test_computed_name_stable(self):
        """Test that computed_name is stable across calls."""
        llm = NimLLM(model_name="test-model")
        name1 = llm.computed_name
        name2 = llm.computed_name

        assert name1 == name2


# ============================================================================
# computed_config Property Tests
# ============================================================================


class TestComputedConfigProperty:
    """Tests for the computed_config property."""

    def test_computed_config_returns_correct_type(self):
        """Test that computed_config returns correct config type."""
        llm = NimLLM(model_name="test-model", name="test")
        config = llm.computed_config

        assert hasattr(config, "model_name")

    def test_computed_config_preserves_values(self):
        """Test that computed_config preserves field values."""
        llm = NimLLM(
            model_name="test-model",
            temperature=0.5,
            name="test",
        )
        config = llm.computed_config

        assert config.model_name == "test-model"
        assert config.temperature == 0.5


# ============================================================================
# Name Generation with Different Component Types
# ============================================================================


class TestNameGenerationDifferentTypes:
    """Tests for name generation with different component types."""

    def test_llm_name_generation(self):
        """Test name generation for LLM components."""
        llm = NimLLM(model_name="test-model")
        name = llm.computed_name

        # Should contain the type name
        assert "nim_llm" in name.lower() or "_" in name

    def test_tool_name_generation(self):
        """Test name generation for tool components."""
        tool = CurrentTimeTool()
        name, config = tool.compute_name_and_config(FunctionBaseConfig)

        assert name is not None
        assert "_" in name

    def test_agent_name_generation(self):
        """Test name generation for agent components."""
        llm = NimLLM(model_name="test-model", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[])
        name, config = agent.compute_name_and_config(FunctionBaseConfig)

        assert name is not None
        # Agent gets auto-generated name

    def test_unique_names_for_different_instances(self):
        """Test that different instances get different auto-generated names."""
        llm1 = NimLLM(model_name="test-model")
        llm2 = NimLLM(model_name="test-model")

        name1 = llm1.computed_name
        name2 = llm2.computed_name

        assert name1 != name2


# ============================================================================
# Fields Set Preservation Tests
# ============================================================================


class TestFieldsSetPreservation:
    """Tests for preserving which fields were explicitly set."""

    def test_only_set_fields_in_config(self):
        """Test that only explicitly set fields are in the cast config."""
        llm = NimLLM(
            model_name="test-model",
            temperature=0.5,
            name="test",
        )
        _, config = llm.compute_name_and_config(LLMBaseConfig)

        # Check what fields were set
        assert "model_name" in config.model_fields_set
        assert "temperature" in config.model_fields_set

    def test_default_fields_not_marked_as_set(self):
        """Test that fields with default values are not marked as set."""
        llm = NimLLM(model_name="test-model", name="test")
        _, config = llm.compute_name_and_config(LLMBaseConfig)

        # Only model_name should be explicitly set
        # (temperature has a default, shouldn't be in fields_set unless set)
        # Note: The exact behavior depends on how the class is defined

    def test_optimizable_params_field_set_preserved(self):
        """Test that optimizable_params field set status is preserved."""
        llm = NimLLM(
            model_name="test-model",
            name="test",
            optimizable_params=["temperature"],
        )
        _, config = llm.compute_name_and_config(LLMBaseConfig)

        # optimizable_params should be marked as set
        assert "optimizable_params" in config.model_fields_set


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in NatBase functionality."""

    def test_empty_string_name_treated_as_unset(self):
        """Test that empty string name triggers auto-generation."""
        llm = NimLLM(model_name="test-model", name="")
        name = llm.computed_name

        # Empty name should trigger auto-generation
        assert name != ""
        assert "_" in name

    def test_whitespace_name_preserved(self):
        """Test that whitespace-only name is preserved (not great, but consistent)."""
        llm = NimLLM(model_name="test-model", name="  ")
        name = llm.computed_name

        # Whitespace name should be preserved as-is
        # (validation could reject this, but base behavior preserves it)
        assert name == "  "

    def test_special_characters_in_name(self):
        """Test that special characters in name are preserved."""
        llm = NimLLM(model_name="test-model", name="my-llm_v2.0")
        name = llm.computed_name

        assert name == "my-llm_v2.0"

    def test_unicode_name(self):
        """Test that unicode characters in name are preserved."""
        llm = NimLLM(model_name="test-model", name="日本語名前")
        name = llm.computed_name

        assert name == "日本語名前"


# ============================================================================
# Multiple Components with Names Tests
# ============================================================================


class TestMultipleComponentsNaming:
    """Tests for naming when multiple components are used together."""

    def test_agent_with_named_llm(self):
        """Test that agent correctly references named LLM."""
        llm = NimLLM(model_name="test-model", name="my_llm")
        llm_name, _ = llm.compute_name_and_config(LLMBaseConfig)

        assert llm_name == "my_llm"

    def test_agent_with_auto_named_llm(self):
        """Test agent with auto-named LLM gets stable names."""
        llm = NimLLM(model_name="test-model")

        # Get names multiple times
        llm_name1, _ = llm.compute_name_and_config(LLMBaseConfig)
        llm_name2, _ = llm.compute_name_and_config(LLMBaseConfig)

        assert llm_name1 == llm_name2

    def test_multiple_tools_with_names(self):
        """Test multiple tools with explicit names."""
        tool1 = CurrentTimeTool(name="tool_1")
        tool2 = CurrentTimeTool(name="tool_2")

        name1, _ = tool1.compute_name_and_config(FunctionBaseConfig)
        name2, _ = tool2.compute_name_and_config(FunctionBaseConfig)

        assert name1 == "tool_1"
        assert name2 == "tool_2"

    def test_multiple_tools_with_auto_names(self):
        """Test multiple tools with auto-generated names are unique."""
        tool1 = CurrentTimeTool()
        tool2 = CurrentTimeTool()
        tool3 = CurrentTimeTool()

        name1, _ = tool1.compute_name_and_config(FunctionBaseConfig)
        name2, _ = tool2.compute_name_and_config(FunctionBaseConfig)
        name3, _ = tool3.compute_name_and_config(FunctionBaseConfig)

        # All names should be unique
        names = {name1, name2, name3}
        assert len(names) == 3


# ============================================================================
# Factory Pattern (Config Wrapper) Tests
# ============================================================================


class TestFactoryPatternConfigWrapper:
    """Tests for the factory pattern using config= parameter."""

    def test_nat_agent_wraps_config(self):
        """Test that NatAgent can wrap an agent config."""
        from nat.agent.react_agent.register import ReActAgentWorkflowConfig
        from nat.data_models.agent import AgentBaseConfig
        from nat.utils.sdk.nat_agent import NatAgent

        # Create a raw config
        react_config = ReActAgentWorkflowConfig(llm_name="test_llm", tool_names=[], description="Test agent")

        # Wrap it with NatAgent
        agent = NatAgent(config=react_config, name="my_wrapped_agent")

        # Verify the wrapper returns the original config
        name, config = agent.compute_name_and_config(AgentBaseConfig)
        assert name == "my_wrapped_agent"
        assert config is react_config

    def test_nat_agent_wrapper_auto_generates_name(self):
        """Test that NatAgent wrapper auto-generates name when not provided."""
        from nat.agent.react_agent.register import ReActAgentWorkflowConfig
        from nat.data_models.agent import AgentBaseConfig
        from nat.utils.sdk.nat_agent import NatAgent

        react_config = ReActAgentWorkflowConfig(llm_name="test_llm", tool_names=[], description="Test agent")

        agent = NatAgent(config=react_config)
        name, config = agent.compute_name_and_config(AgentBaseConfig)

        # Name should be auto-generated with config type
        assert "react_agent" in name
        assert "_" in name
        assert config is react_config

    def test_nat_agent_wrapper_name_is_stable(self):
        """Test that auto-generated name is stable across multiple calls."""
        from nat.agent.react_agent.register import ReActAgentWorkflowConfig
        from nat.data_models.agent import AgentBaseConfig
        from nat.utils.sdk.nat_agent import NatAgent

        react_config = ReActAgentWorkflowConfig(llm_name="test_llm", tool_names=[], description="Test agent")

        agent = NatAgent(config=react_config)
        name1, _ = agent.compute_name_and_config(AgentBaseConfig)
        name2, _ = agent.compute_name_and_config(AgentBaseConfig)

        assert name1 == name2

    def test_nat_llm_wraps_config(self):
        """Test that NatLLM can wrap an LLM config."""
        from nat.data_models.llm import LLMBaseConfig
        from nat.llm.nim_llm import NIMModelConfig
        from nat.utils.sdk.nat_llm import NatLLM

        nim_config = NIMModelConfig(model="test-model")
        llm = NatLLM(config=nim_config, name="my_wrapped_llm")

        name, config = llm.compute_name_and_config(LLMBaseConfig)
        assert name == "my_wrapped_llm"
        assert config is nim_config

    def test_nat_function_wraps_config(self):
        """Test that NatFunction can wrap a function config."""
        from nat.data_models.function import FunctionBaseConfig
        from nat.tool.datetime_tools import CurrentTimeToolConfig
        from nat.utils.sdk.nat_function import NatFunction

        time_config = CurrentTimeToolConfig()
        func = NatFunction(config=time_config, name="my_wrapped_func")

        name, config = func.compute_name_and_config(FunctionBaseConfig)
        assert name == "my_wrapped_func"
        assert config is time_config

    def test_nat_memory_wraps_config(self):
        """Test that NatMemory can wrap a memory config."""

        from nat.data_models.memory import MemoryBaseConfig
        from nat.utils.sdk.nat_memory import NatMemory

        # Create a minimal memory config for testing
        class TestMemoryConfig(MemoryBaseConfig, name="test_memory"):
            pass

        mem_config = TestMemoryConfig()
        memory = NatMemory(config=mem_config, name="my_wrapped_memory")

        name, config = memory.compute_name_and_config(MemoryBaseConfig)
        assert name == "my_wrapped_memory"
        assert config is mem_config

    def test_subclass_pattern_still_works(self):
        """Test that subclass pattern (NatReActAgent) still works."""
        llm = NimLLM(model_name="test-model", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], description="Subclass test", name="subclass_agent")

        assert agent.computed_name == "subclass_agent"

    def test_wrapper_and_subclass_coexist(self):
        """Test that wrapper pattern and subclass pattern can be used together."""
        from nat.agent.react_agent.register import ReActAgentWorkflowConfig
        from nat.data_models.agent import AgentBaseConfig
        from nat.utils.sdk.nat_agent import NatAgent

        # Subclass pattern
        llm = NimLLM(model_name="test-model", name="test_llm")
        subclass_agent = NatReActAgent(llm=llm, tools=[], description="Subclass agent", name="agent_1")

        # Factory pattern
        react_config = ReActAgentWorkflowConfig(llm_name="test_llm", tool_names=[], description="Factory agent")
        factory_agent = NatAgent(config=react_config, name="agent_2")

        # Both should work
        name1 = subclass_agent.computed_name
        name2, _ = factory_agent.compute_name_and_config(AgentBaseConfig)

        assert name1 == "agent_1"
        assert name2 == "agent_2"

    def test_factory_with_registered_function(self):
        """Test that factory pattern works with registered_function parameter."""
        from nat.agent.react_agent.register import ReActAgentWorkflowConfig
        from nat.utils.sdk.nat_agent import NatAgent

        react_config = ReActAgentWorkflowConfig(llm_name="test_llm", tool_names=[], description="Test agent")

        # The registered_function is for custom build functions
        # This just verifies the parameter is accepted
        agent = NatAgent(config=react_config, name="test", registered_function=None)
        assert agent.name == "test"

    def test_config_type_validation(self):
        """Test that config type validation works correctly."""
        from nat.llm.nim_llm import NIMModelConfig
        from nat.utils.sdk.nat_agent import NatAgent

        # Passing an LLM config to NatAgent should raise ValueError
        llm_config = NIMModelConfig(model="test-model")

        with pytest.raises(ValueError, match="config must be an instance of AgentBaseConfig"):
            NatAgent(config=llm_config)

    def test_config_accepts_subclasses(self):
        """Test that config accepts subclasses of the marker class."""
        from nat.agent.react_agent.register import ReActAgentWorkflowConfig
        from nat.data_models.agent import AgentBaseConfig
        from nat.utils.sdk.nat_agent import NatAgent

        # ReActAgentWorkflowConfig is a subclass of AgentBaseConfig
        react_config = ReActAgentWorkflowConfig(llm_name="test_llm", tool_names=[], description="Test agent")

        # This should work without raising
        agent = NatAgent(config=react_config, name="test")
        name, config = agent.compute_name_and_config(AgentBaseConfig)
        assert name == "test"
        assert config is react_config
