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
"""Tests for component type handling in the workflow builder."""

import pytest

from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.llm import LLMBaseConfig
from nat.utils.sdk.nat_llm import NatLLM
from nat.workflow_builder_api.constants import CATEGORY_TO_CONFIG_SECTION
from nat.workflow_builder_api.constants import CATEGORY_TO_REF_TYPE
from nat.workflow_builder_api.constants import SDK_CLASS_TO_REF_TYPE
from nat.workflow_builder_api.constants import UI_CATEGORIES
from nat.workflow_builder_api.models import ComponentCategory
from nat.workflow_builder_api.models import RefType
from nat.workflow_builder_api.utils.config_parser import BASE_CLASS_TO_COMPONENT_TYPE
from nat.workflow_builder_api.utils.schema import get_display_name_from_docstring
from nat.workflow_builder_api.utils.schema import get_icon_url_from_docstring
from nat.workflow_builder_api.utils.type_builder import get_category_types


class TestComponentCategoryMapping:
    """Tests for component category to config section mapping."""

    def test_category_to_section_mapping_exists(self, set_test_env_vars):
        """Test that all categories have config section mappings."""
        # Check that key categories are mapped
        assert ComponentCategory.LLM in CATEGORY_TO_CONFIG_SECTION
        assert ComponentCategory.EMBEDDER in CATEGORY_TO_CONFIG_SECTION
        assert ComponentCategory.AGENT in CATEGORY_TO_CONFIG_SECTION
        assert ComponentCategory.FUNCTION in CATEGORY_TO_CONFIG_SECTION
        assert ComponentCategory.RETRIEVER in CATEGORY_TO_CONFIG_SECTION
        assert ComponentCategory.EVALUATOR in CATEGORY_TO_CONFIG_SECTION

    def test_category_section_names_are_valid(self, set_test_env_vars):
        """Test that section names are valid strings."""
        for category, section in CATEGORY_TO_CONFIG_SECTION.items():
            assert isinstance(section, str)
            assert len(section) > 0

    def test_llm_maps_to_llms_section(self, set_test_env_vars):
        """Test that LLM category maps to 'llms' section."""
        assert CATEGORY_TO_CONFIG_SECTION[ComponentCategory.LLM] == "llms"

    def test_agent_maps_to_functions_section(self, set_test_env_vars):
        """Test that AGENT category maps to 'functions' section."""
        assert CATEGORY_TO_CONFIG_SECTION[ComponentCategory.AGENT] == "functions"

    def test_evaluator_maps_to_evaluators_section(self, set_test_env_vars):
        """Test that EVALUATOR category maps to 'evaluators' section."""
        assert CATEGORY_TO_CONFIG_SECTION[ComponentCategory.EVALUATOR] == "evaluators"


class TestTypeRegistry:
    """Tests for type registry component discovery."""

    def test_registry_has_llms(self, set_test_env_vars):
        """Test that registry has LLM types registered."""
        registry = GlobalTypeRegistry.get()
        llms = registry.get_registered_llm_providers()

        assert len(llms) > 0

        # Should have NIM LLM
        local_names = {llm.local_name for llm in llms}
        assert "nim" in local_names

    def test_registry_has_agents(self, set_test_env_vars):
        """Test that registry has agent types registered."""
        registry = GlobalTypeRegistry.get()
        # Agents are functions in the registry
        functions = registry.get_registered_functions()

        # Find functions that are agents (have 'agent' in local_name)
        agent_names = [f.local_name for f in functions if "agent" in f.local_name.lower()]
        assert len(agent_names) > 0

        # Should have react agent
        assert "react_agent" in agent_names

    def test_registry_has_functions(self, set_test_env_vars):
        """Test that registry has function types registered."""
        registry = GlobalTypeRegistry.get()
        functions = registry.get_registered_functions()

        assert len(functions) > 0

    def test_registry_has_retrievers(self, set_test_env_vars):
        """Test that registry has retriever types registered."""
        registry = GlobalTypeRegistry.get()
        retrievers = registry.get_registered_retriever_providers()

        assert len(retrievers) > 0

    def test_registry_has_embedders(self, set_test_env_vars):
        """Test that registry has embedder types registered."""
        registry = GlobalTypeRegistry.get()
        embedders = registry.get_registered_embedder_providers()

        assert len(embedders) > 0

    def test_registry_has_evaluators(self, set_test_env_vars):
        """Test that registry has evaluator types registered."""
        registry = GlobalTypeRegistry.get()
        evaluators = registry.get_registered_evaluators()

        assert len(evaluators) > 0


class TestBaseClassMapping:
    """Tests for base class to component type mapping."""

    def test_agent_base_config_maps_to_agent(self, set_test_env_vars):
        """Test that AgentBaseConfig subclasses map to 'agent' type."""
        # Find mapping for AgentBaseConfig
        for base_class, component_type in BASE_CLASS_TO_COMPONENT_TYPE:
            if base_class == AgentBaseConfig:
                assert component_type == "agent"
                break
        else:
            pytest.fail("AgentBaseConfig not in mapping")

    def test_llm_base_config_maps_to_llm(self, set_test_env_vars):
        """Test that LLMBaseConfig subclasses map to 'llm' type."""
        # Find mapping for LLMBaseConfig
        for base_class, component_type in BASE_CLASS_TO_COMPONENT_TYPE:
            if base_class == LLMBaseConfig:
                assert component_type == "llm"
                break
        else:
            pytest.fail("LLMBaseConfig not in mapping")

    def test_function_base_config_maps_to_function(self, set_test_env_vars):
        """Test that FunctionBaseConfig subclasses map to 'function' type."""
        # Find mapping for FunctionBaseConfig
        for base_class, component_type in BASE_CLASS_TO_COMPONENT_TYPE:
            if base_class == FunctionBaseConfig:
                assert component_type == "function"
                break
        else:
            pytest.fail("FunctionBaseConfig not in mapping")


class TestComponentTypeInfo:
    """Tests for component type info building."""

    def test_build_llm_type_info(self, set_test_env_vars):
        """Test building type info for LLMs."""
        type_info = get_category_types(ComponentCategory.LLM)

        assert type_info is not None
        assert type_info.category == ComponentCategory.LLM
        assert len(type_info.registered_types) > 0

    def test_build_agent_type_info(self, set_test_env_vars):
        """Test building type info for agents."""
        type_info = get_category_types(ComponentCategory.AGENT)

        assert type_info is not None
        assert type_info.category == ComponentCategory.AGENT
        assert len(type_info.registered_types) > 0

    def test_build_function_type_info(self, set_test_env_vars):
        """Test building type info for functions."""
        type_info = get_category_types(ComponentCategory.FUNCTION)

        assert type_info is not None
        assert type_info.category == ComponentCategory.FUNCTION
        assert len(type_info.registered_types) > 0

    def test_build_evaluator_type_info(self, set_test_env_vars):
        """Test building type info for evaluators."""
        type_info = get_category_types(ComponentCategory.EVALUATOR)

        assert type_info is not None
        assert type_info.category == ComponentCategory.EVALUATOR


class TestRegisteredTypeInfo:
    """Tests for individual registered type info."""

    def test_registered_type_has_full_type(self, set_test_env_vars):
        """Test that registered types have full_type."""
        type_info = get_category_types(ComponentCategory.LLM)

        for reg_type in type_info.registered_types:
            assert reg_type.full_type is not None
            assert len(reg_type.full_type) > 0

    def test_registered_type_has_display_name(self, set_test_env_vars):
        """Test that registered types have display_name."""
        type_info = get_category_types(ComponentCategory.LLM)

        for reg_type in type_info.registered_types:
            assert reg_type.display_name is not None
            assert len(reg_type.display_name) > 0


class TestRefTypeMapping:
    """Tests for RefType mappings."""

    def test_category_to_ref_type_exists(self, set_test_env_vars):
        """Test that categories have RefType mappings."""
        # Check key categories
        assert ComponentCategory.LLM in CATEGORY_TO_REF_TYPE
        assert CATEGORY_TO_REF_TYPE[ComponentCategory.LLM] == RefType.LLM

        assert ComponentCategory.EMBEDDER in CATEGORY_TO_REF_TYPE
        assert CATEGORY_TO_REF_TYPE[ComponentCategory.EMBEDDER] == RefType.EMBEDDER

    def test_sdk_class_to_ref_type_exists(self, set_test_env_vars):
        """Test that SDK classes have RefType mappings."""
        assert NatLLM in SDK_CLASS_TO_REF_TYPE
        assert SDK_CLASS_TO_REF_TYPE[NatLLM] == RefType.LLM


class TestUICategories:
    """Tests for UI category configuration."""

    def test_ui_categories_include_core_types(self, set_test_env_vars):
        """Test that UI categories include core component types."""
        # Core types should be in UI
        assert ComponentCategory.LLM in UI_CATEGORIES
        assert ComponentCategory.EMBEDDER in UI_CATEGORIES
        assert ComponentCategory.AGENT in UI_CATEGORIES
        assert ComponentCategory.FUNCTION in UI_CATEGORIES
        assert ComponentCategory.RETRIEVER in UI_CATEGORIES

    def test_ui_categories_include_observability(self, set_test_env_vars):
        """Test that UI categories include observability types."""
        assert ComponentCategory.LOGGER in UI_CATEGORIES
        assert ComponentCategory.TELEMETRY_EXPORTER in UI_CATEGORIES

    def test_ui_categories_include_evaluation(self, set_test_env_vars):
        """Test that UI categories include evaluation types."""
        assert ComponentCategory.EVALUATOR in UI_CATEGORIES


class TestComponentDisplayNames:
    """Tests for component display name extraction."""

    def test_extract_display_name_from_docstring(self, set_test_env_vars):
        """Test extracting display name from docstring."""
        from pydantic import BaseModel

        class MockComponent(BaseModel):
            """Test component.

            ## Details
            Name: My Custom Component
            ![Icon](/icons/custom.svg)
            """

        display_name = get_display_name_from_docstring(MockComponent)
        assert display_name == "My Custom Component"

    def test_extract_display_name_without_details(self, set_test_env_vars):
        """Test display name extraction when no Details section."""
        from pydantic import BaseModel

        class SimpleComponent(BaseModel):
            """A simple component without details."""

        display_name = get_display_name_from_docstring(SimpleComponent)
        assert display_name is None

    def test_extract_icon_from_docstring(self, set_test_env_vars):
        """Test extracting icon URL from docstring."""
        from pydantic import BaseModel

        class IconComponent(BaseModel):
            """Test component.

            ## Details
            Name: My Component
            ![Icon](/icons/test.svg)
            """

        icon_url = get_icon_url_from_docstring(IconComponent)
        assert icon_url == "/icons/test.svg"
