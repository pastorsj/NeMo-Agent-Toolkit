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
"""
Constants and configuration for the Workflow Builder API.

This package contains:
- categories: UI categories, metadata, and SDK class mappings
- ref_types: RefType mappings and type detection utilities
"""

from nat.workflow_builder_api.constants.categories import CATEGORY_METADATA
from nat.workflow_builder_api.constants.categories import CATEGORY_TO_CONFIG_SECTION
from nat.workflow_builder_api.constants.categories import CATEGORY_TO_REF_TYPE
from nat.workflow_builder_api.constants.categories import SDK_CLASS_METADATA
from nat.workflow_builder_api.constants.categories import UI_CATEGORIES
from nat.workflow_builder_api.constants.ref_types import COMPONENT_REF_TO_REF_TYPE
from nat.workflow_builder_api.constants.ref_types import SDK_CLASS_TO_REF_TYPE
from nat.workflow_builder_api.constants.ref_types import get_all_component_ref_types
from nat.workflow_builder_api.constants.ref_types import get_type_name
from nat.workflow_builder_api.constants.ref_types import is_component_ref_type

__all__ = [
    # Categories
    "UI_CATEGORIES",
    "CATEGORY_TO_REF_TYPE",
    "CATEGORY_TO_CONFIG_SECTION",
    "CATEGORY_METADATA",
    "SDK_CLASS_METADATA",  # Ref types
    "COMPONENT_REF_TO_REF_TYPE",
    "SDK_CLASS_TO_REF_TYPE",
    "get_type_name",
    "get_all_component_ref_types",
    "is_component_ref_type",
]
