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
Component metadata registries for field options.

This module provides mappings for:
- Field options (suggested dropdown values for common fields)

Note: Component icons are now defined in each component's docstring using
markdown image syntax: ![Icon](https://cdn.simpleicons.org/provider/COLOR)
"""

import re
from typing import Any

# =============================================================================
# FIELD OPTIONS REGISTRY
# =============================================================================
# Maps (module_pattern, field_name) to suggested dropdown options
# These are suggestions, not strict enums - users can still enter custom values

FIELD_OPTIONS_REGISTRY: list[tuple[str, str, list[str]]] = [
    # OpenAI models
    (r"openai",
     "model_name",
     [
         "gpt-4.1",
         "gpt-4.1-mini",
         "gpt-4.1-nano",
         "gpt-4o",
         "gpt-4o-mini",
         "gpt-4-turbo",
         "gpt-4",
         "gpt-3.5-turbo",
         "o1",
         "o1-mini",
         "o1-preview",
         "o3",
         "o3-mini",
     ]),

    # Anthropic models
    (r"anthropic|claude",
     "model_name",
     [
         "claude-sonnet-4-20250514",
         "claude-opus-4-20250514",
         "claude-3-7-sonnet-20250219",
         "claude-3-5-sonnet-20241022",
         "claude-3-5-haiku-20241022",
         "claude-3-opus-20240229",
         "claude-3-sonnet-20240229",
         "claude-3-haiku-20240307",
     ]),

    # NVIDIA NIM models
    (r"nim|nvidia",
     "model_name",
     [
         "meta/llama-3.3-70b-instruct",
         "meta/llama-3.1-405b-instruct",
         "meta/llama-3.1-70b-instruct",
         "meta/llama-3.1-8b-instruct",
         "nvidia/llama-3.1-nemotron-70b-instruct",
         "nvidia/llama-3.1-nemotron-ultra-253b-v1",
         "nvidia/nemotron-4-340b-instruct",
         "google/gemma-2-27b-it",
         "mistralai/mixtral-8x22b-instruct-v0.1",
         "mistralai/mistral-large-2-instruct",
         "deepseek-ai/deepseek-r1",
         "qwen/qwen2.5-72b-instruct",
     ]),

    # Google models
    (r"google|gemini",
     "model_name",
     [
         "gemini-2.5-pro-preview-06-05",
         "gemini-2.5-flash-preview-05-20",
         "gemini-2.0-flash",
         "gemini-2.0-flash-lite",
         "gemini-1.5-pro",
         "gemini-1.5-flash",
         "gemini-1.5-flash-8b",
     ]),

    # Mistral models
    (r"mistral",
     "model_name",
     [
         "mistral-large-latest",
         "mistral-medium-latest",
         "mistral-small-latest",
         "codestral-latest",
         "ministral-8b-latest",
         "ministral-3b-latest",
         "pixtral-large-latest",
     ]),

    # Groq models
    (r"groq",
     "model_name",
     [
         "llama-3.3-70b-versatile",
         "llama-3.1-8b-instant",
         "llama-3.2-90b-vision-preview",
         "mixtral-8x7b-32768",
         "gemma2-9b-it",
     ]),

    # OpenAI embedding models
    (r"openai", "embedding_model", [
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ]),

    # NVIDIA embedding models
    (r"nim|nvidia",
     "embedding_model", [
         "nvidia/nv-embedqa-e5-v5",
         "nvidia/nv-embedqa-mistral-7b-v2",
         "nvidia/nv-embed-v1",
         "nvidia/embed-qa-4",
     ]),
]


def get_field_options(module_name: str, field_name: str) -> list[str] | None:
    """
    Get suggested options for a field.

    Args:
        module_name: The module path of the component
        field_name: The name of the field

    Returns:
        List of suggested values if found, None otherwise.
    """
    search_text = module_name.lower()

    for pattern, target_field, options in FIELD_OPTIONS_REGISTRY:
        if target_field == field_name and re.search(pattern, search_text, re.IGNORECASE):
            return options

    return None


def apply_field_options(fields: list[Any], module_name: str) -> None:
    """
    Apply field options to a list of FieldInfo objects in-place.

    Args:
        fields: List of FieldInfo objects to update
        module_name: The module path for looking up options
    """
    for field in fields:
        options = get_field_options(module_name, field.name)
        if options and not field.enum:  # Don't override strict enums
            field.options = options
