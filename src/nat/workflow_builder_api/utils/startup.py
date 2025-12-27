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
Startup utilities for the Workflow Builder API.

This module provides functions that need to run before uvloop starts,
such as pre-loading caches for libraries that can't be imported under uvloop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


def prewarm_caches() -> int:
    """
    Pre-warm all field option caches for registered component types.

    This discovers all registered types and calls any get_*_options methods
    they have to populate caches before uvloop starts. Some libraries (like
    ragas) can't be imported under uvloop, so this ensures their data is
    cached before the event loop starts.

    Should be called before uvicorn.run().

    Returns:
        Number of config types that had options pre-warmed.
    """
    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    # Load all plugins to ensure all types are registered
    discover_and_register_plugins(PluginTypes.ALL)

    registry = GlobalTypeRegistry.get()
    config_types: set[type[BaseModel]] = set()

    # Collect all registered config types
    registry_getters = [
        registry.get_registered_llm_providers,
        registry.get_registered_embedder_providers,
        registry.get_registered_functions,
        registry.get_registered_function_groups,
        registry.get_registered_retriever_providers,
        registry.get_registered_memorys,
        registry.get_registered_object_stores,
        registry.get_registered_auth_providers,
        registry.get_registered_middleware,
        registry.get_registered_front_ends,
        registry.get_registered_logging_method,
        registry.get_registered_telemetry_exporters,
        registry.get_registered_evaluators,
        registry.get_registered_trainers,
        registry.get_registered_trajectory_builders,
        registry.get_registered_trainer_adapters,
        registry.get_registered_ttc_strategies,
    ]

    for getter in registry_getters:
        try:
            for info in getter():
                if hasattr(info, 'config_type') and info.config_type is not None:
                    config_types.add(info.config_type)
        except Exception as e:
            logger.debug("Error getting registered types: %s", e)

    # Pre-warm each config type
    count = 0
    for config_type in config_types:
        if _prewarm_config_options(config_type):
            count += 1

    logger.info("Pre-warmed field options for %d config types", count)
    return count


def _prewarm_config_options(config_type: type[BaseModel]) -> bool:
    """
    Pre-warm a config type by calling all its get_*_options methods.

    This triggers any lazy-loaded caches in the config class.

    Args:
        config_type: The config class to pre-warm.

    Returns:
        True if any options were pre-warmed, False otherwise.
    """
    prewarmed = False

    # Find all get_*_options methods on the class
    for attr_name in dir(config_type):
        if not attr_name.startswith('get_') or not attr_name.endswith('_options'):
            continue

        method = getattr(config_type, attr_name, None)
        if method is None or not callable(method):
            continue

        try:
            result = method()
            if result and isinstance(result, list) and len(result) > 0:
                logger.debug(
                    "Pre-warmed %s.%s: %d options",
                    config_type.__name__,
                    attr_name,
                    len(result),
                )
                prewarmed = True
        except Exception as e:
            logger.debug("Could not pre-warm %s.%s: %s", config_type.__name__, attr_name, e)

    return prewarmed
