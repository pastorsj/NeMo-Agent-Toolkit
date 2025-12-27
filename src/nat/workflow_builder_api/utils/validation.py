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
Configuration validation logic.

Validates YAML configuration files against the NAT Config schema.
"""

import logging
import traceback
from typing import Any

import yaml
from pydantic import ValidationError

from nat.workflow_builder_api.models import ConfigValidationResponse

logger = logging.getLogger(__name__)


def validate_yaml_config(yaml_content: str) -> ConfigValidationResponse:
    """
    Validate a YAML configuration string against the NAT Config schema.

    Steps:
    1. Parse the YAML content
    2. Load all NAT plugins
    3. Validate against the Config schema

    Returns:
        ConfigValidationResponse with validation result and any error messages.
    """
    from nat.data_models.config import Config
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    from nat.utils.data_models.schema_validator import validate_schema
    from nat.utils.io.yaml_tools import yaml_loads

    # Step 1: Parse YAML
    config_dict: dict[str, Any] | None = None
    try:
        config_dict = yaml_loads(yaml_content)
    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML syntax: {str(e)}"
        return ConfigValidationResponse(
            valid=False,
            error_message=error_msg,
            error_details=[error_msg],
            config_dict=None,
        )
    except ValueError as e:
        return ConfigValidationResponse(
            valid=False,
            error_message=str(e),
            error_details=[str(e)],
            config_dict=None,
        )
    except Exception as e:
        logger.error("Unexpected error parsing YAML: %s", e)
        return ConfigValidationResponse(
            valid=False,
            error_message=f"Failed to parse YAML: {str(e)}",
            error_details=[str(e)],
            config_dict=None,
        )

    # Step 2: Ensure ALL plugins are loaded (including external packages like nvidia-nat-langchain)
    try:
        discover_and_register_plugins(PluginTypes.ALL)
    except Exception as e:
        logger.warning("Error loading plugins: %s", e)

    # Step 3: Validate against the Config schema
    try:
        validate_schema(config_dict, Config)
        return ConfigValidationResponse(
            valid=True,
            error_message=None,
            error_details=None,
            config_dict=config_dict,
        )
    except ValidationError as e:
        error_details = _extract_pydantic_errors(e)
        return ConfigValidationResponse(
            valid=False,
            error_message=f"Configuration validation failed with {len(error_details)} error(s)",
            error_details=error_details,
            config_dict=None,
        )
    except ValueError as e:
        # validate_schema wraps ValidationError in ValueError
        # Check if the cause is a ValidationError and extract details from it
        if e.__cause__ and isinstance(e.__cause__, ValidationError):
            error_details = _extract_pydantic_errors(e.__cause__)
            return ConfigValidationResponse(
                valid=False,
                error_message=f"Configuration validation failed with {len(error_details)} error(s)",
                error_details=error_details,
                config_dict=None,
            )
        # Otherwise treat as a generic ValueError
        error_msg = str(e)
        return ConfigValidationResponse(
            valid=False,
            error_message=f"Validation error: {error_msg}",
            error_details=[error_msg],
            config_dict=None,
        )
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error("Config validation error: %s\n%s", error_msg, tb)
        return ConfigValidationResponse(
            valid=False,
            error_message=f"Validation error: {error_msg}",
            error_details=[error_msg],
            config_dict=None,
        )


def _extract_pydantic_errors(e: ValidationError) -> list[str]:
    """Extract detailed error messages from a Pydantic ValidationError."""
    error_details = []
    for error in e.errors():
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Unknown error")
        error_type = error.get("type", "")

        # Check for missing plugin errors (union_tag_invalid means the type isn't registered)
        if error_type == "union_tag_invalid":
            ctx = error.get("ctx", {})
            unknown_tag = ctx.get("tag", "")
            if unknown_tag:
                msg = _format_missing_plugin_error(unknown_tag, loc)

        if loc:
            error_details.append(f"{loc}: {msg}")
        else:
            error_details.append(msg)
    return error_details


def _format_missing_plugin_error(unknown_tag: str, location: str) -> str:
    """Format a helpful error message for missing plugin types."""
    # Known plugins and their install commands
    plugin_hints: dict[str, str] = {
        "tavily_internet_search": "pip install nvidia-nat-langchain",
        "code_generation": "pip install nvidia-nat-langchain",
        "wikipedia_search": "pip install nvidia-nat-langchain",
        "langgraph": "pip install nvidia-nat-langchain",
    }

    install_cmd = plugin_hints.get(unknown_tag)
    if install_cmd:
        return (f"Unknown type '{unknown_tag}'. This type requires an additional plugin. "
                f"Install it with: {install_cmd}")

    # Check if it looks like a langchain tool
    if "langchain" in unknown_tag.lower() or unknown_tag in ["tavily", "arxiv", "pubmed", "duckduckgo"]:
        return (f"Unknown type '{unknown_tag}'. This appears to be a LangChain tool. "
                f"Install with: pip install nvidia-nat-langchain")

    return f"Unknown type '{unknown_tag}'. This type is not registered. Make sure the required plugin is installed."
