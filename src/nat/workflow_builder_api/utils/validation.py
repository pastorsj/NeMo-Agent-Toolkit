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
Supports configs with environment variables (${VAR_NAME} patterns).
"""

import logging
import traceback
from typing import Any

import yaml
from pydantic import ValidationError

from nat.workflow_builder_api.models import ConfigValidationResponse
from nat.workflow_builder_api.models import EnvironmentVariable
from nat.workflow_builder_api.models import EnvVarLocation as ModelEnvVarLocation
from nat.workflow_builder_api.utils.env_var_processor import enrich_env_vars_with_field_types
from nat.workflow_builder_api.utils.env_var_processor import is_sensitive_field
from nat.workflow_builder_api.utils.env_var_processor import preprocess_yaml_env_vars

logger = logging.getLogger(__name__)


class EnvVarAwareValidationResult:
    """Result of validation that includes environment variable information."""

    def __init__(
        self,
        response: ConfigValidationResponse,
        environment_variables: list[EnvironmentVariable] | None = None,
        original_config_dict: dict[str, Any] | None = None,
        env_var_warnings: list[str] | None = None,
    ):
        self.response = response
        self.environment_variables = environment_variables or []
        # Original config (before env var substitution) for later restoration
        self.original_config_dict = original_config_dict
        # Warnings about env vars used in non-secret fields
        self.env_var_warnings = env_var_warnings or []


def validate_yaml_config(yaml_content: str) -> ConfigValidationResponse:
    """
    Validate a YAML configuration string against the NAT Config schema.

    Note: This function does NOT handle environment variables. Use
    validate_yaml_config_with_env_vars() for configs that may contain
    ${VAR_NAME} patterns.

    Steps:
    1. Parse the YAML content
    2. Load all NAT plugins
    3. Validate against the Config schema

    Returns:
        ConfigValidationResponse with validation result and any error messages.
    """
    result = validate_yaml_config_with_env_vars(yaml_content)
    return result.response


def validate_yaml_config_with_env_vars(yaml_content: str) -> EnvVarAwareValidationResult:
    """
    Validate a YAML configuration string, handling environment variables.

    This function:
    1. Preprocesses the raw YAML to replace ${VAR_NAME} with placeholders
    2. Parses the preprocessed YAML
    3. Validates the processed config
    4. Returns both validation result and discovered env vars

    The preprocessing step is critical because NAT's yaml_loads() resolves
    environment variables to their values (or None if not set), which would
    cause validation to fail for configs with unset env vars.

    Returns:
        EnvVarAwareValidationResult with validation response, env vars, and original config.
    """
    from nat.data_models.config import Config
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    from nat.utils.data_models.schema_validator import validate_schema
    from nat.utils.io.yaml_tools import yaml_loads

    # Step 1: Preprocess YAML to replace ${VAR_NAME} with placeholders
    # This must happen BEFORE yaml_loads because it resolves env vars
    preprocess_result = preprocess_yaml_env_vars(yaml_content)
    preprocessed_yaml = preprocess_result.processed_yaml

    # Build environment variables list from preprocessing with location info
    env_var_map: dict[str, EnvironmentVariable] = {}
    for occurrence in preprocess_result.occurrences:
        var_name = occurrence.var_name
        # Build the path from key_path and field_name
        path = ".".join(occurrence.key_path +
                        [occurrence.field_name]) if occurrence.field_name else ".".join(occurrence.key_path)

        # Determine component type and id from key_path
        component_type = occurrence.key_path[0] if len(occurrence.key_path) > 0 else None
        component_id = occurrence.key_path[1] if len(occurrence.key_path) > 1 else None

        location = ModelEnvVarLocation(
            path=path,
            component_type=component_type,
            component_id=component_id,
            field_name=occurrence.field_name,
            is_secret_field=False,  # Will be enriched later
        )

        if var_name not in env_var_map:
            env_var_map[var_name] = EnvironmentVariable(
                name=var_name,
                locations=[location],
                value=None,
                is_sensitive=is_sensitive_field(occurrence.field_name, path),
                export_as_variable=True,
                original_reference=f"${{{var_name}}}",
            )
        else:
            # Add location if not already present
            existing_paths = [loc.path for loc in env_var_map[var_name].locations]
            if path not in existing_paths:
                env_var_map[var_name].locations.append(location)

    environment_variables: list[EnvironmentVariable] = list(env_var_map.values())

    if environment_variables:
        logger.info(
            "Detected %d environment variables in config: %s",
            len(environment_variables),
            [v.name for v in environment_variables],
        )

    # Step 2: Parse the preprocessed YAML
    processed_config: dict[str, Any] | None = None
    try:
        processed_config = yaml_loads(preprocessed_yaml)
    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML syntax: {str(e)}"
        return EnvVarAwareValidationResult(
            response=ConfigValidationResponse(
                valid=False,
                error_message=error_msg,
                error_details=[error_msg],
                config_dict=None,
            ),
            environment_variables=environment_variables,
        )
    except ValueError as e:
        return EnvVarAwareValidationResult(
            response=ConfigValidationResponse(
                valid=False,
                error_message=str(e),
                error_details=[str(e)],
                config_dict=None,
            ),
            environment_variables=environment_variables,
        )
    except Exception as e:
        logger.error("Unexpected error parsing YAML: %s", e)
        return EnvVarAwareValidationResult(
            response=ConfigValidationResponse(
                valid=False,
                error_message=f"Failed to parse YAML: {str(e)}",
                error_details=[str(e)],
                config_dict=None,
            ),
            environment_variables=environment_variables,
        )

    # Step 3: Ensure ALL plugins are loaded (before enrichment)
    # Note: We now get location info directly from preprocessing
    try:
        discover_and_register_plugins(PluginTypes.ALL)
    except Exception as e:
        logger.warning("Error loading plugins: %s", e)

    # Step 5: Validate against the Config schema (using processed config with placeholders)
    try:
        validate_schema(processed_config, Config)

        # Step 6: Enrich env vars with field type information and generate warnings
        env_var_warnings: list[str] = []
        if environment_variables:
            enriched_vars, env_var_warnings = enrich_env_vars_with_field_types(environment_variables, processed_config)
            environment_variables = enriched_vars

            if env_var_warnings:
                logger.warning("")
                logger.warning("=" * 80)
                logger.warning("ENVIRONMENT VARIABLE WARNINGS")
                logger.warning("=" * 80)
                for i, warning in enumerate(env_var_warnings, 1):
                    logger.warning("")
                    logger.warning("  [%d] %s", i, warning)
                logger.warning("")
                logger.warning("=" * 80)

        return EnvVarAwareValidationResult(
            response=ConfigValidationResponse(
                valid=True,
                error_message=None,
                error_details=None,
                config_dict=processed_config,  # Return processed config for parsing
            ),
            environment_variables=environment_variables,
            original_config_dict=None,  # We don't need the original since we preprocess
            env_var_warnings=env_var_warnings,
        )
    except ValidationError as e:
        error_details = _extract_pydantic_errors(e)
        return EnvVarAwareValidationResult(
            response=ConfigValidationResponse(
                valid=False,
                error_message=f"Configuration validation failed with {len(error_details)} error(s)",
                error_details=error_details,
                config_dict=None,
            ),
            environment_variables=environment_variables,
        )
    except ValueError as e:
        # validate_schema wraps ValidationError in ValueError
        if e.__cause__ and isinstance(e.__cause__, ValidationError):
            error_details = _extract_pydantic_errors(e.__cause__)
            return EnvVarAwareValidationResult(
                response=ConfigValidationResponse(
                    valid=False,
                    error_message=f"Configuration validation failed with {len(error_details)} error(s)",
                    error_details=error_details,
                    config_dict=None,
                ),
                environment_variables=environment_variables,
            )
        error_msg = str(e)
        return EnvVarAwareValidationResult(
            response=ConfigValidationResponse(
                valid=False,
                error_message=f"Validation error: {error_msg}",
                error_details=[error_msg],
                config_dict=None,
            ),
            environment_variables=environment_variables,
        )
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error("Config validation error: %s\n%s", error_msg, tb)
        return EnvVarAwareValidationResult(
            response=ConfigValidationResponse(
                valid=False,
                error_message=f"Validation error: {error_msg}",
                error_details=[error_msg],
                config_dict=None,
            ),
            environment_variables=environment_variables,
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


def _format_missing_plugin_error(unknown_tag: str, _location: str) -> str:
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
