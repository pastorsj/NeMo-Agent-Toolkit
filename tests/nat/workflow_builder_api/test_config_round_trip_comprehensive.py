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
Comprehensive round-trip tests for ALL example configurations.

This module discovers all YAML config files in the examples directory
and runs round-trip tests on each one to ensure data integrity.

Field alias handling:
- Pydantic models use validation_alias (AliasChoices) to accept multiple field names
- During import, Pydantic normalizes aliases to canonical field names
- During export, we output canonical field names
- During comparison, we dynamically discover all aliases from registered types
  and treat them as equivalent
"""

import os
import sys
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get project root (assuming tests/nat/workflow_builder_api/test_*.py structure)
# Go up: test file -> workflow_builder_api -> nat -> tests -> project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src is in path for imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Known fields that NAT's schema doesn't support (silently ignored during validation)
# These fields are in example configs but not in the Pydantic models
IGNORED_FIELDS = {
    "general.telemetry.enabled",  # TelemetryConfig doesn't have this field
}

# Cache for dynamically discovered field aliases
_FIELD_ALIASES_CACHE: dict[str, str] | None = None


def _discover_all_field_aliases() -> dict[str, str]:
    """
    Dynamically discover all field aliases from registered Pydantic models.

    Scans all registered types in the GlobalTypeRegistry and extracts
    validation_alias (AliasChoices) information from their fields.

    Returns a bidirectional mapping where each alias maps to the canonical
    field name and vice versa.

    Returns:
        Dictionary mapping field names/aliases to their equivalents.
    """
    from pydantic import AliasChoices

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    # Ensure plugins are loaded
    discover_and_register_plugins(PluginTypes.ALL)

    aliases: dict[str, str] = {}
    registry = GlobalTypeRegistry.get()

    # Collect all config types from all registered components
    config_types: list[type] = []

    # Get config_type from all registry getters
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
        registry.get_registered_ttc_strategies,
        registry.get_registered_trainers,
        registry.get_registered_trajectory_builders,
        registry.get_registered_trainer_adapters,
    ]

    for getter in registry_getters:
        try:
            registered_types = getter()
            for registered in registered_types:
                if hasattr(registered, "config_type"):
                    config_types.append(registered.config_type)
        except Exception:
            continue

    # Also include core config models that aren't in the registry
    try:
        from nat.data_models.finetuning import FinetuneConfig
        config_types.append(FinetuneConfig)
    except ImportError:
        pass

    # Extract aliases from all config types
    seen_types: set[type] = set()
    for config_type in config_types:
        if config_type in seen_types:
            continue
        seen_types.add(config_type)

        if not hasattr(config_type, "model_fields"):
            continue

        for field_name, field_info in config_type.model_fields.items():
            if field_info.validation_alias and isinstance(field_info.validation_alias, AliasChoices):
                # AliasChoices contains a list of valid names for this field
                # Choices can be strings or AliasPath objects - only use strings
                choices = field_info.validation_alias.choices
                # Create bidirectional mappings between all string choices
                for choice in choices:
                    if isinstance(choice, str) and choice != field_name:
                        # Map alias -> canonical name
                        aliases[choice] = field_name
                        # Map canonical name -> alias (for reverse lookup)
                        aliases[field_name] = choice

    return aliases


def get_field_aliases() -> dict[str, str]:
    """
    Get the field alias mapping, discovering it lazily on first call.

    Returns:
        Dictionary mapping field names/aliases to their equivalents.
    """
    global _FIELD_ALIASES_CACHE  # noqa: PLW0603
    if _FIELD_ALIASES_CACHE is None:
        _FIELD_ALIASES_CACHE = _discover_all_field_aliases()
    return _FIELD_ALIASES_CACHE


# Known configs that have issues and should be skipped (with reason)
SKIP_CONFIGS = {
    # Config inheritance uses $extends which isn't standard YAML
    "config-with-tracing.yml": "Uses $extends config inheritance",
    "config-debug.yml": "Uses $extends config inheritance",
    "config-high-temp-debug.yml": "Uses $extends config inheritance",
    "config-different-model.yml": "Uses $extends config inheritance",
    "config-high-temp.yml": "Uses $extends config inheritance",
    "base-config.yml": "Base config template, not standalone",  # Optimizer results are generated configs
    "optimized_config.yml": "Auto-generated optimizer output",
    "config_numeric_trial_0.yml": "Auto-generated optimizer output",
    "config_numeric_trial_1.yml": "Auto-generated optimizer output",
    "config_numeric_trial_2.yml": "Auto-generated optimizer output",
    "config_numeric_trial_3.yml": "Auto-generated optimizer output",
    "config_numeric_trial_4.yml": "Auto-generated optimizer output",
    # Configs requiring plugins not typically installed in test environment
    "config-mcp-service-account-jama.yml": "Requires MCP service account plugin",
    "config-mcp-service-account-jira-function.yml": "Requires MCP service account plugin",
    "config-mcp-service-account-jira.yml": "Requires MCP service account plugin",
    "config-mcp-auth-jira.yml": "Requires MCP auth plugin",
}

# =============================================================================
# NORMALIZATION & COMPARISON UTILITIES
# =============================================================================


def normalize_value(value: Any, key: str | None = None) -> Any:
    """
    Normalize a config value for comparison.

    Handles:
    - Int vs float equivalence (0 == 0.0)
    - _type field normalization (extract short name from full path)
    - Path normalization (strip trailing slashes)
    - Float precision rounding
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: normalize_value(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, str):
        # Normalize _type fields (extract short name from full path)
        if key == "_type" and "/" in value:
            return value.split("/")[-1]
        # Normalize paths (strip trailing slashes and leading ./)
        if "/" in value or "\\" in value:
            normalized = value.rstrip("/\\")
            if normalized.startswith("./"):
                normalized = normalized[2:]
            return normalized
        # Normalize multiline strings (strip trailing/leading whitespace)
        return value.strip()
    return value


def compare_dicts(
    original: dict,
    exported: dict,
    path: str = "",
    ignored_fields: set[str] | None = None,
    field_aliases: dict[str, str] | None = None,
) -> list[str]:
    """
    Deep compare two dictionaries, returning list of differences.

    Handles field aliases dynamically discovered from Pydantic models.
    When a key is missing in exported, checks if it exists under an alias.

    Args:
        original: The original config dict
        exported: The exported config dict
        path: Current path for error messages
        ignored_fields: Set of field paths to ignore
        field_aliases: Dictionary mapping field names to their aliases (discovered from Pydantic)

    Returns:
        List of difference descriptions
    """
    if ignored_fields is None:
        ignored_fields = IGNORED_FIELDS

    if field_aliases is None:
        field_aliases = get_field_aliases()

    # Check if this path should be ignored
    if path in ignored_fields:
        return []

    errors = []
    orig_norm = normalize_value(original, path.split(".")[-1] if "." in path else path)
    exp_norm = normalize_value(exported, path.split(".")[-1] if "." in path else path)

    # Type mismatch (after normalization)
    if type(orig_norm) is not type(exp_norm):
        return [f"{path}: type {type(original).__name__} vs {type(exported).__name__}"]

    if isinstance(original, dict):
        # Track keys that have been matched (including via aliases)
        matched_exported_keys = set()

        # Check all original keys exist in exported (or under an alias)
        for key, orig_value in original.items():
            new_path = f"{path}.{key}" if path else key
            if new_path in ignored_fields:
                continue

            if key in exported:
                matched_exported_keys.add(key)
                errors.extend(compare_dicts(orig_value, exported[key], new_path, ignored_fields, field_aliases))
            elif key in field_aliases and field_aliases[key] in exported:
                # Found under alias - compare values
                alias_key = field_aliases[key]
                matched_exported_keys.add(alias_key)
                errors.extend(compare_dicts(orig_value, exported[alias_key], new_path, ignored_fields, field_aliases))
            else:
                errors.append(f"{new_path}: missing in exported")

        # Check for unexpected keys in exported (excluding matched keys and their aliases)
        for key in exported.keys():
            new_path = f"{path}.{key}" if path else key
            if key not in matched_exported_keys:
                # Check if this key is an alias for something in original
                if key in field_aliases and field_aliases[key] in original:
                    continue  # This was already matched via alias
                if key not in original:
                    errors.append(f"{new_path}: unexpected in exported")

    elif isinstance(original, list):
        if len(original) != len(exported):
            errors.append(f"{path}: list length {len(original)} vs {len(exported)}")
        else:
            for i, (o, e) in enumerate(zip(original, exported)):
                errors.extend(compare_dicts(o, e, f"{path}[{i}]", ignored_fields, field_aliases))
    elif orig_norm != exp_norm:
        errors.append(f"{path}: {original} vs {exported}")

    return errors


# =============================================================================
# ROUND-TRIP TEST LOGIC
# =============================================================================


def run_round_trip(config_path: Path) -> tuple[bool, list[str], str | None]:
    """
    Run a full round-trip test on a config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Tuple of (success, errors, exception_message)
    """
    # Lazy imports to avoid import-time issues
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    from nat.workflow_builder_api.models import ExportComponent
    from nat.workflow_builder_api.models import ExportConnection
    from nat.workflow_builder_api.models import ExportWorkflowRequest
    from nat.workflow_builder_api.utils.config_exporter import export_workflow_to_yaml
    from nat.workflow_builder_api.utils.config_parser import parse_config_to_workflow_state

    try:
        # Load original YAML
        with open(config_path, encoding="utf-8") as f:
            original = yaml.safe_load(f)

        if original is None:
            return True, [], None  # Empty file is valid

        # Ensure plugins are loaded
        discover_and_register_plugins(PluginTypes.ALL)

        # Import: parse config to workflow state
        workflow_state = parse_config_to_workflow_state(original)

        # Export: convert back to config
        components = [
            ExportComponent(
                id=c.id,
                component_type=c.component_type,
                full_type=c.full_type,
                config=c.config or {},
            ) for c in workflow_state.components
        ]
        connections = [
            ExportConnection(
                source_id=c.source_id,
                target_id=c.target_id,
                target_field=c.target_field,
            ) for c in workflow_state.connections
        ]

        export_request = ExportWorkflowRequest(components=components, connections=connections)
        export_result = export_workflow_to_yaml(export_request)

        if not export_result.success:
            return False, [export_result.error_message or "Export failed"], None

        # Parse exported YAML
        if export_result.yaml_content is None:
            return False, ["Export returned empty content"], None
        exported = yaml.safe_load(export_result.yaml_content)

        # Compare
        errors = compare_dicts(original, exported)

        return len(errors) == 0, errors, None

    except Exception as e:
        return False, [], str(e)


def discover_config_files(examples_dir: Path) -> list[Path]:
    """
    Discover all YAML config files in the examples directory.

    Returns:
        List of paths to config files
    """
    config_files = []

    for pattern in ["**/*.yml", "**/*.yaml"]:
        for path in examples_dir.glob(pattern):
            # Only include files in config-related directories
            if "config" in str(path).lower():
                config_files.append(path)

    return sorted(config_files)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================


def run_all_round_trip_tests(verbose: bool = False) -> tuple[int, int, int, list[tuple[Path, list[str]]]]:
    """
    Run round-trip tests on all discovered config files.

    Args:
        verbose: If True, print progress for each file

    Returns:
        Tuple of (passed, failed, skipped, failures_with_details)
    """
    # Set required environment variables
    test_env_vars = {
        "NVIDIA_API_KEY": "test-api-key",
        "NGC_API_KEY": "test-ngc-key",
        "OPENAI_API_KEY": "test-openai-key",
        "NAT_OAUTH_CLIENT_ID": "test-client-id",
        "NAT_OAUTH_CLIENT_SECRET": "test-client-secret",
        "CUSTOMIZER_HOST": "http://localhost:8080",
        "DATASTORE_HOST": "http://localhost:8081",
        "LANGFUSE_PUBLIC_KEY": "test-langfuse-public",
        "LANGFUSE_SECRET_KEY": "test-langfuse-secret",
        "LANGSMITH_API_KEY": "test-langsmith-key",
        "WEAVE_API_KEY": "test-weave-key",
        "GALILEO_API_KEY": "test-galileo-key",
        "PATRONUS_API_KEY": "test-patronus-key",
        "DBNL_API_KEY": "test-dbnl-key",
        "KAGGLE_USERNAME": "test-kaggle-user",
        "KAGGLE_KEY": "test-kaggle-key",
        "JIRA_API_KEY": "test-jira-key",
        "JIRA_EMAIL": "test@example.com",
        "JIRA_URL": "https://example.atlassian.net",
        "OPENPIPE_API_KEY": "test-openpipe-key",
        "AWS_ACCESS_KEY_ID": "test-aws-key",
        "AWS_SECRET_ACCESS_KEY": "test-aws-secret",
        "MYSQL_USER": "test-mysql-user",
        "MYSQL_PASSWORD": "test-mysql-password",
        "MYSQL_HOST": "localhost",
        "MYSQL_DATABASE": "test_db",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "S3_BUCKET": "test-bucket",
        "GITHUB_TOKEN": "test-github-token",
    }

    for key, value in test_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    # Discover config files
    config_files = discover_config_files(EXAMPLES_DIR)

    passed = 0
    failed = 0
    skipped = 0
    failures: list[tuple[Path, list[str]]] = []

    for config_path in config_files:
        filename = config_path.name
        relative_path = config_path.relative_to(PROJECT_ROOT)

        # Check if should skip
        if filename in SKIP_CONFIGS:
            if verbose:
                print(f"SKIP: {relative_path} ({SKIP_CONFIGS[filename]})")
            skipped += 1
            continue

        # Run round-trip test
        success, errors, exception = run_round_trip(config_path)

        if success:
            if verbose:
                print(f"PASS: {relative_path}")
            passed += 1
        else:
            if verbose:
                if exception:
                    print(f"FAIL: {relative_path} - Exception: {exception}")
                else:
                    print(f"FAIL: {relative_path} - {len(errors)} errors")
                    for error in errors[:3]:
                        print(f"      {error}")
                    if len(errors) > 3:
                        print(f"      ... and {len(errors) - 3} more")
            failed += 1
            failures.append((config_path, errors if not exception else [exception]))

    return passed, failed, skipped, failures


# =============================================================================
# PYTEST INTEGRATION
# =============================================================================


def test_all_example_configs_round_trip():
    """
    Pytest test that runs round-trip tests on all example configs.

    This is the main integration test entry point.
    """
    passed, failed, skipped, failures = run_all_round_trip_tests(verbose=True)

    print(f"\n{'='*60}")
    print("ROUND-TRIP TEST RESULTS")
    print("=" * 60)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total:   {passed + failed + skipped}")

    if failures:
        print(f"\n{'='*60}")
        print("FAILURES:")
        print("=" * 60)
        for path, errors in failures:
            relative = path.relative_to(PROJECT_ROOT)
            print(f"\n{relative}:")
            for error in errors[:5]:
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")

    assert failed == 0, f"{failed} config files failed round-trip test"


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive round-trip tests on all example configs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # For standalone running, discover plugins first
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    discover_and_register_plugins(PluginTypes.ALL)

    # Run tests
    passed, failed, skipped, failures = run_all_round_trip_tests(verbose=args.verbose or True)

    print(f"\n{'='*60}")
    print("COMPREHENSIVE ROUND-TRIP TEST RESULTS")
    print("=" * 60)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total:   {passed + failed + skipped}")

    if failures:
        print(f"\n{'='*60}")
        print("FAILURES:")
        print("=" * 60)
        for path, errors in failures:
            relative = path.relative_to(PROJECT_ROOT)
            print(f"\n{relative}:")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

    sys.exit(0 if failed == 0 else 1)
