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
"""Tests for optimizer helper functions.

This module tests the helper functions used in parameter optimization,
including dictionary manipulation, optimization field cleaning, and
minimal config serialization.
"""

from pathlib import Path

import pytest
import yaml

from nat.data_models.config import Config
from nat.profiler.parameter_optimization.parameter_optimizer import _apply_to_dict
from nat.profiler.parameter_optimization.parameter_optimizer import _clear_optimization_fields_from_dict
from nat.profiler.parameter_optimization.parameter_optimizer import _get_minimal_dict_with_types
from nat.profiler.parameter_optimization.parameter_optimizer import _save_dict_as_yaml
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins


# Ensure plugins are discovered for tests
@pytest.fixture(scope="module", autouse=True)
def discover_plugins():
    """Discover and register all plugins before running tests."""
    discover_and_register_plugins(PluginTypes.ALL)


# ============================================================================
# _apply_to_dict Tests
# ============================================================================


class TestApplyToDict:
    """Tests for the _apply_to_dict helper function."""

    def test_apply_single_level_key(self):
        """Test applying a single-level dotted key."""
        cfg_dict = {"a": 1, "b": 2}
        result = _apply_to_dict(cfg_dict, {"a": 10})

        assert result["a"] == 10
        assert result["b"] == 2
        # Original should be unchanged
        assert cfg_dict["a"] == 1

    def test_apply_nested_key(self):
        """Test applying a nested dotted key."""
        cfg_dict = {"level1": {"level2": {"value": 1}}}
        result = _apply_to_dict(cfg_dict, {"level1.level2.value": 42})

        assert result["level1"]["level2"]["value"] == 42

    def test_apply_creates_intermediate_dicts(self):
        """Test that missing intermediate dicts are created."""
        cfg_dict = {"existing": 1}
        result = _apply_to_dict(cfg_dict, {"new.nested.key": "value"})

        assert result["new"]["nested"]["key"] == "value"
        assert result["existing"] == 1

    def test_apply_multiple_updates(self):
        """Test applying multiple updates at once."""
        cfg_dict = {"a": 1, "b": {"c": 2}}
        result = _apply_to_dict(cfg_dict, {
            "a": 10,
            "b.c": 20,
            "b.d": 30,
        })

        assert result["a"] == 10
        assert result["b"]["c"] == 20
        assert result["b"]["d"] == 30

    def test_apply_does_not_modify_original(self):
        """Test that the original dict is not modified."""
        cfg_dict = {"a": {"b": 1}}
        result = _apply_to_dict(cfg_dict, {"a.b": 2})

        assert cfg_dict["a"]["b"] == 1
        assert result["a"]["b"] == 2

    def test_apply_with_empty_updates(self):
        """Test applying empty updates returns a copy."""
        cfg_dict = {"a": 1}
        result = _apply_to_dict(cfg_dict, {})

        assert result == cfg_dict
        assert result is not cfg_dict

    def test_apply_deep_nested_path(self):
        """Test applying a deeply nested path."""
        cfg_dict = {}
        result = _apply_to_dict(cfg_dict, {"a.b.c.d.e.f": "deep"})

        assert result["a"]["b"]["c"]["d"]["e"]["f"] == "deep"

    def test_apply_preserves_sibling_keys(self):
        """Test that sibling keys are preserved when updating nested paths."""
        cfg_dict = {
            "llms": {
                "llm1": {
                    "temperature": 0.5, "max_tokens": 100
                },
                "llm2": {
                    "temperature": 0.7
                },
            }
        }
        result = _apply_to_dict(cfg_dict, {"llms.llm1.temperature": 0.9})

        assert result["llms"]["llm1"]["temperature"] == 0.9
        assert result["llms"]["llm1"]["max_tokens"] == 100
        assert result["llms"]["llm2"]["temperature"] == 0.7


# ============================================================================
# _clear_optimization_fields_from_dict Tests
# ============================================================================


class TestClearOptimizationFieldsFromDict:
    """Tests for the _clear_optimization_fields_from_dict helper function."""

    def test_clears_optimizable_params(self):
        """Test that optimizable_params is removed from nested dicts."""
        cfg_dict = {
            "llms": {
                "my_llm": {
                    "temperature": 0.5,
                    "optimizable_params": ["temperature", "top_p"],
                }
            }
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        assert "optimizable_params" not in result["llms"]["my_llm"]
        assert result["llms"]["my_llm"]["temperature"] == 0.5

    def test_clears_search_space(self):
        """Test that search_space is removed from nested dicts."""
        cfg_dict = {
            "llms": {
                "my_llm": {
                    "temperature": 0.5,
                    "search_space": {
                        "temperature": {
                            "low": 0.1, "high": 0.9
                        }
                    },
                }
            }
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        assert "search_space" not in result["llms"]["my_llm"]
        assert result["llms"]["my_llm"]["temperature"] == 0.5

    def test_clears_optimizer_section(self):
        """Test that top-level optimizer section is removed."""
        cfg_dict = {
            "workflow": {
                "_type": "react_agent"
            },
            "optimizer": {
                "output_path": "./results",
                "eval_metrics": {
                    "accuracy": {
                        "direction": "maximize"
                    }
                },
            },
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        assert "optimizer" not in result
        assert "workflow" in result

    def test_clears_all_optimization_fields(self):
        """Test that all optimization-related fields are cleared."""
        cfg_dict = {
            "llms": {
                "llm1": {
                    "temperature": 0.5,
                    "optimizable_params": ["temperature"],
                    "search_space": {
                        "temperature": {
                            "low": 0.1
                        }
                    },
                },
                "llm2": {
                    "top_p": 0.9,
                    "optimizable_params": ["top_p"],
                },
            },
            "optimizer": {
                "n_trials": 10
            },
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        # All optimization fields should be gone
        assert "optimizer" not in result
        assert "optimizable_params" not in result["llms"]["llm1"]
        assert "search_space" not in result["llms"]["llm1"]
        assert "optimizable_params" not in result["llms"]["llm2"]
        # Other fields should remain
        assert result["llms"]["llm1"]["temperature"] == 0.5
        assert result["llms"]["llm2"]["top_p"] == 0.9

    def test_clears_from_deeply_nested_structures(self):
        """Test clearing from deeply nested structures."""
        cfg_dict = {
            "level1": {
                "level2": {
                    "level3": {
                        "optimizable_params": ["param1"],
                        "value": 42,
                    }
                }
            }
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        assert "optimizable_params" not in result["level1"]["level2"]["level3"]
        assert result["level1"]["level2"]["level3"]["value"] == 42

    def test_clears_from_lists(self):
        """Test that fields in list items are also cleaned."""
        cfg_dict = {
            "items": [
                {
                    "name": "item1", "optimizable_params": ["x"]
                },
                {
                    "name": "item2", "search_space": {
                        "y": {
                            "low": 0
                        }
                    }
                },
            ]
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        assert "optimizable_params" not in result["items"][0]
        assert "search_space" not in result["items"][1]
        assert result["items"][0]["name"] == "item1"
        assert result["items"][1]["name"] == "item2"

    def test_preserves_non_optimization_fields(self):
        """Test that non-optimization fields are preserved."""
        cfg_dict = {
            "general": {
                "telemetry": {
                    "logging": {}
                }
            },
            "llms": {
                "llm1": {
                    "model_name": "test", "_type": "nim_llm"
                }
            },
            "workflow": {
                "_type": "react_agent", "verbose": True
            },
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        assert result == cfg_dict

    def test_handles_empty_dict(self):
        """Test handling of empty dictionary."""
        result = _clear_optimization_fields_from_dict({})
        assert result == {}

    def test_handles_dict_with_only_optimization_fields(self):
        """Test dict containing only optimization fields."""
        cfg_dict = {
            "optimizer": {
                "n_trials": 10
            },
        }
        result = _clear_optimization_fields_from_dict(cfg_dict)

        assert result == {}


# ============================================================================
# _get_minimal_dict_with_types Tests
# ============================================================================


class TestGetMinimalDictWithTypes:
    """Tests for the _get_minimal_dict_with_types helper function."""

    def test_returns_dict_with_type_fields(self):
        """Test that the returned dict includes _type fields."""
        # Create a simple config with workflow
        from nat.agent.react_agent.register import NatReActAgent
        from nat.llm.nim_llm import NimLLM
        from nat.utils.sdk.nat_workflow import NatWorkflow

        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._config  # pylint: disable=protected-access
        result = _get_minimal_dict_with_types(config)

        # Should have workflow with _type
        assert "workflow" in result
        assert "_type" in result["workflow"]

    def test_excludes_unset_fields(self):
        """Test that unset fields are excluded."""
        # Create config with minimal settings
        config = Config()
        result = _get_minimal_dict_with_types(config)

        # Empty dicts like functions, llms should not be present or be empty
        # The behavior depends on what's set vs not set
        assert "functions" not in result
        assert "llms" not in result

    def test_uses_json_mode_for_serialization(self):
        """Test that JSON mode is used (no Python objects)."""
        from nat.agent.react_agent.register import NatReActAgent
        from nat.llm.nim_llm import NimLLM
        from nat.utils.sdk.nat_workflow import NatWorkflow

        llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="test_llm")
        agent = NatReActAgent(llm=llm, tools=[], verbose=True)
        workflow = NatWorkflow(entrypoint=agent)

        config = workflow._config  # pylint: disable=protected-access
        result = _get_minimal_dict_with_types(config)

        # Convert to YAML and back - should not have Python object tags
        yaml_str = yaml.dump(result)
        assert "!!python" not in yaml_str


# ============================================================================
# _save_dict_as_yaml Tests
# ============================================================================


class TestSaveDictAsYaml:
    """Tests for the _save_dict_as_yaml helper function."""

    def test_saves_dict_to_yaml_file(self, tmp_path: Path):
        """Test that dict is saved as valid YAML."""
        data = {"key1": "value1", "nested": {"key2": 42}}
        filepath = tmp_path / "test.yaml"

        _save_dict_as_yaml(data, filepath)

        assert filepath.exists()
        with open(filepath, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_preserves_key_order(self, tmp_path: Path):
        """Test that key order is preserved (sort_keys=False)."""
        # Use ordered keys
        data = {"z_key": 1, "a_key": 2, "m_key": 3}
        filepath = tmp_path / "test.yaml"

        _save_dict_as_yaml(data, filepath)

        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        # Keys should appear in original order
        z_pos = content.find("z_key")
        a_pos = content.find("a_key")
        m_pos = content.find("m_key")
        assert z_pos < a_pos < m_pos

    def test_handles_complex_nested_structure(self, tmp_path: Path):
        """Test saving complex nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42,
                        "list": [1, 2, 3],
                    }
                },
                "sibling": "text",
            },
            "root_list": ["a", "b", "c"],
        }
        filepath = tmp_path / "test.yaml"

        _save_dict_as_yaml(data, filepath)

        with open(filepath, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_uses_utf8_encoding(self, tmp_path: Path):
        """Test that UTF-8 encoding is used for special characters."""
        data = {"name": "日本語テスト", "emoji": "🚀"}
        filepath = tmp_path / "test.yaml"

        _save_dict_as_yaml(data, filepath)

        with open(filepath, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["name"] == "日本語テスト"
        assert loaded["emoji"] == "🚀"


# ============================================================================
# Integration Tests
# ============================================================================


class TestOptimizerHelpersIntegration:
    """Integration tests for optimizer helper functions working together."""

    def test_apply_and_clear_workflow(self, tmp_path: Path):
        """Test applying updates and then clearing optimization fields."""
        # Start with a config dict that has optimization fields
        cfg_dict = {
            "llms": {
                "my_llm": {
                    "_type": "nim_llm",
                    "model_name": "test",
                    "temperature": 0.5,
                    "optimizable_params": ["temperature"],
                    "search_space": {
                        "temperature": {
                            "low": 0.1, "high": 0.9
                        }
                    },
                }
            },
            "optimizer": {
                "n_trials": 10
            },
        }

        # Apply optimized values
        updated = _apply_to_dict(cfg_dict, {"llms.my_llm.temperature": 0.75})

        # Clear optimization fields for final output
        final = _clear_optimization_fields_from_dict(updated)

        # Should have new value but no optimization fields
        assert final["llms"]["my_llm"]["temperature"] == 0.75
        assert "optimizable_params" not in final["llms"]["my_llm"]
        assert "search_space" not in final["llms"]["my_llm"]
        assert "optimizer" not in final

    def test_full_workflow_save(self, tmp_path: Path):
        """Test full workflow: apply updates, clear fields, save."""
        cfg_dict = {
            "llms": {
                "llm1": {
                    "_type": "nim_llm",
                    "temperature": 0.5,
                    "optimizable_params": ["temperature"],
                }
            },
            "optimizer": {
                "output_path": str(tmp_path)
            },
        }

        # Apply optimized values
        updated = _apply_to_dict(cfg_dict, {"llms.llm1.temperature": 0.8})

        # Clear optimization fields
        final = _clear_optimization_fields_from_dict(updated)

        # Save to file
        filepath = tmp_path / "optimized.yaml"
        _save_dict_as_yaml(final, filepath)

        # Load and verify
        with open(filepath, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert loaded["llms"]["llm1"]["temperature"] == 0.8
        assert "optimizable_params" not in loaded["llms"]["llm1"]
        assert "optimizer" not in loaded

    def test_multiple_llms_optimization(self, tmp_path: Path):
        """Test optimization workflow with multiple LLMs."""
        cfg_dict = {
            "llms": {
                "llm1": {
                    "_type": "nim_llm",
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "optimizable_params": ["temperature"],
                },
                "llm2": {
                    "_type": "nim_llm",
                    "temperature": 0.7,
                    "optimizable_params": ["temperature", "top_p"],
                    "search_space": {
                        "temperature": {
                            "low": 0.1
                        },
                        "top_p": {
                            "low": 0.5
                        },
                    },
                },
            },
            "optimizer": {
                "n_trials": 20
            },
        }

        # Apply optimized values for both LLMs
        updated = _apply_to_dict(cfg_dict, {
            "llms.llm1.temperature": 0.3,
            "llms.llm2.temperature": 0.6,
            "llms.llm2.top_p": 0.95,
        })

        # Clear and save
        final = _clear_optimization_fields_from_dict(updated)
        filepath = tmp_path / "multi_llm.yaml"
        _save_dict_as_yaml(final, filepath)

        # Verify
        with open(filepath, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert loaded["llms"]["llm1"]["temperature"] == 0.3
        assert loaded["llms"]["llm1"]["top_p"] == 0.9  # Unchanged
        assert loaded["llms"]["llm2"]["temperature"] == 0.6
        assert loaded["llms"]["llm2"]["top_p"] == 0.95
        assert "optimizer" not in loaded
        for llm_config in loaded["llms"].values():
            assert "optimizable_params" not in llm_config
            assert "search_space" not in llm_config
