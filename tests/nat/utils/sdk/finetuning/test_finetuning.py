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
"""Tests for SDK finetuning configuration classes."""

from pathlib import Path

import pytest
from pydantic import Field

from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.utils.sdk.nat_evaluator import NatEvaluator
from nat.utils.sdk.nat_finetuner import CurriculumLearning
from nat.utils.sdk.nat_finetuner import NatFinetuner
from nat.utils.sdk.nat_trainer import NatTrainer
from nat.utils.sdk.nat_trainer import NatTrainerAdapter
from nat.utils.sdk.nat_trainer import NatTrajectoryBuilder


# Mock implementations for testing
class MockTrainerConfig(TrainerConfig, name="mock_trainer"):
    """Mock trainer config for testing."""
    mock_param: str = Field(default="default_value")


class MockTrainer(MockTrainerConfig, NatTrainer):
    """Mock trainer for testing object-based API."""
    pass


class MockTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="mock_traj_builder"):
    """Mock trajectory builder config for testing."""
    num_generations: int = Field(default=2)


class MockTrajectoryBuilder(MockTrajectoryBuilderConfig, NatTrajectoryBuilder):
    """Mock trajectory builder for testing object-based API."""
    pass


class MockTrainerAdapterConfig(TrainerAdapterConfig, name="mock_trainer_adapter"):
    """Mock trainer adapter config for testing."""
    backend_url: str = Field(default="http://localhost:8000")


class MockTrainerAdapter(MockTrainerAdapterConfig, NatTrainerAdapter):
    """Mock trainer adapter for testing object-based API."""
    pass


class MockEvaluatorConfig(EvaluatorBaseConfig, name="mock_evaluator"):
    """Mock evaluator config for testing."""
    threshold: float = Field(default=0.5)


class MockEvaluator(MockEvaluatorConfig, NatEvaluator):
    """Mock evaluator for testing object-based API."""
    pass


class TestCurriculumLearning:
    """Tests for CurriculumLearning SDK class."""

    def test_inherits_from_config(self):
        """Test that CurriculumLearning inherits from CurriculumLearningConfig."""
        assert issubclass(CurriculumLearning, CurriculumLearningConfig)

    def test_default_initialization(self):
        """Test CurriculumLearning with default values."""
        curriculum = CurriculumLearning()

        assert curriculum.enabled is False
        assert curriculum.initial_percentile == 0.3
        assert curriculum.increment_percentile == 0.2
        assert curriculum.expansion_interval == 5
        assert curriculum.min_reward_diff == 0.1
        assert curriculum.sort_ascending is False
        assert curriculum.random_subsample is None

    def test_custom_initialization(self):
        """Test CurriculumLearning with custom values."""
        curriculum = CurriculumLearning(
            enabled=True,
            initial_percentile=0.5,
            increment_percentile=0.1,
            expansion_interval=3,
            min_reward_diff=0.05,
            sort_ascending=True,
            random_subsample=0.8,
        )

        assert curriculum.enabled is True
        assert curriculum.initial_percentile == 0.5
        assert curriculum.increment_percentile == 0.1
        assert curriculum.expansion_interval == 3
        assert curriculum.min_reward_diff == 0.05
        assert curriculum.sort_ascending is True
        assert curriculum.random_subsample == 0.8

    def test_is_valid_config(self):
        """Test that CurriculumLearning can be used wherever CurriculumLearningConfig is expected."""
        curriculum = CurriculumLearning(enabled=True, initial_percentile=0.4)

        # Should be assignable to CurriculumLearningConfig
        config: CurriculumLearningConfig = curriculum
        assert config.enabled is True
        assert config.initial_percentile == 0.4


class TestNatFinetunerStructure:
    """Tests for NatFinetuner structure."""

    def test_is_base_model(self):
        """Test that NatFinetuner is a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert issubclass(NatFinetuner, BaseModel)

    def test_can_convert_to_finetune_config(self):
        """Test that NatFinetuner can convert to FinetuneConfig."""
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder()
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator()

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
        )

        config = finetuning.to_finetune_config()
        assert isinstance(config, FinetuneConfig)


class TestNatFinetunerWithObjects:
    """Tests for NatFinetuner SDK class with object references."""

    def test_required_fields_with_objects(self):
        """Test that required fields work with objects."""
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder(num_generations=4)
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator(threshold=0.8)

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
        )

        # Objects should be stored
        assert finetuning.trainer is trainer
        assert finetuning.trajectory_builder is traj_builder
        assert finetuning.trainer_adapter is trainer_adapter
        assert finetuning.reward_function is evaluator

        # Name refs should be populated
        assert finetuning.trainer_name is not None
        assert finetuning.trajectory_builder_name is not None
        assert finetuning.trainer_adapter_name is not None
        assert finetuning.reward_function_name is not None

    def test_objects_converted_to_name_refs(self):
        """Test that objects are converted to name references."""
        trainer = MockTrainer(name="my_trainer")
        traj_builder = MockTrajectoryBuilder(name="my_traj_builder")
        trainer_adapter = MockTrainerAdapter(name="my_adapter")
        evaluator = MockEvaluator(name="my_evaluator")

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
        )

        # Name fields should be populated from objects
        assert finetuning.trainer_name == "my_trainer"
        assert finetuning.trajectory_builder_name == "my_traj_builder"
        assert finetuning.trainer_adapter_name == "my_adapter"
        assert finetuning.reward_function_name == "my_evaluator"

    def test_config_enabled_is_true(self):
        """Test that the converted config has enabled=True."""
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder()
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator()

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
        )

        config = finetuning.to_finetune_config()
        assert config.enabled is True

    def test_to_finetune_config_with_objects(self):
        """Test conversion to FinetuneConfig with objects."""
        trainer = MockTrainer(name="my_trainer")
        traj_builder = MockTrajectoryBuilder(name="my_traj_builder")
        trainer_adapter = MockTrainerAdapter(name="my_adapter")
        evaluator = MockEvaluator(name="my_evaluator")

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
            num_epochs=15,
        )

        config = finetuning.to_finetune_config()

        assert isinstance(config, FinetuneConfig)
        assert config.enabled is True
        assert config.trainer_name == "my_trainer"
        assert config.trajectory_builder_name == "my_traj_builder"
        assert config.trainer_adapter_name == "my_adapter"
        assert config.reward_function_name == "my_evaluator"
        assert config.num_epochs == 15

    def test_get_components_returns_objects(self):
        """Test that get_components returns provided objects."""
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder()
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator()

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
        )

        components = finetuning.get_components()

        assert "trainer" in components
        assert "trajectory_builder" in components
        assert "trainer_adapter" in components
        assert "reward_function" in components
        assert components["trainer"] is trainer
        assert components["trajectory_builder"] is traj_builder
        assert components["trainer_adapter"] is trainer_adapter
        assert components["reward_function"] is evaluator


class TestNatFinetunerWithCurriculum:
    """Tests for NatFinetuner with curriculum learning."""

    def test_with_curriculum_learning(self):
        """Test finetuning with curriculum learning."""
        curriculum = CurriculumLearning(
            enabled=True,
            initial_percentile=0.5,
        )
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder()
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator()

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
            curriculum_learning=curriculum,
        )

        assert finetuning.curriculum_learning.enabled is True
        assert finetuning.curriculum_learning.initial_percentile == 0.5


class TestNatFinetunerValidation:
    """Tests for NatFinetuner validation."""

    def test_num_epochs_validation(self):
        """Test that num_epochs must be at least 1."""
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder()
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator()

        with pytest.raises(ValueError):
            NatFinetuner(
                trainer=trainer,
                trajectory_builder=traj_builder,
                trainer_adapter=trainer_adapter,
                reward_function=evaluator,
                num_epochs=0,
            )

    def test_preserves_target_model_name(self):
        """Test that target_model_name is preserved in conversion."""
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder()
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator()

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
            target_model_name="specific-model",
        )

        config = finetuning.to_finetune_config()
        assert config.target_model_name == "specific-model"

    def test_preserves_target_function_names(self):
        """Test that target_function_names are preserved in conversion."""
        trainer = MockTrainer()
        traj_builder = MockTrajectoryBuilder()
        trainer_adapter = MockTrainerAdapter()
        evaluator = MockEvaluator()

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
            target_function_names=["func1", "func2", "func3"],
        )

        config = finetuning.to_finetune_config()
        assert config.target_function_names == ["func1", "func2", "func3"]


class TestFinetuningIntegration:
    """Integration tests for finetuning configuration."""

    def test_full_configuration_with_objects(self):
        """Test creating full config with objects and converting to internal format."""
        curriculum = CurriculumLearning(
            enabled=True,
            initial_percentile=0.3,
            increment_percentile=0.2,
            expansion_interval=5,
            min_reward_diff=0.1,
            sort_ascending=False,
        )

        trainer = MockTrainer(name="art_trainer", mock_param="custom")
        traj_builder = MockTrajectoryBuilder(name="art_traj", num_generations=4)
        trainer_adapter = MockTrainerAdapter(name="art_adapter", backend_url="http://gpu:8000")
        evaluator = MockEvaluator(name="accuracy", threshold=0.9)

        finetuning = NatFinetuner(
            trainer=trainer,
            trajectory_builder=traj_builder,
            trainer_adapter=trainer_adapter,
            reward_function=evaluator,
            target_function_names=["<workflow>"],
            target_model_name=None,
            num_epochs=10,
            output_dir=Path(".tmp/nat/finetuning"),
            curriculum_learning=curriculum,
        )

        config = finetuning.to_finetune_config()

        # Verify the complete configuration
        assert config.enabled is True
        assert config.trainer_name == "art_trainer"
        assert config.trajectory_builder_name == "art_traj"
        assert config.trainer_adapter_name == "art_adapter"
        assert config.reward_function_name == "accuracy"
        assert config.target_function_names == ["<workflow>"]
        assert config.target_model_name is None
        assert config.num_epochs == 10
        assert config.output_dir == Path(".tmp/nat/finetuning")
        assert config.curriculum_learning.enabled is True
        assert config.curriculum_learning.initial_percentile == 0.3

        # Verify components are discoverable
        components = finetuning.get_components()
        assert len(components) == 4
        assert components["trainer"].mock_param == "custom"
        assert components["trajectory_builder"].num_generations == 4
        assert components["trainer_adapter"].backend_url == "http://gpu:8000"
        assert components["reward_function"].threshold == 0.9


class TestFinetuneConfigBackwardCompatibility:
    """Test that FinetuneConfig can still be loaded with old field names."""

    def test_old_field_names_still_work(self):
        """Test that old YAML field names are mapped to new names."""
        # Simulate loading from YAML with old field names
        config = FinetuneConfig(
            enabled=True,
            trainer="my_trainer",  # Old name
            trajectory_builder="my_traj_builder",  # Old name
            trainer_adapter="my_adapter",  # Old name
            reward_function={"name": "accuracy"},  # Old nested format
            target_functions=["<workflow>"],  # Old name
            target_model="my_model",  # Old name
        )

        # New field names should be populated
        assert config.trainer_name == "my_trainer"
        assert config.trajectory_builder_name == "my_traj_builder"
        assert config.trainer_adapter_name == "my_adapter"
        assert config.reward_function_name == "accuracy"
        assert config.target_function_names == ["<workflow>"]
        assert config.target_model_name == "my_model"
