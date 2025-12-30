<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Finetuning with the SDK

:::{warning}
**Experimental Feature**: The Finetuning Harness is experimental and may change in future releases. Future versions may introduce breaking changes without notice.
:::

The NeMo Agent Toolkit Python SDK provides programmatic access to the finetuning harness, enabling in-situ reinforcement learning of agent workflows directly from Python code.

## Overview

The SDK exposes finetuning through:

- `NatFinetuner`: Main configuration class for finetuning settings
- `CurriculumLearning`: Configuration for curriculum learning
- `workflow.finetune()`: Method to run finetuning on a workflow

## Quick Start

The SDK uses an **object-based approach** where you **must** pass component instances directly (not string names):

```python
from nat.utils.sdk.nat_workflow import NatWorkflow
from nat.utils.sdk.nat_finetuner import NatFinetuner, CurriculumLearning
from nat.utils.sdk.nat_evaluation import NatEvaluation

# With plugin implementations (e.g., OpenPipe ART):
from nat.plugins.openpipe.config import ARTTrainer, ARTTrajectoryBuilder, ARTTrainerAdapter

# Create your workflow (agent, LLM, etc.)
# ... (see Creating Workflows documentation)

# Create your evaluator for computing rewards
my_evaluator = MyAccuracyEvaluator(threshold=0.8)

# Configure evaluation
evaluation = NatEvaluation(
    evaluators=[my_evaluator],
    dataset=EvalDatasetJsonConfig(file_path="data/training_data.json"),
)
workflow.add_evaluator(evaluation)

# Configure finetuning with objects (required)
finetuning = NatFinetuner(
    trainer=ARTTrainer(),
    trajectory_builder=ARTTrajectoryBuilder(num_generations=2),
    trainer_adapter=ARTTrainerAdapter(backend=backend_config),
    reward_function=my_evaluator,  # Pass the evaluator object
    num_epochs=10,
)
workflow.add_finetuning(finetuning)

# Names are computed automatically from objects:
print(finetuning.trainer_name)  # e.g., "openpipe_art_trainer"

# Run finetuning
await workflow.finetune()
```

## Configuration Classes

### NatFinetuner

The main configuration class for finetuning. Components **must** be passed as objects:

```python
from nat.utils.sdk.nat_finetuner import NatFinetuner

finetuning = NatFinetuner(
    trainer=ARTTrainer(),
    trajectory_builder=ARTTrajectoryBuilder(num_generations=2),
    trainer_adapter=ARTTrainerAdapter(backend=backend_config),
    reward_function=my_evaluator,
    num_epochs=10,
    curriculum_learning=CurriculumLearning(enabled=True),
)

# Name properties are computed from the objects:
print(finetuning.trainer_name)
print(finetuning.trajectory_builder_name)
print(finetuning.trainer_adapter_name)
print(finetuning.reward_function_name)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `trainer` | `NatTrainer` | Trainer object (required) |
| `trajectory_builder` | `NatTrajectoryBuilder` | TrajectoryBuilder object (required) |
| `trainer_adapter` | `NatTrainerAdapter` | TrainerAdapter object (required) |
| `reward_function` | `NatEvaluator` | Evaluator object for computing rewards (required) |
| `num_epochs` | `int` | Number of training epochs (default: 1) |
| `target_function_names` | `list[str]` | Functions to extract trajectories from (default: `["<workflow>"]`) |
| `target_model_name` | `str \| None` | Specific model to target |
| `output_dir` | `Path` | Directory for outputs and checkpoints |
| `curriculum_learning` | `CurriculumLearning \| None` | Curriculum learning settings |

### CurriculumLearning

Configure curriculum learning for progressive training:

```python
from nat.utils.sdk.nat_finetuner import CurriculumLearning

curriculum = CurriculumLearning(
    enabled=True,
    initial_percentile=0.3,      # Start with easiest 30%
    increment_percentile=0.2,     # Add 20% more each expansion
    expansion_interval=5,         # Expand every 5 epochs
    min_reward_diff=0.1,         # Skip groups with no variance
    sort_ascending=False,         # False = easy-to-hard
    random_subsample=None,        # Optional: subsample trajectories
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | `bool` | Whether to enable curriculum learning |
| `initial_percentile` | `float` | Fraction of examples to start with (0.0-1.0) |
| `increment_percentile` | `float` | Fraction to add at each expansion |
| `expansion_interval` | `int` | Epochs between expansions |
| `min_reward_diff` | `float` | Minimum reward variance to include a group |
| `sort_ascending` | `bool` | Sort direction (False=easy-to-hard) |
| `random_subsample` | `float \| None` | Optional random subsampling fraction |

## Running Finetuning

### From Python

```python
# Add configuration to workflow
workflow.add_evaluator(evaluation)
workflow.add_finetuning(finetuning)

# Run finetuning with default settings
await workflow.finetune()

# Run with custom parameters
await workflow.finetune(
    dataset="data/custom_training.json",  # Override dataset
    result_json_path="$.output",           # Extract specific field
    endpoint="http://localhost:8000/generate",  # Use remote endpoint
    endpoint_timeout=600,
    validation_dataset="data/validation.json",
    validation_interval=3,
)
```

### Saving Configuration for CLI

You can also save the SDK configuration to a YAML file and run finetuning via the CLI:

```python
# Save workflow with finetuning configuration
workflow.save_to_config_file("finetuning_workflow.yaml")
```

Then run via CLI:

```bash
nat finetune --config_file=finetuning_workflow.yaml
```

## Example: OpenPipe ART Training

Here's a complete example using OpenPipe ART for GRPO training:

```python
import asyncio
from nat.llm.sdk import OpenAILLM
from nat.agent.sdk import NatReActAgent
from nat.utils.sdk.nat_workflow import NatWorkflow
from nat.utils.sdk.nat_finetuner import NatFinetuner, CurriculumLearning
from nat.utils.sdk.nat_evaluation import NatEvaluation, EvalDatasetJsonConfig

# Import plugin components
from nat.plugins.openpipe.config import ARTTrainer, ARTTrajectoryBuilder, ARTTrainerAdapter

async def main():
    # Create LLM with log probabilities enabled
    llm = OpenAILLM(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",  # vLLM endpoint
        api_key="default",
    )

    # Create agent
    agent = NatReActAgent(llm=llm, tools=[], verbose=True)

    # Create workflow
    workflow = NatWorkflow(entrypoint=agent)

    # Create evaluator for rewards
    my_evaluator = MyAccuracyEvaluator(threshold=0.8)

    # Configure evaluation (reward function)
    evaluation = NatEvaluation(
        evaluators=[my_evaluator],
        dataset=EvalDatasetJsonConfig(file_path="data/training_data.json"),
        max_concurrency=16,
        output_dir=".tmp/nat/finetuning/eval",
    )
    workflow.add_evaluator(evaluation)

    # Configure curriculum learning
    curriculum = CurriculumLearning(
        enabled=True,
        initial_percentile=0.3,
        increment_percentile=0.2,
        expansion_interval=5,
    )

    # Configure finetuning with objects
    finetuning = NatFinetuner(
        trainer=ARTTrainer(),
        trajectory_builder=ARTTrajectoryBuilder(num_generations=2),
        trainer_adapter=ARTTrainerAdapter(backend=backend_config),
        reward_function=my_evaluator,
        num_epochs=20,
        curriculum_learning=curriculum,
    )
    workflow.add_finetuning(finetuning)

    # Run finetuning
    await workflow.finetune()

asyncio.run(main())
```

## Requirements

Before running finetuning, ensure you have:

1. **Training Backend**: A running training backend (e.g., OpenPipe ART server)
2. **LLM Endpoint**: An inference endpoint with log probability support
3. **Training Dataset**: Data in JSON/JSONL format
4. **Reward Function**: A custom evaluator for computing rewards

## See Also

- [Finetuning Concepts](../improve-workflows/finetuning/concepts.md) - Core concepts and architecture
- [OpenPipe ART Integration](../improve-workflows/finetuning/rl_with_openpipe.md) - Using the ART backend
- [Evaluation with the SDK](./evaluation.md) - Configuring evaluators for reward functions

