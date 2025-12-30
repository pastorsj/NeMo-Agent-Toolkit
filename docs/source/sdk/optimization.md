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

# Optimization with the SDK

:::{warning}
**Experimental Feature**: The Optimization API is experimental and may change in future releases.
:::

Optimization automatically tunes hyperparameters to improve your agent's performance. The SDK provides a programmatic interface for configuring and running optimizations.

## Prerequisites

Install the profiling extras:

```bash
pip install "nvidia-nat[profiling]"
```

## Optimization Overview

NeMo Agent toolkit supports two types of optimization:

1. **Numeric Optimization**: Tunes numerical parameters like `temperature`, `top_p`, and `max_tokens` using Optuna
2. **Prompt Optimization**: Evolves prompts using a genetic algorithm

## Configuring Optimizable Parameters

Mark LLM parameters as optimizable when creating the LLM:

```python
from nat.llm.sdk import NimLLM
from nat.utils.sdk.nat_optimizer import SearchSpace

llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.5,      # Starting value
    max_tokens=1024,
    # Specify which parameters to optimize
    optimizable_params=["temperature", "top_p"],
    # Define custom search ranges (optional)
    search_space={
        "temperature": SearchSpace(low=0.1, high=0.9, step=0.1),
        "top_p": SearchSpace(low=0.7, high=1.0, step=0.1),
    },
)
```

### SearchSpace Options

| Parameter | Description | Example |
|-----------|-------------|---------|
| `low` | Minimum value | `0.1` |
| `high` | Maximum value | `0.9` |
| `step` | Step size | `0.1` |
| `values` | Categorical options | `["a", "b", "c"]` |
| `log` | Use log scale | `True` |

**Continuous range:**

```python
SearchSpace(low=0.1, high=0.9, step=0.1)
```

**Categorical values:**

```python
SearchSpace(values=["model-a", "model-b", "model-c"])
```

## Configuring the Optimizer

Use `NatOptimizer` to configure optimization:

```python
from pathlib import Path
from nat.utils.sdk.nat_optimizer import (
    NatOptimizer,
    OptimizerMetric,
    NumericOptimizationConfig,
    PromptGAOptimizationConfig,
)

optimizer = NatOptimizer(
    output_path=Path("./opt_results"),

    # Metrics to optimize
    eval_metrics={
        "accuracy": OptimizerMetric(
            evaluator_name="accuracy",   # Must match evaluator name
            direction="maximize",        # "maximize" or "minimize"
            weight=1.0,                  # Weight for multi-objective
        )
    },

    # Numeric optimization (Optuna)
    numeric=NumericOptimizationConfig(
        enabled=True,
        n_trials=10,  # Number of trials
    ),

    # Prompt optimization (Genetic Algorithm)
    prompt=PromptGAOptimizationConfig(
        enabled=False,  # Disabled by default
        ga_population_size=8,
        ga_generations=5,
    ),

    # Evaluation settings
    reps_per_param_set=1,
    target=0.95,  # Early stopping target
)

workflow.add_optimizer(optimizer)
```

### Optimizer Parameters

| Parameter | Description |
|-----------|-------------|
| `output_path` | Directory for results |
| `eval_metrics` | Dict of metrics to optimize |
| `numeric` | Numeric optimization config |
| `prompt` | Prompt optimization config |
| `reps_per_param_set` | Evaluations per parameter set |
| `target` | Early stopping target score |

## Running Optimization

Optimization requires both an evaluator and optimizer:

```python
# Add evaluator first
workflow.add_evaluator(evaluation)

# Add optimizer
workflow.add_optimizer(optimizer)

# Run optimization
await workflow.optimize()
```

## Complete Example

```python
import asyncio
import json
from pathlib import Path
from nat.llm.sdk import NimLLM
from nat.agent.sdk import NatReActAgent
from nat.tool.sdk import CurrentTimeTool
from nat.eval.sdk import RagasEvaluator
from nat.utils.sdk.nat_evaluation import NatEvaluation, EvalDatasetJsonConfig
from nat.utils.sdk.nat_optimizer import (
    NatOptimizer,
    OptimizerMetric,
    NumericOptimizationConfig,
    SearchSpace,
)
from nat.utils.sdk.nat_workflow import NatWorkflow

# Create LLM with optimizable parameters
llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.5,
    max_tokens=1024,
    optimizable_params=["temperature", "top_p"],
    search_space={
        "temperature": SearchSpace(low=0.1, high=0.8, step=0.1),
        "top_p": SearchSpace(low=0.7, high=1.0, step=0.1),
    },
)

# Create tools and agent
time_tool = CurrentTimeTool()

try:
    from nat_simple_calculator.register import CalculatorToolGroup
    calculator = CalculatorToolGroup()
    tools = [time_tool, calculator]
except ImportError:
    tools = [time_tool]

agent = NatReActAgent(
    tools=tools,
    llm=llm,
    verbose=True,
)

workflow = NatWorkflow(entrypoint=agent)

# Create evaluation dataset
eval_data = [
    {"id": "add_001", "question": "What is 2 + 2?", "answer": "The answer is 4."},
    {"id": "mul_001", "question": "What is 10 * 5?", "answer": "The answer is 50."},
]

data_dir = Path("./data")
data_dir.mkdir(parents=True, exist_ok=True)
with open(data_dir / "opt_dataset.json", "w") as f:
    json.dump(eval_data, f, indent=2)

# Configure evaluator
accuracy_evaluator = RagasEvaluator(
    llm=llm,
    metric="AnswerAccuracy",
    name="accuracy",
)

evaluation = NatEvaluation(
    output_dir=Path("./opt_results/eval"),
    dataset=EvalDatasetJsonConfig(file_path=data_dir / "opt_dataset.json"),
    evaluators=[accuracy_evaluator],
)

workflow.add_evaluator(evaluation)

# Configure optimizer
optimizer = NatOptimizer(
    output_path=Path("./opt_results"),
    eval_metrics={
        "accuracy": OptimizerMetric(
            evaluator_name="accuracy",
            direction="maximize",
            weight=1.0,
        )
    },
    numeric=NumericOptimizationConfig(
        enabled=True,
        n_trials=5,
    ),
    reps_per_param_set=1,
    target=0.95,
)

workflow.add_optimizer(optimizer)

# Run optimization
async def main():
    await workflow.optimize()
    print("Optimization complete! Check ./opt_results for results.")

asyncio.run(main())
```

## Understanding Results

After optimization, results are saved in the output directory:

```
opt_results/
├── optimized_config.yml           # Best configuration
├── trials_dataframe_params.csv    # All trial results
├── config_numeric_trial_0.yml     # Individual trial configs
├── config_numeric_trial_1.yml
├── eval/                          # Evaluation results
└── plots/                         # Pareto front visualizations
```

### Reading the Optimized Configuration

```python
import yaml

with open("./opt_results/optimized_config.yml") as f:
    optimized = yaml.safe_load(f)

print("Best parameters found:")
print(f"  temperature: {optimized['llms']['nim_llm']['temperature']}")
print(f"  top_p: {optimized['llms']['nim_llm']['top_p']}")
```

### Using the Optimized Configuration

Run with the optimized configuration:

```bash
nat run --config_file opt_results/optimized_config.yml --input "What is 5 + 5?"
```

## Multi-Objective Optimization

Optimize for multiple metrics simultaneously:

```python
optimizer = NatOptimizer(
    output_path=Path("./opt_results"),
    eval_metrics={
        "accuracy": OptimizerMetric(
            evaluator_name="accuracy",
            direction="maximize",
            weight=0.7,
        ),
        "similarity": OptimizerMetric(
            evaluator_name="similarity",
            direction="maximize",
            weight=0.3,
        ),
    },
    multi_objective_combination_mode="harmonic",  # or "weighted_sum"
)
```

## Jupyter Notebook Usage

When running in Jupyter notebooks, apply `nest_asyncio` first:

```python
import nest_asyncio
nest_asyncio.apply()

# Now you can use await directly
await workflow.optimize()
```

## CLI Alternative

After configuring, you can also run optimization via CLI:

```python
workflow.save_to_config_file("opt_workflow.yaml")
```

```bash
nat optimize --config_file opt_workflow.yaml
```

## Next Steps

- [Evaluation](./evaluation.md): Set up evaluation for optimization
- [Optimizer Reference](../improve-workflows/optimizer.md): Detailed optimizer documentation
- [Components](./components.md): Configure optimizable LLMs

