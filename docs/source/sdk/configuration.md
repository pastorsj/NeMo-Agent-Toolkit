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

# Exporting Configuration

The SDK allows you to export workflows as YAML configuration files. This enables deployment with the CLI, sharing configurations, and version control.

## The `name` Parameter

When exporting workflows to YAML, components need unique names to be referenced in the configuration. You can either:

1. **Let the SDK generate names automatically**: Works for prototyping, but names may change between runs
2. **Provide explicit names**: Recommended for production configurations

```python
# Without name (auto-generated) - good for prototyping
llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")

# With explicit name - recommended for production
llm = NimLLM(model_name="meta/llama-3.1-70b-instruct", name="nim_llm")
```

:::{tip}
For pure Python development and prototyping, you can omit the `name` parameter entirely. Only add explicit names when you need stable, reproducible YAML exports for production deployment.
:::

## Saving to YAML

Export your SDK workflow using `save_to_config_file()`:

```python
workflow.save_to_config_file("workflow.yaml")
```

This generates a YAML file that can be used with the CLI:

```bash
nat run --config_file workflow.yaml --input "Hello!"
```

## Example Export

When exporting for production, provide explicit names for stable configuration:

```python
from nat.llm.nim_llm import NimLLM
from nat.agent.react_agent.register import NatReActAgent
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_workflow import NatWorkflow

# Use explicit names for production export
llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.0,
    max_tokens=1024,
    name="nim_llm",  # Explicit name for YAML export
)

time_tool = CurrentTimeTool(name="current_time")  # Explicit name

agent = NatReActAgent(
    tools=[time_tool],
    llm=llm,
    verbose=True,
    additional_instructions="Be concise.",
)

workflow = NatWorkflow(entrypoint=agent)
workflow.save_to_config_file("workflow.yaml")
```

The generated `workflow.yaml`:

```yaml
functions:
  current_time:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model: meta/llama-3.1-70b-instruct
    max_tokens: 1024
    temperature: 0.0

workflow:
  _type: react_agent
  llm_name: nim_llm
  verbose: true
  tool_names:
  - current_time
  additional_instructions: Be concise.
```

## What Gets Exported

The SDK exports only the fields that were explicitly set, keeping the YAML clean and minimal:

| Section | Contents |
|---------|----------|
| `functions` | Tools (functions) used by the agent |
| `function_groups` | Function groups (bundled tools) |
| `llms` | Language model configurations |
| `embedders` | Embedding model configurations |
| `workflow` | The agent configuration |
| `eval` | Evaluation configuration (if added) |
| `optimizer` | Optimizer configuration (if added) |

## Exporting with Evaluation

If you've added an evaluator, it will be included:

```python
from nat.utils.sdk.nat_evaluation import NatEvaluation, EvalDatasetJsonConfig
from nat.eval.rag_evaluator.register import RagasEvaluator

evaluator = RagasEvaluator(llm=llm, metric="AnswerAccuracy", name="accuracy")
evaluation = NatEvaluation(
    output_dir=Path("./eval_results"),
    dataset=EvalDatasetJsonConfig(file_path=Path("./data/test.json")),
    evaluators=[evaluator],
)

workflow.add_evaluator(evaluation)
workflow.save_to_config_file("eval_workflow.yaml")
```

The generated YAML includes the eval section:

```yaml
# ... other sections ...

eval:
  general:
    output_dir: eval_results
    dataset:
      _type: json
      file_path: data/test.json
  evaluators:
    accuracy:
      _type: ragas
      llm_name: nim_llm
      metric: AnswerAccuracy
```

## Exporting with Optimization

If you've added an optimizer, it will be included:

```python
from nat.utils.sdk.nat_optimizer import NatOptimizer, OptimizerMetric

optimizer = NatOptimizer(
    output_path=Path("./opt_results"),
    eval_metrics={
        "accuracy": OptimizerMetric(
            evaluator_name="accuracy",
            direction="maximize",
            weight=1.0,
        )
    },
)

workflow.add_optimizer(optimizer)
workflow.save_to_config_file("opt_workflow.yaml")
```

## Using Exported Configurations

### Running with CLI

```bash
# Run the workflow
nat run --config_file workflow.yaml --input "What time is it?"

# Serve as API
nat serve --config_file workflow.yaml

# Run evaluation
nat eval --config_file eval_workflow.yaml

# Run optimization
nat optimize --config_file opt_workflow.yaml
```

### Loading in Another Script

Exported configurations work with the standard NeMo Agent toolkit loaders:

```python
from nat.config_loader import load_config

config = load_config("workflow.yaml")
```

## Environment Variables

The SDK preserves environment variable references in exports. If you use `NatEnvironmentVariable`, the exported YAML contains `${VAR_NAME}` placeholders:

```python
from nat.utils.sdk.nat_env import NatEnvironmentVariable

llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    api_key=NatEnvironmentVariable("NVIDIA_API_KEY"),
    name="nim_llm",
)
```

Exported YAML:

```yaml
llms:
  nim_llm:
    _type: nim
    model: meta/llama-3.1-70b-instruct
    api_key: ${NVIDIA_API_KEY}
```

## Best Practices

### Version Control

Export configurations to version control alongside your code:

```python
# After making changes
workflow.save_to_config_file("configs/production.yaml")
```

### Development vs. Production

Create separate configurations for different environments:

```python
# Development (verbose, local)
workflow.save_to_config_file("configs/dev.yaml")

# Modify for production
agent.verbose = False
workflow.save_to_config_file("configs/prod.yaml")
```

### Reviewing Generated YAML

Always review the generated YAML to ensure it matches your expectations:

```python
workflow.save_to_config_file("workflow.yaml")

# Print the generated config
with open("workflow.yaml") as f:
    print(f.read())
```

## Next Steps

- [CLI Reference](../reference/cli.md): Command-line interface documentation
- [Workflow Configuration](../workflows/workflow-configuration.md): YAML configuration reference
- [Run Workflows](../workflows/run-workflows.md): Running workflows with the CLI

