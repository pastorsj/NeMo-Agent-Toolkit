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

# Creating Workflows with the SDK

The `NatWorkflow` class is the central component of the Python SDK. It manages the agent entrypoint, discovers components, and provides methods for running, evaluating, and optimizing workflows.

## Creating a Workflow

A workflow requires an entrypoint, which is typically an agent:

```python
from nat.utils.sdk.nat_workflow import NatWorkflow

workflow = NatWorkflow(entrypoint=agent)
```

## Running Workflows

### Using `prompt()`

The `prompt()` method runs the workflow with a single input:

```python
response = await workflow.prompt("What is 2 + 2?")
print(response)
```

### Multi-Turn Conversations with Memory

For chatbots and assistants with long-term memory, pass a `conversation_id` to maintain context across sessions:

```python
# Use the same conversation_id across multiple prompts
THREAD_ID = "user_session_123"

response = await workflow.prompt(
    "Hi! My name is Alex and I'm a software engineer.",
    conversation_id=THREAD_ID,
)

# Later in the same session or a different session
response = await workflow.prompt(
    "What do you remember about me?",
    conversation_id=THREAD_ID,
)
```

The `conversation_id` is passed to memory backends (such as Zep or Mem0) to store and retrieve conversation history and user context. See the [Multi-Turn Chatbot notebook](../../../examples/notebooks/sdk/15_multiturn_chatbot.ipynb) for a complete example.

### Accessing the Configuration

The workflow automatically discovers all components (LLMs, tools, agents) and builds a configuration:

```python
# Get the internal Config object
config = workflow._config

# Print a summary
config.print_summary()
```

## Workflow Operations

### Save to YAML

Export the workflow configuration to a YAML file:

```python
workflow.save_to_config_file("workflow.yaml")
```

The generated YAML can be used with the CLI:

```bash
nat run --config_file workflow.yaml --input "Hello!"
```

### Serve as an API

Serve the workflow as a REST API:

```python
await workflow.serve(host="localhost", port=8000)
```

Or use the CLI after exporting:

```bash
nat serve --config_file workflow.yaml
```

### Evaluate Performance

Add evaluators and run evaluation:

```python
from nat.eval.rag_evaluator.register import RagasEvaluator
from nat.utils.sdk.nat_evaluation import NatEvaluation, EvalDatasetJsonConfig

# Create evaluator
evaluator = RagasEvaluator(
    llm=llm,
    metric="AnswerAccuracy",
    name="accuracy",
)

# Configure evaluation
evaluation = NatEvaluation(
    output_dir=Path("./eval_results"),
    dataset=EvalDatasetJsonConfig(file_path="data/test.json"),
    evaluators=[evaluator],
)

workflow.add_evaluator(evaluation)
await workflow.evaluate()
```

See [Evaluation](./evaluation.md) for more details.

### Optimize Parameters

Add an optimizer and run optimization:

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
await workflow.optimize()
```

See [Optimization](./optimization.md) for more details.

## Component Discovery

When you create a `NatWorkflow`, it automatically discovers all components referenced by the entrypoint:

- **LLMs**: Language models used by agents
- **Tools**: Functions available to agents
- **Function Groups**: Collections of related tools
- **Embedders**: Embedding models for retrieval
- **Memory**: Memory stores for conversation history
- **Evaluators**: Metrics for evaluation

This discovery happens automatically when you access `workflow._config` or call methods like `save_to_config_file()`.

## Example: Complete Workflow

```python
import asyncio
from pathlib import Path
from nat.llm.nim_llm import NimLLM
from nat.agent.react_agent.register import NatReActAgent
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_workflow import NatWorkflow

# Create components
llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.0,
)

time_tool = CurrentTimeTool()

agent = NatReActAgent(
    tools=[time_tool],
    llm=llm,
    verbose=True,
)

# Create workflow
workflow = NatWorkflow(entrypoint=agent)

async def main():
    # Run the workflow
    response = await workflow.prompt("What time is it?")
    print(f"Response: {response}")

    # Save configuration
    workflow.save_to_config_file("my_workflow.yaml")
    print("Configuration saved!")

asyncio.run(main())
```

## Next Steps

- [Components](./components.md): Learn about LLMs, tools, and agents
- [Evaluation](./evaluation.md): Evaluate workflow performance
- [Optimization](./optimization.md): Tune hyperparameters
- [Exporting Configuration](./configuration.md): Save and deploy workflows

