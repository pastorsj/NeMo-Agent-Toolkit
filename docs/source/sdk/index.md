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

# Python SDK

The NeMo Agent toolkit Python SDK provides a fluent, Pythonic interface for building agent workflows entirely in code. Instead of writing YAML configuration files, you can create LLMs, tools, agents, and workflows using Python objects.

## Why Use the SDK?

- **Pure Python Development**: Build and test agents without switching between Python and YAML
- **IDE Support**: Get full autocomplete, type hints, and inline documentation
- **Programmatic Control**: Dynamically create and modify workflows at runtime
- **Rapid Prototyping**: Quickly iterate on agent designs in Jupyter notebooks
- **Seamless Export**: Save SDK workflows as YAML for deployment or CLI usage

## Quick Example

```python
import asyncio
from nat.llm.sdk import NimLLM
from nat.agent.sdk import NatReActAgent
from nat.tool.sdk import CurrentTimeTool
from nat.utils.sdk.nat_workflow import NatWorkflow

# Create components
llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")
time_tool = CurrentTimeTool()

# Create agent and workflow
agent = NatReActAgent(tools=[time_tool], llm=llm, verbose=True)
workflow = NatWorkflow(entrypoint=agent)

# Run the workflow
async def main():
    response = await workflow.prompt("What time is it?")
    print(response)

asyncio.run(main())
```

:::{note}
**About the `name` parameter**: Components like LLMs and tools accept an optional `name` parameter. This is only required when exporting workflows to YAML configuration files using `save_to_config_file()`. For pure Python development and prototyping, you can omit `name` and let the SDK generate unique identifiers automatically.
:::

## SDK Documentation

```{toctree}
:hidden:
:caption: Python SDK

Getting Started <./getting-started.md>
Creating Workflows <./workflows.md>
Components (LLMs, Tools, Agents) <./components.md>
Evaluation <./evaluation.md>
Optimization <./optimization.md>
Finetuning <./finetuning.md>
Exporting Configuration <./configuration.md>
```

- [Getting Started](./getting-started.md): Install and build your first SDK workflow
- [Creating Workflows](./workflows.md): Learn how to create and manage workflows
- [Components](./components.md): Work with LLMs, tools, and agents
- [Evaluation](./evaluation.md): Evaluate workflow performance with the SDK
- [Optimization](./optimization.md): Optimize hyperparameters and prompts
- [Finetuning](./finetuning.md): Train agents with reinforcement learning
- [Exporting Configuration](./configuration.md): Save workflows as YAML for deployment

## Interactive Tutorials

For hands-on learning, explore the [SDK tutorial notebooks](../../../examples/notebooks/sdk/) in the repository:

| Notebook | Description |
|----------|-------------|
| `01_installation_getting_started.ipynb` | Environment setup and first workflow |
| `03_agents.ipynb` | Different agent types |
| `04_tools.ipynb` | Working with tools |
| `05_llms.ipynb` | Configuring LLMs |
| `08_middleware.ipynb` | Using middleware for caching and more |
| `10_evaluation.ipynb` | Evaluating workflows |
| `15_finetuning.ipynb` | Finetuning with reinforcement learning |
| `16_multiturn_chatbot.ipynb` | Multi-turn chatbot with long-term memory |

