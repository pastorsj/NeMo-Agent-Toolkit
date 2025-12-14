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

# Getting Started with the Python SDK

This guide walks you through installing NeMo Agent toolkit and building your first agent workflow using the Python SDK.

## Prerequisites

- Python 3.11, 3.12, or 3.13
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/)

## Installation

Install NeMo Agent toolkit with pip:

```bash
pip install nvidia-nat
```

For additional features, install optional dependencies:

```bash
# For LangChain tools (such as Wikipedia search)
pip install "nvidia-nat[langchain]"

# For evaluation and optimization
pip install "nvidia-nat[profiling]"

# For all optional dependencies
pip install "nvidia-nat[all]"
```

## Set Up Your API Key

Set the `NVIDIA_API_KEY` environment variable:

```bash
export NVIDIA_API_KEY=<your_api_key>
```

Or in Python:

```python
import os
os.environ["NVIDIA_API_KEY"] = "<your_api_key>"
```

## Your First Workflow

Let's build a simple agent that can tell the current time.

### Step 1: Import Components

```python
from nat.llm.nim_llm import NimLLM
from nat.agent.react_agent.register import NatReActAgent
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_workflow import NatWorkflow
```

### Step 2: Create an LLM

```python
llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.0,
    max_tokens=1024,
)
```

### Step 3: Create Tools

```python
time_tool = CurrentTimeTool()
```

### Step 4: Create an Agent

```python
agent = NatReActAgent(
    tools=[time_tool],
    llm=llm,
    verbose=True,
)
```

### Step 5: Create and Run the Workflow

```python
import asyncio

workflow = NatWorkflow(entrypoint=agent)

async def main():
    response = await workflow.prompt("What is the current time?")
    print(response)

asyncio.run(main())
```

## Complete Example

Here's the full code:

```python
import asyncio
from nat.llm.nim_llm import NimLLM
from nat.agent.react_agent.register import NatReActAgent
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_workflow import NatWorkflow

# Create the LLM
llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.0,
    max_tokens=1024,
)

# Create tools
time_tool = CurrentTimeTool()

# Create the agent
agent = NatReActAgent(
    tools=[time_tool],
    llm=llm,
    verbose=True,
)

# Create the workflow
workflow = NatWorkflow(entrypoint=agent)

# Run the workflow
async def main():
    response = await workflow.prompt("What is the current time?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Running in Jupyter Notebooks

When running in Jupyter notebooks, use `await` directly instead of `asyncio.run()`:

```python
# In a Jupyter notebook cell
response = await workflow.prompt("What is the current time?")
print(response)
```

If you encounter event loop issues, add this at the start of your notebook:

```python
import nest_asyncio
nest_asyncio.apply()
```

## Saving Your Workflow

Export your SDK workflow to a YAML configuration file:

```python
workflow.save_to_config_file("my_workflow.yaml")
```

This allows you to run the workflow using the CLI:

```bash
nat run --config_file my_workflow.yaml --input "What time is it?"
```

## Next Steps

- [Creating Workflows](./workflows.md): Learn about workflow management
- [Components](./components.md): Explore LLMs, tools, and agents
- [Evaluation](./evaluation.md): Evaluate your workflow's performance
- [Optimization](./optimization.md): Tune hyperparameters automatically

