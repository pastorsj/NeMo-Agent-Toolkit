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

# SDK Components: LLMs, Tools, and Agents

The NeMo Agent toolkit SDK provides Python classes for creating LLMs, tools, and agents. This guide covers the main components and how to use them.

## Two Ways to Create Components

SDK components can be created in two ways:

### 1. Subclass Pattern (Recommended for most use cases)

Use specific classes like `NatReActAgent`, `NimLLM`, etc. for the most convenient API:

```python
from nat.llm.nim_llm import NimLLM
from nat.agent.react_agent.register import NatReActAgent

llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")
agent = NatReActAgent(llm=llm, tools=[...])
```

### 2. Factory Pattern (For advanced use cases)

Wrap any config object using the base classes (`NatAgent`, `NatLLM`, `NatFunction`, etc.):

```python
from nat.utils.sdk.nat_agent import NatAgent
from nat.agent.react_agent.register import ReActAgentWorkflowConfig

# Create the config directly
config = ReActAgentWorkflowConfig(
    llm_name="my_llm",
    tool_names=["tool1", "tool2"],
    description="My agent",
)

# Wrap it with NatAgent for SDK integration
agent = NatAgent(config=config, name="my_agent")
```

The factory pattern is useful when:
- You need to work with config objects from YAML or other sources
- You're building dynamic workflows where config types are determined at runtime
- You want to separate config creation from SDK wrapper creation

Both patterns work interchangeably in workflows and produce the same results.

## LLMs (Language Models)

### NimLLM

The most common LLM for NVIDIA NIMs:

```python
from nat.llm.nim_llm import NimLLM

llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.0,
    max_tokens=1024,
)
```

**Key Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | str | Model identifier (such as `meta/llama-3.1-70b-instruct`) |
| `temperature` | float | Randomness (0.0-1.0) |
| `max_tokens` | int | Maximum response length |
| `top_p` | float | Nucleus sampling parameter |

### Optimizable Parameters

LLMs support hyperparameter optimization:

```python
from nat.utils.sdk.nat_optimizer import SearchSpace

llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.5,
    # Specify which parameters to optimize
    optimizable_params=["temperature", "top_p"],
    # Define custom search ranges (optional)
    search_space={
        "temperature": SearchSpace(low=0.1, high=0.9, step=0.1),
        "top_p": SearchSpace(low=0.7, high=1.0, step=0.1),
    },
)
```

## Tools (Functions)

Tools are functions that agents can call. NeMo Agent toolkit provides built-in tools and supports custom tools.

### Built-in Tools

**CurrentTimeTool**: Get the current date and time:

```python
from nat.tool.datetime_tools import CurrentTimeTool

time_tool = CurrentTimeTool()
```

**WikiSearchTool**: Search Wikipedia (requires `nvidia-nat-langchain` plugin):

```python
from nat.plugins.langchain.tools.wikipedia_search import WikiSearchTool

wiki_tool = WikiSearchTool(max_results=3)
```

**TavilyInternetSearchTool**: Search the web using Tavily (requires `nvidia-nat-langchain` plugin and Tavily API key):

```python
from pydantic import SecretStr
from nat.plugins.langchain.tools.tavily_internet_search import TavilyInternetSearchTool

tavily_tool = TavilyInternetSearchTool(
    name="web_search",
    max_results=3,
    api_key=SecretStr("your-tavily-api-key"),
)
```

### Memory Tools

Memory tools allow agents to store and retrieve information about users:

**AddMemoryTool**: Store information in the memory backend:

```python
from nat.tool.memory_tools.add_memory_tool import AddMemoryTool

add_memory = AddMemoryTool(
    nat_memory=memory_backend,
    name="add_memory",
    description="Store important information about the user.",
)
```

**GetMemoryTool**: Retrieve stored information:

```python
from nat.tool.memory_tools.get_memory_tool import GetMemoryTool

get_memory = GetMemoryTool(
    nat_memory=memory_backend,
    name="get_memory",
    description="Retrieve previously stored information about the user.",
)
```

### Function Groups

Function groups bundle related tools together:

```python
# Calculator function group (from examples)
from nat_simple_calculator.register import CalculatorToolGroup

calculator = CalculatorToolGroup()
```

### Custom Tools

Create custom tools using the `@register_function` decorator:

```python
from nat.registry import register_function

@register_function(name="my_custom_tool")
def my_custom_tool(query: str) -> str:
    """Search for information about a topic.

    Args:
        query: The search query

    Returns:
        Search results as a string
    """
    # Your implementation here
    return f"Results for: {query}"
```

For more details on creating custom tools, see [Writing Custom Functions](../extend/custom-components/custom-functions/functions.md).

## Agents

Agents use LLMs to reason about tasks and call tools. NeMo Agent toolkit provides several agent types.

### NatReActAgent

The ReAct (Reasoning and Acting) agent:

```python
from nat.agent.react_agent.register import NatReActAgent

agent = NatReActAgent(
    tools=[time_tool, calculator],
    llm=llm,
    verbose=True,
    additional_instructions="Be concise in your responses.",
)
```

**Key Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tools` | list | List of tools available to the agent |
| `llm` | NimLLM | The language model to use |
| `verbose` | bool | Enable detailed logging |
| `additional_instructions` | str | Extra instructions for the agent |
| `max_tool_calls` | int | Maximum number of tool calls (default: 15) |

### Other Agent Types

**Tool Calling Agent**: Uses the LLM's native tool calling:

```python
from nat.agent.tool_calling_agent.register import NatToolCallingAgent

agent = NatToolCallingAgent(
    tools=[time_tool],
    llm=llm,
)
```

**Reasoning Agent**: Enhanced reasoning with chain-of-thought:

```python
from nat.agent.reasoning_agent.register import NatReasoningAgent

agent = NatReasoningAgent(
    tools=[time_tool],
    llm=llm,
)
```

**Router Agent**: Routes requests to specialized sub-agents:

```python
from nat.control_flow.router_agent.register import NatRouterAgent

router = NatRouterAgent(
    routes=[math_agent, search_agent],
    llm=llm,
)
```

For more details on agent types, see:
- [ReAct Agent](../components/agents/react-agent/index.md)
- [Reasoning Agent](../components/agents/reasoning-agent/index.md)
- [Tool Calling Agent](../components/agents/tool-calling-agent/index.md)
- [Router Agent](../components/agents/router-agent/index.md)

## Middleware

Middleware provides a mechanism for adding cross-cutting concerns to functions without modifying their implementation. You can use middleware for caching, logging, rate limiting, and more.

### Cache Middleware

The cache middleware memoizes function outputs based on input similarity:

```python
from nat.middleware.cache_middleware import CacheMiddleware

cache_middleware = CacheMiddleware(
    enabled_mode="always",       # "always" or "eval" (only during evaluation)
    similarity_threshold=1.0,    # 1.0 = exact matching, < 1.0 = fuzzy matching
    name="my_cache",
)
```

**Key Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled_mode` | str | When caching is active: `"always"` or `"eval"` |
| `similarity_threshold` | float | Input matching threshold (0.0–1.0) |

### Applying Middleware to Functions

Use the `mw` parameter to attach middleware to any function:

```python
from nat.tool.datetime_tools import CurrentTimeTool

# Create a function with middleware attached
time_tool = CurrentTimeTool(
    name="cached_time_tool",
    mw=[cache_middleware],  # Attach middleware using 'mw' parameter
)

print(time_tool.middleware)  # ['my_cache']
```

### Applying Middleware to Function Groups

Middleware can also be applied to function groups, which applies the middleware to all functions in the group:

```python
from nat.plugins.mcp.mcp_client.register import NatMCPClient

mcp_client = NatMCPClient(
    server_url="http://localhost:8000",
    name="my_mcp_client",
    mw=[cache_middleware],  # Applied to all functions in the group
)
```

### Multiple Middleware

You can attach multiple middleware components. They execute in order (first to last for preprocessing, last to first for postprocessing):

```python
# Create additional middleware
eval_cache = CacheMiddleware(
    enabled_mode="eval",  # Only cache during evaluation
    name="eval_cache",
)

# Attach multiple middleware
tool = CurrentTimeTool(
    name="multi_mw_tool",
    mw=[cache_middleware, eval_cache],  # Order matters
)
```

For more details on middleware concepts and custom middleware development, see the [Middleware documentation](../build-workflows/advanced/middleware.md).

## Embedders

Embedders create vector embeddings for retrieval:

```python
from nat.embedder.nim_embedder import NimEmbedder

embedder = NimEmbedder(model_name="nvidia/nv-embedqa-e5-v5")
```

## Memory Backends

Memory backends provide long-term storage for conversation history and user context. They integrate with memory tools to enable persistent memory across sessions.

### Zep Cloud Memory

Zep Cloud provides automatic conversation summarization, semantic search, and user preference extraction:

```python
from nat.plugins.zep_cloud.memory import ZepMemory

memory_backend = ZepMemory(name="zep_memory")
```

Requires the `nvidia-nat-zep-cloud` plugin and a `ZEP_API_KEY` environment variable.

### Mem0 Memory

Mem0 provides memory management with semantic search:

```python
from nat.plugins.mem0ai.memory import Mem0Memory

memory_backend = Mem0Memory(name="mem0_memory")
```

Requires the `nvidia-nat-mem0ai` plugin and a `MEM0_API_KEY` environment variable.

### Using Memory with Agents

Combine a memory backend with memory tools to create agents that remember users:

```python
from nat.tool.memory_tools.add_memory_tool import AddMemoryTool
from nat.tool.memory_tools.get_memory_tool import GetMemoryTool

# Create memory tools
add_memory = AddMemoryTool(nat_memory=memory_backend, name="add_memory")
get_memory = GetMemoryTool(nat_memory=memory_backend, name="get_memory")

# Add to agent's tools
agent = NatReActAgent(
    tools=[add_memory, get_memory, time_tool],
    llm=llm,
)
```

See the [Multi-Turn Chatbot notebook](../../../examples/notebooks/sdk/16_multiturn_chatbot.ipynb) for a complete example with Zep Cloud.

## Putting It Together

Here's a complete example combining components:

```python
import asyncio
from nat.llm.nim_llm import NimLLM
from nat.agent.react_agent.register import NatReActAgent
from nat.tool.datetime_tools import CurrentTimeTool
from nat.utils.sdk.nat_workflow import NatWorkflow

# Create LLM
llm = NimLLM(
    model_name="meta/llama-3.1-70b-instruct",
    temperature=0.0,
    max_tokens=1024,
)

# Create tools
time_tool = CurrentTimeTool()

# Try to import calculator (optional)
tools = [time_tool]
try:
    from nat_simple_calculator.register import CalculatorToolGroup
    calculator = CalculatorToolGroup()
    tools.append(calculator)
except ImportError:
    pass

# Create agent
agent = NatReActAgent(
    tools=tools,
    llm=llm,
    verbose=True,
    additional_instructions="Be helpful and concise.",
)

# Create workflow
workflow = NatWorkflow(entrypoint=agent)

async def main():
    response = await workflow.prompt("What is 25 * 4?")
    print(response)

asyncio.run(main())
```

:::{note}
**About the `name` parameter**: All SDK components accept an optional `name` parameter. This is only required when you need to export your workflow to a YAML configuration file using `save_to_config_file()`. For pure Python prototyping and development, you can omit `name` and let the SDK generate unique identifiers automatically. See [Exporting Configuration](./configuration.md) for details on when to use explicit names.
:::

## Next Steps

- [Creating Workflows](./workflows.md): Manage workflows with the SDK
- [Evaluation](./evaluation.md): Evaluate your agents
- [Optimization](./optimization.md): Tune hyperparameters
- [Finetuning](./finetuning.md): Train agents with reinforcement learning
- [Writing Custom Functions](../extend/custom-components/custom-functions/functions.md): Create your own tools
- [Memory Documentation](../build-workflows/memory.md): Advanced memory configuration

