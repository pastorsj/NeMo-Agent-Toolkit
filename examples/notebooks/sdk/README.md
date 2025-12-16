# NeMo Agent Toolkit SDK Tutorials

This directory contains comprehensive tutorials for the NeMo Agent Toolkit (NAT) Python SDK.

## Why an SDK?

NeMo Agent Toolkit uses a **configuration-driven paradigm** where workflows are defined declaratively in YAML files. The configuration file is a flattened representation of all components—functions, LLMs, embedders, retrievers, and memory—loaded asynchronously at startup. This design enables:

- **Performance**: Parallel initialization, dependency resolution, and shared resources
- **Reproducibility**: Version-controllable, shareable workflow specifications
- **Production-Ready**: Same config works in development and production

The SDK bridges the gap for developers who prefer building workflows programmatically. It generates the exact same YAML configurations you would write manually, providing a seamless transition from development to production.

## Component Overview

From the [documentation](../../../docs/source/index.md):

| Component | Documentation | Description |
|-----------|--------------|-------------|
| **Functions** | [Functions](../../../docs/source/build-workflows/functions-and-function-groups/functions.md) | Main building blocks with input/output schemas |
| **Function Groups** | [Function Groups](../../../docs/source/build-workflows/functions-and-function-groups/function-groups.md) | Package related functions to share config |
| **Agents** | [About Workflows](../../../docs/source/build-workflows/about-building-workflows.md) | ReAct, Tool Calling, ReWOO, Router, and more |
| **LLMs** | [LLMs](../../../docs/source/build-workflows/llms/index.md) | NIM, OpenAI, AWS Bedrock, Azure, LiteLLM |
| **Embedders** | [Embedders](../../../docs/source/build-workflows/embedders.md) | NIM, OpenAI, Azure OpenAI |
| **Retrievers** | [Retrievers](../../../docs/source/build-workflows/retrievers.md) | NeMo Retriever, Milvus |
| **Memory** | [Memory](../../../docs/source/build-workflows/memory.md) | Mem0, Redis, Zep |
| **MCP** | [MCP](../../../docs/source/build-workflows/mcp-client.md) | Model Context Protocol integration |
| **A2A** | [A2A](../../../docs/source/components/integrations/a2a.md) | Agent-to-Agent Protocol |

## Tutorial Overview

| # | Tutorial | Description |
|---|----------|-------------|
| 01 | [Installation & Getting Started](./01_installation_getting_started.ipynb) | Setup, API keys, first workflow |
| 02 | [Your First Agent](./02_your_first_agent.ipynb) | Build a calculator agent step-by-step |
| 03 | [Agent Types](./03_agents.ipynb) | ReAct, Tool Calling, ReWOO agents |
| 04 | [Functions & Tools](./04_functions_and_tools.ipynb) | Built-in tools, MCP, A2A |
| 05 | [LLM Providers](./05_llms.ipynb) | NIM, OpenAI configuration |
| 06 | [Memory](./06_memory.ipynb) | Mem0, Redis memory |
| 07 | [Retrievers](./07_retrievers.ipynb) | RAG with Milvus |
| 08 | [Configuration Guide](./08_configuration_guide.ipynb) | YAML structure and options |
| 09 | [Evaluation](./09_evaluation.ipynb) | Test workflow performance |
| 10 | [Profiling](./10_profiling.ipynb) | Measure latency and costs |
| 11 | [Optimization](./11_optimization.ipynb) | Improve prompts and parameters |
| 12 | [Observability](./12_observability.ipynb) | Phoenix tracing and debugging |
| 13 | [Custom Functions Inline](./13_custom_functions_inline.ipynb) | Create and register functions inline for prototyping |
| 14 | [Finetuning](./14_finetuning.ipynb) | Configure RL finetuning with curriculum learning |
| 15 | [Multi-Turn Chatbot](./15_multiturn_chatbot.ipynb) | Build a chatbot with long-term memory and web search |

## Quick Start

1. **Install NAT SDK**:
   ```bash
   uv pip install nvidia-nemo-agent-toolkit
   ```

2. **Set API Key**:
   ```bash
   export NVIDIA_API_KEY="your-api-key"
   ```

3. **Start with Tutorial 01**:
   Open `01_installation_getting_started.ipynb` in Jupyter or VS Code.

## Prerequisites

- Python 3.10+
- NVIDIA API Key (get one at [build.nvidia.com](https://build.nvidia.com/))
- Jupyter Notebook or VS Code with Jupyter extension

## Development Workflow

| Phase | Approach | Why |
|-------|----------|-----|
| **Prototyping** | SDK (Python) | IDE support, rapid iteration, familiar syntax |
| **Testing** | SDK or CLI | Run workflows, validate behavior |
| **Production** | CLI + YAML | Version control, reproducibility, CI/CD integration |

**Recommended Workflow**:
1. **Develop** - Build and iterate in Python using the SDK
2. **Export** - Generate YAML configuration with `workflow.save_to_config_file()`
3. **Deploy** - Run with CLI (`nat run`, `nat serve`) or integrate into your infrastructure

## CLI Commands Reference

After generating a config file from the SDK:

```bash
# Run a workflow
nat run --config_file config.yaml --input "Your question"

# Serve as API
nat serve --config_file config.yaml --port 8000

# Evaluate
nat eval --config_file config.yaml

# Optimize
nat optimize --config_file config.yaml
```

## Additional Resources

- [NAT Documentation](../../../docs/source/index.md)
- [Example Workflows](../../)
