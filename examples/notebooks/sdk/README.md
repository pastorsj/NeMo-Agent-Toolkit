# NeMo Agent Toolkit SDK Tutorials

This directory contains comprehensive tutorials for the NeMo Agent Toolkit (NAT) Python SDK.

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

## Quick Start

1. **Install NAT SDK**:
   ```bash
   pip install nvidia-nemo-agent-toolkit
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

## SDK vs CLI Workflow

| Task | SDK (Python) | CLI |
|------|--------------|-----|
| **Prototyping** | ✅ Best | ⚠️ Limited |
| **Production** | ⚠️ Limited | ✅ Best |
| **Notebooks** | ✅ Yes | ❌ No |
| **Config Generation** | ✅ Yes | ❌ No |

**Typical Workflow**:
1. Build & prototype in Python (SDK)
2. Export to YAML configuration
3. Deploy with CLI (`nat run`, `nat serve`)

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

- [NAT Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html)
- [Example Workflows](../../)
