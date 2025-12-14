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

# Evaluation with the SDK

Evaluation measures how well your agent performs on a test dataset. The SDK provides a programmatic interface for configuring and running evaluations.

## Prerequisites

Install the profiling extras:

```bash
pip install "nvidia-nat[profiling]"
```

## Evaluation Overview

Evaluation requires three components:

1. **Dataset**: Test cases with questions and expected answers
2. **Evaluators**: Metrics to measure performance (such as accuracy)
3. **Workflow**: The agent workflow to evaluate

## Creating an Evaluation Dataset

Evaluation datasets contain test cases with questions and expected answers:

```python
import json
from pathlib import Path

# Create dataset
eval_data = [
    {"id": "test_001", "question": "What is 2 + 2?", "answer": "The answer is 4."},
    {"id": "test_002", "question": "What is 10 * 5?", "answer": "The answer is 50."},
    {"id": "test_003", "question": "What is 100 / 4?", "answer": "The answer is 25."},
]

# Save to file
data_dir = Path("./data")
data_dir.mkdir(parents=True, exist_ok=True)

with open(data_dir / "eval_dataset.json", "w") as f:
    json.dump(eval_data, f, indent=2)
```

### Dataset Format

Each test case requires:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier for the test case |
| `question` | The input prompt to send to the agent |
| `answer` | The expected answer (ground truth) |

## Configuring Evaluators

Evaluators measure different aspects of performance:

```python
from nat.eval.rag_evaluator.register import RagasEvaluator

# Create an accuracy evaluator
accuracy_evaluator = RagasEvaluator(
    llm=llm,                    # LLM used for evaluation
    metric="AnswerAccuracy",    # Metric to compute
    name="accuracy",            # Unique name
)
```

### Available Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `AnswerAccuracy` | Correctness of answers | General Q&A |
| `AnswerSimilarity` | Semantic similarity to expected | Flexible matching |
| `Faithfulness` | Groundedness in context | RAG applications |
| `ContextRelevancy` | Relevance of retrieved context | RAG applications |

## Configuring Evaluation

Use `NatEvaluation` to configure the evaluation:

```python
from pathlib import Path
from nat.utils.sdk.nat_evaluation import NatEvaluation, EvalDatasetJsonConfig

evaluation = NatEvaluation(
    output_dir=Path("./eval_results"),
    dataset=EvalDatasetJsonConfig(
        file_path=Path("./data/eval_dataset.json"),
    ),
    evaluators=[accuracy_evaluator],
)

# Add to workflow
workflow.add_evaluator(evaluation)
```

### Dataset Configuration Options

**JSON Dataset:**

```python
from nat.utils.sdk.nat_evaluation import EvalDatasetJsonConfig

dataset = EvalDatasetJsonConfig(
    file_path=Path("./data/test.json"),
)
```

**CSV Dataset:**

```python
from nat.utils.sdk.nat_evaluation import EvalDatasetCsvConfig

dataset = EvalDatasetCsvConfig(
    file_path=Path("./data/test.csv"),
)
```

## Running Evaluation

Run the evaluation with `workflow.evaluate()`:

```python
await workflow.evaluate()
```

### Evaluation Parameters

```python
await workflow.evaluate(
    dataset="./data/custom_dataset.json",  # Override dataset
    reps=3,                                  # Run multiple times
    skip_workflow=False,                     # Skip if results exist
)
```

## Complete Example

```python
import asyncio
import json
from pathlib import Path
from nat.llm.nim_llm import NimLLM
from nat.agent.react_agent.register import NatReActAgent
from nat.tool.datetime_tools import CurrentTimeTool
from nat.eval.rag_evaluator.register import RagasEvaluator
from nat.utils.sdk.nat_evaluation import NatEvaluation, EvalDatasetJsonConfig
from nat.utils.sdk.nat_workflow import NatWorkflow

# Create workflow components
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

workflow = NatWorkflow(entrypoint=agent)

# Create evaluation dataset
eval_data = [
    {"id": "time_001", "question": "What time is it?", "answer": "The current time is displayed."},
]

data_dir = Path("./data")
data_dir.mkdir(parents=True, exist_ok=True)
with open(data_dir / "eval_dataset.json", "w") as f:
    json.dump(eval_data, f, indent=2)

# Configure evaluator
accuracy_evaluator = RagasEvaluator(
    llm=llm,
    metric="AnswerAccuracy",
    name="accuracy",
)

# Configure evaluation
evaluation = NatEvaluation(
    output_dir=Path("./eval_results"),
    dataset=EvalDatasetJsonConfig(file_path=data_dir / "eval_dataset.json"),
    evaluators=[accuracy_evaluator],
)

workflow.add_evaluator(evaluation)

# Run evaluation
async def main():
    await workflow.evaluate()
    print("Evaluation complete! Check ./eval_results for results.")

asyncio.run(main())
```

## Understanding Results

After evaluation, results are saved in the output directory:

```
eval_results/
├── workflow_output.json      # Raw workflow outputs
├── accuracy_output.json      # Per-metric results
└── ...
```

### Reading Results

```python
import json

with open("./eval_results/accuracy_output.json") as f:
    results = json.load(f)

print(f"Average Score: {results['average_score']:.1%}")

for item in results["eval_output_items"]:
    print(f"  {item['id']}: {item['score']:.1%}")
```

## Multiple Evaluators

Use multiple evaluators for comprehensive analysis:

```python
accuracy_eval = RagasEvaluator(llm=llm, metric="AnswerAccuracy", name="accuracy")
similarity_eval = RagasEvaluator(llm=llm, metric="AnswerSimilarity", name="similarity")

evaluation = NatEvaluation(
    output_dir=Path("./eval_results"),
    dataset=EvalDatasetJsonConfig(file_path=dataset_path),
    evaluators=[accuracy_eval, similarity_eval],
)
```

## CLI Alternative

After configuring evaluation, you can also run it via CLI:

```python
# Save configuration
workflow.save_to_config_file("eval_workflow.yaml")
```

```bash
# Run via CLI
nat eval --config_file eval_workflow.yaml
```

## Next Steps

- [Optimization](./optimization.md): Use evaluation results to optimize parameters
- [Evaluate Workflows](../workflows/evaluate.md): CLI-based evaluation reference
- [Custom Evaluators](../extend/custom-evaluator.md): Create your own metrics

