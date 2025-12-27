# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Category configuration for the Workflow Builder API.

Defines:
- UI_CATEGORIES: Which component categories are available in the UI
- CATEGORY_TO_REF_TYPE: Maps categories to their output RefType
- CATEGORY_METADATA: Display names and descriptions for each category
- SDK_CLASS_METADATA: SDK workflow classes (not registered in type registry)
"""

from pydantic import BaseModel

from nat.utils.sdk.nat_evaluation import NatEvaluation
from nat.utils.sdk.nat_finetuner import NatFinetuner
from nat.utils.sdk.nat_general_configuraton import NatGeneralConfiguration
from nat.utils.sdk.nat_optimizer import NatOptimizer
from nat.utils.sdk.nat_workflow import NatWorkflow
from nat.workflow_builder_api.models import ComponentCategory
from nat.workflow_builder_api.models import RefType

# =============================================================================
# UI CATEGORIES
# =============================================================================

UI_CATEGORIES: set[ComponentCategory] = {
    ComponentCategory.LLM,
    ComponentCategory.EMBEDDER,
    ComponentCategory.AGENT,
    ComponentCategory.FUNCTION,
    ComponentCategory.FUNCTION_GROUP,
    ComponentCategory.RETRIEVER,
    ComponentCategory.MEMORY,
    ComponentCategory.OBJECT_STORE,
    ComponentCategory.AUTHENTICATION,
    ComponentCategory.MIDDLEWARE,
    # Front-end and observability
    ComponentCategory.FRONT_END,
    ComponentCategory.LOGGER,
    ComponentCategory.TELEMETRY_EXPORTER,
    # Evaluation
    ComponentCategory.EVALUATOR,
    # Finetuning components
    ComponentCategory.TRAINER,
    ComponentCategory.TRAJECTORY_BUILDER,
    ComponentCategory.TRAINER_ADAPTER,
    # Test-Time Compute strategies
    ComponentCategory.TTC_STRATEGY,
    # Workflow-level configuration containers
    ComponentCategory.NAT_WORKFLOW,
    ComponentCategory.GENERAL_CONFIG,
    ComponentCategory.EVALUATION_CONFIG,
    ComponentCategory.OPTIMIZER_CONFIG,
    ComponentCategory.FINETUNER_CONFIG,
}

# =============================================================================
# CATEGORY TO REF TYPE MAPPING
# =============================================================================

CATEGORY_TO_REF_TYPE: dict[ComponentCategory, RefType] = {
    ComponentCategory.LLM: RefType.LLM,
    ComponentCategory.EMBEDDER: RefType.EMBEDDER,
    ComponentCategory.FUNCTION: RefType.FUNCTION,
    ComponentCategory.FUNCTION_GROUP: RefType.FUNCTION_GROUP,
    ComponentCategory.AGENT: RefType.FUNCTION,  # Agents can be used as functions/tools
    ComponentCategory.RETRIEVER: RefType.RETRIEVER,
    ComponentCategory.MEMORY: RefType.MEMORY,
    ComponentCategory.OBJECT_STORE: RefType.OBJECT_STORE,
    ComponentCategory.AUTHENTICATION: RefType.AUTHENTICATION,
    ComponentCategory.MIDDLEWARE: RefType.MIDDLEWARE,
    # Front-end and observability
    ComponentCategory.FRONT_END: RefType.FRONT_END,
    ComponentCategory.LOGGER: RefType.LOGGER,
    ComponentCategory.TELEMETRY_EXPORTER: RefType.TELEMETRY_EXPORTER,
    # Evaluation
    ComponentCategory.EVALUATOR: RefType.EVALUATOR,
    # Finetuning components
    ComponentCategory.TRAINER: RefType.TRAINER,
    ComponentCategory.TRAJECTORY_BUILDER: RefType.TRAJECTORY_BUILDER,
    ComponentCategory.TRAINER_ADAPTER: RefType.TRAINER_ADAPTER,
    # Test-Time Compute strategies
    ComponentCategory.TTC_STRATEGY: RefType.TTC_STRATEGY,
    # Workflow-level configuration containers
    ComponentCategory.NAT_WORKFLOW: RefType.NAT_WORKFLOW,
    ComponentCategory.GENERAL_CONFIG: RefType.GENERAL_CONFIG,
    ComponentCategory.EVALUATION_CONFIG: RefType.EVALUATION_CONFIG,
    ComponentCategory.OPTIMIZER_CONFIG: RefType.OPTIMIZER_CONFIG,
    ComponentCategory.FINETUNER_CONFIG: RefType.FINETUNER_CONFIG,
}

# =============================================================================
# CATEGORY DISPLAY METADATA
# =============================================================================

CATEGORY_METADATA: dict[ComponentCategory, tuple[str, str]] = {
    ComponentCategory.LLM: ("LLM Providers", "Large Language Model providers for reasoning and generation"),
    ComponentCategory.EMBEDDER: ("Embedders", "Text embedding models for semantic search and retrieval"),
    ComponentCategory.AGENT: ("Agents", "AI agents that can reason, use tools, and complete complex tasks"),
    ComponentCategory.FUNCTION: ("Functions", "Custom functions and tools that agents can use"),
    ComponentCategory.FUNCTION_GROUP: ("Function Groups", "Groups of related functions"),
    ComponentCategory.RETRIEVER: ("Retrievers", "Document retrievers for RAG workflows"),
    ComponentCategory.MEMORY: ("Memory", "Persistent memory storage for agent context"),
    ComponentCategory.OBJECT_STORE: ("Object Stores", "Object storage backends (S3, etc.)"),
    ComponentCategory.AUTHENTICATION: ("Authentication", "API authentication providers"),
    ComponentCategory.MIDDLEWARE: ("Middleware", "Request/response middleware components"),
    # Front-end and observability
    ComponentCategory.FRONT_END: ("Front Ends", "Deployment front ends (FastAPI, Console, MCP)"),
    ComponentCategory.LOGGER: ("Loggers", "Logging configurations for runtime observability"),
    ComponentCategory.TELEMETRY_EXPORTER: ("Telemetry", "Telemetry exporters for tracing and metrics"),
    # Evaluation
    ComponentCategory.EVALUATOR: ("Evaluators", "Evaluation metrics for testing workflow quality"),
    # Finetuning components
    ComponentCategory.TRAINER: ("Trainers", "Training loop orchestrators for finetuning"),
    ComponentCategory.TRAJECTORY_BUILDER: ("Trajectory Builders", "Training data collectors"),
    ComponentCategory.TRAINER_ADAPTER: ("Trainer Adapters", "Training backend adapters"),
    # Test-Time Compute strategies
    ComponentCategory.TTC_STRATEGY: ("TTC Strategies", "Test-Time Compute strategies for enhanced reasoning"),
    # Workflow-level configuration containers
    ComponentCategory.NAT_WORKFLOW: ("Workflow", "Main workflow entry point"),
    ComponentCategory.GENERAL_CONFIG: ("General Config", "Loggers, telemetry, and front-end configuration"),
    ComponentCategory.EVALUATION_CONFIG: ("Evaluation", "Evaluation configuration with evaluators"),
    ComponentCategory.OPTIMIZER_CONFIG: ("Optimizer", "Hyperparameter optimization configuration"),
    ComponentCategory.FINETUNER_CONFIG: ("Finetuner", "Model finetuning configuration"),
}

# =============================================================================
# CATEGORY TO CONFIG SECTION MAPPING
# =============================================================================

# Maps component categories to their YAML config section names.
# This is used when exporting workflows to YAML config files.
CATEGORY_TO_CONFIG_SECTION: dict[ComponentCategory, str] = {
    ComponentCategory.LLM: "llms",
    ComponentCategory.EMBEDDER: "embedders",
    ComponentCategory.AGENT: "functions",  # Agents are registered as functions
    ComponentCategory.FUNCTION: "functions",
    ComponentCategory.FUNCTION_GROUP: "function_groups",
    ComponentCategory.RETRIEVER: "retrievers",
    ComponentCategory.MEMORY: "memory",
    ComponentCategory.OBJECT_STORE: "object_stores",
    ComponentCategory.AUTHENTICATION: "authentication",
    ComponentCategory.MIDDLEWARE: "middleware",
    # Front-end and observability (nested under general)
    ComponentCategory.FRONT_END: "front_end",
    ComponentCategory.LOGGER: "loggers",
    ComponentCategory.TELEMETRY_EXPORTER: "telemetry_exporters",
    # Evaluation (nested under eval)
    ComponentCategory.EVALUATOR: "evaluators",
    # Finetuning components (nested under finetuner)
    ComponentCategory.TRAINER: "trainers",
    ComponentCategory.TRAJECTORY_BUILDER: "trajectory_builders",
    ComponentCategory.TRAINER_ADAPTER: "trainer_adapters",
    # Test-Time Compute strategies
    ComponentCategory.TTC_STRATEGY: "ttc_strategies",
    # Workflow-level (special handling)
    ComponentCategory.NAT_WORKFLOW: "workflow",
    ComponentCategory.GENERAL_CONFIG: "general",
    ComponentCategory.EVALUATION_CONFIG: "eval",
    ComponentCategory.OPTIMIZER_CONFIG: "optimizer",
    ComponentCategory.FINETUNER_CONFIG: "finetuner",
}

# =============================================================================
# SDK CLASS METADATA
# =============================================================================

# Workflow SDK classes that are NOT registered in the type registry.
# Format: (sdk_class, module_name, icon_url)
SDK_CLASS_METADATA: dict[ComponentCategory, tuple[type[BaseModel], str, str | None]] = {
    ComponentCategory.NAT_WORKFLOW: (
        NatWorkflow,
        "nat.utils.sdk.nat_workflow",
        "https://cdn.simpleicons.org/nvidia/76B900",
    ),
    ComponentCategory.GENERAL_CONFIG: (
        NatGeneralConfiguration,
        "nat.utils.sdk.nat_general_configuraton",
        "https://cdn.simpleicons.org/gnubash/4EAA25",
    ),
    ComponentCategory.EVALUATION_CONFIG: (
        NatEvaluation,
        "nat.utils.sdk.nat_evaluation",
        "https://cdn.simpleicons.org/pytest/0A9EDC",
    ),
    ComponentCategory.OPTIMIZER_CONFIG: (
        NatOptimizer,
        "nat.utils.sdk.nat_optimizer",
        "https://cdn.simpleicons.org/apachespark/E25A1C",
    ),
    ComponentCategory.FINETUNER_CONFIG: (
        NatFinetuner,
        "nat.utils.sdk.nat_finetuner",
        "https://cdn.simpleicons.org/pytorch/EE4C2C",
    ),
}
