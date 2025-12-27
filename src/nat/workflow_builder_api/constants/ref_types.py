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
Reference type mappings and type detection utilities.

Provides:
- COMPONENT_REF_TO_REF_TYPE: Maps ComponentRef class names to RefType
- SDK_CLASS_TO_REF_TYPE: Maps SDK class names (Nat*) to RefType
- Type detection utilities for annotations
"""

import types
import typing
from typing import Any
from typing import get_args
from typing import get_origin

from nat.workflow_builder_api.models import RefType

# =============================================================================
# COMPONENT REF MAPPINGS
# =============================================================================

COMPONENT_REF_TO_REF_TYPE: dict[str, RefType] = {
    "LLMRef": RefType.LLM,
    "EmbedderRef": RefType.EMBEDDER,
    "FunctionRef": RefType.FUNCTION,
    "FunctionGroupRef": RefType.FUNCTION_GROUP,
    "RetrieverRef": RefType.RETRIEVER,
    "MemoryRef": RefType.MEMORY,
    "ObjectStoreRef": RefType.OBJECT_STORE,
    "AuthenticationRef": RefType.AUTHENTICATION,
    "MiddlewareRef": RefType.MIDDLEWARE,
    "TTCStrategyRef": RefType.TTC_STRATEGY,
}

# =============================================================================
# SDK CLASS MAPPINGS
# =============================================================================

SDK_CLASS_TO_REF_TYPE: dict[str, RefType] = {
    # Core components
    "NatLLM": RefType.LLM,
    "NatEmbedder": RefType.EMBEDDER,
    "NatFunction": RefType.FUNCTION,
    "NatFunctionGroup": RefType.FUNCTION_GROUP,
    "NatAgent": RefType.FUNCTION,
    "NatRetriever": RefType.RETRIEVER,
    "NatMemory": RefType.MEMORY,
    "NatObjectStore": RefType.OBJECT_STORE,
    "NatAuthProvider": RefType.AUTHENTICATION,
    "NatMiddleware": RefType.MIDDLEWARE,  # Front-end and observability
    "NatFrontEnd": RefType.FRONT_END,
    "NatLogger": RefType.LOGGER,
    "NatTelemetryExporter": RefType.TELEMETRY_EXPORTER,  # Evaluation
    "NatEvaluator": RefType.EVALUATOR,  # Workflow configuration containers
    "NatGeneralConfiguration": RefType.GENERAL_CONFIG,
    "NatEvaluation": RefType.EVALUATION_CONFIG,
    "NatOptimizer": RefType.OPTIMIZER_CONFIG,
    "NatFinetuner": RefType.FINETUNER_CONFIG,  # Finetuning components
    "NatTrainer": RefType.TRAINER,
    "NatTrajectoryBuilder": RefType.TRAJECTORY_BUILDER,
    "NatTrainerAdapter": RefType.TRAINER_ADAPTER,
}

# =============================================================================
# TYPE DETECTION UTILITIES
# =============================================================================


def get_type_name(annotation: Any) -> str:
    """Get the type name from an annotation."""
    if annotation is None:
        return ""
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def get_all_component_ref_types(annotation: Any) -> tuple[bool, list[RefType], bool]:
    """
    Check if an annotation is a ComponentRef type and return ALL ref types it accepts.

    Handles union types like `FunctionRef | FunctionGroupRef` and lists of unions.

    Returns:
        Tuple of (is_ref, ref_types, is_list)
    """
    if annotation is None:
        return False, [], False

    type_name = get_type_name(annotation)

    # Direct ComponentRef type
    if type_name in COMPONENT_REF_TO_REF_TYPE:
        return True, [COMPONENT_REF_TO_REF_TYPE[type_name]], False

    # Direct SDK type
    if type_name in SDK_CLASS_TO_REF_TYPE:
        return True, [SDK_CLASS_TO_REF_TYPE[type_name]], False

    # Check for list/sequence types
    origin = get_origin(annotation)
    if origin in (list, list, typing.Sequence):
        args = get_args(annotation)
        if args:
            # Check for Union inside list
            if len(args) == 1:
                inner_origin = get_origin(args[0])
                is_union = inner_origin is typing.Union or isinstance(args[0], types.UnionType)
                if is_union:
                    inner_args = get_args(args[0])
                    ref_types = []
                    for inner_arg in inner_args:
                        inner_type_name = get_type_name(inner_arg)
                        if inner_type_name in SDK_CLASS_TO_REF_TYPE:
                            ref_types.append(SDK_CLASS_TO_REF_TYPE[inner_type_name])
                        elif inner_type_name in COMPONENT_REF_TO_REF_TYPE:
                            ref_types.append(COMPONENT_REF_TO_REF_TYPE[inner_type_name])
                    if ref_types:
                        return True, ref_types, True

            # Single type in list
            for arg in args:
                inner_type_name = get_type_name(arg)
                if inner_type_name in COMPONENT_REF_TO_REF_TYPE:
                    return True, [COMPONENT_REF_TO_REF_TYPE[inner_type_name]], True
                if inner_type_name in SDK_CLASS_TO_REF_TYPE:
                    return True, [SDK_CLASS_TO_REF_TYPE[inner_type_name]], True

    # Check for Optional/Union types
    is_union = origin is typing.Union or isinstance(annotation, types.UnionType)
    if is_union:
        args = get_args(annotation)
        ref_types = []
        for arg in args:
            if arg is type(None):
                continue
            is_ref, inner_ref_types, _ = get_all_component_ref_types(arg)
            if is_ref:
                ref_types.extend(inner_ref_types)
        if ref_types:
            return True, ref_types, False

    return False, [], False


def is_component_ref_type(annotation: Any) -> tuple[bool, RefType | None, bool]:
    """
    Check if an annotation is a ComponentRef type, SDK type, or a list of them.

    Returns:
        Tuple of (is_ref, ref_type, is_list) where ref_type is the primary type.
    """
    is_ref, ref_types, is_list = get_all_component_ref_types(annotation)
    primary_ref_type = ref_types[0] if ref_types else None
    return is_ref, primary_ref_type, is_list
