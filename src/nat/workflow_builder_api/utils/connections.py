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
Connection port extraction utilities.

Extracts connection port information from Pydantic models,
including both config types (ComponentRef) and SDK types (Nat* classes).
"""

import typing

from pydantic import BaseModel

from nat.workflow_builder_api.constants.ref_types import get_all_component_ref_types
from nat.workflow_builder_api.constants.ref_types import get_type_name
from nat.workflow_builder_api.models import ConnectionPort


def _extract_ports_from_model(
    model_type: type[BaseModel],
    field_path: str = "",
) -> list[ConnectionPort]:
    """
    Recursively extract connection ports from a Pydantic model and its nested models.

    Args:
        model_type: The Pydantic model type to inspect
        field_path: Dot-separated path prefix for nested fields

    Returns:
        List of ConnectionPort objects
    """
    ports = []

    for field_name, field_info in model_type.model_fields.items():
        if field_name.startswith("_"):
            continue

        annotation = field_info.annotation
        current_path = f"{field_path}.{field_name}" if field_path else field_name

        # Check if this field is a ComponentRef
        is_ref, all_ref_types, is_list = get_all_component_ref_types(annotation)

        if is_ref and all_ref_types:
            primary_ref_type = all_ref_types[0]
            # Use the full path as field_name for nested fields
            ports.append(
                ConnectionPort(
                    field_name=current_path,
                    ref_type=primary_ref_type,
                    accepts_ref_types=all_ref_types,
                    required=field_info.is_required(),
                    is_list=is_list,
                    description=field_info.description,
                    title=field_name.replace("_", " ").title(),
                ))
        else:
            # Check if this is a nested BaseModel that might contain refs
            # Get the actual type, handling Optional and other wrappers
            inner_type = annotation
            origin = typing.get_origin(annotation)
            if origin is type(None) or origin is typing.Union:
                # For Union types (including Optional), check each arg
                args = typing.get_args(annotation)
                for arg in args:
                    if arg is type(None):
                        continue
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        inner_type = arg
                        break

            # If it's a BaseModel subclass, recurse
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                nested_ports = _extract_ports_from_model(inner_type, current_path)
                ports.extend(nested_ports)

    return ports


def extract_connection_ports(config_type: type[BaseModel]) -> list[ConnectionPort]:
    """
    Extract connection port information from a Pydantic model.

    Inspects the model's fields to find ComponentRef types
    (LLMRef, EmbedderRef, FunctionRef, etc.) and creates ConnectionPort objects.

    This function recursively scans nested BaseModel fields to find all ComponentRef
    fields (e.g., server.auth_provider in MCP configs).
    """
    return _extract_ports_from_model(config_type)


def find_sdk_class_for_config(config_type: type[BaseModel]) -> type | None:
    """
    Find the SDK class that corresponds to a config type.

    For example, given ReActAgentWorkflowConfig, find NatReActAgent.
    """
    try:
        for subclass in config_type.__subclasses__():
            class_name = subclass.__name__
            if class_name.startswith("Nat"):
                return subclass
            for parent in subclass.__mro__:
                parent_name = parent.__name__
                if parent_name.startswith("Nat") and parent_name not in ("NatBase", "NatFunction", "NatAgent"):
                    return subclass
    except Exception:
        pass
    return None


def extract_sdk_connection_ports(sdk_class: type) -> list[ConnectionPort]:
    """
    Extract connection ports from an SDK class.

    SDK classes have fields like `llm: NatLLM` that represent component dependencies.
    """
    ports = []

    if not hasattr(sdk_class, "model_fields"):
        return ports

    for field_name, field_info in sdk_class.model_fields.items():
        if field_name.startswith("_"):
            continue
        if field_name.endswith("_name") or field_name.endswith("_names"):
            continue

        annotation = field_info.annotation
        is_ref, all_ref_types, is_list = get_all_component_ref_types(annotation)

        if is_ref and all_ref_types:
            primary_ref_type = all_ref_types[0]
            description = field_info.description
            if not description:
                type_name = get_type_name(annotation)
                if len(all_ref_types) > 1:
                    type_names = " or ".join(rt.value.replace("_", " ").title() for rt in all_ref_types)
                    description = f"{type_names} component to use"
                else:
                    description = f"{type_name.replace('Nat', '')} component to use"

            ports.append(
                ConnectionPort(
                    field_name=field_name,
                    ref_type=primary_ref_type,
                    accepts_ref_types=all_ref_types,
                    required=field_info.is_required(),
                    is_list=is_list,
                    description=description,
                    title=field_name.replace("_", " ").title(),
                ))

    return ports


def extract_all_connection_ports(config_type: type[BaseModel]) -> list[ConnectionPort]:
    """
    Extract all connection ports from both config type and its SDK class.

    Combines ports from:
    1. Config fields with ComponentRef types (LLMRef, FunctionRef, etc.)
    2. SDK class fields with Nat* types (NatLLM, NatFunction, etc.)
    """
    config_ports = extract_connection_ports(config_type)

    sdk_class = find_sdk_class_for_config(config_type)
    if sdk_class:
        sdk_ports = extract_sdk_connection_ports(sdk_class)

        for sdk_port in sdk_ports:
            existing = next((p for p in config_ports if p.ref_type == sdk_port.ref_type), None)
            if existing:
                if not existing.field_name.endswith("_name") and not existing.field_name.endswith("_names"):
                    continue
                config_ports.remove(existing)
            config_ports.append(sdk_port)

    return config_ports
