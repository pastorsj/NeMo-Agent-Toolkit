# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import ClassVar

from nat.data_models.common import TypedBaseModel
from nat.data_models.function import FunctionBaseConfig
from nat.utils.sdk.nat_base import NatBase


class NatFunction(TypedBaseModel, NatBase):
    """Base class for function configurations that inherit from both FunctionBaseConfig and NatBase."""

    _marker_class: ClassVar[type] = FunctionBaseConfig


# class NatFunction(BaseModel):

#     config: FunctionBaseConfig = Field(description="Configuration for the tool")
#     function: Callable = Field(description="Generator yielding FunctionInfo instances")
#     name: str | None = Field(description="Name of the tool", default=None)

#     @property
#     def tool_name(self) -> str:
#         if self.name is not None:
#             return self.name
#         else:
#             return self.config.type

#     # Inspect `webquery_tool` to find the parameter whose type is a subclass of FunctionBaseConfig

#     def determine_function_name_param(self, function):
#         """Return (param_name, annotation, inferred_name) for the first parameter whose annotation is
#         a subclass of FunctionBaseConfig. If none found, returns (None, None, None).
#         """
#         sig = inspect.signature(function)
#         try:
#             hints = get_type_hints(function, globalns=globals(), localns=locals())
#         except Exception:
#             hints = {}

#         for param_name, param in sig.parameters.items():
#             ann = hints.get(param_name, param.annotation)
#             if ann is inspect._empty:
#                 continue
#             if self._is_subclass_function_base(ann):
#                 # If annotation is a typing wrapper, try to extract the raw class
#                 type_name = getattr(ann, '_typed_model_name', None)
#                 return param_name, ann, type_name
#         return None, None, None

#     def _is_subclass_function_base(self, tp) -> bool:
#         """Return True if `tp` is (or contains) a subclass of FunctionBaseConfig.
#         Handles direct classes, typing.Union, and typing.Annotated wrappers.
#         """
#         if tp is None:
#             return False
#         # Unwrap Annotated and similar wrappers
#         origin = get_origin(tp)
#         if origin is typing.Annotated:
#             args = get_args(tp)
#             if args:
#                 return self._is_subclass_function_base(args[0])
#         # Handle Union / Optional
#         if origin is typing.Union:
#             for a in get_args(tp):
#                 if self._is_subclass_function_base(a):
#                     return True
#             return False
#         # Direct class check
#         try:
#             if isinstance(tp, type) and issubclass(tp, FunctionBaseConfig):
#                 return True
#         except Exception:
#             pass
#         return False
