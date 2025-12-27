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

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import MiddlewareRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.sdk.nat_base import NatBase
from nat.utils.sdk.nat_middleware import NatMiddleware


class NatFunction(NatBase[FunctionBaseConfig]):
    """Base class for function configurations.

    Can be used in two ways:

    1. **Subclass pattern**: Use specific function classes like CurrentTimeTool
       ```python
       tool = CurrentTimeTool()
       ```

    2. **Factory pattern**: Pass a config object directly
       ```python
       config = CurrentTimeToolConfig()
       func = NatFunction(config=config, name="my_func")
       ```

    Middleware can be attached to functions:
       ```python
       middleware = MyMiddleware(...)
       tool = CurrentTimeTool(mw=[middleware])
       ```

    The factory pattern is useful when you want to use an existing config
    without creating a custom class.
    """

    _marker_class: ClassVar[type] = FunctionBaseConfig

    # SDK field for middleware - accepts NatMiddleware objects
    # Named 'mw' to avoid conflict with inherited 'middleware: list[str]' field
    mw: list[NatMiddleware] = Field(
        default_factory=list,
        exclude=True,
        description="List of middleware to apply to this function",
    )

    @model_validator(mode="after")
    def _set_middleware_refs(self) -> NatFunction:
        """Convert NatMiddleware objects to MiddlewareRef references."""
        if self.mw:
            middleware_refs = [MiddlewareRef(value=m.computed_name) for m in self.mw if isinstance(m, NatMiddleware)]

            # For subclass pattern: set on self.middleware (inherited from config)
            # For factory pattern: set on the wrapped config
            if hasattr(self, "middleware") and "middleware" in self.model_fields:
                self.middleware = middleware_refs
            elif self.config is not None and hasattr(self.config, "middleware"):
                self.config.middleware = middleware_refs
        return self
