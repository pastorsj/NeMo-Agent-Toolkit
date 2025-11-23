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

from collections.abc import Callable

from pydantic import BaseModel
from pydantic import Field

from nat.observability.register import ConsoleLoggingMethodConfig
from nat.observability.register import FileLoggingMethod


class NatLogger(BaseModel):
    config: ConsoleLoggingMethodConfig | FileLoggingMethod = Field(description="Configuration of the logger")
    function: Callable = Field(description="Generator yielding FunctionInfo instances")
    name: str | None = Field(description="Name of the logger", default=None)

    @property
    def logger_name(self) -> str:
        if self.name is not None:
            return self.name
        return self.type
