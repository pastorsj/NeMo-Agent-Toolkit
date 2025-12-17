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

from nat.data_models.function import FunctionGroupBaseConfig
from nat.utils.sdk.nat_base import NatBase


class NatFunctionGroup(NatBase[FunctionGroupBaseConfig]):
    """Base class for function group configurations.

    Can be used in two ways:

    1. **Subclass pattern**: Use specific function group classes like MCPFunctionGroup
       ```python
       group = MCPFunctionGroup(...)
       ```

    2. **Factory pattern**: Pass a config object directly
       ```python
       config = MyFunctionGroupConfig(...)
       group = NatFunctionGroup(config=config, name="my_group")
       ```

    The factory pattern is useful when you want to use an existing config
    without creating a custom class.
    """

    _marker_class: ClassVar[type] = FunctionGroupBaseConfig
