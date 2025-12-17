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

from nat.data_models.llm import LLMBaseConfig
from nat.utils.sdk.nat_base import NatBase


class NatLLM(NatBase[LLMBaseConfig]):
    """Base class for LLM configurations.

    Can be used in two ways:

    1. **Subclass pattern**: Use specific LLM classes like NimLLM
       ```python
       llm = NimLLM(model_name="meta/llama-3.1-70b-instruct")
       ```

    2. **Factory pattern**: Pass a config object directly
       ```python
       config = NIMModelConfig(model="meta/llama-3.1-70b-instruct")
       llm = NatLLM(config=config, name="my_llm")
       ```

    The factory pattern is useful when you want to use an existing config
    without creating a custom class.
    """

    _marker_class: ClassVar[type] = LLMBaseConfig
