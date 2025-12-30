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
"""
SDK classes for Test plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

from .embedder import EmbedderTestConfig
from .functions import ConstantFunctionConfig
from .functions import EchoFunctionConfig
from .functions import StreamingConstantFunctionConfig
from .functions import StreamingEchoFunctionConfig
from .llm import TestLLMConfig


class TestLLM(TestLLMConfig, NatLLM):
    """Test LLM Provider"""


class EchoFunction(EchoFunctionConfig, NatFunction):
    """Echo Function"""


class StreamingEchoFunction(StreamingEchoFunctionConfig, NatFunction):
    """Streaming Echo Function"""


class ConstantFunction(ConstantFunctionConfig, NatFunction):
    """Constant Function"""


class StreamingConstantFunction(StreamingConstantFunctionConfig, NatFunction):
    """Streaming Constant Function"""


class EmbedderTest(EmbedderTestConfig, NatEmbedder):
    """Embedder Test Provider"""
