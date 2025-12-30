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
SDK classes for Redis plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import EmbedderRef
from nat.utils.sdk.nat_embedder import NatEmbedder
from nat.utils.sdk.nat_memory import NatMemory
from nat.utils.sdk.nat_object_store import NatObjectStore

from .memory import RedisMemoryClientConfig
from .object_store import RedisObjectStoreClientConfig


class RedisObjectStore(RedisObjectStoreClientConfig, NatObjectStore):
    """Redis Object Store Provider"""


class RedisMemory(RedisMemoryClientConfig, NatMemory):
    """Redis Memory Provider"""

    embedder: EmbedderRef = Field(
        default=EmbedderRef(value=""),
        description=("Instance name of the memory client instance from the workflow "
                     "configuration object."),
        init=False,
    )

    embedder_obj: NatEmbedder = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set embedder reference from embedder object if provided."""
        if self.embedder_obj:
            self.embedder = EmbedderRef(value=self.embedder_obj.computed_name)
        return self
