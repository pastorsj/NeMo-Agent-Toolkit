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
"""SDK classes for this example package."""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.utils.sdk.nat_function_group import NatFunctionGroup

from .user_report_tools import UserReportConfig


class UserReportToolGroup(UserReportConfig, NatFunctionGroup):
    """User Report Function Group"""

    object_store: ObjectStoreRef = Field(default=ObjectStoreRef(value=""),
                                         description="The object store to use for storing user reports",
                                         init=False)

    object_store_obj: ObjectStoreBaseConfig | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set object store reference from object store object if provided."""
        if self.object_store_obj:
            self.object_store = ObjectStoreRef(value=self.object_store_obj.computed_name)
        return self
