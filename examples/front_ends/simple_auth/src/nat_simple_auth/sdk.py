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

from nat.data_models.component_ref import AuthenticationRef
from nat.utils.sdk.nat_auth_provider import NatAuthProvider
from nat.utils.sdk.nat_function import NatFunction

from .ip_lookup import WhoAmIConfig


class WhoAmITool(WhoAmIConfig, NatFunction):

    auth_provider: AuthenticationRef = Field(description=("Reference to the authentication provider to use for "
                                                          "authentication before making the who am i request."),
                                             init=False)

    nat_auth_provider: NatAuthProvider = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set auth provider name from nat_auth_provider object if provided."""
        if self.nat_auth_provider:
            self.auth_provider = AuthenticationRef(value=self.nat_auth_provider.computed_name)
        return self
