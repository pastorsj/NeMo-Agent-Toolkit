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

from pydantic import Field
from pydantic import model_validator

from nat.authentication.interfaces import AuthProviderBaseConfig
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.a2a.client.client_config import A2AClientConfig
from nat.utils.sdk.nat_function_group import NatFunctionGroup


class A2AClient(A2AClientConfig, NatFunctionGroup):
    """A2A Client Function Group.

    SDK wrapper for connecting to remote A2A agents and using their capabilities
    as tools in your workflow.

    Example:
        ```python
        from nat.plugins.a2a.client.a2a_client import A2AClient

        # Connect to a remote A2A agent
        currency_agent = A2AClient(
            url="http://localhost:11000",
            task_timeout=60,
            name="currency_agent",
        )

        # Use it as a tool in a ReAct agent
        agent = NatReActAgent(
            tools=[currency_agent],
            llm=llm,
        )
        ```
    """

    # Optional auth provider object for SDK usage
    auth_provider_obj: AuthProviderBaseConfig | None = Field(
        default=None,
        exclude=True,
        description="Authentication provider object for SDK usage.",
    )

    @model_validator(mode='after')
    def set_references(self):
        """Set auth provider reference from object if provided."""
        if self.auth_provider_obj:
            self.auth_provider = AuthenticationRef(value=self.auth_provider_obj.computed_name)
        return self
