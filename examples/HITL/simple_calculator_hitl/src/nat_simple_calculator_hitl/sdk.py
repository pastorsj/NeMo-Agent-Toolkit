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

from nat.data_models.component_ref import FunctionRef
from nat.utils.sdk.nat_function import NatFunction

from .retry_react_agent import RetryReactAgentConfig
from .retry_react_agent import TimeZonePromptConfig


class RetryReactAgent(RetryReactAgentConfig, NatFunction):
    """Retry React Agent"""

    hitl_approval_fn: FunctionRef = Field(description="The hitl approval function",
                                          default=FunctionRef(value=""),
                                          init=False)
    react_agent_fn: FunctionRef = Field(description="The react agent to retry",
                                        default=FunctionRef(value=""),
                                        init=False)

    hitl_approval_function: NatFunction = Field(exclude=True)
    react_agent_function: NatFunction = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.hitl_approval_function:
            self.hitl_approval_fn = FunctionRef(value=self.hitl_approval_function.computed_name)
        if self.react_agent_function:
            self.react_agent_fn = FunctionRef(value=self.react_agent_function.computed_name)
        return self


class TimeZonePrompt(TimeZonePromptConfig, NatFunction):
    """Time Zone Prompt"""
    pass
