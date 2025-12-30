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
from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_function import NatFunction

from .hotel_price_tool import HotelPriceToolConfig
from .local_events_tool import LocalEventsToolConfig
from .register import SKTravelPlanningWorkflowConfig


class SKTravelPlanningWorkflow(SKTravelPlanningWorkflowConfig, NatFunction):
    """Semantic Kernel Travel Planning Workflow"""

    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="The list of tools to provide to the semantic kernel.",
                                          init=False)
    llm_name: LLMRef = Field(description="The LLM model to use with the semantic kernel.",
                             default=LLMRef(value=""),
                             init=False)

    tools: list[NatFunction] | None = Field(default=None, exclude=True)
    llm: NatFunction = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set component names from objects if they are provided."""
        if self.tools and len(self.tools) > 0:
            self.tool_names = [FunctionRef(value=tool.computed_name) for tool in self.tools]
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class HotelPriceTool(HotelPriceToolConfig, NatFunction):
    pass


class LocalEventsTool(LocalEventsToolConfig, NatFunction):
    pass
