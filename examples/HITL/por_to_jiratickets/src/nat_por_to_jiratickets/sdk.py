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
from nat.utils.sdk.nat_llm import NatLLM

from .extract_por_tool import ExtractPORToolConfig
from .extract_por_tool import ShowTicketsToolConfig
from .hitl_approval_tool import HITLApprovalFnConfig
from .jira_tickets_tool import CreateJiraToolConfig
from .jira_tickets_tool import GetJiraToolConfig


class ExtractPORTool(ExtractPORToolConfig, NatFunction):

    llm: LLMRef = Field(description="LLM to use for extraction", init=False)

    nat_llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.nat_llm:
            self.llm = LLMRef(value=self.nat_llm.computed_name)
        return self


class ShowTicketsTool(ShowTicketsToolConfig, NatFunction):
    pass


class HITLApprovalTool(HITLApprovalFnConfig, NatFunction):
    pass


class CreateJiraTicketsTool(CreateJiraToolConfig, NatFunction):

    hitl_approval_fn: FunctionRef = Field(description="Reference to the HITL approval function", init=False)

    hitl_approval_tool: NatFunction = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        if self.hitl_approval_tool:
            self.hitl_approval_fn = FunctionRef(value=self.hitl_approval_tool.computed_name)
        return self


class GetJiraTicketsTool(GetJiraToolConfig, NatFunction):
    pass
