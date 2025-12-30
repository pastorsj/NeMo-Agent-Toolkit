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
SDK classes for Agno plugin.

This module contains SDK wrapper classes that combine configuration with
NatBase classes for use in the Python SDK.
"""

from pydantic import Field
from pydantic import PrivateAttr
from pydantic import model_validator

from nat.data_models.common import OptionalSecretStr
from nat.utils.sdk.nat_env_var import NatEnvironmentVariable
from nat.utils.sdk.nat_function import NatFunction

from .tools.serp_api_tool import SerpApiToolConfig


class SerpApiTool(SerpApiToolConfig, NatFunction):
    """SerpAPI Search Tool.

    Supports environment variable references for API keys:

    Example:
        ```python
        # Using env var reference (recommended for portability):
        tool = SerpApiTool(
            api_key_env=NatEnvironmentVariable("SERP_API_KEY"),
            max_results=5,
        )
        ```
    """

    # Override api_key from parent with init=False to signal users should use api_key_env
    api_key: OptionalSecretStr = Field(
        default=None,
        init=False,
        description="The API key for SerpAPI - use api_key_env instead for SDK usage.",
    )

    # Environment variable reference for api_key
    api_key_env: NatEnvironmentVariable | None = Field(
        default=None,
        exclude=True,
        description="Environment variable name for the API key (e.g., 'SERP_API_KEY').",
    )

    # Private attribute to track env var references for serialization
    _env_var_refs: dict[str, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def resolve_env_vars(self):
        """Resolve environment variable references to their actual values."""
        if self.api_key_env is not None:
            # Store the env var name for later serialization
            self._env_var_refs["api_key"] = self.api_key_env.name
            # Resolve the env var to set the api_key field
            self.api_key = self.api_key_env.to_secret_str()
        return self
