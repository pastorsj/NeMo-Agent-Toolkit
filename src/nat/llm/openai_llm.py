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

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.common import OptionalSecretStr
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.utils.sdk.nat_env_var import NatEnvironmentVariable
from nat.utils.sdk.nat_llm import NatLLM


class OpenAIModelConfig(LLMBaseConfig, RetryMixin, OptimizableMixin, ThinkingMixin, name="openai"):
    """An OpenAI LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: OptionalSecretStr = Field(default=None, description="OpenAI API key to interact with hosted model.")
    base_url: str | None = Field(default=None, description="Base url to the hosted model.")
    model_name: str = OptimizableField(validation_alias=AliasChoices("model_name", "model"),
                                       serialization_alias="model",
                                       description="The OpenAI hosted model name.")
    seed: int | None = Field(default=None, description="Random seed to set for generation.")
    max_retries: int = Field(default=10, description="The max number of retries for the request.")
    temperature: float | None = OptimizableField(
        default=None,
        ge=0.0,
        description="Sampling temperature to control randomness in the output.",
        space=SearchSpace(high=0.9, low=0.1, step=0.2))
    top_p: float | None = OptimizableField(default=None,
                                           ge=0.0,
                                           le=1.0,
                                           description="Top-p for distribution sampling.",
                                           space=SearchSpace(high=1.0, low=0.5, step=0.1))


class OpenAILLM(OpenAIModelConfig, NatLLM):
    """OpenAI Model LLM Provider.

    Supports environment variable references for API keys:

    Example:
        ```python
        # Using env var reference (recommended for portability):
        llm = OpenAILLM(
            api_key_env=NatEnvironmentVariable("OPENAI_API_KEY"),
            model_name="gpt-4o",
        )
        ```
    """

    # Override api_key from parent with init=False to signal users should use api_key_env
    api_key: OptionalSecretStr = Field(
        default=None,
        init=False,
        description="OpenAI API key - use api_key_env instead for SDK usage.",
    )

    # Environment variable reference for api_key
    api_key_env: NatEnvironmentVariable | None = Field(
        default=None,
        exclude=True,
        description="Environment variable name for the API key (e.g., 'OPENAI_API_KEY').",
    )

    # Private attribute to track env var references for serialization
    _env_var_refs: dict[str, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode='after')
    def resolve_env_vars(self):
        """Resolve environment variable references to their actual values."""
        if self.api_key_env is not None:
            # Store the env var name for later serialization
            self._env_var_refs["api_key"] = self.api_key_env.name
            # Resolve the env var to set the api_key field
            self.api_key = self.api_key_env.to_secret_str()
        return self


@register_llm_provider(config_type=OpenAIModelConfig)
async def openai_llm(config: OpenAIModelConfig, _builder: Builder):

    yield LLMProviderInfo(config=config, description="An OpenAI model for use with an LLM client.")
