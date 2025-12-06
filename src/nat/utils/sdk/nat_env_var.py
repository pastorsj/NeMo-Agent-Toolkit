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
NatEnvironmentVariable - A wrapper for environment variable references.

This class allows SDK users to specify API keys and other secrets using environment
variable names instead of actual values. When the config is saved, the env var
reference (e.g., `${OPENAI_API_KEY}`) is preserved instead of the resolved value.

Example:
    ```python
    from nat.utils.sdk.nat_env_var import NatEnvironmentVariable
    from nat.llm.openai_llm import OpenAILLM

    # Instead of passing the actual API key:
    # llm = OpenAILLM(api_key=SecretStr(os.environ["OPENAI_API_KEY"]))

    # Use an env var reference:
    llm = OpenAILLM(
        api_key_env=NatEnvironmentVariable("OPENAI_API_KEY"),
        model_name="gpt-4o",
    )

    # When saved, the config will contain:
    # api_key: ${OPENAI_API_KEY}
    ```
"""

import os
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr


class NatEnvironmentVariable(BaseModel):
    """A reference to an environment variable.

    This class wraps an environment variable name and provides methods to:
    - Resolve the env var to get its actual value
    - Format it for YAML serialization as `${VAR_NAME}`

    Attributes:
        name: The name of the environment variable (e.g., "OPENAI_API_KEY")
        required: Whether the env var must be set (raises error if missing)
    """

    name: str = Field(description="The name of the environment variable.")
    required: bool = Field(default=True, description="Whether the env var must be set.")

    def __init__(self, name: str, required: bool = True, **kwargs: Any):
        """Create a new environment variable reference.

        Args:
            name: The name of the environment variable (without ${} wrapper)
            required: Whether to raise an error if the env var is not set
        """
        super().__init__(name=name, required=required, **kwargs)

    def resolve(self) -> str | None:
        """Resolve the environment variable to its actual value.

        Returns:
            The value of the environment variable, or None if not set and not required.

        Raises:
            ValueError: If the env var is required but not set.
        """
        value = os.environ.get(self.name)
        if value is None and self.required:
            raise ValueError(f"Environment variable '{self.name}' is required but not set. "
                             f"Please set it before using this configuration.")
        return value

    def to_secret_str(self) -> SecretStr | None:
        """Resolve the env var and wrap it in a SecretStr.

        Returns:
            A SecretStr containing the env var value, or None if not set.
        """
        value = self.resolve()
        return SecretStr(value) if value else None

    def to_yaml_reference(self) -> str:
        """Format the env var name for YAML serialization.

        Returns:
            The env var reference in `${VAR_NAME}` format.
        """
        return f"${{{self.name}}}"

    def __str__(self) -> str:
        """Return the YAML reference format."""
        return self.to_yaml_reference()

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"NatEnvironmentVariable(name='{self.name}', required={self.required})"


# Type alias for optional env var
OptionalNatEnvVar = NatEnvironmentVariable | None
