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

from pydantic import BaseModel
from pydantic import Field

from nat.utils.sdk.nat_front_end import NatFrontEnd
from nat.utils.sdk.nat_logger import NatLogger
from nat.utils.sdk.nat_telemetry_exporter import NatTelemetryExporter


class NatGeneralConfiguration(BaseModel):

    loggers: list[NatLogger] = Field(description="A list of loggers", default=[])
    telemetry_exporters: list[NatTelemetryExporter] = Field(description="A list of telemetry exporters", default=[])
    front_end_configuration: NatFrontEnd | None = Field(description="A configuration for a FAST API", default=None)
