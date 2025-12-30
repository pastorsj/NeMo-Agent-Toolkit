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

from nat.data_models.component_ref import LLMRef
from nat.utils.sdk.nat_evaluator import NatEvaluator
from nat.utils.sdk.nat_function import NatFunction
from nat.utils.sdk.nat_llm import NatLLM

from .categorizer import CategorizerToolConfig
from .classification_evaluator import ClassificationEvaluatorConfig
from .hardware_check_tool import HardwareCheckToolConfig
from .host_performance_check_tool import HostPerformanceCheckToolConfig
from .maintenance_check import MaintenanceCheckToolConfig
from .monitoring_process_check_tool import MonitoringProcessCheckToolConfig
from .register import AlertTriageAgentWorkflowConfig
from .telemetry_metrics_analysis_agent import TelemetryMetricsAnalysisAgentConfig
from .telemetry_metrics_host_heartbeat_check_tool import TelemetryMetricsHostHeartbeatCheckToolConfig
from .telemetry_metrics_host_performance_check_tool import TelemetryMetricsHostPerformanceCheckToolConfig


class HostPerformanceCheckTool(HostPerformanceCheckToolConfig, NatFunction):
    """Host Performance Check Tool"""

    llm_name: LLMRef = Field(description="LLM to use for the host performance check task.",
                             default=LLMRef(value=""),
                             init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class HardwareCheckTool(HardwareCheckToolConfig, NatFunction):

    llm_name: LLMRef = Field(description="", default=LLMRef(value=""), init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class AlertTriageAgentWorkflow(AlertTriageAgentWorkflowConfig, NatFunction):
    """Alert Triage Agent Workflow"""

    llm_name: LLMRef = Field(description="LLM to use for the alert triage agent workflow.",
                             default=LLMRef(value=""),
                             init=False,
                             exclude=True)
    llm: NatLLM = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class TelemetryMetricsHostPerformanceCheckTool(TelemetryMetricsHostPerformanceCheckToolConfig, NatFunction):
    """Telemetry Metrics Host Performance Check Tool"""
    llm_name: LLMRef = Field(description="", default=LLMRef(value=""), init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class TelemetryMetricsAnalysisAgent(TelemetryMetricsAnalysisAgentConfig, NatFunction):

    llm_name: LLMRef = Field(description="LLM to use for the telemetry metrics analysis agent.",
                             default=LLMRef(value=""),
                             init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class MonitoringProcessCheckTool(MonitoringProcessCheckToolConfig, NatFunction):
    """Monitoring Process Check Tool"""

    llm_name: LLMRef = Field(description="LLM to use for the monitoring process check task.",
                             default=LLMRef(value=""),
                             init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class TelemetryMetricsHostHeartbeatCheckTool(TelemetryMetricsHostHeartbeatCheckToolConfig, NatFunction):
    """Telemetry Metrics Host Heartbeat Check Tool"""
    llm_name: LLMRef = Field(description="LLM to use for the telemetry metrics host heartbeat check task.",
                             default=LLMRef(value=""),
                             init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode="after")
    def set_references(self):
        """Set llm name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class ClassificationEvaluator(ClassificationEvaluatorConfig, NatEvaluator):
    """Classification Evaluator"""


class MaintenanceCheckTool(MaintenanceCheckToolConfig, NatFunction):
    """Maintenance Check Tool"""

    llm_name: LLMRef = Field(description="LLM to use for the maintenance check task.",
                             default=LLMRef(value=""),
                             init=False)

    llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self


class CategorizerTool(CategorizerToolConfig, NatFunction):
    """Categorizer Tool"""
    llm_name: LLMRef = Field(description="LLM to use for the categorization task.",
                             default=LLMRef(value=""),
                             init=False)
    llm: NatLLM = Field(exclude=True)

    @model_validator(mode='after')
    def set_references(self):
        """Set llm_name from llm object if llm is provided."""
        if self.llm:
            self.llm_name = LLMRef(value=self.llm.computed_name)
        return self
