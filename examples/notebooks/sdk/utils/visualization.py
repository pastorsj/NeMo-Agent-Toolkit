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
"""Visualization utilities for evaluation and profiling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML
from IPython.display import display

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _safe_score(item: dict[str, Any], default: float = 0.0) -> float:
    """Safely get score from item, handling None values."""
    score = item.get("score")
    return score if score is not None else default


# =============================================================================
# HIGH-LEVEL FUNCTIONS (pass file paths, everything handled automatically)
# =============================================================================


def display_evaluation_dashboard(
    eval_results_dir: str | Path,
    pass_threshold: float = 0.8,
    warn_threshold: float = 0.5,
    show_summary: bool = True,
    show_table: bool = True,
    show_chart: bool = True,
    show_reasoning: bool = True,
) -> dict[str, Any] | None:
    """Display a complete evaluation dashboard from result files.

    Args:
        eval_results_dir: Path to evaluation results directory containing
            accuracy_output.json and workflow_output.json
        pass_threshold: Score threshold for passing (default: 0.8)
        warn_threshold: Score threshold for warning vs fail (default: 0.5)
        show_summary: Whether to show summary card
        show_table: Whether to show detailed results table
        show_chart: Whether to show score chart
        show_reasoning: Whether to show agent reasoning for first test

    Returns:
        Dictionary with loaded data, or None if files not found
    """
    eval_dir = Path(eval_results_dir)
    accuracy_file = eval_dir / "accuracy_output.json"
    workflow_file = eval_dir / "workflow_output.json"

    if not accuracy_file.exists():
        print(f"⚠️ No evaluation results found at {accuracy_file}")
        return None

    with open(accuracy_file, encoding="utf-8") as f:
        accuracy_data = json.load(f)

    workflow_data = []
    if workflow_file.exists():
        with open(workflow_file, encoding="utf-8") as f:
            workflow_data = json.load(f)

    # Calculate metrics (handle None scores from rate limit errors, etc.)
    avg_score = accuracy_data.get("average_score") or 0
    items = accuracy_data.get("eval_output_items", [])
    total_tests = len(items)
    passed = sum(1 for item in items if _safe_score(item) >= pass_threshold)

    # Display components
    if show_summary:
        display_eval_summary(
            avg_score=avg_score,
            total_tests=total_tests,
            passed=passed,
            pass_threshold=pass_threshold,
            warn_threshold=warn_threshold,
        )

    if show_table and items:
        results = _build_eval_results_list(items, workflow_data, pass_threshold)
        display_eval_results_table(results, pass_threshold=pass_threshold, warn_threshold=warn_threshold)

    if show_chart and items:
        ids = [item.get("id", f"test_{i}") for i, item in enumerate(items)]
        scores = [_safe_score(item) for item in items]
        display_score_chart(ids, scores, pass_threshold=pass_threshold, warn_threshold=warn_threshold)

    if show_reasoning and items:
        display_agent_reasoning(items[0])

    return {"accuracy_data": accuracy_data, "workflow_data": workflow_data}


def display_profiling_dashboard(
    profiling_results_dir: str | Path,
    fast_threshold_ms: float = 2000,
    medium_threshold_ms: float = 5000,
    show_summary: bool = True,
    show_table: bool = True,
    show_chart: bool = True,
) -> dict[str, Any] | None:
    """Display a complete profiling dashboard from result files.

    Args:
        profiling_results_dir: Path to profiling results directory containing
            workflow_output.json
        fast_threshold_ms: Latency threshold for "fast" (default: 2000ms)
        medium_threshold_ms: Latency threshold for "medium" vs "slow" (default: 5000ms)
        show_summary: Whether to show summary card
        show_table: Whether to show detailed results table
        show_chart: Whether to show latency chart

    Returns:
        Dictionary with loaded data and computed metrics, or None if files not found
    """
    profiling_dir = Path(profiling_results_dir)
    workflow_file = profiling_dir / "workflow_output.json"

    if not workflow_file.exists():
        print(f"⚠️ No profiling results found at {workflow_file}")
        return None

    with open(workflow_file, encoding="utf-8") as f:
        workflow_data = json.load(f)

    # Extract metrics
    metrics = _extract_profiling_metrics(workflow_data)

    # Display components
    if show_summary:
        display_profiling_summary(
            total_requests=metrics["total_requests"],
            avg_latency_ms=metrics["avg_latency_ms"],
            total_tokens=metrics["total_tokens"],
            total_llm_calls=metrics["total_llm_calls"],
            total_tool_calls=metrics["total_tool_calls"],
        )

    if show_table:
        results = _build_profiling_results_list(workflow_data, metrics["latencies"])
        display_profiling_results(
            results,
            fast_threshold_ms=fast_threshold_ms,
            medium_threshold_ms=medium_threshold_ms,
        )

    if show_chart:
        ids = [item.get("id", f"req_{i}") for i, item in enumerate(workflow_data)]
        display_latency_chart(
            ids,
            metrics["latencies"],
            fast_threshold_ms=fast_threshold_ms,
            medium_threshold_ms=medium_threshold_ms,
        )

    return {"workflow_data": workflow_data, "metrics": metrics}


# =============================================================================
# HELPER FUNCTIONS (internal data processing)
# =============================================================================


def _build_eval_results_list(
    items: list[dict[str, Any]],
    _workflow_data: list[dict[str, Any]],
    pass_threshold: float,
) -> list[dict[str, Any]]:
    """Build a list of result dicts for the eval table."""
    results = []
    for i, item in enumerate(items):
        reasoning = item.get("reasoning", {})
        score = _safe_score(item)
        results.append({
            "ID": item.get("id", f"test_{i}"),
            "Question": reasoning.get("user_input", ""),
            "Expected": reasoning.get("reference", ""),
            "Agent Response": reasoning.get("response", ""),
            "Score": score,
            "Status": "✅ Pass" if score >= pass_threshold else "❌ Fail",
        })
    return results


def _extract_profiling_metrics(workflow_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract profiling metrics from workflow output data."""
    latencies = []
    llm_calls_list = []
    tool_calls_list = []

    for item in workflow_data:
        steps = item.get("intermediate_steps", [])
        llm_calls = sum(1 for s in steps if s.get("payload", {}).get("event_type") == "LLM_END")
        tool_calls = sum(1 for s in steps if s.get("payload", {}).get("event_type") == "TOOL_END")

        # Estimate latency from timestamps
        if steps:
            timestamps = [s.get("payload", {}).get("event_timestamp", 0) for s in steps]
            latency_ms = (max(timestamps) - min(timestamps)) * 1000 if len(timestamps) > 1 else 0
        else:
            latency_ms = 0

        latencies.append(latency_ms)
        llm_calls_list.append(llm_calls)
        tool_calls_list.append(tool_calls)

    total_requests = len(workflow_data)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "total_requests": total_requests,
        "avg_latency_ms": avg_latency,
        "latencies": latencies,
        "total_tokens": 0,  # Token info may not be available in workflow output
        "total_llm_calls": sum(llm_calls_list),
        "total_tool_calls": sum(tool_calls_list),
        "llm_calls_per_request": llm_calls_list,
        "tool_calls_per_request": tool_calls_list,
    }


def _build_profiling_results_list(
    workflow_data: list[dict[str, Any]],
    latencies: list[float],
) -> list[dict[str, Any]]:
    """Build a list of result dicts for the profiling table."""
    results = []
    for i, item in enumerate(workflow_data):
        steps = item.get("intermediate_steps", [])
        llm_calls = sum(1 for s in steps if s.get("payload", {}).get("event_type") == "LLM_END")
        tool_calls = sum(1 for s in steps if s.get("payload", {}).get("event_type") == "TOOL_END")

        question = item.get("question", "")
        if len(question) > 50:
            question = question[:50] + "..."

        results.append({
            "ID": item.get("id", f"req_{i}"),
            "Question": question,
            "Latency (ms)": latencies[i],
            "LLM Calls": llm_calls,
            "Tool Calls": tool_calls,
        })
    return results


# =============================================================================
# LOW-LEVEL DISPLAY FUNCTIONS (for custom usage)
# =============================================================================


def display_eval_summary(
    avg_score: float,
    total_tests: int,
    passed: int,
    title: str = "Evaluation Summary",
    pass_threshold: float = 0.8,
    warn_threshold: float = 0.5,
) -> None:
    """Display a styled summary card for evaluation results.

    Args:
        avg_score: Average score (0.0 to 1.0)
        total_tests: Total number of test cases
        passed: Number of passed tests
        title: Title for the summary card
        pass_threshold: Threshold for pass status (default: 0.8)
        warn_threshold: Threshold for warning vs fail (default: 0.5)
    """
    if avg_score >= pass_threshold:
        status = "✅"
    elif avg_score >= warn_threshold:
        status = "⚠️"
    else:
        status = "❌"

    bg_style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
    container_style = f"{bg_style} padding: 20px; border-radius: 12px; margin: 10px 0; color: white;"
    font_style = "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;"
    card_style = "background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 8px; text-align: center;"

    summary_html = f"""
    <div style="{container_style} {font_style}">
        <h2 style="margin: 0 0 15px 0; font-size: 24px;">📊 {title}</h2>
        <div style="display: flex; gap: 30px; flex-wrap: wrap;">
            <div style="{card_style}">
                <div style="font-size: 36px; font-weight: bold;">{avg_score:.0%}</div>
                <div style="font-size: 14px; opacity: 0.9;">Average Score</div>
            </div>
            <div style="{card_style}">
                <div style="font-size: 36px; font-weight: bold;">{passed}/{total_tests}</div>
                <div style="font-size: 14px; opacity: 0.9;">Tests Passed (≥{pass_threshold:.0%})</div>
            </div>
            <div style="{card_style}">
                <div style="font-size: 36px; font-weight: bold;">{status}</div>
                <div style="font-size: 14px; opacity: 0.9;">Status</div>
            </div>
        </div>
    </div>
    """
    display(HTML(summary_html))


def display_eval_results_table(
    results: list[dict[str, Any]],
    pass_threshold: float = 0.8,
    warn_threshold: float = 0.7,
) -> None:
    """Display a styled results table with color-coded scores.

    Args:
        results: List of result dictionaries with keys:
            - ID: Test case identifier
            - Question: The question asked
            - Expected: Expected answer
            - Agent Response: Actual agent response
            - Score: Score value (0.0 to 1.0)
            - Status: Pass/Fail status string
        pass_threshold: Score threshold for green/pass (default: 0.8)
        warn_threshold: Score threshold for yellow/warning (default: 0.7)
    """
    df = pd.DataFrame(results)

    def style_score(val: float) -> str:
        if val >= pass_threshold:
            return "background-color: #d4edda; color: #155724"
        elif val >= warn_threshold:
            return "background-color: #fff3cd; color: #856404"
        else:
            return "background-color: #f8d7da; color: #721c24"

    styled_df = (df.style.map(style_score, subset=["Score"]).set_properties(**{
        "text-align": "left", "font-size": "12px", "padding": "8px"
    }).set_table_styles([
        {
            "selector":
                "th",
            "props": [
                ("background-color", "#4a5568"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "left"),
                ("padding", "10px"),
            ],
        },
        {
            "selector": "tr:hover", "props": [("background-color", "#f7fafc")]
        },
    ]).format({"Score": "{:.0%}"}))

    print("📋 Detailed Test Results:\n")
    display(styled_df)


def display_score_chart(
    ids: list[str],
    scores: list[float],
    pass_threshold: float = 0.8,
    warn_threshold: float = 0.5,
) -> None:
    """Display a horizontal bar chart of scores.

    Args:
        ids: List of test case identifiers
        scores: List of corresponding scores (0.0 to 1.0)
        pass_threshold: Score threshold for green/pass (default: 0.8)
        warn_threshold: Score threshold for yellow/warning (default: 0.5)
    """
    bars_html = ""
    for id_, score in zip(ids, scores):
        if score >= pass_threshold:
            color = "#48bb78"
        elif score >= warn_threshold:
            color = "#ecc94b"
        else:
            color = "#f56565"
        width = score * 100
        bar_inner = (f"width: {width}%; background: {color}; height: 100%; border-radius: 4px; "
                     "display: flex; align-items: center; justify-content: flex-end; "
                     "padding-right: 8px; transition: width 0.3s ease;")
        bars_html += f"""
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 100px; font-size: 13px; color: #4a5568; font-weight: 500;">{id_}</div>
            <div style="flex: 1; background: #e2e8f0; border-radius: 4px; height: 24px; margin: 0 10px;">
                <div style="{bar_inner}">
                    <span style="color: white; font-size: 12px; font-weight: bold;">{score:.0%}</span>
                </div>
            </div>
        </div>
        """

    chart_style = ("background: white; padding: 20px; border-radius: 12px; "
                   "box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; "
                   "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;")
    legend_style = ("display: flex; gap: 20px; margin-top: 15px; padding-top: 15px; "
                    "border-top: 1px solid #e2e8f0;")
    legend_item = "display: flex; align-items: center; gap: 6px;"
    dot_base = "width: 12px; height: 12px; border-radius: 2px;"

    chart_html = f"""
    <div style="{chart_style}">
        <h3 style="margin: 0 0 15px 0; color: #2d3748;">📈 Score by Test Case</h3>
        {bars_html}
        <div style="{legend_style}">
            <div style="{legend_item}">
                <div style="{dot_base} background: #48bb78;"></div>
                <span style="font-size: 12px; color: #718096;">Pass (≥{pass_threshold:.0%})</span>
            </div>
            <div style="{legend_item}">
                <div style="{dot_base} background: #ecc94b;"></div>
                <span style="font-size: 12px; color: #718096;">Warn ({warn_threshold:.0%}-{pass_threshold:.0%})</span>
            </div>
            <div style="{legend_item}">
                <div style="{dot_base} background: #f56565;"></div>
                <span style="font-size: 12px; color: #718096;">Fail (&lt;{warn_threshold:.0%})</span>
            </div>
        </div>
    </div>
    """
    display(HTML(chart_html))


def display_agent_reasoning(sample: dict[str, Any]) -> None:
    """Display the agent's reasoning process for a test case.

    Args:
        sample: Evaluation output item with keys:
            - id: Test case identifier
            - reasoning: Dict with user_input, reference, response, retrieved_contexts
    """
    steps_html = ""
    contexts = sample.get("reasoning", {}).get("retrieved_contexts", [])

    step_style = ("background: #f7fafc; padding: 12px 15px; border-radius: 8px; "
                  "margin: 8px 0; border-left: 3px solid #667eea;")
    for i, ctx in enumerate(contexts):
        step_content = ctx.replace("\n", "<br>")
        steps_html += f"""
        <div style="{step_style}">
            <div style="font-size: 11px; color: #718096; margin-bottom: 5px;">Step {i}</div>
            <div style="font-size: 13px; color: #2d3748; line-height: 1.5;">{step_content}</div>
        </div>
        """

    reasoning = sample.get("reasoning", {})
    container_style = ("background: white; padding: 20px; border-radius: 12px; "
                       "box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; "
                       "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;")
    reasoning_html = f"""
    <div style="{container_style}">
        <h3 style="margin: 0 0 5px 0; color: #2d3748;">🔍 Agent Reasoning Explorer</h3>
        <p style="color: #718096; font-size: 13px; margin: 0 0 15px 0;">
            Sample test case: <strong>{sample.get("id", "unknown")}</strong>
        </p>

        <div style="background: #ebf4ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 15px;">
            <div style="font-size: 11px; color: #667eea; font-weight: 600;">Question</div>
            <div style="font-size: 14px; color: #2d3748;">{reasoning.get("user_input", "")}</div>
        </div>

        <div style="font-size: 12px; color: #718096; font-weight: 600; margin-bottom: 8px;">
            AGENT THOUGHT PROCESS:
        </div>
        {steps_html}

        <div style="display: flex; gap: 15px; margin-top: 15px;">
            <div style="flex: 1; background: #c6f6d5; padding: 12px 15px; border-radius: 8px;">
                <div style="font-size: 11px; color: #276749; font-weight: 600;">Expected Answer</div>
                <div style="font-size: 14px; color: #22543d;">{reasoning.get("reference", "")}</div>
            </div>
            <div style="flex: 1; background: #bee3f8; padding: 12px 15px; border-radius: 8px;">
                <div style="font-size: 11px; color: #2b6cb0; font-weight: 600;">Agent's Answer</div>
                <div style="font-size: 14px; color: #2c5282;">{reasoning.get("response", "")}</div>
            </div>
        </div>
    </div>
    """
    display(HTML(reasoning_html))


def display_profiling_summary(
    total_requests: int,
    avg_latency_ms: float,
    total_tokens: int,
    total_llm_calls: int,
    total_tool_calls: int,
) -> None:
    """Display a styled summary card for profiling results.

    Args:
        total_requests: Total number of requests processed
        avg_latency_ms: Average latency in milliseconds
        total_tokens: Total tokens used
        total_llm_calls: Total LLM invocations
        total_tool_calls: Total tool invocations
    """
    bg_style = "background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);"
    container_style = f"{bg_style} padding: 20px; border-radius: 12px; margin: 10px 0; color: white;"
    font_style = "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;"
    card_style = "background: rgba(255,255,255,0.15); padding: 12px 20px; border-radius: 8px; text-align: center;"

    summary_html = f"""
    <div style="{container_style} {font_style}">
        <h2 style="margin: 0 0 15px 0; font-size: 24px;">⚡ Profiling Summary</h2>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="{card_style}">
                <div style="font-size: 28px; font-weight: bold;">{total_requests}</div>
                <div style="font-size: 12px; opacity: 0.9;">Requests</div>
            </div>
            <div style="{card_style}">
                <div style="font-size: 28px; font-weight: bold;">{avg_latency_ms:.0f}ms</div>
                <div style="font-size: 12px; opacity: 0.9;">Avg Latency</div>
            </div>
            <div style="{card_style}">
                <div style="font-size: 28px; font-weight: bold;">{total_tokens:,}</div>
                <div style="font-size: 12px; opacity: 0.9;">Total Tokens</div>
            </div>
            <div style="{card_style}">
                <div style="font-size: 28px; font-weight: bold;">{total_llm_calls}</div>
                <div style="font-size: 12px; opacity: 0.9;">LLM Calls</div>
            </div>
            <div style="{card_style}">
                <div style="font-size: 28px; font-weight: bold;">{total_tool_calls}</div>
                <div style="font-size: 12px; opacity: 0.9;">Tool Calls</div>
            </div>
        </div>
    </div>
    """
    display(HTML(summary_html))


def display_profiling_results(
    results: list[dict[str, Any]],
    fast_threshold_ms: float = 2000,
    medium_threshold_ms: float = 5000,
) -> None:
    """Display detailed profiling results in a styled table.

    Args:
        results: List of profiling result dictionaries with keys:
            - ID: Request identifier
            - Question: The question asked
            - Latency (ms): Request latency
            - LLM Calls: Number of LLM calls
            - Tool Calls: Number of tool calls
        fast_threshold_ms: Latency threshold for "fast" (default: 2000ms)
        medium_threshold_ms: Latency threshold for "medium" vs "slow" (default: 5000ms)
    """
    df = pd.DataFrame(results)

    def style_latency(val: float) -> str:
        if val < fast_threshold_ms:
            return "background-color: #d4edda; color: #155724"
        elif val < medium_threshold_ms:
            return "background-color: #fff3cd; color: #856404"
        else:
            return "background-color: #f8d7da; color: #721c24"

    # Only apply latency styling if the column exists
    subset = ["Latency (ms)"] if "Latency (ms)" in df.columns else []

    styled_df = df.style
    if subset:
        styled_df = styled_df.map(style_latency, subset=subset)

    styled_df = styled_df.set_properties(**{
        "text-align": "left", "font-size": "12px", "padding": "8px"
    }).set_table_styles([
        {
            "selector":
                "th",
            "props": [
                ("background-color", "#2d3748"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "left"),
                ("padding", "10px"),
            ],
        },
        {
            "selector": "tr:hover", "props": [("background-color", "#f7fafc")]
        },
    ])

    print("📋 Profiling Details:\n")
    display(styled_df)


def display_latency_chart(
    ids: list[str],
    latencies: list[float],
    fast_threshold_ms: float = 2000,
    medium_threshold_ms: float = 5000,
) -> None:
    """Display a horizontal bar chart of latencies.

    Args:
        ids: List of request identifiers
        latencies: List of latency values in milliseconds
        fast_threshold_ms: Latency threshold for "fast" (default: 2000ms)
        medium_threshold_ms: Latency threshold for "medium" vs "slow" (default: 5000ms)
    """
    max_latency = max(latencies) if latencies else 1

    bars_html = ""
    for id_, latency in zip(ids, latencies):
        if latency < fast_threshold_ms:
            color = "#48bb78"
        elif latency < medium_threshold_ms:
            color = "#ecc94b"
        else:
            color = "#f56565"
        width = (latency / max_latency) * 100
        bar_inner = (f"width: {width}%; background: {color}; height: 100%; border-radius: 4px; "
                     "display: flex; align-items: center; justify-content: flex-end; "
                     "padding-right: 8px; transition: width 0.3s ease; min-width: 60px;")
        bars_html += f"""
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 100px; font-size: 13px; color: #4a5568; font-weight: 500;">{id_}</div>
            <div style="flex: 1; background: #e2e8f0; border-radius: 4px; height: 24px; margin: 0 10px;">
                <div style="{bar_inner}">
                    <span style="color: white; font-size: 12px; font-weight: bold;">{latency:.0f}ms</span>
                </div>
            </div>
        </div>
        """

    chart_style = ("background: white; padding: 20px; border-radius: 12px; "
                   "box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; "
                   "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;")
    legend_style = ("display: flex; gap: 20px; margin-top: 15px; padding-top: 15px; "
                    "border-top: 1px solid #e2e8f0;")
    legend_item = "display: flex; align-items: center; gap: 6px;"
    dot_base = "width: 12px; height: 12px; border-radius: 2px;"
    fast_s = fast_threshold_ms / 1000
    med_s = medium_threshold_ms / 1000

    chart_html = f"""
    <div style="{chart_style}">
        <h3 style="margin: 0 0 15px 0; color: #2d3748;">⏱️ Latency by Request</h3>
        {bars_html}
        <div style="{legend_style}">
            <div style="{legend_item}">
                <div style="{dot_base} background: #48bb78;"></div>
                <span style="font-size: 12px; color: #718096;">Fast (&lt;{fast_s:.0f}s)</span>
            </div>
            <div style="{legend_item}">
                <div style="{dot_base} background: #ecc94b;"></div>
                <span style="font-size: 12px; color: #718096;">Medium ({fast_s:.0f}-{med_s:.0f}s)</span>
            </div>
            <div style="{legend_item}">
                <div style="{dot_base} background: #f56565;"></div>
                <span style="font-size: 12px; color: #718096;">Slow (&gt;{med_s:.0f}s)</span>
            </div>
        </div>
    </div>
    """
    display(HTML(chart_html))


def load_optimization_results(results_dir: str | Path) -> pd.DataFrame:
    """Load optimization results from the trials CSV file.

    Args:
        results_dir: Path to the optimization results directory.

    Returns:
        DataFrame containing trial results.
    """
    results_dir = Path(results_dir)
    csv_path = results_dir / "trials_dataframe_params.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    return pd.read_csv(csv_path)


def plot_optimization_results(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create a visualization of optimization results.

    Args:
        df: DataFrame with optimization trial results.
        output_path: Optional path to save the figure.
        show: Whether to display the plot.

    Returns:
        The matplotlib Figure object.
    """
    # Extract parameter columns (they start with 'params_')
    param_cols = [col for col in df.columns if col.startswith('params_')]

    # Clean up parameter names for display
    param_names = [col.replace('params_llms.nim_llm.', '') for col in param_cols]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Optimization Results Summary', fontsize=14, fontweight='bold')

    # Plot 1: Accuracy across trials
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if v == df['value'].max() else '#3498db' for v in df['value']]
    bars = ax1.bar(df['number'], df['value'], color=colors, edgecolor='white', linewidth=1.2)
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Accuracy Score', fontsize=11)
    ax1.set_title('Accuracy by Trial', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=df['value'].max(), color='#e74c3c', linestyle='--', alpha=0.7, label=f'Best: {df["value"].max():.2f}')
    ax1.legend(loc='lower right')

    # Add value labels on bars
    for bar, val in zip(bars, df['value']):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{val:.2f}',
                 ha='center',
                 va='bottom',
                 fontsize=9)

    # Plot 2: Parameter values across trials
    ax2 = axes[0, 1]
    x = df['number']
    width = 0.35
    colors_params = ['#9b59b6', '#e67e22']

    for i, (col, name) in enumerate(zip(param_cols, param_names)):
        offset = (i - len(param_cols) / 2 + 0.5) * width
        ax2.bar(x + offset, df[col], width, label=name, color=colors_params[i % len(colors_params)], alpha=0.8)

    ax2.set_xlabel('Trial Number', fontsize=11)
    ax2.set_ylabel('Parameter Value', fontsize=11)
    ax2.set_title('Parameters Explored', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xticks(x)

    # Plot 3: Parameter correlation scatter
    ax3 = axes[1, 0]
    if len(param_cols) >= 2:
        scatter = ax3.scatter(df[param_cols[0]],
                              df[param_cols[1]],
                              c=df['value'],
                              cmap='RdYlGn',
                              s=150,
                              edgecolors='black',
                              linewidth=1.5)
        ax3.set_xlabel(param_names[0], fontsize=11)
        ax3.set_ylabel(param_names[1], fontsize=11)
        ax3.set_title('Parameter Space Exploration', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Accuracy', fontsize=10)

        # Highlight best trial
        best_idx = df['value'].idxmax()
        ax3.scatter(df.loc[best_idx, param_cols[0]],
                    df.loc[best_idx, param_cols[1]],
                    s=300,
                    facecolors='none',
                    edgecolors='red',
                    linewidth=3,
                    label='Best')
        ax3.legend()

    # Plot 4: Trial duration
    ax4 = axes[1, 1]

    # Parse duration - handle the timedelta string format
    def parse_duration_seconds(dur_str):
        """Parse duration string to seconds."""
        if pd.isna(dur_str):
            return 0
        try:
            # Format: "0 days 00:00:02.656489"
            parts = str(dur_str).split()
            if len(parts) >= 3:
                time_part = parts[2]
                h, m, s = time_part.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
        except (ValueError, IndexError):
            pass
        return 0

    durations_sec = df['duration'].apply(parse_duration_seconds)
    max_duration = durations_sec.max()

    # Determine the best unit for display
    if max_duration < 60:
        durations = durations_sec
        unit_label = "seconds"
        unit_short = "s"
    elif max_duration < 3600:
        durations = durations_sec / 60
        unit_label = "minutes"
        unit_short = "m"
    else:
        durations = durations_sec / 3600
        unit_label = "hours"
        unit_short = "h"

    ax4.bar(df['number'], durations, color='#1abc9c', edgecolor='white', linewidth=1.2)
    ax4.set_xlabel('Trial Number', fontsize=11)
    ax4.set_ylabel(f'Duration ({unit_label})', fontsize=11)
    ax4.set_title('Trial Duration', fontsize=12, fontweight='bold')

    # Add average line
    avg_duration = durations.mean()
    ax4.axhline(y=avg_duration,
                color='#e74c3c',
                linestyle='--',
                alpha=0.7,
                label=f'Avg: {avg_duration:.2f}{unit_short}')
    ax4.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 Visualization saved to: {output_path}")

    if show:
        plt.show()

    return fig


def format_duration_human(dur_str) -> str:
    """Format duration string in human-readable format (seconds, minutes, or hours)."""
    if pd.isna(dur_str):
        return "N/A"
    try:
        # Format: "0 days 00:00:02.656489"
        parts = str(dur_str).split()
        if len(parts) >= 3:
            time_part = parts[2]
            h, m, s = time_part.split(':')
            total_seconds = float(h) * 3600 + float(m) * 60 + float(s)

            if total_seconds < 60:
                return f"{total_seconds:.2f} seconds"
            elif total_seconds < 3600:
                minutes = total_seconds / 60
                remaining_sec = total_seconds % 60
                if remaining_sec > 0:
                    return f"{int(minutes)} min {remaining_sec:.1f} sec"
                return f"{minutes:.2f} minutes"
            else:
                hours = total_seconds / 3600
                remaining_min = (total_seconds % 3600) / 60
                if remaining_min > 0:
                    return f"{int(hours)} hr {remaining_min:.1f} min"
                return f"{hours:.2f} hours"
    except (ValueError, IndexError):
        pass
    return str(dur_str)


def print_best_config(df: pd.DataFrame) -> dict:
    """Print and return the best configuration from optimization results.

    Args:
        df: DataFrame with optimization trial results.

    Returns:
        Dictionary with the best parameters.
    """
    best_idx = df['value'].idxmax()
    best_trial = df.loc[best_idx]

    param_cols = [col for col in df.columns if col.startswith('params_')]

    print("=" * 60)
    print("🏆 BEST CONFIGURATION FOUND")
    print("=" * 60)
    print(f"\n📈 Trial #{int(best_trial['number'])} achieved the best score!\n")
    print(f"   Accuracy Score: {best_trial['value']:.4f}")
    print(f"   Duration: {format_duration_human(best_trial['duration'])}")
    print("\n📋 Optimal Parameters:")
    print("-" * 40)

    best_params = {}
    for col in param_cols:
        param_name = col.replace('params_llms.nim_llm.', '').replace('params_', '')
        value = best_trial[col]
        best_params[param_name] = value
        print(f"   {param_name}: {value}")

    print("-" * 40)
    print("\n💡 Use these parameters in your workflow for optimal performance!")
    print("=" * 60)

    return best_params


def visualize_optimization(results_dir: str | Path = "opt_results") -> dict:
    """Main function to visualize optimization results.

    Args:
        results_dir: Path to the optimization results directory.

    Returns:
        Dictionary with the best parameters.
    """
    print("📊 Loading optimization results...")
    df = load_optimization_results(results_dir)

    print(f"   Found {len(df)} trials\n")

    # Print best config
    best_params = print_best_config(df)

    # Create visualization
    print("\n📈 Creating visualization...")
    output_path = Path(results_dir) / "optimization_summary.png"
    plot_optimization_results(df, output_path=output_path)

    return best_params
