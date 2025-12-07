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
Visualization utilities for NAT optimization results.

This module provides functions to visualize optimization trial results,
including parameter exploration and accuracy scores.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

    def format_duration(seconds: float) -> str:
        """Format duration in appropriate units (seconds, minutes, or hours)."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

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


if __name__ == "__main__":
    visualize_optimization()
