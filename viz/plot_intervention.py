"""
Plot Circuit Intervention Results (Combined Ablation)

Generates a bar chart showing combined ablation effect per emotion.
This validates the causal circuit by showing the total effect when
ablating Top 3 causal contributors simultaneously.

Usage:
    python plot_intervention.py

Output:
    results/figures/circuit_intervention.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from theme import apply_theme, EMOTIONS

apply_theme()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Circuit Intervention Results")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "..", "results", "circuit_intervention.csv")
    default_output = os.path.join(script_dir, "..", "results", "figures")
    parser.add_argument("--input", type=str, default=default_input,
                        help="Path to circuit_intervention.csv")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for figure")
    return parser.parse_args()


def compute_intervention_stats(df):
    """
    Compute mean logit change for combined ablation per emotion.

    Returns:
        dict: {emotion: {"combined": float, "combined_name": str}}
    """
    stats = {}

    for emotion in EMOTIONS:
        emotion_df = df[df["emotion"] == emotion]
        if len(emotion_df) == 0:
            continue

        combined_df = emotion_df[emotion_df["intervention_type"] == "combined_ablation"]

        if len(combined_df) == 0:
            continue

        # Get the components used in combined ablation (e.g., "L21.MLP+L23.H0+L16.MLP")
        combined_name = combined_df["ablated_component"].iloc[0]
        combined_mean = combined_df["logit_change"].mean()

        stats[emotion] = {
            "combined": combined_mean,
            "combined_name": combined_name,
        }

    return stats


def plot_combined_ablation(stats, output_path):
    """Create bar chart showing combined ablation effect per emotion."""
    emotions_with_data = [e for e in EMOTIONS if e in stats]

    if len(emotions_with_data) == 0:
        print("No intervention data to plot")
        return

    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    x = np.arange(len(emotions_with_data))
    width = 0.6

    combined = [stats[e]["combined"] for e in emotions_with_data]

    bars = ax.bar(x, combined, width, color="#5B7DB1", alpha=0.9,
                  edgecolor="#5B7DB1", linewidth=0.8)

    for bar, val in zip(bars, combined):
        height = bar.get_height()
        y_pos = height / 2
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}',
                ha='center', va='center', fontsize=7, fontweight='bold', color='white')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(emotions_with_data, fontsize=8)
    ax.set_ylabel("Δ Logit", fontsize=9, labelpad=2)
    ax.tick_params(axis='y', labelsize=7, pad=1)
    ax.tick_params(axis='x', which='both', length=0, pad=1)

    ax.set_ylim(bottom=min(combined) * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    plt.close()


def main():
    args = parse_args()

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)

    # Compute statistics
    stats = compute_intervention_stats(df)

    # Print summary
    print("\n" + "=" * 60)
    print("Combined Ablation Results (Top 3 Components):")
    print("=" * 60)
    print(f"{'Emotion':<10} {'Combined Δlogit':>15} {'Ablated Components'}")
    print("-" * 60)

    for emotion in EMOTIONS:
        if emotion not in stats:
            continue
        data = stats[emotion]
        print(f"{emotion:<10} {data['combined']:>+15.3f}   {data['combined_name']}")

    print("-" * 60)
    print("Negative Δlogit = ablation reduces emotion word probability")
    print("(confirms causal contribution of ablated components)")

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "circuit_intervention.png")
    plot_combined_ablation(stats, output_path)


if __name__ == "__main__":
    main()
