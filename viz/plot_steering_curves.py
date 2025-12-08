"""
Plot Steering Effect Curves (α vs Δlog_prob)

Generates a single line plot with 5 emotion curves overlaid, showing how
feature activation/ablation (α) affects target emotion word probability.

Usage:
    python plot_steering_curves.py

Output:
    results/figures/steering_curves.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from theme import apply_theme, EMOTIONS, EMOTION_COLORS

apply_theme()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Steering Effect Curves")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "..", "results", "steering_results.csv")
    default_output = os.path.join(script_dir, "..", "results", "figures")
    parser.add_argument("--input", type=str, default=default_input,
                        help="Path to steering_results.csv")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for figure")
    parser.add_argument("--no_std", action="store_true",
                        help="Hide standard deviation shaded area (shown by default)")
    return parser.parse_args()


def compute_steering_stats(df):
    """
    Compute mean and std of Δlog_prob for each emotion and α value.

    Returns:
        dict: {emotion: {"alphas": [...], "means": [...], "stds": [...]}}
    """
    stats = {}

    for emotion in EMOTIONS:
        emotion_df = df[df["emotion"] == emotion]
        if len(emotion_df) == 0:
            continue

        # Get unique alpha values (sorted)
        alphas = sorted(emotion_df["alpha"].unique())

        means = []
        stds = []
        for alpha in alphas:
            alpha_df = emotion_df[emotion_df["alpha"] == alpha]
            means.append(alpha_df["delta_log_prob"].mean())
            stds.append(alpha_df["delta_log_prob"].std())

        stats[emotion] = {
            "alphas": alphas,
            "means": means,
            "stds": stds
        }

    return stats


def plot_curves(stats, output_path, show_std=False):
    """
    Create line plot with 5 emotion curves.
    """
    # NeurIPS single-column subfigure: ~2.7 inches wide (half of 5.5)
    fig, ax = plt.subplots(figsize=(2.7, 2.4))

    for emotion in EMOTIONS:
        if emotion not in stats:
            continue

        data = stats[emotion]
        alphas = data["alphas"]
        means = data["means"]
        stds = data["stds"]
        color = EMOTION_COLORS[emotion]

        # Plot main line
        ax.plot(alphas, means, marker='o', markersize=3, linewidth=1.2,
                color=color, label=emotion)

        # Optional: show std as shaded area
        if show_std:
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.fill_between(alphas, means_arr - stds_arr, means_arr + stds_arr,
                            color=color, alpha=0.15)

    # Reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Reference line at x=0 (baseline)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # Labels (sized for subfigure)
    ax.set_xlabel("Steering Coefficient (α)", fontsize=8)
    ax.set_ylabel("Δlog P(emotion)", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    # Legend (outside to save space)
    ax.legend(loc='upper left', fontsize=6, framealpha=0.9, handlelength=1)

    # Grid
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    # Save PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    # Save PDF
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
    stats = compute_steering_stats(df)

    # Print summary
    print("\n" + "=" * 60)
    print("Steering Effect Summary (Mean Δlog_prob):")
    print("=" * 60)

    # Get all unique alphas
    all_alphas = sorted(set(a for s in stats.values() for a in s["alphas"]))
    header = "Emotion    " + "  ".join([f"α={a:+.0f}" if a != 0 else " α=0" for a in all_alphas])
    print(header)
    print("-" * len(header))

    for emotion in EMOTIONS:
        if emotion not in stats:
            continue
        data = stats[emotion]
        values = []
        for alpha in all_alphas:
            if alpha in data["alphas"]:
                idx = data["alphas"].index(alpha)
                values.append(f"{data['means'][idx]:+.3f}")
            else:
                values.append("  -  ")
        row = f"{emotion:10s} " + "  ".join([f"{v:>5s}" for v in values])
        print(row)

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    show_std = not args.no_std
    filename = "steering_curves.png"
    output_path = os.path.join(args.output_dir, filename)
    plot_curves(stats, output_path, show_std=show_std)


if __name__ == "__main__":
    main()
