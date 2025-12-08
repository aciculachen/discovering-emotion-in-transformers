"""
Plot Validation Pass Rate Bar Chart

Generates a grouped bar chart showing confound and cross-emotion pass rates
for each emotion.

Usage:
    python plot_validation_passrate.py

Output:
    results/figures/validation_passrate.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from theme import apply_theme, EMOTIONS, EMOTION_COLORS, TEXT_PRIMARY

apply_theme()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot validation pass rate bar chart")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(script_dir, "..", "results")
    parser.add_argument("--results_dir", type=str, default=default_results,
                        help="Results directory containing validation CSVs")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(default_results, "figures"),
                        help="Output directory for figure")
    return parser.parse_args()


def compute_pass_rates(confound_df, cross_df):
    """
    Compute pass rates for each emotion.

    Returns:
        dict with keys: emotion -> {confound_rate, cross_rate, both_pass, total}
    """
    results = {}

    for emotion in EMOTIONS:
        # Confound pass rate
        conf_emo = confound_df[confound_df["emotion"] == emotion]
        conf_total = len(conf_emo)
        conf_pass = (conf_emo["status"] == "PASS").sum()
        conf_rate = conf_pass / conf_total * 100 if conf_total > 0 else 0

        # Cross-emotion pass rate
        cross_emo = cross_df[cross_df["emotion"] == emotion]
        cross_total = len(cross_emo)
        cross_pass = (cross_emo["status"] == "PASS").sum()
        cross_rate = cross_pass / cross_total * 100 if cross_total > 0 else 0

        # Both pass (intersection)
        conf_pass_set = set(
            (row["emotion"], row["layer"], row["feature_id"])
            for _, row in conf_emo[conf_emo["status"] == "PASS"].iterrows()
        )
        cross_pass_set = set(
            (row["emotion"], row["layer"], row["feature_id"])
            for _, row in cross_emo[cross_emo["status"] == "PASS"].iterrows()
        )
        both_pass = len(conf_pass_set & cross_pass_set)

        results[emotion] = {
            "confound_rate": conf_rate,
            "cross_rate": cross_rate,
            "confound_pass": conf_pass,
            "cross_pass": cross_pass,
            "both_pass": both_pass,
            "total": conf_total,
        }

    return results


def plot_passrate_bars(pass_rates, output_path):
    """Create grouped bar chart for pass rates."""
    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    x = np.arange(len(EMOTIONS))
    width = 0.38

    confound_rates = [pass_rates[e]["confound_rate"] for e in EMOTIONS]
    cross_rates = [pass_rates[e]["cross_rate"] for e in EMOTIONS]

    bars1 = ax.bar(x - width/2, confound_rates, width, label="Confound",
                   color="#9E9E9E", alpha=0.9, edgecolor=TEXT_PRIMARY, linewidth=0.3)
    bars2 = ax.bar(x + width/2, cross_rates, width, label="Cross-Emotion",
                   color="#5B7DB1", alpha=0.9, edgecolor=TEXT_PRIMARY, linewidth=0.3)

    ax.set_ylabel("Pass Rate (%)", fontsize=7, labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, fontsize=5)
    ax.tick_params(axis='y', labelsize=6, pad=1)
    ax.tick_params(axis='x', pad=1)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=5.5,
              ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    total_features = sum(pass_rates[e]["total"] for e in EMOTIONS)
    total_both_pass = sum(pass_rates[e]["both_pass"] for e in EMOTIONS)
    both_rate = total_both_pass / total_features * 100 if total_features > 0 else 0

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    plt.close()

    return total_both_pass, total_features, both_rate


def main():
    args = parse_args()

    # Load validation CSVs
    confound_path = os.path.join(args.results_dir, "validation_confound.csv")
    cross_path = os.path.join(args.results_dir, "validation_cross_emotion.csv")

    print(f"Loading: {confound_path}")
    confound_df = pd.read_csv(confound_path)

    print(f"Loading: {cross_path}")
    cross_df = pd.read_csv(cross_path)

    # Compute pass rates
    pass_rates = compute_pass_rates(confound_df, cross_df)

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Pass Rates:")
    print("=" * 60)
    print(f"{'Emotion':<10} {'Confound':>10} {'Cross-Emo':>10} {'Both':>8}")
    print("-" * 40)
    for emotion in EMOTIONS:
        pr = pass_rates[emotion]
        print(f"{emotion:<10} {pr['confound_rate']:>9.1f}% {pr['cross_rate']:>9.1f}% {pr['both_pass']:>6}/{pr['total']}")

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "validation_passrate.png")
    total_both, total_all, both_rate = plot_passrate_bars(pass_rates, output_path)

    print("\n" + "=" * 60)
    print(f"Summary: {total_both}/{total_all} features ({both_rate:.1f}%) passed both tests.")
    print("=" * 60)


if __name__ == "__main__":
    main()
