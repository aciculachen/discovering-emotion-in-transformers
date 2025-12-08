"""
Plot Cross-Emotion Steering Control Heatmap

Generates a 5×5 heatmap showing the effect of steering with one emotion's
feature on other emotions' target words. Diagonal entries show same-emotion
steering (expected high), off-diagonal show cross-emotion steering (expected low).

Usage:
    python plot_steering_cross_emotion.py

Output:
    results/figures/steering_cross_emotion.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from theme import apply_theme, EMOTIONS, TEXT_PRIMARY, TEXT_LIGHT

apply_theme()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Cross-Emotion Steering Heatmap")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_controls = os.path.join(script_dir, "..", "results", "steering_controls.csv")
    default_results = os.path.join(script_dir, "..", "results", "steering_results.csv")
    default_output = os.path.join(script_dir, "..", "results", "figures")
    parser.add_argument("--controls", type=str, default=default_controls,
                        help="Path to steering_controls.csv")
    parser.add_argument("--results", type=str, default=default_results,
                        help="Path to steering_results.csv (for diagonal)")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for figure")
    return parser.parse_args()


def compute_cross_emotion_matrix(controls_df, results_df=None):
    """
    Compute mean Δlog_prob for each Source × Target emotion pair.

    Args:
        controls_df: DataFrame from steering_controls.csv (cross-emotion data)
        results_df: DataFrame from steering_results.csv (for diagonal, α=4)

    Returns:
        2D numpy array (5×5): rows = source feature, cols = target emotion
    """
    # Filter to cross_emotion experiments only
    cross_df = controls_df[controls_df["experiment_type"] == "cross_emotion"]

    matrix = np.full((len(EMOTIONS), len(EMOTIONS)), np.nan)

    # First, identify which (layer, feature_id) is used for each source emotion
    # in the cross-emotion experiments (to ensure diagonal uses the same feature)
    source_features = {}
    for source in EMOTIONS:
        source_rows = cross_df[cross_df["source_emotion"] == source]
        if len(source_rows) > 0:
            # Get the unique (layer, feature_id) used for this source
            layer = source_rows["layer"].iloc[0]
            feature_id = source_rows["feature_id"].iloc[0]
            source_features[source] = (layer, feature_id)

    # Fill off-diagonal from cross-emotion controls
    for i, source in enumerate(EMOTIONS):
        for j, target in enumerate(EMOTIONS):
            if i == j:
                continue  # Skip diagonal for now
            pair_df = cross_df[
                (cross_df["source_emotion"] == source) &
                (cross_df["target_emotion"] == target)
            ]
            if len(pair_df) > 0:
                matrix[i, j] = pair_df["delta_log_prob"].mean()

    # Fill diagonal from steering_results (α=4, same-emotion steering)
    # IMPORTANT: Use the SAME feature (layer, feature_id) as used in cross-emotion
    if results_df is not None:
        alpha4_df = results_df[results_df["alpha"] == 4]
        for i, emotion in enumerate(EMOTIONS):
            if emotion in source_features:
                layer, feature_id = source_features[emotion]
                # Filter to the specific feature used in cross-emotion experiments
                emotion_df = alpha4_df[
                    (alpha4_df["emotion"] == emotion) &
                    (alpha4_df["layer"] == layer) &
                    (alpha4_df["feature_id"] == feature_id)
                ]
                if len(emotion_df) > 0:
                    matrix[i, i] = emotion_df["delta_log_prob"].mean()
            else:
                # Fallback: use all features for this emotion (if no cross-emotion data)
                emotion_df = alpha4_df[alpha4_df["emotion"] == emotion]
                if len(emotion_df) > 0:
                    matrix[i, i] = emotion_df["delta_log_prob"].mean()

    return matrix


def plot_heatmap(matrix, output_path):
    """
    Create 5×5 heatmap with diverging colormap.
    """
    # NeurIPS single-column subfigure: ~2.7 inches wide (half of 5.5)
    fig, ax = plt.subplots(figsize=(2.7, 2.4))

    # Diverging colormap: blue (negative) - white (zero) - red (positive)
    vmax = np.nanmax(np.abs(matrix))
    vmax = max(vmax, 0.1)  # Ensure minimum range

    cmap = plt.cm.RdBu_r  # Red-White-Blue (reversed so red=positive)

    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')

    # Add value annotations
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            value = matrix[i, j]
            if not np.isnan(value):
                # Darker text for cells close to white
                text_color = TEXT_LIGHT if abs(value) > vmax * 0.5 else TEXT_PRIMARY
                # Highlight diagonal
                fontweight = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        fontsize=6, color=text_color, fontweight=fontweight)

    # Axis labels (sized for subfigure)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels([e[:3] for e in EMOTIONS], fontsize=7)  # Abbreviated labels
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels([e[:3] for e in EMOTIONS], fontsize=7)  # Abbreviated labels

    # Hide tick marks but keep labels
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_xlabel("Target", fontsize=8, labelpad=5)
    ax.set_ylabel("Source", fontsize=8)

    # Colorbar removed - values are annotated directly in cells

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(EMOTIONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(EMOTIONS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

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
    print(f"Loading controls: {args.controls}")
    controls_df = pd.read_csv(args.controls)

    results_df = None
    if os.path.exists(args.results):
        print(f"Loading results: {args.results}")
        results_df = pd.read_csv(args.results)

    # Compute matrix (with diagonal from steering_results)
    matrix = compute_cross_emotion_matrix(controls_df, results_df)

    # Print summary
    print("\n" + "=" * 60)
    print("Cross-Emotion Steering Matrix (Mean Δlog_prob):")
    print("=" * 60)
    print("Rows: Source Feature | Columns: Target Emotion")
    print("-" * 60)

    header = "Source     " + "  ".join([f"{e:>8s}" for e in EMOTIONS])
    print(header)
    print("-" * len(header))

    for i, source in enumerate(EMOTIONS):
        values = [f"{matrix[i,j]:+.4f}" if not np.isnan(matrix[i,j]) else "    -   "
                  for j in range(len(EMOTIONS))]
        row = f"{source:10s} " + "  ".join(values)
        print(row)

    # Summary statistics
    print("\n" + "-" * 60)
    diagonal = np.array([matrix[i, i] for i in range(len(EMOTIONS)) if not np.isnan(matrix[i, i])])
    off_diagonal = np.array([matrix[i, j] for i in range(len(EMOTIONS))
                             for j in range(len(EMOTIONS))
                             if i != j and not np.isnan(matrix[i, j])])

    if len(diagonal) > 0:
        print(f"Same-emotion (diagonal) mean: {diagonal.mean():+.4f}")
    if len(off_diagonal) > 0:
        print(f"Cross-emotion (off-diag) mean: {off_diagonal.mean():+.4f}")
    if len(diagonal) > 0 and len(off_diagonal) > 0:
        print(f"Specificity ratio: {diagonal.mean() / off_diagonal.mean():.2f}x")

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "steering_cross_emotion.png")
    plot_heatmap(matrix, output_path)


if __name__ == "__main__":
    main()
