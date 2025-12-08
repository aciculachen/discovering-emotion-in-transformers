"""
Plot Emotion × Top-N Feature Heatmap

Generates a heatmap showing Top-4 features per emotion with selectivity values.
Each emotion row uses its own color theme (consistent with circuit diagrams).

Usage:
    python plot_heatmap_topn.py
    python plot_heatmap_topn.py --top_n 4

Output:
    results/figures/heatmap_emotion_topn.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from theme import apply_theme, EMOTIONS, EMOTION_COLORS, TEXT_PRIMARY, TEXT_LIGHT

apply_theme()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Emotion × Top-N Feature Heatmap")
    parser.add_argument("--top_n", type=int, default=4,
                        help="Number of top features per emotion (default: 4)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "..", "results", "validated_features.csv")
    default_output = os.path.join(script_dir, "..", "results", "figures")
    parser.add_argument("--input", type=str, default=default_input,
                        help="Path to validated_features.csv")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for figure")
    return parser.parse_args()


def create_emotion_cmap(base_color, name="custom"):
    """Create a colormap from white to the emotion's color."""
    rgb = mcolors.to_rgb(base_color)
    colors = [(1, 1, 1), rgb]  # white to color
    return mcolors.LinearSegmentedColormap.from_list(name, colors, N=256)


def get_topn_features(df, top_n):
    """
    Get Top-N features per emotion.

    Returns:
        matrix: 2D numpy array (5 emotions × top_n)
        labels: 2D list of feature_id labels for annotation
    """
    matrix = np.zeros((len(EMOTIONS), top_n))
    labels = [[None for _ in range(top_n)] for _ in range(len(EMOTIONS))]

    for i, emotion in enumerate(EMOTIONS):
        emotion_df = df[df["emotion"] == emotion].head(top_n)
        for j, (_, row) in enumerate(emotion_df.iterrows()):
            if j < top_n:
                matrix[i, j] = row["selectivity"]
                labels[i][j] = f"L{int(row['layer'])} #{int(row['feature_id'])}"

    return matrix, labels


def plot_heatmap(matrix, labels, top_n, output_path):
    """
    Create heatmap with emotion-specific colors per row.
    """
    # NeurIPS single-column subfigure: ~2.7 inches wide
    fig, ax = plt.subplots(figsize=(2.7, 2.0))

    # Global max for consistent color scaling
    vmax = matrix.max()

    # Draw each row separately with its own colormap
    for i, emotion in enumerate(EMOTIONS):
        cmap = create_emotion_cmap(EMOTION_COLORS[emotion])
        row_data = matrix[i:i+1, :]

        im = ax.imshow(row_data, aspect='auto', cmap=cmap,
                       extent=[-0.5, top_n-0.5, i+0.5, i-0.5],
                       vmin=0, vmax=vmax)

    # Add value and layer annotations
    for i in range(len(EMOTIONS)):
        for j in range(top_n):
            value = matrix[i, j]
            layer_label = labels[i][j]
            if value > 0 and layer_label:
                # Use black text for light colors, white for dark
                text_color = TEXT_PRIMARY if value < vmax * 0.6 else TEXT_LIGHT
                # Main value
                ax.text(j, i - 0.1, f'{value:.0f}', ha='center', va='center',
                        fontsize=6, color=text_color, fontweight='bold')
                # Layer label below
                ax.text(j, i + 0.15, layer_label, ha='center', va='center',
                        fontsize=4, color=text_color, alpha=0.8)

    # Axis labels (sized for subfigure)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f'F{i+1}' for i in range(top_n)], fontsize=7)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=7)

    # Hide tick marks but keep labels
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_xlabel("Feature Rank", fontsize=8)
    ax.set_ylabel("Emotion", fontsize=8)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, top_n, 1), minor=True)
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
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)

    # Get top-N features
    matrix, labels = get_topn_features(df, args.top_n)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Emotion × Top-{args.top_n} Feature Selectivity:")
    print("=" * 60)
    header = "Emotion    " + "  ".join([f"  F{i+1}  " for i in range(args.top_n)])
    print(header)
    print("-" * len(header))
    for i, emotion in enumerate(EMOTIONS):
        values = [f"{matrix[i,j]:6.1f}" if matrix[i,j] > 0 else "     -"
                  for j in range(args.top_n)]
        layers = [f"({labels[i][j]})" if labels[i][j] else "     "
                  for j in range(args.top_n)]
        row = f"{emotion:10s} " + "  ".join([f"{v}{l}" for v, l in zip(values, layers)])
        print(row)

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "heatmap_emotion_topn.png")
    plot_heatmap(matrix, labels, args.top_n, output_path)


if __name__ == "__main__":
    main()
