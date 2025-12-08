"""
Plot Emotion × Layer Heatmap

Generates a heatmap showing max selectivity for each emotion across layers L18-L25.
Each emotion row uses its own color theme (consistent with circuit diagrams).

Usage:
    python plot_heatmap_layer.py

Output:
    results/figures/heatmap_emotion_layer.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from theme import apply_theme, EMOTIONS, EMOTION_COLORS, TEXT_PRIMARY, TEXT_LIGHT

apply_theme()

LAYERS = list(range(18, 26))  # L18 to L25


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Emotion × Layer Heatmap")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "..", "results", "validated_features.csv")
    default_output = os.path.join(script_dir, "..", "results", "figures")
    parser.add_argument("--input", type=str, default=default_input,
                        help="Path to feature_lists.csv")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for figure")
    return parser.parse_args()


def create_emotion_cmap(base_color, name="custom"):
    """Create a colormap from white to the emotion's color."""
    rgb = mcolors.to_rgb(base_color)
    colors = [(1, 1, 1), rgb]  # white to color
    return mcolors.LinearSegmentedColormap.from_list(name, colors, N=256)


def compute_layer_matrix(df):
    """
    Compute max selectivity for each emotion × layer.

    Returns:
        2D numpy array (5 emotions × 8 layers)
    """
    matrix = np.zeros((len(EMOTIONS), len(LAYERS)))

    for i, emotion in enumerate(EMOTIONS):
        emotion_df = df[df["emotion"] == emotion]
        for j, layer in enumerate(LAYERS):
            layer_df = emotion_df[emotion_df["layer"] == layer]
            if len(layer_df) > 0:
                matrix[i, j] = layer_df["selectivity"].max()

    return matrix


def plot_heatmap(matrix, output_path):
    """
    Create heatmap with emotion-specific colors per row.
    """
    # NeurIPS single-column subfigure: ~2.7 inches wide
    fig, ax = plt.subplots(figsize=(2.7, 2.0))

    # We'll draw each row separately with its own colormap
    for i, emotion in enumerate(EMOTIONS):
        cmap = create_emotion_cmap(EMOTION_COLORS[emotion])
        row_data = matrix[i:i+1, :]

        # Normalize row to [0, 1] based on global max for consistent scaling
        vmax = matrix.max()

        im = ax.imshow(row_data, aspect='auto', cmap=cmap,
                       extent=[-0.5, len(LAYERS)-0.5, i+0.5, i-0.5],
                       vmin=0, vmax=vmax)

    # Add value annotations
    for i in range(len(EMOTIONS)):
        for j in range(len(LAYERS)):
            value = matrix[i, j]
            if value > 0:
                # Use black text for light colors, white for dark
                text_color = TEXT_PRIMARY if value < matrix.max() * 0.6 else TEXT_LIGHT
                ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                        fontsize=6, color=text_color, fontweight='medium')
            else:
                # Fill empty cells with light gray background
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                            facecolor='#F0F0F0', edgecolor='none'))

    # Axis labels (sized for subfigure)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f'L{l}' for l in LAYERS], fontsize=6)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=7)

    # Hide tick marks but keep labels
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("Emotion", fontsize=8)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(LAYERS), 1), minor=True)
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

    # Compute matrix
    matrix = compute_layer_matrix(df)

    # Print summary
    print("\n" + "=" * 60)
    print("Emotion × Layer Max Selectivity Matrix:")
    print("=" * 60)
    header = "Emotion    " + "  ".join([f"L{l:2d}" for l in LAYERS])
    print(header)
    print("-" * len(header))
    for i, emotion in enumerate(EMOTIONS):
        row = f"{emotion:10s} " + "  ".join([f"{matrix[i,j]:5.1f}" if matrix[i,j] > 0 else "    -"
                                              for j in range(len(LAYERS))])
        print(row)

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "heatmap_emotion_layer.png")
    plot_heatmap(matrix, output_path)


if __name__ == "__main__":
    main()
