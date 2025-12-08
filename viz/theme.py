"""
Unified Theme Configuration for Visualization

All plot scripts should import from this module to ensure consistent styling.

Usage:
    from theme import apply_theme, EMOTIONS, EMOTION_COLORS
    apply_theme()
"""

import matplotlib.pyplot as plt

# ============================================================
# Emotion Definitions
# ============================================================

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Disgust"]

# Material Design inspired colors
EMOTION_COLORS = {
    "Joy": "#FBC02D",      # Material Amber 600
    "Sadness": "#42A5F5",  # Material Blue 400
    "Anger": "#E53935",    # Material Red 600
    "Fear": "#7E57C2",     # Material Deep Purple 400
    "Disgust": "#689F38",  # Material Light Green 700
}

# ============================================================
# Color Palette
# ============================================================

# Text colors (avoid pure black)
TEXT_PRIMARY = "#444444"
TEXT_SECONDARY = "#666666"
TEXT_LIGHT = "white"

# Grid and reference lines
GRID_COLOR = "gray"
GRID_ALPHA = 0.3

# ============================================================
# Matplotlib rcParams
# ============================================================

RCPARAMS = {
    # Font settings (Helvetica: standard for academic figures)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,

    # Axes settings
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.labelcolor': TEXT_PRIMARY,

    # Tick settings
    'xtick.color': TEXT_PRIMARY,
    'ytick.color': TEXT_PRIMARY,

    # Legend settings
    'legend.labelcolor': TEXT_PRIMARY,

    # Figure settings
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}


def apply_theme():
    """Apply the unified theme to matplotlib."""
    plt.rcParams.update(RCPARAMS)


# ============================================================
# Helper Functions
# ============================================================

def get_text_color(value, threshold=0.5, vmax=1.0):
    """
    Get appropriate text color based on background brightness.

    Args:
        value: The value determining background color intensity
        threshold: Fraction of vmax where text switches from dark to light
        vmax: Maximum value for normalization

    Returns:
        str: TEXT_PRIMARY for light backgrounds, TEXT_LIGHT for dark
    """
    if abs(value) > vmax * threshold:
        return TEXT_LIGHT
    return TEXT_PRIMARY


def get_emotion_color(emotion):
    """Get the color for a specific emotion."""
    return EMOTION_COLORS.get(emotion, TEXT_PRIMARY)
