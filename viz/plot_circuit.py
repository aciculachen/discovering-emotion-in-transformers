"""
Plot Circuit Analysis Results

Generates visualizations from circuit analysis CSVs:
- circuit_upstream.csv (causal attribution)
- circuit_downstream.csv (token effects)
- circuit_intervention.csv (validation experiments)

Usage:
    python plot_circuit.py                  # Use default paths
    python plot_circuit.py --input_dir DIR  # Specify input directory

Output:
    - figures/circuit_joy.png
    - figures/circuit_sadness.png
    - figures/circuit_anger.png
    - figures/circuit_fear.png
    - figures/circuit_causal_heatmap.pdf
    - figures/circuit_intervention.pdf
"""

import os
import re
import argparse
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Avenir Next', 'Helvetica Neue', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.labelcolor': '#444444',
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Disgust"]

# Example prompts for display (no explicit emotion keywords)
EXAMPLE_PROMPTS = {
    "Joy": "Haha, well it sounds like its done too good of a job!",
    "Sadness": "Same here. It's very hard. I really feel for you.",
    "Anger": "Dude, wtf is wrong with you?",
    "Fear": "That's nightmare fuel right there",
    "Disgust": "This is repulsive what the fuck is wrong with places like this.",
}

EMOTION_COLORS = {
    "Joy": "#FBC02D",      # Material Amber 600
    "Sadness": "#42A5F5",  # Material Blue 400
    "Anger": "#E53935",    # Material Red 600
    "Fear": "#7E57C2",     # Material Deep Purple 400
    "Disgust": "#689F38",  # Material Light Green 700
}

# Curated whitelist for PROMOTED tokens - based on circuit_downstream.csv
# Non-English tokens filtered: autorytatywna (Polish), peur (French), pourquoi (French)
PROMOTED_WHITELIST = {
    "Joy": {':)', ':]', '^_^', '=)', ':))', '^^', ';)', ':>', '☺️', '🙂',
            ':-)', '😊', '♥', '❤', ":')" },  # Top 15
    "Sadness": {'sad', ':(', ':-(', 'bummer', '😔', ":'(", '😢', 'disappointing', 'saddened', '🙁'},  # Top 10
    "Anger": {'Stop', 'why', 'wtf', 'Seriously', 'quit', 'Come', 'FFS', 'how', 'please'},  # Top 9 (filtered: autorytatywna)
    "Fear": {'fear', 'scared', 'terrified', 'afraid', 'frightened', 'anxiety', 'panic', 'worry', 'scare', 'worried'},  # Top 10 (filtered: peur)
    "Disgust": {'Seriously', 'wtf', 'ffs', 'why', 'smh', 'ridiculous', '🤦', 'pathetic',
                'ludicrous', 'laughable', '🙄', 'unbelievable', 'wow', 'idiots', 'nonsense'},
}

# =============================================================================
# FEATURE OVERRIDES (must match circuit.py)
# =============================================================================
# Override the default feature selection (first by selectivity) when a different
# feature has been determined to have better causal dynamics.
# Format: {"layer": int, "feature_id": int} or None
FEATURE_OVERRIDE = {
    "Joy": {"layer": 25, "feature_id": 13068},
    "Sadness": None,  # Use CSV default (L25:F66)
    "Anger": {"layer": 23, "feature_id": 11903},  # Better causal dynamics than L25:F8662
    "Fear": None,  # Use CSV default (L25:F3410)
    "Disgust": {"layer": 24, "feature_id": 10476},
}

# Font paths (cross-platform)
if platform.system() == "Darwin":  # macOS
    EMOJI_FONT_PATH = '/System/Library/Fonts/Apple Color Emoji.ttc'
    TEXT_FONT_PATH = '/System/Library/Fonts/Avenir Next.ttc'
    TEXT_FONT_FALLBACK = '/System/Library/Fonts/Helvetica.ttc'
else:  # Linux
    EMOJI_FONT_PATH = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'
    TEXT_FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    TEXT_FONT_FALLBACK = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'

# Emoji pattern (compiled once)
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF\U00002300-\U000023FF\U0000FE00-\U0000FE0F"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002764\U00002665\U00002661]+"
)


def render_emoji_text(text, fontsize=24, color=(51, 51, 51)):
    """Render text with emoji using Pillow (composite: text font + emoji font)."""
    if not os.path.exists(EMOJI_FONT_PATH):
        return None

    try:
        # Load text font
        text_font = None
        for font_path in [TEXT_FONT_PATH, TEXT_FONT_FALLBACK]:
            if os.path.exists(font_path):
                try:
                    text_font = ImageFont.truetype(font_path, fontsize)
                    break
                except:
                    continue
        if text_font is None:
            text_font = ImageFont.load_default()

        # Load emoji font (bitmap with specific sizes)
        valid_sizes = [16, 20, 32, 40, 48, 64, 96, 160]
        emoji_size = min(valid_sizes, key=lambda x: abs(x - fontsize))
        emoji_font = ImageFont.truetype(EMOJI_FONT_PATH, emoji_size)

        # Split into emoji vs text segments
        emoji_capture = re.compile(f"({EMOJI_PATTERN.pattern})")
        segments = []
        last_end = 0
        for match in emoji_capture.finditer(text):
            if match.start() > last_end:
                segments.append(("text", text[last_end:match.start()]))
            segments.append(("emoji", match.group()))
            last_end = match.end()
        if last_end < len(text):
            segments.append(("text", text[last_end:]))
        if not segments:
            segments = [("text", text)]

        # Measure segments
        temp_img = Image.new('RGBA', (1, 1), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        total_width = 10
        max_height = fontsize + 10
        segment_info = []

        for seg_type, seg_text in segments:
            if seg_type == "emoji":
                bbox = temp_draw.textbbox((0, 0), seg_text, font=emoji_font)
                scale = fontsize / emoji_size if emoji_size > 0 else 1.0
                w = int((bbox[2] - bbox[0]) * scale)
                h = int((bbox[3] - bbox[1]) * scale)
            else:
                bbox = temp_draw.textbbox((0, 0), seg_text, font=text_font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            segment_info.append((seg_type, seg_text, w, h))
            total_width += w
            max_height = max(max_height, h + 10)

        total_width += 10

        # Render
        img = Image.new('RGBA', (max(total_width, 1), max(max_height, 1)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        x_pos = 10
        y_center = max_height // 2

        for seg_type, seg_text, w, h in segment_info:
            if seg_type == "emoji":
                emoji_img = Image.new('RGBA', (emoji_size * 2, emoji_size * 2), (255, 255, 255, 0))
                emoji_draw = ImageDraw.Draw(emoji_img)
                emoji_draw.text((0, 0), seg_text, font=emoji_font, embedded_color=True)
                scale = fontsize / emoji_size if emoji_size > 0 else 1.0
                new_size = (int(emoji_img.width * scale), int(emoji_img.height * scale))
                if new_size[0] > 0 and new_size[1] > 0:
                    emoji_img = emoji_img.resize(new_size, Image.Resampling.LANCZOS)
                y_pos = y_center - emoji_img.height // 2
                img.paste(emoji_img, (x_pos, max(0, y_pos)), emoji_img)
            else:
                bbox = draw.textbbox((0, 0), seg_text, font=text_font)
                y_pos = y_center - (bbox[3] - bbox[1]) // 2 - bbox[1]
                draw.text((x_pos, y_pos), seg_text, font=text_font, fill=color)
            x_pos += w

        return img
    except Exception as e:
        print(f"Warning: Could not render emoji text: {e}")
        return None


def add_emoji_text_to_ax(ax, x, y, text, fontsize=11, color="#444444", ha="left"):
    """Add text (with emoji support) to matplotlib axes."""
    if isinstance(color, str) and color.startswith('#'):
        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    else:
        color_rgb = (51, 51, 51)

    if EMOJI_PATTERN.search(text):
        img = render_emoji_text(text, fontsize=int(fontsize * 3), color=color_rgb)
        if img is not None:
            img_array = np.array(img)
            imagebox = OffsetImage(img_array, zoom=0.33)
            imagebox.image.axes = ax
            box_alignment = (0.5, 0.5) if ha == "center" else (1.0, 0.5) if ha == "right" else (0.0, 0.5)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=box_alignment)
            ax.add_artist(ab)
            return

    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color)


def render_highlighted_sentence(ax, x, y, sentence, salient_tokens, emotion_color,
                                 base_color="#B0BEC5", fontsize=11, ha="center"):
    """
    Render a sentence with salient tokens highlighted in emotion color + bold.

    Uses PIL to render as image, avoiding matplotlib coordinate issues.

    Args:
        ax: matplotlib axes
        x, y: center position
        sentence: the full sentence to display
        salient_tokens: list of tokens to highlight (cleaned, no quotes)
        emotion_color: color for highlighted tokens
        base_color: light gray for non-highlighted text
        fontsize: font size
        ha: horizontal alignment ("center", "left", "right")
    """
    # Tokenize sentence: split into words and punctuation separately
    tokens = re.findall(r"[\w]+|[^\w\s]+|\s+", sentence)

    # Build salient set with multiple matching strategies
    salient_normalized = set()
    salient_raw = set()
    for t in salient_tokens:
        t_clean = t.strip()
        salient_raw.add(t_clean)
        salient_raw.add(t_clean.lower())
        salient_raw.add(t_clean.upper())
        t_word = re.sub(r'[^\w]', '', t_clean)
        if t_word:
            salient_normalized.add(t_word.lower())

    def is_salient_token(token):
        token_stripped = token.strip()
        if not token_stripped:
            return False
        if token_stripped in salient_raw:
            return True
        token_word = re.sub(r'[^\w]', '', token_stripped)
        if token_word and len(token_word) > 1 and token_word.lower() in salient_normalized:
            return True
        return False

    # Collect token info
    token_info = [(token, is_salient_token(token)) for token in tokens]

    # Render using PIL
    pil_fontsize = int(fontsize * 4)  # Higher resolution
    try:
        font_regular = ImageFont.truetype(TEXT_FONT_PATH, pil_fontsize)
        font_bold = ImageFont.truetype(TEXT_FONT_PATH, pil_fontsize)
    except:
        font_regular = ImageFont.load_default()
        font_bold = font_regular

    # Convert colors to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    emotion_rgb = hex_to_rgb(emotion_color)
    base_rgb = hex_to_rgb(base_color)

    # Calculate total width first
    temp_img = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_img)

    total_width = 0
    token_widths = []
    for token, is_sal in token_info:
        font = font_bold if is_sal else font_regular
        bbox = temp_draw.textbbox((0, 0), token, font=font)
        w = bbox[2] - bbox[0]
        token_widths.append(w)
        total_width += w

    # Create image with proper size
    height = pil_fontsize + 20
    img = Image.new("RGBA", (total_width + 20, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Draw each token
    current_x = 10
    for i, (token, is_sal) in enumerate(token_info):
        color = emotion_rgb if is_sal else base_rgb
        font = font_bold if is_sal else font_regular
        draw.text((current_x, 5), token, font=font, fill=color + (255,))
        current_x += token_widths[i]

    # Add to matplotlib as image
    img_array = np.array(img)
    imagebox = OffsetImage(img_array, zoom=0.25)
    imagebox.image.axes = ax
    box_alignment = (0.5, 0.5) if ha == "center" else (1.0, 0.5) if ha == "right" else (0.0, 0.5)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=box_alignment)
    ax.add_artist(ab)


def render_promotes_line(ax, x, y, tokens, label_color="#37474F", token_color="#444444",
                          fontsize=13, ha="left"):
    """
    Render "PROMOTES: token1, token2, ..." as a unified PIL image.

    Ensures consistent baseline, tight spacing, and proper emoji rendering.
    """
    if not tokens:
        return

    # Build the full text: "PROMOTES: tok1, tok2, tok3"
    tokens_str = ", ".join(tokens[:6])  # Max 6 tokens
    full_text = f"PROMOTES: {tokens_str}"

    # Render using PIL for consistent baseline
    pil_fontsize = int(fontsize * 4)  # Higher resolution
    try:
        font_regular = ImageFont.truetype(TEXT_FONT_PATH, pil_fontsize)
        font_bold = ImageFont.truetype(TEXT_FONT_PATH, pil_fontsize)
    except:
        font_regular = ImageFont.load_default()
        font_bold = font_regular

    # Check if text contains emoji
    has_emoji = any(ord(c) > 0x1F300 for c in full_text)

    if has_emoji:
        try:
            emoji_font = ImageFont.truetype(EMOJI_FONT_PATH, pil_fontsize)
        except:
            emoji_font = font_regular
    else:
        emoji_font = None

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    label_rgb = hex_to_rgb(label_color)
    token_rgb = hex_to_rgb(token_color)

    # Calculate sizes
    temp_img = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_img)

    # "PROMOTES: " part
    label_part = "PROMOTES: "
    label_bbox = temp_draw.textbbox((0, 0), label_part, font=font_bold)
    label_width = label_bbox[2] - label_bbox[0]

    # Tokens part - measure each segment for emoji handling
    tokens_width = 0
    if emoji_font:
        for char in tokens_str:
            if ord(char) > 0x1F300:
                bbox = temp_draw.textbbox((0, 0), char, font=emoji_font)
            else:
                bbox = temp_draw.textbbox((0, 0), char, font=font_regular)
            tokens_width += bbox[2] - bbox[0]
    else:
        tokens_bbox = temp_draw.textbbox((0, 0), tokens_str, font=font_regular)
        tokens_width = tokens_bbox[2] - tokens_bbox[0]

    total_width = label_width + tokens_width + 20
    height = pil_fontsize + 20

    # Create image
    img = Image.new("RGBA", (int(total_width), height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Draw "PROMOTES: " in bold + label color
    draw.text((5, 5), label_part, font=font_bold, fill=label_rgb + (255,))

    # Draw tokens part - handle emoji char by char if needed
    current_x = 5 + label_width
    if emoji_font:
        for char in tokens_str:
            if ord(char) > 0x1F300:
                draw.text((current_x, 5), char, font=emoji_font, fill=token_rgb + (255,))
                bbox = temp_draw.textbbox((0, 0), char, font=emoji_font)
            else:
                draw.text((current_x, 5), char, font=font_regular, fill=token_rgb + (255,))
                bbox = temp_draw.textbbox((0, 0), char, font=font_regular)
            current_x += bbox[2] - bbox[0]
    else:
        draw.text((current_x, 5), tokens_str, font=font_regular, fill=token_rgb + (255,))

    # Add to matplotlib
    img_array = np.array(img)
    imagebox = OffsetImage(img_array, zoom=0.25)
    imagebox.image.axes = ax
    box_alignment = (0.5, 0.5) if ha == "center" else (1.0, 0.5) if ha == "right" else (0.0, 0.5)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=box_alignment)
    ax.add_artist(ab)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Circuit Analysis Results")
    default_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    parser.add_argument("--input_dir", type=str, default=default_input)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_circuit_data(input_dir):
    """Load circuit analysis CSVs."""
    data = {}
    files = {
        "upstream": "circuit_upstream.csv",
        "downstream": "circuit_downstream.csv",
        "intervention": "circuit_intervention.csv",
        "validated_features": "validated_features.csv",
        "token_drivers": "circuit_token_drivers.csv",
        "attention_sources": "circuit_attention_sources.csv",
    }
    for key, filename in files.items():
        path = os.path.join(input_dir, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"Loaded: {filename} ({len(data[key])} rows)")
        else:
            if key not in ["token_drivers", "attention_sources"]:
                print(f"Warning: {filename} not found")
            data[key] = None
    return data


def _is_valid_salient_token(token_str):
    """Filter for salient tokens from input prompts."""
    if not token_str or len(str(token_str).strip()) == 0:
        return False
    token_str = str(token_str).strip()
    cleaned = token_str.lstrip('\u2581').strip()
    if len(cleaned) == 0:
        return False
    # Filter special tokens
    if token_str in ["", "<", ">", "[", "]", "\u2581", "<unk>", "<pad>", "<s>", "</s>", "<bos>", "<eos>", "bos", "eos"]:
        return False
    if token_str.startswith("<") and token_str.endswith(">"):
        return False
    if any(ord(c) < 32 for c in token_str):
        return False
    # Filter single non-meaningful chars
    if len(cleaned) == 1 and cleaned not in {'!', '?', '.', ',', ':', ';'}:
        return False
    # Filter pure punctuation
    if all(c in '.,;:\'"()[]{}<>-_/\\|@#$%^&*+=~`' for c in cleaned):
        return False
    return True


def _clean_token(token_str):
    """Clean token string for display."""
    return str(token_str).strip().lstrip('\u2581').strip()


def get_feature_info(validated_df, emotion):
    """Get top feature info for an emotion.

    Priority order:
    1. FEATURE_OVERRIDE (if specified for the emotion)
    2. First row for each emotion in validated_features.csv
    """
    if validated_df is None:
        return None

    # Check for override first
    if FEATURE_OVERRIDE.get(emotion) is not None:
        override = FEATURE_OVERRIDE[emotion]
        # Find selectivity from CSV for the overridden feature
        match = validated_df[(validated_df["emotion"] == emotion) &
                             (validated_df["layer"] == override["layer"]) &
                             (validated_df["feature_id"] == override["feature_id"])]
        if len(match) > 0:
            selectivity = float(match.iloc[0]["selectivity"])
        else:
            selectivity = 0.0
        return {
            "layer": override["layer"],
            "feature_id": override["feature_id"],
            "selectivity": selectivity,
        }

    # Fall back to CSV (first row for this emotion)
    emotion_df = validated_df[validated_df["emotion"] == emotion]
    if len(emotion_df) == 0:
        return None
    top = emotion_df.iloc[0]
    return {
        "layer": int(top["layer"]),
        "feature_id": int(top["feature_id"]),
        "selectivity": float(top["selectivity"]),
    }


def plot_causal_circuit(emotion, feature_info, upstream_df, downstream_df, output_dir,
                        token_drivers_df=None, attention_sources_df=None):
    """Create circuit diagram for an emotion."""
    os.makedirs(output_dir, exist_ok=True)

    if feature_info is None:
        print(f"  No feature info for {emotion}, skipping")
        return

    layer = feature_info["layer"]
    feature_id = feature_info["feature_id"]

    if upstream_df is None or len(upstream_df) == 0:
        print(f"  No upstream data for {emotion}, skipping")
        return

    up_data = upstream_df[upstream_df["emotion"] == emotion]
    if len(up_data) == 0:
        print(f"  No upstream data for {emotion}, skipping")
        return

    # Aggregate causal effects (only MLP and Head, filter out Attn)
    if "component_type" in up_data.columns:
        # Only keep MLP and Head, filter out Attn
        filtered_data = up_data[up_data["component_type"].isin(["MLP", "Head"])]
        up_agg = filtered_data.groupby(["component_name", "component_type", "upstream_layer"]).agg({
            "causal_effect": "mean",
        }).reset_index()
        all_components = [{"name": r["component_name"], "layer": int(r["upstream_layer"]),
                          "type": r["component_type"], "effect": r["causal_effect"]}
                         for _, r in up_agg.iterrows()]
    else:
        # Legacy format - only keep MLP
        up_agg = up_data.groupby("upstream_layer").agg({
            "mlp_causal_effect": "mean",
        }).reset_index()
        all_components = []
        for _, r in up_agg.iterrows():
            l = int(r["upstream_layer"])
            all_components.append({"name": f"L{l}.MLP", "layer": l, "type": "MLP", "effect": r["mlp_causal_effect"]})

    # Filter for only positive causal effects (components that PROMOTE the emotion)
    # Negative effects = suppression, which we exclude per Option A
    all_components = [c for c in all_components if c["effect"] > 0]

    # Sort by effect value (descending) - all positive now
    all_components.sort(key=lambda x: x["effect"], reverse=True)

    total_effect = sum(c["effect"] for c in all_components)
    for c in all_components:
        c["percentage"] = (c["effect"] / total_effect * 100) if total_effect > 1e-6 else 0.0

    top_components = all_components[:5]

    # Get promoted tokens using curated whitelist
    promoted_tokens = []
    whitelist = PROMOTED_WHITELIST.get(emotion, set())
    if downstream_df is not None and whitelist:
        down_data = downstream_df[downstream_df["emotion"] == emotion]
        promoted = down_data[down_data["direction"] == "promoted"]
        # Filter by whitelist
        def in_whitelist(token_str):
            cleaned = _clean_token(token_str)
            return cleaned in whitelist or cleaned.lower() in {w.lower() for w in whitelist}
        promoted = promoted[promoted["token"].apply(in_whitelist)].head(10)
        for _, row in promoted.iterrows():
            token_display = _clean_token(row["token"])
            promoted_tokens.append(token_display)

    # Get salient tokens (from input prompt) - sorted by position in prompt
    driver_tokens = []
    if token_drivers_df is not None and len(token_drivers_df) > 0:
        drivers_data = token_drivers_df[token_drivers_df["emotion"] == emotion]
        if len(drivers_data) > 0:
            # Filter by activation threshold and valid tokens
            # Only include tokens with meaningful activation (> 0.1)
            drivers_filtered = drivers_data[drivers_data["activation"] > 0.1]
            drivers_filtered = drivers_filtered[drivers_filtered["token"].apply(_is_valid_salient_token)]
            # Need at least 2 meaningful tokens to show salient section
            if len(drivers_filtered) >= 2:
                # Get top 8 by activation, then sort by position (prompt order)
                top_by_activation = drivers_filtered.nlargest(8, "activation")
                top_by_position = top_by_activation.sort_values("position", ascending=True)
                for _, row in top_by_position.iterrows():
                    token_display = _clean_token(row["token"])
                    driver_tokens.append(token_display)

    example_prompt = EXAMPLE_PROMPTS.get(emotion, "...")

    # Determine figure height based on whether we have token drivers
    has_token_evidence = len(driver_tokens) > 0
    fig_height = 9.0 if has_token_evidence else 7.2

    # Create figure (tight layout)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_xlim(0, 14)
    ax.set_ylim(-1.5, fig_height)
    ax.set_aspect("equal")
    ax.axis("off")

    emotion_color = EMOTION_COLORS.get(emotion, "#4a4a4a")
    box_color     = "#FAFAFA"

    # === REFINED COLOR PALETTE ===
    gray_text     = "#9AA0A6"
    neutral_dark  = "#546E7A"
    neutral_label = "#455A64"
    head_color    = "#78909C"
    mlp_color     = "#90A4AE"
    promote_color = "#37474F"
    arrow_color   = emotion_color

    # === Y POSITIONS ===
    y_offset = 1.25 if has_token_evidence else 0.0

    # === INPUT PROMPT BOX ===
    input_y = 5.9 + y_offset
    ax.add_patch(FancyBboxPatch((3.28, input_y), 7.44, 1.05, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor=box_color, edgecolor=neutral_dark, linewidth=1.0))
    ax.text(7, input_y + 0.73, "INPUT PROMPT", ha="center", va="bottom",
            fontsize=13, fontweight="medium", color=neutral_label)
    ax.text(7, input_y + 0.48, f'"{example_prompt}"', ha="center", va="center",
            fontsize=11, style="italic", color=gray_text)

    # === SALIENT TOKENS SECTION (same size as INPUT PROMPT) ===
    salient_y = 5.50  # Adjusted so both arrows have equal length (0.6)
    salient_height = 1.05  # Same as INPUT PROMPT
    causal_top = 2.90 + 2.0  # causal_y + height = top edge of CAUSAL box

    if has_token_evidence:
        # Arrow from INPUT bottom to SALIENT top (on block edges)
        ax.annotate("", xy=(7, salient_y + salient_height), xytext=(7, input_y),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0, mutation_scale=15))

        ax.add_patch(FancyBboxPatch((3.28, salient_y), 7.44, salient_height, boxstyle="round,pad=0.02,rounding_size=0.12",
                                     facecolor=box_color, edgecolor=neutral_dark, linewidth=1.0))
        ax.text(7, salient_y + 0.73, "SALIENT TOKENS", ha="center", va="bottom",
                fontsize=13, fontweight="medium", color=neutral_label)

        if driver_tokens:
            # Display full sentence with salient tokens highlighted
            render_highlighted_sentence(
                ax, 7, salient_y + 0.48, example_prompt, driver_tokens, emotion_color,
                base_color="#B0BEC5", fontsize=11, ha="center"
            )
        else:
            ax.text(7, salient_y + 0.48, "(computing...)", ha="center", va="center", fontsize=11, color=gray_text)

        # Arrow from SALIENT bottom to CAUSAL top (on block edges)
        ax.annotate("", xy=(7, causal_top), xytext=(7, salient_y),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0, mutation_scale=15))
    else:
        # Arrow from INPUT bottom to CAUSAL top (on block edges)
        ax.annotate("", xy=(7, causal_top), xytext=(7, input_y),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0, mutation_scale=15))

    # === CAUSAL CONTRIBUTORS ===
    causal_y = 2.90
    ax.add_patch(FancyBboxPatch((1.4, causal_y), 11.2, 2.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor=box_color, edgecolor=neutral_dark, linewidth=1.0))
    ax.text(7, causal_y + 1.68, "TOP 5 CAUSAL CONTRIBUTORS", ha="center", va="bottom",
            fontsize=13, fontweight="medium", color=neutral_label)

    n_comp = len(top_components)
    node_y = 0.90
    node_top = node_y + 0.75  # Top of emotion node circle

    # Store component x positions for curved arrows
    comp_x_positions = []

    if n_comp > 0:
        comp_width = 1.7
        spacing = min(2.2, 11.5 / n_comp)
        start_x = 7 - (n_comp - 1) * spacing / 2
        max_pct = max(c["percentage"] for c in top_components)

        for i, comp in enumerate(top_components):
            cx = start_x + i * spacing
            cy = causal_y + 0.95
            comp_x_positions.append(cx)
            # Subtle distinction: Head = cool gray, MLP = warm gray
            if comp["type"] == "Head":
                edge_color = head_color
                face_color = "#E8EEF2"  # Cool blue-gray tint (stronger)
            else:
                edge_color = mlp_color
                face_color = "#F5F0E8"  # Warm cream tint (stronger)
            opacity = 0.6 + 0.4 * (comp["percentage"] / max(max_pct, 1))

            ax.add_patch(FancyBboxPatch((cx - comp_width/2, cy - 0.45), comp_width, 1.0,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=face_color, edgecolor=edge_color, linewidth=1.0, alpha=opacity))
            ax.text(cx, cy + 0.24, comp["name"], ha="center", va="center",
                    fontsize=11, fontweight="semibold", color=neutral_label)
            ax.text(cx, cy, f"({comp['percentage']:.1f}%)", ha="center", va="center",
                    fontsize=10, color=gray_text)
            ax.text(cx, cy - 0.24, f"{comp['effect']:+.2f}", ha="center", va="center",
                    fontsize=11, fontweight="normal", color=emotion_color)

            # Curved arrow from component box bottom to emotion node top
            arrow_start_y = cy - 0.45  # Each arrow starts from its own box bottom
            # Calculate curvature based on distance from center (x=7)
            # Positive rad = curve to the right, negative = curve to the left
            distance_from_center = cx - 7
            if abs(distance_from_center) < 0.3:
                # Near center: straight arrow
                rad = 0.0
            else:
                # Farther from center: more curvature, direction based on position
                rad = -0.25 * (distance_from_center / abs(distance_from_center)) * min(1.0, abs(distance_from_center) / 3.5)

            # Line width proportional to contribution percentage (0.5 to 2.0)
            arrow_lw = 0.5 + 1.5 * (comp["percentage"] / max(max_pct, 1))
            arrow = FancyArrowPatch(
                (cx, arrow_start_y), (7, node_top),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                color=arrow_color,
                lw=arrow_lw,
                mutation_scale=10,
                alpha=0.35,
                zorder=1
            )
            ax.add_patch(arrow)

    # === EMOTION NODE ===
    EMOTION_BG_COLORS = {
        "Joy": "#FFF3C4", "Sadness": "#D6E8F5", "Anger": "#FFD9D9",
        "Fear": "#E1D5F0", "Disgust": "#E5F0D8",
    }
    bg_color = EMOTION_BG_COLORS.get(emotion, "#F5F5F5")
    from matplotlib.patches import Circle
    # node_y already defined above for curved arrows
    circle = Circle((7, node_y), 0.75, facecolor=bg_color, edgecolor=emotion_color, linewidth=1.8)
    ax.add_patch(circle)
    ax.text(7, node_y + 0.15, f"{emotion}", ha="center", va="center",
            fontsize=15, fontweight="semibold", color=emotion_color)
    ax.text(7, node_y - 0.2, f"F{feature_id} (L{layer})", ha="center", va="center",
            fontsize=10, color=gray_text)

    # Arrow to DOWNSTREAM
    ax.annotate("", xy=(7, -0.42), xytext=(7, node_y - 0.75),
                arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0, mutation_scale=15))

    # === DOWNSTREAM EFFECT ===
    box_height = 0.9
    box_y = -1.32

    ax.add_patch(FancyBboxPatch((1.42, box_y), 11.17, box_height, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor=box_color, edgecolor=neutral_dark, linewidth=1.0))
    ax.text(7, box_y + box_height - 0.32, "DOWNSTREAM EFFECT", ha="center", va="bottom",
            fontsize=12, fontweight="medium", color=neutral_label)

    if promoted_tokens:
        tokens_str = ", ".join(promoted_tokens[:15])
        y_pos = box_y + 0.32
        ax.text(1.62, y_pos, "PROMOTES:", ha="left", va="center",
                fontsize=12, fontweight="medium", color=promote_color)
        add_emoji_text_to_ax(ax, 3.52, y_pos, tokens_str, fontsize=12, color=gray_text, ha="left")
    else:
        ax.text(7, box_y + 0.32, "no interpretable tokens", ha="center", va="center",
                fontsize=11, style="italic", color=gray_text)

    plt.tight_layout()
    # Save PNG
    png_path = os.path.join(output_dir, f"circuit_{emotion.lower()}.png")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"  Saved: circuit_{emotion.lower()}.png")
    # Save PDF
    pdf_path = os.path.join(output_dir, f"circuit_{emotion.lower()}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"  Saved: circuit_{emotion.lower()}.pdf")
    plt.close()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "figures")

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    data = load_circuit_data(input_dir)
    upstream_df = data.get("upstream")
    downstream_df = data.get("downstream")
    intervention_df = data.get("intervention")
    validated_df = data.get("validated_features")
    token_drivers_df = data.get("token_drivers")
    attention_sources_df = data.get("attention_sources")

    if upstream_df is None:
        print("\nNo circuit_upstream.csv found. Run src/circuit.py first.")
        return

    print("\nGenerating circuit diagrams...")
    for emotion in EMOTIONS:
        feature_info = get_feature_info(validated_df, emotion)
        plot_causal_circuit(emotion, feature_info, upstream_df, downstream_df, output_dir,
                           token_drivers_df=token_drivers_df, attention_sources_df=attention_sources_df)

    print(f"\nCircuit diagrams saved to: {output_dir}")
    print("Run plot_causal_heatmap.py and plot_intervention.py for additional figures.")


if __name__ == "__main__":
    main()
