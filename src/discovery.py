"""
SAE Emotion Feature Discovery (English Only, Streaming Mode)

Uses Google's Gemma Scope SAEs to discover emotion-selective features
in Gemma 2 2B. Analyzes 5 emotions (Joy, Sadness, Anger, Fear, Disgust) using GoEmotions.

Key Features:
- Streaming SAE loading: One layer at a time to avoid OOM
- Uses GoEmotions dataset (Reddit comments)
- Discovers Top-12 features per emotion across all 26 layers

Usage:
    python discovery.py --test         # Test mode (20 samples per emotion)
    python discovery.py --full         # Full mode (all GoEmotions samples)
    python discovery.py --device cuda  # Use GPU
    python discovery.py --width 16k    # SAE width (16k, 32k, 65k, etc.)

Output:
    results/feature_lists.csv
    results/heatmap_joy.csv
    results/heatmap_sadness.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE

from prompts import get_prompts_with_labels, DEFAULT_EMOTIONS


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 30          # Top-K candidates per emotion
N_FINAL = 12        # Final number of features to select per emotion
EMOTIONS = DEFAULT_EMOTIONS  # ["joy", "sadness", "anger", "fear", "disgust"]

# Gemma Scope configuration
GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"


def parse_args():
    parser = argparse.ArgumentParser(description="SAE Emotion Feature Discovery")
    parser.add_argument("--test", action="store_true", help="Use test dataset (20 per emotion)")
    parser.add_argument("--full", action="store_true", help="Use full GoEmotions dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, or mps")
    parser.add_argument("--width", type=str, default="16k",
                        choices=["16k", "32k", "65k", "131k", "262k", "524k", "1m"],
                        help="Gemma Scope SAE width (default: 16k)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Samples per emotion (default: 100)")
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Output directory")
    return parser.parse_args()


# =============================================================================
# Model and SAE Loading
# =============================================================================

def get_available_sae_layers():
    """Return all 26 layers for Gemma 2 2B."""
    return list(range(26))


def load_model(device):
    """Load Gemma 2 2B model."""
    print(f"Loading Gemma 2 2B on {device}...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    return model


def load_single_sae(layer, width, device):
    """Load a single SAE for the specified layer."""
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae, _, _ = SAE.from_pretrained(
        release=GEMMA_SCOPE_RELEASE,
        sae_id=sae_id,
        device=device
    )
    return sae


# =============================================================================
# Collect Residual Activations
# =============================================================================

def collect_residuals(model, prompts, layers, device):
    """
    Collect residual activations from all layers for all prompts.
    Store on CPU to save GPU memory.

    Returns:
        residuals: Dict[layer] -> np.array of shape (n_prompts, d_model)
        metadata: List of dicts with prompt info
    """
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    residuals = {l: [] for l in layers}
    metadata = []

    print(f"Collecting residuals for {len(prompts)} prompts across {len(layers)} layers...")

    for p in tqdm(prompts):
        text = p["text"]

        with torch.no_grad():
            _, cache = model.run_with_cache(text, names_filter=hook_names)

            for layer in layers:
                hook_name = f"blocks.{layer}.hook_resid_post"
                # Get last token residual, move to CPU
                resid = cache[hook_name][0, -1, :].cpu().numpy()
                residuals[layer].append(resid)

            metadata.append({
                "text": text,
                "emotion": p["emotion"],
            })

    # Stack into arrays
    for layer in layers:
        residuals[layer] = np.stack(residuals[layer])

    return residuals, metadata


# =============================================================================
# Streaming SAE Processing
# =============================================================================

def process_layer_streaming(layer, residuals, metadata, device, width="16k"):
    """
    Process a single layer with streaming SAE loading.

    1. Load SAE
    2. Encode all residuals
    3. Compute selectivity
    4. Release SAE

    Returns:
        selectivity: Dict[feature_id] -> {Joy_sel, Sadness_sel}
    """
    print(f"  Processing layer {layer}...")

    # Load SAE
    sae = load_single_sae(layer, width, device)

    # Encode residuals
    resid_tensor = torch.tensor(residuals[layer], device=device, dtype=torch.float32)

    with torch.no_grad():
        feature_acts = sae.encode(resid_tensor)
        feature_acts = feature_acts.cpu().numpy()

    # Build emotion masks for all emotions
    df = pd.DataFrame(metadata)
    emotions_cap = [e.capitalize() for e in EMOTIONS]

    # Compute mean activations per emotion
    mean_acts = {}
    for emotion in emotions_cap:
        mask = (df["emotion"] == emotion).values
        if mask.sum() > 0:
            mean_acts[emotion] = feature_acts[mask].mean(axis=0)
        else:
            mean_acts[emotion] = np.zeros(feature_acts.shape[1])

    # Compute overall mean (for all samples)
    overall_mean = feature_acts.mean(axis=0)

    # Compute selectivity: mean(emotion) - mean(others)
    num_features = feature_acts.shape[1]
    selectivity = {}

    for f in range(num_features):
        selectivity[f] = {}
        for emotion in emotions_cap:
            # Selectivity = mean activation for this emotion - mean of other emotions
            other_means = [mean_acts[e][f] for e in emotions_cap if e != emotion]
            other_mean = np.mean(other_means) if other_means else 0
            selectivity[f][emotion] = float(mean_acts[emotion][f] - other_mean)

    # Release SAE
    del sae
    del resid_tensor
    del feature_acts
    torch.cuda.empty_cache()

    return selectivity


# =============================================================================
# Feature Selection
# =============================================================================

def select_top_features(all_selectivity, emotion, top_k=N_FINAL):
    """
    Select Top-K (layer, feature) pairs for the given emotion.

    Returns:
        List of (layer, feature_id, score) tuples
    """
    candidates = []

    for layer in all_selectivity:
        for f, scores in all_selectivity[layer].items():
            score = scores[emotion]
            candidates.append((layer, f, score))

    # Sort by score descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    return candidates[:top_k]


# =============================================================================
# Build Heatmap Matrix
# =============================================================================

def build_heatmap(all_selectivity, features, emotion, layers):
    """
    Build heatmap matrix for selected features.

    Args:
        all_selectivity: Dict[layer][feature] -> {Joy, Sadness}
        features: List of (layer, feature_id, score) for this emotion
        emotion: "Joy" or "Sadness"
        layers: All layer indices

    Returns:
        DataFrame with rows=features, cols=layers
    """
    n_features = len(features)
    n_layers = len(layers)

    # Initialize matrix with NaN
    matrix = np.full((n_features, n_layers), np.nan)

    # Fill in selectivity values
    for i, (feat_layer, feat_id, _) in enumerate(features):
        for j, layer in enumerate(layers):
            if layer == feat_layer:
                matrix[i, j] = all_selectivity[layer][feat_id][emotion]

    # Create DataFrame
    row_labels = [f"L{layer}_F{fid}" for (layer, fid, _) in features]
    col_labels = [f"L{l}" for l in layers]

    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.index.name = "feature"

    return df


# =============================================================================
# Export Results
# =============================================================================

def export_results(all_features, all_selectivity, layers, output_dir):
    """Export feature lists and heatmaps to CSV for all emotions."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export feature lists
    feature_rows = []

    for emotion, features in all_features.items():
        for layer, fid, score in features:
            feature_rows.append({
                "emotion": emotion,
                "layer": int(layer),
                "feature_id": int(fid),
                "selectivity": float(score),
            })

    feature_df = pd.DataFrame(feature_rows)
    feature_path = os.path.join(output_dir, "feature_lists.csv")
    feature_df.to_csv(feature_path, index=False)
    print(f"Saved: {feature_path}")

    # 2. Export heatmaps for each emotion
    for emotion, features in all_features.items():
        heatmap = build_heatmap(all_selectivity, features, emotion, layers)
        heatmap_path = os.path.join(output_dir, f"heatmap_{emotion.lower()}.csv")
        heatmap.to_csv(heatmap_path)
        print(f"Saved: {heatmap_path}")

    return feature_df


# =============================================================================
# Print Summary
# =============================================================================

def print_summary(all_features, all_selectivity):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("  DISCOVERY RESULTS")
    print("=" * 70)

    for emotion, features in all_features.items():
        print(f"\n  {emotion.upper()} FEATURES (Top {len(features)}):")
        print(f"  {'-' * 50}")
        for layer, fid, score in features:
            print(f"    L{layer:2d} F{fid:5d}: selectivity = {score:.4f}")

    # Layer distribution
    print(f"\n  LAYER DISTRIBUTION:")
    print(f"  {'-' * 50}")

    for emotion, features in all_features.items():
        layers_used = sorted(set([l for l, _, _ in features]))
        print(f"    {emotion:10s}: {layers_used}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Determine mode
    if args.test:
        n_samples = 20
        print("=== TEST MODE (20 samples per emotion) ===")
    elif args.full:
        n_samples = None
        print("=== FULL MODE (all GoEmotions samples) ===")
    else:
        n_samples = args.n_samples
        print(f"=== DEFAULT MODE ({n_samples} samples per emotion) ===")

    device = args.device
    width = args.width
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Get all layers
    layers = get_available_sae_layers()
    print(f"Will analyze {len(layers)} layers: {layers[0]}-{layers[-1]}")
    print(f"SAE width: {width}")

    # Load model (once)
    model = load_model(device)

    # Load prompts
    if n_samples is None:
        prompts = get_prompts_with_labels(use_full=True)
    else:
        prompts = get_prompts_with_labels(use_full=False, n_samples=n_samples)

    print(f"Loaded {len(prompts)} prompts")

    # Count by emotion
    emotions_cap = [e.capitalize() for e in EMOTIONS]
    for emotion in emotions_cap:
        count = sum(1 for p in prompts if p["emotion"] == emotion)
        print(f"  {emotion}: {count}")

    # Step 1: Collect all residuals (model forward pass, once)
    residuals, metadata = collect_residuals(model, prompts, layers, device)
    print(f"Residuals collected: {residuals[0].shape}")

    # Free model from GPU (optional, if memory is tight)
    # del model
    # torch.cuda.empty_cache()

    # Step 2: Process each layer with streaming SAE
    print(f"\nProcessing {len(layers)} layers with streaming SAE...")
    all_selectivity = {}

    for layer in tqdm(layers, desc="Layers"):
        all_selectivity[layer] = process_layer_streaming(
            layer, residuals, metadata, device, width
        )

    # Step 3: Select top features for all emotions
    print("\nSelecting top features...")
    all_features = {}
    for emotion in emotions_cap:
        all_features[emotion] = select_top_features(all_selectivity, emotion, top_k=N_FINAL)

    # Step 4: Print summary
    print_summary(all_features, all_selectivity)

    # Step 5: Export results
    print("\nExporting results...")
    export_results(all_features, all_selectivity, layers, output_dir)

    print("\n" + "=" * 70)
    print("  DISCOVERY COMPLETE!")
    print("=" * 70)
    print(f"  Results saved to: {output_dir}")
    print(f"\n  To generate heatmap visualizations, run:")
    print(f"    python viz/plot_discovery_heatmap.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
