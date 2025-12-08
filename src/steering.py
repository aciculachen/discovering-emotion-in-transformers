"""
Steering Experiments for Validated Emotion Features

Uses validated features from validation pipeline to test causality:
- Activation (positive α): Add feature direction → emotion should increase
- Ablation (negative α): Subtract feature direction → emotion should decrease

Consistency with discovery.py:
- Model: Gemma 2 2B (gemma-2-2b)
- SAE: Gemma Scope (gemma-scope-2b-pt-res-canonical)
- Dataset: GoEmotions (validation/test split to avoid train overlap)
- Hook: blocks.{layer}.hook_resid_post

Output:
    - results/steering_results.csv
    - results/steering_controls.csv
    - results/steering_activations.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch.nn.functional as F

from prompts import get_prompts_with_labels, DEFAULT_EMOTIONS

# =============================================================================
# CONFIGURATION
# =============================================================================

GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"
EMOTIONS = [e.capitalize() for e in DEFAULT_EMOTIONS]

# Emotion words for probability measurement
EMOTION_WORDS = {
    "Joy": ["happy", "glad", "excited", "joyful", "delighted"],
    "Sadness": ["sad", "unhappy", "depressed", "miserable", "sorrowful"],
    "Anger": ["angry", "furious", "mad", "irritated", "outraged"],
    "Fear": ["afraid", "scared", "terrified", "frightened", "anxious"],
    "Disgust": ["disgusting", "revolting", "repulsive", "nauseating", "gross"],
}

# Neutral prompts for control condition
NEUTRAL_PROMPTS = [
    "The weather is",
    "Today I feel",
    "This news is",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Steering experiments")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--test", action="store_true", help="Run minimal test")
    parser.add_argument("--width", type=str, default="16k",
                        choices=["16k", "32k", "65k", "131k"],
                        help="Gemma Scope SAE width")
    parser.add_argument("--n_prompts", type=int, default=10,
                        help="Number of prompts per emotion for steering test")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


# =============================================================================
# Model and SAE Loading
# =============================================================================

def load_model(device):
    """Load Gemma 2 2B model."""
    print(f"Loading Gemma 2 2B on {device}...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    return model


def load_sae(layer, width, device):
    """Load a single Gemma Scope SAE for the specified layer."""
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae, _, _ = SAE.from_pretrained(
        release=GEMMA_SCOPE_RELEASE,
        sae_id=sae_id,
        device=device
    )
    return sae


# =============================================================================
# Feature Selection
# =============================================================================

def load_validated_features(results_dir):
    """
    Load validated features and compute combined score.

    combined_score = discrimination_ratio × specificity_ratio
    """
    validated_path = os.path.join(results_dir, "validated_features.csv")
    confound_path = os.path.join(results_dir, "validation_confound.csv")
    cross_path = os.path.join(results_dir, "validation_cross_emotion.csv")

    if not os.path.exists(validated_path):
        raise FileNotFoundError(f"Run validation.py first: {validated_path}")

    validated_df = pd.read_csv(validated_path)
    confound_df = pd.read_csv(confound_path)
    cross_df = pd.read_csv(cross_path)

    # Merge to get both ratios
    merged = validated_df.merge(
        confound_df[["emotion", "layer", "feature_id", "discrimination_ratio"]],
        on=["emotion", "layer", "feature_id"],
        how="left"
    ).merge(
        cross_df[["emotion", "layer", "feature_id", "specificity_ratio"]],
        on=["emotion", "layer", "feature_id"],
        how="left"
    )

    # Compute combined score
    merged["combined_score"] = merged["discrimination_ratio"] * merged["specificity_ratio"]

    return merged


def select_top_features(features_df, n_per_emotion=3):
    """Select top N features per emotion by combined_score."""
    selected = []

    for emotion in EMOTIONS:
        emotion_df = features_df[features_df["emotion"] == emotion].copy()
        emotion_df = emotion_df.sort_values("combined_score", ascending=False)
        top_n = emotion_df.head(n_per_emotion)
        selected.append(top_n)

        print(f"  {emotion}: {len(top_n)} features selected")
        for _, row in top_n.iterrows():
            print(f"    L{int(row['layer'])}:F{int(row['feature_id'])} "
                  f"(score={row['combined_score']:.2f})")

    return pd.concat(selected, ignore_index=True)


# =============================================================================
# Steering Implementation
# =============================================================================

def get_steering_hook(steering_dir, alpha):
    """
    Create a hook that adds normalized steering direction to residual stream.

    steering_dir should be pre-normalized.
    """
    def hook(resid, hook):
        # Add steering direction scaled by alpha
        return resid + alpha * steering_dir
    return hook


def get_word_log_prob(model, prompt, word, hooks=None):
    """
    Calculate sequence log-prob for a word (handles multi-token words).

    log P(word|prompt) = Σᵢ log P(tᵢ|prompt, t<i)
    """
    try:
        word_tokens = model.to_tokens(word, prepend_bos=False)[0]
    except:
        return float('-inf')

    if len(word_tokens) == 0:
        return float('-inf')

    total_log_prob = 0.0
    current_prompt = prompt

    for token_id in word_tokens:
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                logits = model(current_prompt)[0, -1, :]
        else:
            logits = model(current_prompt)[0, -1, :]

        log_probs = F.log_softmax(logits, dim=-1)
        total_log_prob += log_probs[token_id.item()].item()
        current_prompt = current_prompt + model.to_string(token_id)

    return total_log_prob


def compute_emotion_log_prob(model, prompt, emotion, hooks=None):
    """Compute average log-prob for emotion words."""
    words = EMOTION_WORDS.get(emotion, [])
    if not words:
        return float('-inf')

    log_probs = []
    for word in words:
        lp = get_word_log_prob(model, prompt, word, hooks)
        if lp > float('-inf'):
            log_probs.append(lp)

    if not log_probs:
        return float('-inf')

    return sum(log_probs) / len(log_probs)


def run_steering_experiment(model, sae, feature_row, prompt, alphas, device):
    """
    Run steering experiment for a single feature/prompt combination.

    Returns list of result dicts for each alpha value.
    """
    layer = int(feature_row["layer"])
    feature_id = int(feature_row["feature_id"])
    emotion = feature_row["emotion"]

    hook_name = f"blocks.{layer}.hook_resid_post"

    # Get normalized steering direction
    steering_dir = sae.W_dec[feature_id].clone()
    steering_dir = steering_dir / steering_dir.norm()

    # === IMPORTANT: Compute baseline (α=0) FIRST ===
    # This ensures delta calculations are correct for all α values
    baseline_log_prob = compute_emotion_log_prob(model, prompt, emotion, hooks=None)

    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        resid = cache[hook_name][0, -1, :].to(device)
        baseline_activation = sae.encode(resid)[feature_id].item()

    results = []

    for alpha in alphas:
        if alpha == 0:
            # Use pre-computed baseline
            log_prob = baseline_log_prob
            activation = baseline_activation
        else:
            hook = get_steering_hook(steering_dir, alpha)
            hooks = [(hook_name, hook)]

            # Compute emotion log-prob with steering
            log_prob = compute_emotion_log_prob(model, prompt, emotion, hooks)

            # Get feature activation with steering
            with torch.no_grad():
                with model.hooks(fwd_hooks=hooks):
                    _, cache = model.run_with_cache(prompt)
                resid = cache[hook_name][0, -1, :].to(device)
                activation = sae.encode(resid)[feature_id].item()

        # Compute deltas (baseline is always available now)
        delta_log_prob = log_prob - baseline_log_prob
        delta_activation = activation - baseline_activation

        results.append({
            "emotion": emotion,
            "layer": layer,
            "feature_id": feature_id,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "alpha": alpha,
            "log_prob": log_prob,
            "delta_log_prob": delta_log_prob,
            "activation": activation,
            "delta_activation": delta_activation,
        })

    return results


def run_control_experiment(model, sae, feature_row, prompt, target_emotion,
                           experiment_type, alpha, device):
    """
    Run control experiment.

    experiment_type: "random" or "cross_emotion"
    """
    layer = int(feature_row["layer"])
    feature_id = int(feature_row["feature_id"])
    source_emotion = feature_row["emotion"]

    hook_name = f"blocks.{layer}.hook_resid_post"

    # Get normalized steering direction
    steering_dir = sae.W_dec[feature_id].clone()
    steering_dir = steering_dir / steering_dir.norm()

    # Baseline
    baseline_log_prob = compute_emotion_log_prob(model, prompt, target_emotion, hooks=None)

    # Steered
    hook = get_steering_hook(steering_dir, alpha)
    hooks = [(hook_name, hook)]
    steered_log_prob = compute_emotion_log_prob(model, prompt, target_emotion, hooks)

    delta = steered_log_prob - baseline_log_prob

    return {
        "experiment_type": experiment_type,
        "source_emotion": source_emotion,
        "target_emotion": target_emotion,
        "layer": layer,
        "feature_id": feature_id,
        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
        "alpha": alpha,
        "baseline_log_prob": baseline_log_prob,
        "steered_log_prob": steered_log_prob,
        "delta_log_prob": delta,
    }


# =============================================================================
# Main Execution
# =============================================================================

def main():
    args = parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(project_root, args.output_dir)

    print("=" * 70)
    print("  STEERING EXPERIMENTS")
    print("=" * 70)
    print(f"  Device: {args.device}")
    print(f"  SAE width: {args.width}")

    # Load model
    model = load_model(args.device)

    # Load validated features
    print("\nLoading validated features...")
    features_df = load_validated_features(results_dir)
    print(f"  Total validated features: {len(features_df)}")

    # Select top 3 per emotion
    print("\nSelecting top features per emotion:")
    n_per_emotion = 1 if args.test else 3
    selected_features = select_top_features(features_df, n_per_emotion=n_per_emotion)

    # Get prompts from GoEmotions validation set
    print("\nLoading prompts from GoEmotions validation set...")
    n_prompts = 2 if args.test else args.n_prompts
    prompts_data = get_prompts_with_labels(
        use_full=False,
        n_samples=n_prompts,
        split="validation"
    )

    # Group prompts by emotion
    prompts_by_emotion = {e: [] for e in EMOTIONS}
    for p in prompts_data:
        prompts_by_emotion[p["emotion"]].append(p["text"])

    # Alpha values
    if args.test:
        alphas = [0, 2, 4]
    else:
        alphas = [-2, -1, 0, 1, 2, 4]

    print(f"\nAlpha values: {alphas}")

    # ==========================================================================
    # Main Steering Experiments
    # ==========================================================================

    print("\n" + "=" * 70)
    print("  RUNNING MAIN STEERING EXPERIMENTS")
    print("=" * 70)

    all_results = []

    # Group features by layer to minimize SAE loading
    layer_groups = selected_features.groupby("layer")

    for layer, layer_features in tqdm(layer_groups, desc="Layers"):
        sae = load_sae(int(layer), args.width, args.device)

        for _, feature_row in layer_features.iterrows():
            emotion = feature_row["emotion"]
            prompts = prompts_by_emotion.get(emotion, [])[:n_prompts]

            for prompt in prompts:
                results = run_steering_experiment(
                    model, sae, feature_row, prompt, alphas, args.device
                )
                all_results.extend(results)

        # Free SAE memory
        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)

    # ==========================================================================
    # Control Experiments
    # ==========================================================================

    print("\n" + "=" * 70)
    print("  RUNNING CONTROL EXPERIMENTS")
    print("=" * 70)

    control_results = []
    control_alpha = 4  # Fixed alpha for controls

    # Cross-emotion steering: test all source→target pairs (excluding same emotion)
    cross_pairs = [(s, t) for s in EMOTIONS for t in EMOTIONS if s != t]

    for source_emotion, target_emotion in cross_pairs:
        # Get a feature from source emotion
        source_features = selected_features[selected_features["emotion"] == source_emotion]
        if len(source_features) == 0:
            continue

        feature_row = source_features.iloc[0]
        layer = int(feature_row["layer"])

        # Get prompts from target emotion
        target_prompts = prompts_by_emotion.get(target_emotion, [])[:3]

        sae = load_sae(layer, args.width, args.device)

        for prompt in target_prompts:
            result = run_control_experiment(
                model, sae, feature_row, prompt, target_emotion,
                "cross_emotion", control_alpha, args.device
            )
            control_results.append(result)

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Neutral prompt steering
    print("  Testing neutral prompts...")
    for emotion in EMOTIONS[:2]:  # Just Joy and Sadness for brevity
        emotion_features = selected_features[selected_features["emotion"] == emotion]
        if len(emotion_features) == 0:
            continue

        feature_row = emotion_features.iloc[0]
        layer = int(feature_row["layer"])

        sae = load_sae(layer, args.width, args.device)

        for prompt in NEUTRAL_PROMPTS[:2]:
            result = run_control_experiment(
                model, sae, feature_row, prompt, emotion,
                "neutral_prompt", control_alpha, args.device
            )
            control_results.append(result)

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    control_df = pd.DataFrame(control_results)

    # ==========================================================================
    # Save Results
    # ==========================================================================

    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "steering_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    controls_path = os.path.join(results_dir, "steering_controls.csv")
    control_df.to_csv(controls_path, index=False)
    print(f"Saved: {controls_path}")

    # ==========================================================================
    # Print Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("  STEERING RESULTS SUMMARY")
    print("=" * 70)

    for emotion in EMOTIONS:
        emotion_results = results_df[results_df["emotion"] == emotion]
        if len(emotion_results) == 0:
            continue

        print(f"\n  {emotion}:")

        # Activation effect (α > 0)
        activation_results = emotion_results[emotion_results["alpha"] > 0]
        if len(activation_results) > 0:
            mean_delta = activation_results["delta_log_prob"].mean()
            print(f"    Activation (α>0): Δlog-prob = {mean_delta:+.3f}")

        # Ablation effect (α < 0)
        ablation_results = emotion_results[emotion_results["alpha"] < 0]
        if len(ablation_results) > 0:
            mean_delta = ablation_results["delta_log_prob"].mean()
            print(f"    Ablation (α<0):   Δlog-prob = {mean_delta:+.3f}")

    print("\n  Control Experiments:")
    for exp_type in control_df["experiment_type"].unique():
        exp_results = control_df[control_df["experiment_type"] == exp_type]
        mean_delta = exp_results["delta_log_prob"].mean()
        print(f"    {exp_type}: Δlog-prob = {mean_delta:+.3f}")

    print("\n" + "=" * 70)
    print("  Done! Run viz/plot_steering.py to generate figures.")
    print("=" * 70)


if __name__ == "__main__":
    main()
