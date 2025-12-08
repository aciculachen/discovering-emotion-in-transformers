"""
Anger Feature Selection via Causal Circuit Analysis

Problem: The top Anger feature (L25:F8662, selectivity=39.0) shows abnormal
causal dynamics - all top causal components have NEGATIVE Δ values, meaning
ablating them INCREASES the feature activation. This indicates a suppressor
circuit rather than an activator circuit.

Solution: Test all 10 validated Anger features and find one with proper
causal dynamics (positive Δ for causal components, ablation reduces activation).

Usage:
    python anger_circuit.py --device cuda
    python anger_circuit.py --device cpu --test

Output:
    - results/anger_feature_comparison.csv
    - Terminal recommendation for best feature
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

from prompts import get_prompts_with_labels

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "gemma-2-2b"
GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_WIDTH = "16k"

ANGER_WORDS = ["angry", "mad", "furious", "annoyed"]


def parse_args():
    parser = argparse.ArgumentParser(description="Anger Feature Comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=str, default="16k")
    parser.add_argument("--test", action="store_true",
                        help="Test mode with fewer prompts")
    parser.add_argument("--n_prompts", type=int, default=3,
                        help="Number of prompts to test")
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    parser.add_argument("--output_dir", type=str, default=default_output)
    return parser.parse_args()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_pct_change(new_val, baseline_val, cap=1000.0):
    """Compute percentage change with safety bounds."""
    if abs(baseline_val) < 1e-6:
        if abs(new_val) < 1e-6:
            return 0.0
        return cap if new_val > 0 else -cap
    pct = (new_val - baseline_val) / abs(baseline_val) * 100
    return max(-cap, min(cap, pct))


def load_model(device):
    """Load Gemma 2 2B model."""
    print(f"Loading {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    return model


def load_sae(layer, width, device):
    """Load SAE for specified layer."""
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae, _, _ = SAE.from_pretrained(
        release=GEMMA_SCOPE_RELEASE,
        sae_id=sae_id,
        device=device
    )
    return sae


def load_all_anger_features(results_dir):
    """Load all 10 validated Anger features."""
    csv_path = os.path.join(results_dir, "validated_features.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"validated_features.csv not found at {csv_path}. "
            "Run validation.py first."
        )

    df = pd.read_csv(csv_path)
    anger_df = df[df["emotion"] == "Anger"]

    features = []
    for _, row in anger_df.iterrows():
        features.append({
            "layer": int(row["layer"]),
            "feature_id": int(row["feature_id"]),
            "selectivity": float(row["selectivity"]),
        })

    # Sort by selectivity descending
    features.sort(key=lambda x: x["selectivity"], reverse=True)
    return features


def get_ablation_hook(position="last"):
    """Create a hook that zeros out the component output."""
    def hook_fn(activation, hook):
        if position == "last":
            activation[:, -1, :] = 0.0
        else:
            activation[:, :, :] = 0.0
        return activation
    return hook_fn


def compute_feature_activation(model, sae, prompt, feature_id, feature_layer, device, hooks=None):
    """Compute the SAE feature activation for a prompt."""
    hook_name = f"blocks.{feature_layer}.hook_resid_post"

    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                _, cache = model.run_with_cache(prompt)
        else:
            _, cache = model.run_with_cache(prompt)

        resid = cache[hook_name][0, -1, :].to(device)
        activation = sae.encode(resid)[feature_id].item()

    return activation


def compute_emotion_logit(model, prompt, target_words, hooks=None):
    """Compute average logit for emotion words."""
    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                logits = model(prompt)[0, -1, :]
        else:
            logits = model(prompt)[0, -1, :]

    total_logit = 0.0
    count = 0
    for word in target_words:
        token_ids = model.to_tokens(word, prepend_bos=False)[0]
        if len(token_ids) > 0:
            total_logit += logits[token_ids[0]].item()
            count += 1

    return total_logit / count if count > 0 else 0.0


# =============================================================================
# FEATURE EVALUATION
# =============================================================================

def evaluate_feature(model, sae, feature_info, prompts, device):
    """
    Evaluate a single feature's causal dynamics.

    Returns metrics for scoring:
    - positive_ratio: proportion of causal effects that are positive
    - mean_positive_effect: average of positive causal effects
    - intervention_activation_pct: % change when ablating top components
    - intervention_logit_change: logit change when ablating top components
    """
    feature_layer = feature_info["layer"]
    feature_id = feature_info["feature_id"]

    # Only test top-5 upstream layers (to save time)
    upstream_layers = list(range(max(0, feature_layer - 5), feature_layer))

    all_causal_effects = []

    for prompt_data in prompts:
        prompt = prompt_data["text"]

        # Clean run
        clean_activation = compute_feature_activation(
            model, sae, prompt, feature_id, feature_layer, device, hooks=None
        )

        # Test each upstream layer (MLP only for speed)
        for up_layer in upstream_layers:
            mlp_hook_name = f"blocks.{up_layer}.hook_mlp_out"
            mlp_hooks = [(mlp_hook_name, get_ablation_hook("last"))]

            ablated_activation = compute_feature_activation(
                model, sae, prompt, feature_id, feature_layer, device, hooks=mlp_hooks
            )

            causal_effect = clean_activation - ablated_activation

            all_causal_effects.append({
                "layer": up_layer,
                "type": "MLP",
                "effect": causal_effect,
                "clean": clean_activation,
                "ablated": ablated_activation,
            })

    # Calculate positive_ratio
    positive_count = sum(1 for e in all_causal_effects if e["effect"] > 0)
    positive_ratio = positive_count / len(all_causal_effects) if all_causal_effects else 0.0

    # Calculate mean_positive_effect
    positive_effects = [e["effect"] for e in all_causal_effects if e["effect"] > 0]
    mean_positive = np.mean(positive_effects) if positive_effects else 0.0

    # Calculate mean_negative_effect
    negative_effects = [e["effect"] for e in all_causal_effects if e["effect"] < 0]
    mean_negative = np.mean(negative_effects) if negative_effects else 0.0

    # Aggregate by layer
    layer_effects = {}
    for e in all_causal_effects:
        layer = e["layer"]
        if layer not in layer_effects:
            layer_effects[layer] = []
        layer_effects[layer].append(e["effect"])

    layer_avg = {l: np.mean(effects) for l, effects in layer_effects.items()}

    # Get top-3 layers by absolute effect
    sorted_layers = sorted(layer_avg.items(), key=lambda x: abs(x[1]), reverse=True)
    top3_layers = [l for l, _ in sorted_layers[:3]]

    # Intervention: combined ablation of top-3 layers
    combined_hooks = []
    for layer in top3_layers:
        hook_name = f"blocks.{layer}.hook_mlp_out"
        combined_hooks.append((hook_name, get_ablation_hook("last")))

    # Test intervention on all prompts
    activation_changes = []
    logit_changes = []

    for prompt_data in prompts:
        prompt = prompt_data["text"]

        # Baseline
        baseline_activation = compute_feature_activation(
            model, sae, prompt, feature_id, feature_layer, device, hooks=None
        )
        baseline_logit = compute_emotion_logit(model, prompt, ANGER_WORDS, hooks=None)

        # Ablated
        ablated_activation = compute_feature_activation(
            model, sae, prompt, feature_id, feature_layer, device, hooks=combined_hooks
        )
        ablated_logit = compute_emotion_logit(model, prompt, ANGER_WORDS, hooks=combined_hooks)

        activation_changes.append(safe_pct_change(ablated_activation, baseline_activation))
        logit_changes.append(ablated_logit - baseline_logit)

    avg_activation_pct = np.mean(activation_changes)
    avg_logit_change = np.mean(logit_changes)

    return {
        "positive_ratio": positive_ratio,
        "mean_positive_effect": mean_positive,
        "mean_negative_effect": mean_negative,
        "intervention_activation_pct": avg_activation_pct,
        "intervention_logit_change": avg_logit_change,
        "top3_layers": top3_layers,
        "layer_effects": layer_avg,
    }


def compute_score(metrics):
    """
    Compute overall score (0-100).

    Good feature characteristics:
    - High positive_ratio (>50%)
    - intervention_activation_pct is NEGATIVE (ablating reduces activation)
    - intervention_logit_change is NEGATIVE (ablating reduces emotion probability)
    """
    score = 0.0

    # 1. Positive ratio (40 points)
    # 100% positive = 40 points, 50% = 20 points, 0% = 0 points
    score += min(40, metrics["positive_ratio"] * 80)

    # 2. Intervention reduces activation (30 points)
    act_pct = metrics["intervention_activation_pct"]
    if act_pct < -30:
        score += 30  # Strong reduction
    elif act_pct < -10:
        score += 25  # Good reduction
    elif act_pct < 0:
        score += 20  # Some reduction
    elif act_pct < 10:
        score += 10  # Neutral
    elif act_pct < 50:
        score += 5   # Slight increase
    # else: 0 points (bad: large increase)

    # 3. Intervention reduces logit (30 points)
    logit_change = metrics["intervention_logit_change"]
    if logit_change < -1.0:
        score += 30  # Strong reduction
    elif logit_change < -0.5:
        score += 25  # Good reduction
    elif logit_change < 0:
        score += 20  # Some reduction
    elif logit_change < 0.5:
        score += 10  # Neutral
    # else: 0 points (bad: large increase)

    return score


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  ANGER FEATURE COMPARISON")
    print("=" * 70)
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  SAE: {GEMMA_SCOPE_RELEASE}")
    print(f"  Device: {args.device}")

    # Load all Anger features
    print("\n  Loading all Anger features...")
    anger_features = load_all_anger_features(output_dir)
    print(f"  Found {len(anger_features)} Anger features")

    for f in anger_features:
        print(f"    L{f['layer']}:F{f['feature_id']} (selectivity={f['selectivity']:.1f})")

    # Load model
    model = load_model(args.device)

    # Load prompts
    n_prompts = 2 if args.test else args.n_prompts
    print(f"\n  Loading {n_prompts} Anger prompts...")

    all_prompts = get_prompts_with_labels(
        n_samples=n_prompts,
        emotions=["anger"],
        split="validation"
    )

    prompts = [p for p in all_prompts if p["emotion"].lower() == "anger"]
    print(f"  Loaded {len(prompts)} prompts")

    # Evaluate each feature
    print("\n" + "=" * 70)
    print("  EVALUATING FEATURES")
    print("=" * 70)

    results = []
    sae_cache = {}  # Cache SAEs by layer

    for idx, feature in enumerate(anger_features):
        layer = feature["layer"]
        feature_id = feature["feature_id"]
        selectivity = feature["selectivity"]

        print(f"\n  Feature {idx + 1}/{len(anger_features)}: L{layer}:F{feature_id} (selectivity={selectivity:.1f})")

        # Load SAE (with caching)
        if layer not in sae_cache:
            sae_cache[layer] = load_sae(layer, args.width, args.device)
        sae = sae_cache[layer]

        # Evaluate
        metrics = evaluate_feature(model, sae, feature, prompts, args.device)
        score = compute_score(metrics)

        # Print summary
        print(f"    Positive ratio: {metrics['positive_ratio'] * 100:.1f}%")
        print(f"    Mean positive effect: {metrics['mean_positive_effect']:.3f}")
        print(f"    Mean negative effect: {metrics['mean_negative_effect']:.3f}")
        print(f"    Intervention: {metrics['intervention_activation_pct']:+.1f}% activation, "
              f"{metrics['intervention_logit_change']:+.3f} logit")
        print(f"    Top-3 layers: {metrics['top3_layers']}")

        status = "GOOD" if score >= 60 else ("NEUTRAL" if score >= 40 else "BAD")
        print(f"    Score: {score:.0f}/100 <- {status}")

        results.append({
            "layer": layer,
            "feature_id": feature_id,
            "selectivity": selectivity,
            "positive_ratio": metrics["positive_ratio"],
            "mean_positive_effect": metrics["mean_positive_effect"],
            "mean_negative_effect": metrics["mean_negative_effect"],
            "intervention_activation_pct": metrics["intervention_activation_pct"],
            "intervention_logit_change": metrics["intervention_logit_change"],
            "top3_layers": str(metrics["top3_layers"]),
            "score": score,
        })

    # Clear SAE cache
    sae_cache.clear()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("score", ascending=False)
    csv_path = os.path.join(output_dir, "anger_feature_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Recommendation
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    best = results_df.iloc[0]
    print(f"\n  Best Anger Feature: L{int(best['layer'])}:F{int(best['feature_id'])} (score={best['score']:.0f})")
    print(f"    Selectivity: {best['selectivity']:.1f}")
    print(f"    Positive ratio: {best['positive_ratio'] * 100:.1f}%")
    print(f"    Intervention: {best['intervention_activation_pct']:+.1f}% activation, "
          f"{best['intervention_logit_change']:+.3f} logit")

    # Compare with current top-1 (by selectivity)
    current = anger_features[0]  # Already sorted by selectivity
    if best["layer"] != current["layer"] or best["feature_id"] != current["feature_id"]:
        print(f"\n  NOTE: This is different from the current top-1 by selectivity!")
        print(f"  Current top-1: L{current['layer']}:F{current['feature_id']} (selectivity={current['selectivity']:.1f})")
        current_result = results_df[
            (results_df["layer"] == current["layer"]) &
            (results_df["feature_id"] == current["feature_id"])
        ].iloc[0]
        print(f"    Score: {current_result['score']:.0f}/100")
    else:
        print(f"\n  The top-1 by selectivity is also the best by causal dynamics!")

    print("\n  To use in main circuit.py:")
    print(f"    1. Manually update validated_features.csv to move L{int(best['layer'])}:F{int(best['feature_id'])} to top")
    print(f"    2. Or modify circuit.py load_validated_features() to use this feature for Anger")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
