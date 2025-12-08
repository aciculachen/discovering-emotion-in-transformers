"""
Steering Generation: Generate Text with Emotion Feature Manipulation

Generates text completions with steering at different α values to demonstrate
causal control over emotion expression.

Usage:
    python steering_generation.py --device cuda --n_runs 5

Output:
    results/steering_generations.csv
    Console output for easy browsing
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

# =============================================================================
# CONFIGURATION
# =============================================================================

GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"

# Neutral prompts for steering demonstration
NEUTRAL_PROMPTS = [
    "Today I feel",
    "The weather is",
    "This news is",
    "I think that",
    "Looking at this, I",
    "After hearing that,",
    "The situation seems",
    "My reaction is",
]

# Happy prompts for cross-emotion steering (steer toward Sadness)
HAPPY_PROMPTS = [
    "I'm so excited! This is the best day ever because",
    "I just won the lottery and I feel",
    "My friends threw me a surprise party and I",
    "I got accepted to my dream school! Now I",
    "This is amazing news! I can't believe",
    "I'm over the moon with joy because",
    "Best birthday ever! Everyone showed up and",
    "I finally achieved my goal and I feel",
]

# Sad prompts for intensity ladder (gradually increase Sadness)
SAD_PROMPTS = [
    "I just heard some bad news and I feel",
    "Things haven't been going well lately, and I",
    "I lost someone close to me, and now I",
    "Nobody understands what I'm going through, so I",
    "Everything feels hopeless because",
    "I can't stop thinking about what went wrong, and I",
    "The loneliness is overwhelming, and I just want to",
    "I tried my best but failed, and now I feel",
]

# Intensity ladder alpha values (0 to 10)
INTENSITY_ALPHAS = [0, 2, 4, 6, 8, 10]

# Emotions to test (can be subset)
EMOTIONS = ["Joy", "Sadness", "Anger", "Fear"]

# Alpha values for steering
ALPHA_VALUES = [-4, 0, 4]


def parse_args():
    parser = argparse.ArgumentParser(description="Steering Generation")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--width", type=str, default="16k",
                        choices=["16k", "32k", "65k", "131k"],
                        help="Gemma Scope SAE width")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of generation runs per prompt/emotion")
    parser.add_argument("--max_new_tokens", type=int, default=40,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--emotions", type=str, nargs="+", default=None,
                        help="Specific emotions to test (default: all)")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="Specific prompts to use (default: all neutral)")
    parser.add_argument("--cross_emotion", action="store_true",
                        help="Run cross-emotion steering (happy→sad)")
    parser.add_argument("--intensity_ladder", action="store_true",
                        help="Run intensity ladder (sad prompts, α=0→10)")
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


def load_validated_features(results_dir):
    """Load validated features from CSV."""
    validated_path = os.path.join(results_dir, "validated_features.csv")
    if not os.path.exists(validated_path):
        raise FileNotFoundError(f"Run validation.py first: {validated_path}")
    return pd.read_csv(validated_path)


def get_top_feature(validated_df, emotion):
    """Get top-1 validated feature for an emotion."""
    emo_df = validated_df[validated_df["emotion"] == emotion]
    if len(emo_df) == 0:
        return None
    # Sort by selectivity descending
    top = emo_df.sort_values("selectivity", ascending=False).iloc[0]
    return {
        "layer": int(top["layer"]),
        "feature_id": int(top["feature_id"]),
        "selectivity": top["selectivity"]
    }


# =============================================================================
# Steering Generation
# =============================================================================

def get_steering_hook(steering_dir, alpha):
    """Create a hook that adds steering direction to residual stream."""
    def hook(resid, hook):
        return resid + alpha * steering_dir
    return hook


def generate_with_steering(model, sae, prompt, layer, feature_id, alpha,
                           max_new_tokens, temperature, device):
    """
    Generate text with steering applied.

    Args:
        model: HookedTransformer model
        sae: SAE for the target layer
        prompt: Input prompt string
        layer: Layer to apply steering
        feature_id: Feature index in SAE
        alpha: Steering strength
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        device: Device string

    Returns:
        Generated text (continuation only, not including prompt)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Get normalized steering direction
    steering_dir = sae.W_dec[feature_id].clone()
    steering_dir = steering_dir / steering_dir.norm()

    # Tokenize prompt
    tokens = model.to_tokens(prompt)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits with steering
            if alpha != 0:
                hook = get_steering_hook(steering_dir, alpha)
                with model.hooks(fwd_hooks=[(hook_name, hook)]):
                    logits = model(tokens)[0, -1, :]
            else:
                logits = model(tokens)[0, -1, :]

            # Sample next token
            if temperature == 0:
                next_token = logits.argmax().unsqueeze(0)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated_tokens.append(next_token.item())

            # Check for EOS
            if next_token.item() == model.tokenizer.eos_token_id:
                break

            # Append to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    # Decode generated tokens
    generated_text = model.to_string(torch.tensor(generated_tokens))

    return generated_text


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
    print("  STEERING GENERATION")
    print("=" * 70)
    print(f"  Device: {args.device}")
    print(f"  SAE width: {args.width}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_new_tokens}")
    print(f"  Runs per config: {args.n_runs}")

    # Load model
    model = load_model(args.device)

    # Load validated features
    print("\nLoading validated features...")
    validated_df = load_validated_features(results_dir)

    # Select emotions
    emotions = args.emotions if args.emotions else EMOTIONS
    print(f"Emotions: {emotions}")

    # Select prompts
    prompts = args.prompts if args.prompts else NEUTRAL_PROMPTS
    print(f"Prompts: {len(prompts)} neutral prompts")

    # Get top features for each emotion
    feature_info = {}
    for emotion in emotions:
        feat = get_top_feature(validated_df, emotion)
        if feat:
            feature_info[emotion] = feat
            print(f"  {emotion}: L{feat['layer']} #{feat['feature_id']} "
                  f"(selectivity={feat['selectivity']:.2f})")
        else:
            print(f"  {emotion}: No validated feature found")

    # Load SAEs (grouped by layer to minimize loading)
    layers_needed = set(f["layer"] for f in feature_info.values())
    saes = {}
    for layer in layers_needed:
        print(f"Loading SAE for layer {layer}...")
        saes[layer] = load_sae(layer, args.width, args.device)

    # ==========================================================================
    # Generate
    # ==========================================================================

    print("\n" + "=" * 70)
    print("  GENERATING...")
    print("=" * 70)

    all_results = []

    for emotion in emotions:
        if emotion not in feature_info:
            continue

        feat = feature_info[emotion]
        layer = feat["layer"]
        feature_id = feat["feature_id"]
        sae = saes[layer]

        print(f"\n{'─' * 70}")
        print(f"  {emotion.upper()} (L{layer} #{feature_id})")
        print(f"{'─' * 70}")

        for prompt in prompts:
            print(f"\n  Prompt: \"{prompt}\"")

            for run_idx in range(args.n_runs):
                # Set seed for reproducibility within run
                seed = hash(f"{emotion}_{prompt}_{run_idx}") % (2**32)
                torch.manual_seed(seed)

                run_results = {"prompt": prompt, "emotion": emotion,
                               "layer": layer, "feature_id": feature_id,
                               "run": run_idx + 1, "seed": seed}

                for alpha in ALPHA_VALUES:
                    generation = generate_with_steering(
                        model, sae, prompt, layer, feature_id, alpha,
                        args.max_new_tokens, args.temperature, args.device
                    )

                    # Clean up generation
                    generation = generation.strip()

                    run_results[f"alpha_{alpha}"] = generation

                    # Print for quick viewing
                    alpha_str = f"α={alpha:+d}"
                    gen_preview = generation[:60] + "..." if len(generation) > 60 else generation
                    print(f"    [{run_idx+1}] {alpha_str}: {gen_preview}")

                all_results.append(run_results)

    # ==========================================================================
    # Cross-Emotion Steering: Happy prompts → Sadness feature
    # ==========================================================================

    if args.cross_emotion:
        print("\n" + "=" * 70)
        print("  CROSS-EMOTION STEERING: Happy → Sadness")
        print("=" * 70)

        # Get Sadness feature
        if "Sadness" not in feature_info:
            print("  Sadness feature not found, skipping cross-emotion steering")
        else:
            sad_feat = feature_info["Sadness"]
            sad_layer = sad_feat["layer"]
            sad_feature_id = sad_feat["feature_id"]

            # Make sure SAE is loaded
            if sad_layer not in saes:
                saes[sad_layer] = load_sae(sad_layer, args.width, args.device)
            sad_sae = saes[sad_layer]

            print(f"\n  Using Sadness feature: L{sad_layer} #{sad_feature_id}")
            print(f"  Applying to {len(HAPPY_PROMPTS)} happy prompts")

            cross_results = []

            for prompt in HAPPY_PROMPTS:
                print(f"\n  Prompt: \"{prompt[:50]}...\"")

                for run_idx in range(args.n_runs):
                    seed = hash(f"cross_{prompt}_{run_idx}") % (2**32)
                    torch.manual_seed(seed)

                    run_results = {
                        "prompt": prompt,
                        "experiment": "happy_to_sad",
                        "target_emotion": "Sadness",
                        "layer": sad_layer,
                        "feature_id": sad_feature_id,
                        "run": run_idx + 1,
                        "seed": seed
                    }

                    # Only test α=0 (baseline) and α=+4 (steer toward sadness)
                    for alpha in [0, 4]:
                        generation = generate_with_steering(
                            model, sad_sae, prompt, sad_layer, sad_feature_id, alpha,
                            args.max_new_tokens, args.temperature, args.device
                        )
                        generation = generation.strip()
                        run_results[f"alpha_{alpha}"] = generation

                        alpha_str = f"α={alpha:+d}"
                        gen_preview = generation[:60] + "..." if len(generation) > 60 else generation
                        print(f"    [{run_idx+1}] {alpha_str}: {gen_preview}")

                    cross_results.append(run_results)

            # Save cross-emotion results separately
            cross_df = pd.DataFrame(cross_results)
            cross_path = os.path.join(results_dir, "steering_cross_generation.csv")
            cross_df.to_csv(cross_path, index=False)
            print(f"\nSaved cross-emotion results: {cross_path}")

    # ==========================================================================
    # Intensity Ladder: Sad prompts with α = 0 → 10
    # ==========================================================================

    if args.intensity_ladder:
        print("\n" + "=" * 70)
        print("  INTENSITY LADDER: Sadness α=0 → 10")
        print("=" * 70)

        # Get Sadness feature
        if "Sadness" not in feature_info:
            print("  Sadness feature not found, skipping intensity ladder")
        else:
            sad_feat = feature_info["Sadness"]
            sad_layer = sad_feat["layer"]
            sad_feature_id = sad_feat["feature_id"]

            # Make sure SAE is loaded
            if sad_layer not in saes:
                saes[sad_layer] = load_sae(sad_layer, args.width, args.device)
            sad_sae = saes[sad_layer]

            print(f"\n  Using Sadness feature: L{sad_layer} #{sad_feature_id}")
            print(f"  Alpha values: {INTENSITY_ALPHAS}")
            print(f"  Applying to {len(SAD_PROMPTS)} sad prompts")

            ladder_results = []

            for prompt in SAD_PROMPTS:
                print(f"\n  Prompt: \"{prompt[:50]}...\"")

                for run_idx in range(args.n_runs):
                    seed = hash(f"ladder_{prompt}_{run_idx}") % (2**32)
                    torch.manual_seed(seed)

                    run_results = {
                        "prompt": prompt,
                        "experiment": "intensity_ladder",
                        "target_emotion": "Sadness",
                        "layer": sad_layer,
                        "feature_id": sad_feature_id,
                        "run": run_idx + 1,
                        "seed": seed
                    }

                    print(f"    [Run {run_idx+1}]")
                    for alpha in INTENSITY_ALPHAS:
                        generation = generate_with_steering(
                            model, sad_sae, prompt, sad_layer, sad_feature_id, alpha,
                            args.max_new_tokens, args.temperature, args.device
                        )
                        generation = generation.strip()
                        run_results[f"alpha_{alpha}"] = generation

                        gen_preview = generation[:50] + "..." if len(generation) > 50 else generation
                        print(f"      α={alpha:2d}: {gen_preview}")

                    ladder_results.append(run_results)

            # Save intensity ladder results
            ladder_df = pd.DataFrame(ladder_results)
            ladder_path = os.path.join(results_dir, "steering_intensity_ladder.csv")
            ladder_df.to_csv(ladder_path, index=False)
            print(f"\nSaved intensity ladder results: {ladder_path}")

    # ==========================================================================
    # Save Results
    # ==========================================================================

    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)

    output_path = os.path.join(results_dir, "steering_generations.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # ==========================================================================
    # Print Summary for Easy Browsing
    # ==========================================================================

    print("\n" + "=" * 70)
    print("  GENERATION EXAMPLES (for hand-picking)")
    print("=" * 70)

    for emotion in emotions:
        if emotion not in feature_info:
            continue

        feat = feature_info[emotion]
        emo_results = results_df[results_df["emotion"] == emotion]

        print(f"\n{'━' * 70}")
        print(f"  {emotion.upper()} - Feature L{feat['layer']} #{feat['feature_id']}")
        print(f"{'━' * 70}")

        for prompt in prompts:
            prompt_results = emo_results[emo_results["prompt"] == prompt]

            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  {'─' * 60}")

            for _, row in prompt_results.iterrows():
                print(f"\n  [Run {row['run']}]")
                for alpha in ALPHA_VALUES:
                    gen = row[f"alpha_{alpha}"]
                    print(f"    α={alpha:+d}: \"{gen}\"")

    print("\n" + "=" * 70)
    print("  Done! Pick your favorite examples from above.")
    print(f"  Full results: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
