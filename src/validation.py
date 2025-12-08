"""
Validation for Emotion Features

Two validation checks:
1. Confound Check: true emotion vs confound (similar sentiment, not target emotion)
2. Cross-Emotion Specificity: target emotion vs other 3 emotions

Uses same model/SAE as discovery.py:
- Model: Gemma 2 2B
- SAE: Gemma Scope (gemma-scope-2b-pt-res-canonical)

Usage:
    python validation.py --test --device cpu
    python validation.py --full --device cuda

Output:
    - results/validation_confound.csv
    - results/validation_cross_emotion.csv
    - results/validated_features.csv (features passing both checks)
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


EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Disgust"]
GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate emotion features")
    parser.add_argument("--test", action="store_true", help="Use small test set")
    parser.add_argument("--full", action="store_true", help="Use full validation set")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--width", type=str, default="16k",
                        choices=["16k", "32k", "65k", "131k"],
                        help="Gemma Scope SAE width (default: 16k)")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--confound_threshold", type=float, default=1.5,
                        help="Threshold for confound check (default: 1.5)")
    parser.add_argument("--cross_threshold", type=float, default=1.5,
                        help="Threshold for cross-emotion check (default: 1.5)")
    return parser.parse_args()


def get_validation_prompts(use_full=False):
    """
    Get validation prompts: confounds and true emotions for 4 emotions.

    Confounds = sentences with similar sentiment but NOT the target emotion
    True = sentences clearly expressing the target emotion
    """

    # === JOY CONFOUNDS (positive but NOT joy) ===
    joy_confounds = [
        {"text": "I like this book.", "note": "preference"},
        {"text": "I agree with you.", "note": "agreement"},
        {"text": "This is a good idea.", "note": "evaluation"},
        {"text": "I appreciate your help.", "note": "gratitude"},
        {"text": "The food tastes nice.", "note": "sensory"},
        {"text": "I prefer the blue one.", "note": "preference"},
        {"text": "That makes sense.", "note": "intellectual"},
        {"text": "I'm satisfied with the result.", "note": "satisfaction"},
    ]

    # === TRUE JOY ===
    true_joy = [
        {"text": "I am so happy today!", "note": "happy"},
        {"text": "We are thrilled about the news!", "note": "thrilled"},
        {"text": "She feels joyful and excited.", "note": "joyful"},
        {"text": "I'm delighted to meet you.", "note": "delighted"},
        {"text": "This makes me incredibly happy!", "note": "happy"},
        {"text": "I'm overjoyed by the surprise.", "note": "overjoyed"},
        {"text": "What a wonderful feeling of joy!", "note": "joy"},
        {"text": "I feel so cheerful and bright.", "note": "cheerful"},
    ]

    # === SADNESS CONFOUNDS (negative but NOT sadness) ===
    sad_confounds = [
        {"text": "I'm sorry for being late.", "note": "apology"},
        {"text": "This is frustrating.", "note": "frustration"},
        {"text": "I'm angry about this.", "note": "anger"},
        {"text": "I'm worried about the exam.", "note": "anxiety"},
        {"text": "I don't like this.", "note": "dislike"},
        {"text": "This is disappointing.", "note": "disappointment"},
        {"text": "I'm stressed about work.", "note": "stress"},
        {"text": "This situation is unfortunate.", "note": "unfortunate"},
    ]

    # === TRUE SADNESS ===
    true_sad = [
        {"text": "I feel so sad.", "note": "sad"},
        {"text": "She is heartbroken.", "note": "heartbroken"},
        {"text": "He cried with grief.", "note": "grief"},
        {"text": "I'm devastated by the loss.", "note": "devastated"},
        {"text": "This fills me with sorrow.", "note": "sorrow"},
        {"text": "I feel deeply melancholic.", "note": "melancholic"},
        {"text": "My heart aches with sadness.", "note": "heartache"},
        {"text": "I'm mourning the loss.", "note": "mourning"},
    ]

    # === ANGER CONFOUNDS (negative but NOT anger) ===
    anger_confounds = [
        {"text": "I'm disappointed in you.", "note": "disappointment"},
        {"text": "This is frustrating to deal with.", "note": "frustration"},
        {"text": "I feel sad about this.", "note": "sadness"},
        {"text": "I'm worried this won't work.", "note": "worry"},
        {"text": "This is annoying.", "note": "annoyance"},
        {"text": "I'm uncomfortable with this.", "note": "discomfort"},
        {"text": "I disagree with this decision.", "note": "disagreement"},
        {"text": "This is unfair treatment.", "note": "unfair"},
    ]

    # === TRUE ANGER ===
    true_anger = [
        {"text": "I am furious about this!", "note": "furious"},
        {"text": "This makes me so angry!", "note": "angry"},
        {"text": "I'm enraged by their behavior.", "note": "enraged"},
        {"text": "I feel intense rage inside.", "note": "rage"},
        {"text": "I'm livid with anger.", "note": "livid"},
        {"text": "This infuriates me!", "note": "infuriated"},
        {"text": "I'm seething with anger.", "note": "seething"},
        {"text": "My blood is boiling!", "note": "boiling"},
    ]

    # === FEAR CONFOUNDS (negative but NOT fear) ===
    fear_confounds = [
        {"text": "I'm nervous about the presentation.", "note": "nervous"},
        {"text": "This is stressful.", "note": "stress"},
        {"text": "I'm uncertain about the outcome.", "note": "uncertainty"},
        {"text": "I'm anxious about tomorrow.", "note": "anxiety"},
        {"text": "This situation is concerning.", "note": "concern"},
        {"text": "I'm hesitant to proceed.", "note": "hesitation"},
        {"text": "I'm uneasy about this.", "note": "unease"},
        {"text": "I have doubts about this plan.", "note": "doubt"},
    ]

    # === TRUE FEAR ===
    true_fear = [
        {"text": "I am terrified!", "note": "terrified"},
        {"text": "This scares me so much.", "note": "scared"},
        {"text": "I'm filled with dread.", "note": "dread"},
        {"text": "I feel paralyzed by fear.", "note": "paralyzed"},
        {"text": "I'm horrified by what I saw.", "note": "horrified"},
        {"text": "Pure terror gripped me.", "note": "terror"},
        {"text": "I'm frightened beyond words.", "note": "frightened"},
        {"text": "Fear consumed my entire being.", "note": "consumed"},
    ]

    # === DISGUST CONFOUNDS (negative but NOT disgust) ===
    disgust_confounds = [
        {"text": "I don't like this at all.", "note": "dislike"},
        {"text": "This is disappointing.", "note": "disappointment"},
        {"text": "I'm unhappy with the result.", "note": "unhappy"},
        {"text": "This is frustrating to deal with.", "note": "frustration"},
        {"text": "I'm annoyed by this behavior.", "note": "annoyance"},
        {"text": "This is unacceptable.", "note": "unacceptable"},
        {"text": "I find this offensive.", "note": "offense"},
        {"text": "This is inappropriate.", "note": "inappropriate"},
    ]

    # === TRUE DISGUST ===
    true_disgust = [
        {"text": "That is absolutely disgusting!", "note": "disgusting"},
        {"text": "I feel sick to my stomach.", "note": "sick"},
        {"text": "This is revolting.", "note": "revolting"},
        {"text": "I'm repulsed by this.", "note": "repulsed"},
        {"text": "How nauseating!", "note": "nauseating"},
        {"text": "This fills me with revulsion.", "note": "revulsion"},
        {"text": "I find this utterly repugnant.", "note": "repugnant"},
        {"text": "This makes me want to vomit.", "note": "vomit"},
    ]

    # Build prompt list with metadata
    all_prompts = []

    prompt_sets = [
        (joy_confounds, "confound", "Joy"),
        (true_joy, "true", "Joy"),
        (sad_confounds, "confound", "Sadness"),
        (true_sad, "true", "Sadness"),
        (anger_confounds, "confound", "Anger"),
        (true_anger, "true", "Anger"),
        (fear_confounds, "confound", "Fear"),
        (true_fear, "true", "Fear"),
        (disgust_confounds, "confound", "Disgust"),
        (true_disgust, "true", "Disgust"),
    ]

    for prompts, prompt_type, emotion in prompt_sets:
        subset = prompts if use_full else prompts[:3]
        for p in subset:
            all_prompts.append({
                "text": p["text"],
                "type": prompt_type,
                "emotion": emotion,
                "note": p["note"],
            })

    return all_prompts


def load_model(device):
    """Load Gemma 2 2B model (same as discovery.py)."""
    print(f"Loading Gemma 2 2B on {device}...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    return model


def load_sae(layer, width, device):
    """Load SAE for a specific layer."""
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae, _, _ = SAE.from_pretrained(
        release=GEMMA_SCOPE_RELEASE,
        sae_id=sae_id,
        device=device
    )
    return sae


def load_top_features(results_dir):
    """Load discovered top features from CSV with layer info."""
    feature_list_path = os.path.join(results_dir, "feature_lists.csv")
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError(f"Feature list not found: {feature_list_path}")

    df = pd.read_csv(feature_list_path)

    # Group features by emotion and layer
    features = []
    for _, row in df.iterrows():
        features.append({
            "emotion": row["emotion"],
            "layer": int(row["layer"]),
            "feature_id": int(row["feature_id"]),
        })

    # Get unique layers
    layers = sorted(list(set(f["layer"] for f in features)))

    return {
        "features": features,
        "layers": layers,
        "df": df,
    }


def collect_activations(model, prompts, features, width, device):
    """
    Collect activations for features across multiple layers.
    Loads SAE for each layer as needed.
    """
    results = []

    # Group features by layer
    layer_features = {}
    for f in features:
        layer = f["layer"]
        if layer not in layer_features:
            layer_features[layer] = []
        layer_features[layer].append(f)

    layers = sorted(layer_features.keys())
    print(f"Collecting activations from {len(layers)} layers...")

    # Collect residuals for all layers first
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]

    print(f"Running {len(prompts)} prompts through model...")
    prompt_residuals = {}

    for p in tqdm(prompts, desc="Collecting residuals"):
        with torch.no_grad():
            _, cache = model.run_with_cache(p["text"], names_filter=hook_names)

        prompt_residuals[p["text"]] = {}
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            prompt_residuals[p["text"]][layer] = cache[hook_name][0, -1, :].cpu()

    # Process each layer with its SAE
    for layer in tqdm(layers, desc="Processing layers"):
        sae = load_sae(layer, width, device)
        layer_fids = [f["feature_id"] for f in layer_features[layer]]

        for p in prompts:
            resid = prompt_residuals[p["text"]][layer].to(device)
            feature_acts = sae.encode(resid).detach().cpu().numpy()

            for f in layer_features[layer]:
                fid = f["feature_id"]
                results.append({
                    "text": p["text"],
                    "type": p["type"],
                    "emotion": p["emotion"],
                    "note": p["note"],
                    "layer": layer,
                    "feature_id": fid,
                    "activation": feature_acts[fid],
                })

        # Free SAE memory
        del sae
        if device == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(results)


def compute_confound_metrics(activation_df, feature_list, threshold=1.5):
    """
    Compute discrimination ratio for each feature (confound check).

    discrimination_ratio = mean(true) / mean(confound)
    """
    results = []

    for f in feature_list:
        emotion = f["emotion"]
        layer = f["layer"]
        fid = f["feature_id"]

        fid_df = activation_df[
            (activation_df["feature_id"] == fid) &
            (activation_df["layer"] == layer) &
            (activation_df["emotion"] == emotion)
        ]

        true_acts = fid_df[fid_df["type"] == "true"]["activation"].values
        confound_acts = fid_df[fid_df["type"] == "confound"]["activation"].values

        if len(true_acts) == 0 or len(confound_acts) == 0:
            continue

        mean_true = true_acts.mean()
        mean_confound = confound_acts.mean()

        # Discrimination ratio
        eps = 1e-8
        if mean_confound > eps:
            ratio = mean_true / mean_confound
        else:
            ratio = min(mean_true / eps, 100.0) if mean_true > 0 else 0.0

        ratio_display = min(ratio, 100.0)

        # Status based on threshold
        status = "PASS" if ratio >= threshold else "FAIL"

        results.append({
            "emotion": emotion,
            "layer": layer,
            "feature_id": fid,
            "mean_true": mean_true,
            "mean_confound": mean_confound,
            "discrimination_ratio": ratio_display,
            "status": status,
        })

    return pd.DataFrame(results)


def compute_cross_emotion_metrics(activation_df, feature_list, threshold=1.5):
    """
    Compute cross-emotion specificity for each feature.

    specificity_ratio = mean(target_emotion) / mean(other_emotions)

    A good feature should activate strongly on its target emotion
    and weakly on other emotions.
    """
    results = []

    for f in feature_list:
        target_emotion = f["emotion"]
        layer = f["layer"]
        fid = f["feature_id"]

        # Get activations for this feature across all emotions (true prompts only)
        fid_df = activation_df[
            (activation_df["feature_id"] == fid) &
            (activation_df["layer"] == layer) &
            (activation_df["type"] == "true")
        ]

        # Target emotion activations
        target_acts = fid_df[fid_df["emotion"] == target_emotion]["activation"].values

        # Other emotions activations
        other_emotions = [e for e in EMOTIONS if e != target_emotion]
        other_acts = fid_df[fid_df["emotion"].isin(other_emotions)]["activation"].values

        if len(target_acts) == 0 or len(other_acts) == 0:
            continue

        mean_target = target_acts.mean()
        mean_others = other_acts.mean()

        # Specificity ratio
        eps = 1e-8
        if mean_others > eps:
            ratio = mean_target / mean_others
        else:
            ratio = min(mean_target / eps, 100.0) if mean_target > 0 else 0.0

        ratio_display = min(ratio, 100.0)

        # Status based on threshold
        status = "PASS" if ratio >= threshold else "FAIL"

        # Also record per-emotion activations for detailed analysis
        emotion_acts = {}
        for emotion in EMOTIONS:
            acts = fid_df[fid_df["emotion"] == emotion]["activation"].values
            emotion_acts[f"mean_{emotion.lower()}"] = acts.mean() if len(acts) > 0 else 0.0

        result = {
            "emotion": target_emotion,
            "layer": layer,
            "feature_id": fid,
            "mean_target": mean_target,
            "mean_others": mean_others,
            "specificity_ratio": ratio_display,
            "status": status,
        }
        result.update(emotion_acts)
        results.append(result)

    return pd.DataFrame(results)


def print_confound_summary(confound_df):
    """Print confound validation summary"""
    print("\n" + "=" * 60)
    print("  CONFOUND CHECK SUMMARY")
    print("=" * 60)

    for emotion in EMOTIONS:
        emotion_df = confound_df[confound_df["emotion"] == emotion]
        if len(emotion_df) == 0:
            continue

        n_pass = (emotion_df["status"] == "PASS").sum()
        n_fail = (emotion_df["status"] == "FAIL").sum()
        n_total = len(emotion_df)

        print(f"\n  {emotion}: PASS {n_pass}/{n_total}, FAIL {n_fail}/{n_total}")

        for _, row in emotion_df.iterrows():
            icon = "+" if row["status"] == "PASS" else "-"
            print(f"    {icon} L{int(row['layer'])}:F{int(row['feature_id'])}: "
                  f"ratio={row['discrimination_ratio']:.2f}")

    print("=" * 60)


def print_cross_emotion_summary(cross_df):
    """Print cross-emotion validation summary"""
    print("\n" + "=" * 60)
    print("  CROSS-EMOTION SPECIFICITY SUMMARY")
    print("=" * 60)

    for emotion in EMOTIONS:
        emotion_df = cross_df[cross_df["emotion"] == emotion]
        if len(emotion_df) == 0:
            continue

        n_pass = (emotion_df["status"] == "PASS").sum()
        n_fail = (emotion_df["status"] == "FAIL").sum()
        n_total = len(emotion_df)

        print(f"\n  {emotion}: PASS {n_pass}/{n_total}, FAIL {n_fail}/{n_total}")

        for _, row in emotion_df.iterrows():
            icon = "+" if row["status"] == "PASS" else "-"
            print(f"    {icon} L{int(row['layer'])}:F{int(row['feature_id'])}: "
                  f"ratio={row['specificity_ratio']:.2f}")

    print("=" * 60)


def print_final_summary(validated_df, total_features):
    """Print final validation summary"""
    print("\n" + "=" * 60)
    print("  FINAL VALIDATION RESULT")
    print("=" * 60)

    n_validated = len(validated_df)
    print(f"\n  Features passing both checks: {n_validated}/{total_features}")

    for emotion in EMOTIONS:
        emotion_df = validated_df[validated_df["emotion"] == emotion]
        if len(emotion_df) == 0:
            print(f"\n  {emotion}: 0 validated features")
            continue

        print(f"\n  {emotion}: {len(emotion_df)} validated features")
        for _, row in emotion_df.iterrows():
            print(f"    + L{int(row['layer'])}:F{int(row['feature_id'])}")

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    use_full = args.full
    mode_str = "FULL" if use_full else "TEST"
    print(f"=== {mode_str} MODE ===")
    print(f"Confound threshold: {args.confound_threshold}")
    print(f"Cross-emotion threshold: {args.cross_threshold}")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    output_dir = os.path.join(project_root, args.output_dir)

    # Load model
    model = load_model(args.device)

    # Load discovered features
    print("\nLoading discovered features...")
    feature_dict = load_top_features(output_dir)

    print(f"  Total features: {len(feature_dict['features'])}")
    print(f"  Layers: {feature_dict['layers']}")

    for emotion in EMOTIONS:
        count = len([f for f in feature_dict['features'] if f['emotion'] == emotion])
        print(f"  {emotion}: {count} features")

    # Get validation prompts
    prompts = get_validation_prompts(use_full=use_full)
    print(f"\nValidation prompts: {len(prompts)}")

    # Collect activations
    activation_df = collect_activations(
        model, prompts, feature_dict["features"], args.width, args.device
    )

    # === Confound Check ===
    print("\nComputing confound metrics...")
    confound_df = compute_confound_metrics(
        activation_df, feature_dict["features"], threshold=args.confound_threshold
    )

    # === Cross-Emotion Specificity ===
    print("Computing cross-emotion specificity...")
    cross_df = compute_cross_emotion_metrics(
        activation_df, feature_dict["features"], threshold=args.cross_threshold
    )

    # === Combine results ===
    # Features must pass BOTH checks
    confound_pass = set(
        (row["emotion"], row["layer"], row["feature_id"])
        for _, row in confound_df[confound_df["status"] == "PASS"].iterrows()
    )
    cross_pass = set(
        (row["emotion"], row["layer"], row["feature_id"])
        for _, row in cross_df[cross_df["status"] == "PASS"].iterrows()
    )
    both_pass = confound_pass & cross_pass

    # Create validated features dataframe
    validated_features = []
    for _, row in feature_dict["df"].iterrows():
        key = (row["emotion"], int(row["layer"]), int(row["feature_id"]))
        if key in both_pass:
            validated_features.append(row.to_dict())

    validated_df = pd.DataFrame(validated_features)

    # === Save results ===
    os.makedirs(output_dir, exist_ok=True)

    confound_path = os.path.join(output_dir, "validation_confound.csv")
    confound_df.to_csv(confound_path, index=False)
    print(f"\nSaved: {confound_path}")

    cross_path = os.path.join(output_dir, "validation_cross_emotion.csv")
    cross_df.to_csv(cross_path, index=False)
    print(f"Saved: {cross_path}")

    validated_path = os.path.join(output_dir, "validated_features.csv")
    validated_df.to_csv(validated_path, index=False)
    print(f"Saved: {validated_path}")

    # === Print summaries ===
    print_confound_summary(confound_df)
    print_cross_emotion_summary(cross_df)
    print_final_summary(validated_df, len(feature_dict["features"]))

    print("\nDone! Run viz/plot_validation.py to generate figures.")


if __name__ == "__main__":
    main()
