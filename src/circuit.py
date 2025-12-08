"""
Emotion Circuit Analysis with Causal Flow

Two-step approach (following "On the Biology of a Large Language Model"):
1. Attribution Graphs: Use activation patching to trace how features in earlier
   layers causally influence the target emotion feature
2. Intervention Experiments: Validate hypotheses by ablating components and
   observing whether behavior changes as predicted

Method: Activation Patching
- Clean run: Normal forward pass
- Ablated run: Zero out specific component (Attn/MLP output)
- Causal effect = Clean activation - Ablated activation

Consistent with pipeline:
- Model: gemma-2-2b
- SAE: gemma-scope-2b-pt-res-canonical
- Dataset: GoEmotions

Usage:
    python circuit.py --device cuda
    python circuit.py --device cpu --test

Output:
    - results/circuit_upstream.csv (causal attribution)
    - results/circuit_downstream.csv (logit effects)
    - results/circuit_intervention.csv (validation experiments)

For visualization, run: python viz/plot_circuit.py
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

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Disgust"]

# Selected example prompts for circuit visualization (no emotion keywords)
# These prompts express the emotion without using explicit emotion words
EXAMPLE_PROMPTS = {
    "Joy": "Haha, well it sounds like its done too good of a job!",
    "Sadness": "Same here. It's very hard. I really feel for you.",
    "Anger": "Dude, wtf is wrong with you?",
    "Fear": "That's nightmare fuel right there",
    "Disgust": "This is repulsive what the fuck is wrong with places like this.",
}

# =============================================================================
# FEATURE OVERRIDES
# =============================================================================
# Override the default feature selection (first by selectivity) when a different
# feature has been determined to have better causal dynamics via anger_circuit.py
# or manual analysis. Set to None to use validated_features.csv default.
#
# Format: {"layer": int, "feature_id": int} or None
FEATURE_OVERRIDE = {
    # Joy: L25:F13068 - second highest selectivity but better validation metrics
    # than L24:F1306. Promotes emoticons (:), :], ^_^) and positive expressions.
    "Joy": {"layer": 25, "feature_id": 13068},

    # Sadness: Use default from validated_features.csv (L25:F66)
    "Sadness": None,

    # Anger: L23:F11903 - scored 100/100 in causal dynamics analysis (anger_circuit.py)
    # L25:F8662 has higher selectivity (37.70) but promotes ALL-CAPS words, not anger semantics.
    # L23:F11903 promotes "Stop", "why", "wtf", "Seriously" - better semantic fit.
    "Anger": {"layer": 23, "feature_id": 11903},

    # Fear: Use default from validated_features.csv (L25:F3410)
    "Fear": None,

    # Disgust: L24:F10476 - better validation metrics than L25:F11011
    "Disgust": {"layer": 24, "feature_id": 10476},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Circuit Analysis (Causal Flow)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=str, default="16k",
                        choices=["16k", "32k", "65k", "131k"])
    parser.add_argument("--test", action="store_true",
                        help="Test mode with fewer prompts")
    parser.add_argument("--n_prompts", type=int, default=5,
                        help="Number of prompts per emotion")
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    parser.add_argument("--output_dir", type=str, default=default_output)
    return parser.parse_args()


# =============================================================================
# Model and SAE Loading
# =============================================================================

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


def load_validated_features(results_dir):
    """Load validated features, select top 1 per emotion.

    Priority order:
    1. FEATURE_OVERRIDE (if specified for the emotion)
    2. First row for each emotion in validated_features.csv

    This ensures reproducible feature selection without manual CSV editing.
    """
    csv_path = os.path.join(results_dir, "validated_features.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"validated_features.csv not found at {csv_path}. "
            "Run validation.py first."
        )

    df = pd.read_csv(csv_path)

    features = {}
    for emotion in EMOTIONS:
        # Check for override first
        if FEATURE_OVERRIDE.get(emotion) is not None:
            override = FEATURE_OVERRIDE[emotion]
            # Find selectivity from CSV for the overridden feature
            match = df[(df["emotion"] == emotion) &
                       (df["layer"] == override["layer"]) &
                       (df["feature_id"] == override["feature_id"])]
            if len(match) > 0:
                selectivity = float(match.iloc[0]["selectivity"])
            else:
                # Feature not in CSV - shouldn't happen with valid overrides
                selectivity = 0.0
                print(f"  Warning: Override L{override['layer']}:F{override['feature_id']} "
                      f"for {emotion} not in validated_features.csv")

            features[emotion] = {
                "layer": override["layer"],
                "feature_id": override["feature_id"],
                "selectivity": selectivity,
            }
            continue

        # Fall back to CSV (first row for this emotion)
        emotion_df = df[df["emotion"] == emotion]
        if len(emotion_df) == 0:
            print(f"  Warning: No validated features for {emotion}")
            continue

        top = emotion_df.iloc[0]
        features[emotion] = {
            "layer": int(top["layer"]),
            "feature_id": int(top["feature_id"]),
            "selectivity": float(top["selectivity"]),
        }

    return features


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_pct_change(new_val, baseline_val, cap=1000.0):
    """
    Compute percentage change with safety bounds.
    Returns value capped to [-cap, +cap] to avoid division by near-zero issues.
    """
    if abs(baseline_val) < 1e-6:
        # Baseline is essentially zero - use absolute difference instead
        if abs(new_val) < 1e-6:
            return 0.0
        return cap if new_val > 0 else -cap

    pct = (new_val - baseline_val) / abs(baseline_val) * 100
    return max(-cap, min(cap, pct))


# =============================================================================
# STEP 1: CAUSAL ATTRIBUTION (Activation Patching)
# =============================================================================

def get_ablation_hook(position="last"):
    """
    Create a hook that zeros out the component output.
    This is the "ablation" intervention.
    """
    def hook_fn(activation, hook):
        if position == "last":
            activation[:, -1, :] = 0.0
        else:
            activation[:, :, :] = 0.0
        return activation
    return hook_fn


def get_head_ablation_hook(head_idx, position="last"):
    """
    Create a hook that zeros out a specific attention head.
    Used for per-head causal analysis.

    Args:
        head_idx: Index of the head to ablate (0 to n_heads-1)
        position: "last" to ablate only last token, "all" for all tokens
    """
    def hook_fn(activation, hook):
        # activation shape: [batch, seq, n_heads, head_dim]
        if position == "last":
            activation[:, -1, head_idx, :] = 0.0
        else:
            activation[:, :, head_idx, :] = 0.0
        return activation
    return hook_fn


def compute_feature_activation(model, sae, prompt, feature_id, feature_layer, device, hooks=None):
    """
    Compute the SAE feature activation for a prompt.
    Optionally apply hooks for intervention.
    """
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


def causal_attribution_upstream(model, sae, feature_info, prompts, device, n_heads=8, head_threshold=0):
    """
    STEP 1: Attribution Patching for Upstream Analysis

    For each upstream component (Attn/MLP at each layer):
    1. Run clean forward pass → get feature activation A_clean
    2. Ablate component (zero out) → get activation A_ablated
    3. Causal effect = A_clean - A_ablated

    Components with high causal effect are causally important for the feature.

    Args:
        n_heads: Number of attention heads per layer (Gemma 2 2B = 8)
        head_threshold: Only do per-head analysis if layer attn effect > threshold
    """
    emotion = feature_info["emotion"]
    feature_layer = feature_info["layer"]
    feature_id = feature_info["feature_id"]

    print(f"\n  Causal Attribution for {emotion} (L{feature_layer}:F{feature_id})...")

    # Only analyze top-10 upstream layers for efficiency
    upstream_layers = list(range(max(0, feature_layer - 10), feature_layer))
    all_results = []

    for prompt_data in tqdm(prompts, desc=f"  {emotion} prompts"):
        prompt = prompt_data["text"]

        # 1. Clean run (no intervention)
        clean_activation = compute_feature_activation(
            model, sae, prompt, feature_id, feature_layer, device, hooks=None
        )

        # 2. For each upstream component, measure causal effect
        for up_layer in upstream_layers:
            # === MLP (layer-level) ===
            mlp_hook_name = f"blocks.{up_layer}.hook_mlp_out"
            mlp_hooks = [(mlp_hook_name, get_ablation_hook("last"))]

            ablated_mlp_activation = compute_feature_activation(
                model, sae, prompt, feature_id, feature_layer, device, hooks=mlp_hooks
            )
            mlp_causal_effect = clean_activation - ablated_mlp_activation

            all_results.append({
                "emotion": emotion,
                "feature_layer": feature_layer,
                "feature_id": feature_id,
                "upstream_layer": up_layer,
                "component_type": "MLP",
                "component_name": f"L{up_layer}.MLP",
                "head_idx": -1,
                "clean_activation": clean_activation,
                "ablated_activation": ablated_mlp_activation,
                "causal_effect": mlp_causal_effect,
                "prompt": prompt,
            })

            # === Attention (layer-level) ===
            # Fixed: use hook_attn_out instead of attn.hook_result
            attn_hook_name = f"blocks.{up_layer}.hook_attn_out"
            attn_hooks = [(attn_hook_name, get_ablation_hook("last"))]

            ablated_attn_activation = compute_feature_activation(
                model, sae, prompt, feature_id, feature_layer, device, hooks=attn_hooks
            )
            attn_causal_effect = clean_activation - ablated_attn_activation

            all_results.append({
                "emotion": emotion,
                "feature_layer": feature_layer,
                "feature_id": feature_id,
                "upstream_layer": up_layer,
                "component_type": "Attn",
                "component_name": f"L{up_layer}.Attn",
                "head_idx": -1,
                "clean_activation": clean_activation,
                "ablated_activation": ablated_attn_activation,
                "causal_effect": attn_causal_effect,
                "prompt": prompt,
            })

            # === Per-Head Analysis (only if layer attn effect is significant) ===
            if abs(attn_causal_effect) > head_threshold:
                head_hook_name = f"blocks.{up_layer}.attn.hook_z"

                for head_idx in range(n_heads):
                    head_hooks = [(head_hook_name, get_head_ablation_hook(head_idx, "last"))]
                    ablated_head_activation = compute_feature_activation(
                        model, sae, prompt, feature_id, feature_layer, device, hooks=head_hooks
                    )
                    head_causal_effect = clean_activation - ablated_head_activation

                    all_results.append({
                        "emotion": emotion,
                        "feature_layer": feature_layer,
                        "feature_id": feature_id,
                        "upstream_layer": up_layer,
                        "component_type": "Head",
                        "component_name": f"L{up_layer}.H{head_idx}",
                        "head_idx": head_idx,
                        "clean_activation": clean_activation,
                        "ablated_activation": ablated_head_activation,
                        "causal_effect": head_causal_effect,
                        "prompt": prompt,
                    })

    return all_results


# =============================================================================
# STEP 2: INTERVENTION EXPERIMENTS (Validation)
# =============================================================================

def intervention_experiment(model, sae, feature_info, prompts, top_components, device):
    """
    STEP 2: Validate causal hypotheses through intervention

    Hypothesis: If identified components are truly causal, ablating them should:
    1. Reduce feature activation
    2. Change output logits for emotion words

    We test:
    - Single component ablation
    - Combined ablation of top-k components
    """
    emotion = feature_info["emotion"]
    feature_layer = feature_info["layer"]
    feature_id = feature_info["feature_id"]

    print(f"\n  Intervention Experiments for {emotion}...")

    # Emotion word tokens for logit measurement
    emotion_words = {
        "Joy": ["happy", "joy", "excited", "glad"],
        "Sadness": ["sad", "unhappy", "depressed", "miserable"],
        "Anger": ["angry", "mad", "furious", "annoyed"],
        "Fear": ["afraid", "scared", "fearful", "terrified"],
        "Disgust": ["disgusting", "revolting", "repulsive", "nauseating"],
    }

    target_words = emotion_words.get(emotion, ["emotion"])

    results = []

    for prompt_data in tqdm(prompts, desc=f"  {emotion} intervention"):
        prompt = prompt_data["text"]

        # Baseline (no intervention)
        baseline_activation = compute_feature_activation(
            model, sae, prompt, feature_id, feature_layer, device, hooks=None
        )

        # Baseline logits
        with torch.no_grad():
            baseline_logits = model(prompt)[0, -1, :]

        baseline_emotion_logit = 0.0
        for word in target_words:
            token_ids = model.to_tokens(word, prepend_bos=False)[0]
            if len(token_ids) > 0:
                baseline_emotion_logit += baseline_logits[token_ids[0]].item()
        baseline_emotion_logit /= len(target_words)

        # Test each top component individually
        for comp in top_components[:5]:  # Top 5
            comp_name = comp["name"]
            up_layer = comp["layer"]
            comp_type = comp["type"]

            # Create ablation hook based on component type
            if comp_type == "Attn":
                hook_name = f"blocks.{up_layer}.hook_attn_out"
                hooks = [(hook_name, get_ablation_hook("last"))]
            elif comp_type == "Head":
                hook_name = f"blocks.{up_layer}.attn.hook_z"
                head_idx = comp.get("head_idx", 0)
                hooks = [(hook_name, get_head_ablation_hook(head_idx, "last"))]
            else:  # MLP
                hook_name = f"blocks.{up_layer}.hook_mlp_out"
                hooks = [(hook_name, get_ablation_hook("last"))]

            # Ablated activation
            ablated_activation = compute_feature_activation(
                model, sae, prompt, feature_id, feature_layer, device, hooks=hooks
            )

            # Ablated logits
            with torch.no_grad():
                with model.hooks(fwd_hooks=hooks):
                    ablated_logits = model(prompt)[0, -1, :]

            ablated_emotion_logit = 0.0
            for word in target_words:
                token_ids = model.to_tokens(word, prepend_bos=False)[0]
                if len(token_ids) > 0:
                    ablated_emotion_logit += ablated_logits[token_ids[0]].item()
            ablated_emotion_logit /= len(target_words)

            results.append({
                "emotion": emotion,
                "feature_layer": feature_layer,
                "feature_id": feature_id,
                "intervention_type": "single_ablation",
                "ablated_component": comp_name,
                "baseline_activation": baseline_activation,
                "ablated_activation": ablated_activation,
                "activation_change": ablated_activation - baseline_activation,
                "activation_change_pct": safe_pct_change(ablated_activation, baseline_activation),
                "baseline_emotion_logit": baseline_emotion_logit,
                "ablated_emotion_logit": ablated_emotion_logit,
                "logit_change": ablated_emotion_logit - baseline_emotion_logit,
                "prompt": prompt,
            })

        # Combined ablation of top-3 components
        combined_hooks = []
        top3_names = []
        for comp in top_components[:3]:
            up_layer = comp["layer"]
            comp_type = comp["type"]
            if comp_type == "Attn":
                hook_name = f"blocks.{up_layer}.hook_attn_out"
                combined_hooks.append((hook_name, get_ablation_hook("last")))
            elif comp_type == "Head":
                hook_name = f"blocks.{up_layer}.attn.hook_z"
                head_idx = comp.get("head_idx", 0)
                combined_hooks.append((hook_name, get_head_ablation_hook(head_idx, "last")))
            else:  # MLP
                hook_name = f"blocks.{up_layer}.hook_mlp_out"
                combined_hooks.append((hook_name, get_ablation_hook("last")))
            top3_names.append(comp["name"])

        if combined_hooks:
            combined_activation = compute_feature_activation(
                model, sae, prompt, feature_id, feature_layer, device, hooks=combined_hooks
            )

            with torch.no_grad():
                with model.hooks(fwd_hooks=combined_hooks):
                    combined_logits = model(prompt)[0, -1, :]

            combined_emotion_logit = 0.0
            for word in target_words:
                token_ids = model.to_tokens(word, prepend_bos=False)[0]
                if len(token_ids) > 0:
                    combined_emotion_logit += combined_logits[token_ids[0]].item()
            combined_emotion_logit /= len(target_words)

            results.append({
                "emotion": emotion,
                "feature_layer": feature_layer,
                "feature_id": feature_id,
                "intervention_type": "combined_ablation",
                "ablated_component": "+".join(top3_names),
                "baseline_activation": baseline_activation,
                "ablated_activation": combined_activation,
                "activation_change": combined_activation - baseline_activation,
                "activation_change_pct": safe_pct_change(combined_activation, baseline_activation),
                "baseline_emotion_logit": baseline_emotion_logit,
                "ablated_emotion_logit": combined_emotion_logit,
                "logit_change": combined_emotion_logit - baseline_emotion_logit,
                "prompt": prompt,
            })

    return results


# =============================================================================
# DOWNSTREAM ANALYSIS
# =============================================================================

def _simple_stem(word):
    """Simple English stemmer - reduces words to approximate root form.

    Examples: sadness->sad, saddened->sad, happier->happy, excitedly->excit
    """
    word = word.lower().strip()

    # Common suffixes to remove (order matters - longer first)
    suffixes = [
        'ingly', 'edly', 'ness', 'ment', 'tion', 'sion',
        'ally', 'ful', 'less', 'able', 'ible', 'ious', 'eous',
        'ing', 'ely', 'est', 'ity', 'ery', 'ary',
        'ed', 'er', 'ly', 'en', 'es', 's'
    ]

    for suffix in suffixes:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            stem = word[:-len(suffix)]
            # Handle doubling: sadder -> sad, bigger -> big
            if len(stem) >= 3 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem

    return word


def _is_valid_token(token):
    """Filter out invalid tokens (BOM, subword fragments, etc.).

    Allows:
    - Text emoticons: :), ;), ^_^, =), :-), etc.
    - Emoji: 😊, ❤️, etc.
    - Regular words with alphanumeric characters
    """
    import re
    if not token or len(token.strip()) == 0:
        return False
    # Filter BOM and zero-width characters
    if '\ufeff' in token or '\u200b' in token:
        return False
    # Filter special tokenizer symbols
    if token.strip() in ['▁', '##', '<unk>', '<pad>', '<s>', '</s>', '<bos>', '<eos>']:
        return False
    # Clean the token for validation
    cleaned = token.strip().lstrip('▁')
    if len(cleaned) == 0:
        return False

    # Allow text emoticons (common patterns)
    text_emoticons = [
        r'^[:;=][)\-\']?[)D\]>pP]$',      # :) ;) =) :D :] :> :P
        r'^[)D\]>pP]?[)\-\']?[:;=]$',      # (: D:
        r'^\^[_\-]?\^$',                    # ^^ ^_^ ^-^
        r'^[<>]?3$',                        # <3 >3
        r'^x[Dd]$',                         # xD xd
        r'^[Oo][._][Oo]$',                  # O.O o_o
        r'^[:;][)\'\-]?[(\[<]$',            # :( ;( :[
        r'^[)\'\-]?[:;]$',                  # ):
    ]
    for pattern in text_emoticons:
        if re.match(pattern, cleaned, re.IGNORECASE):
            return True

    # Allow emoji (Unicode emoji ranges)
    # Check if token contains emoji
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map symbols
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002702-\U000027B0"   # dingbats
        "\U0001F900-\U0001F9FF"   # supplemental symbols
        "\U00002600-\U000026FF"   # misc symbols
        "\U00002300-\U000023FF"   # misc technical
        "\U0000FE00-\U0000FE0F"   # variation selectors
        "\U0001FA00-\U0001FA6F"   # chess symbols
        "\U0001FA70-\U0001FAFF"   # symbols extended
        "]+", re.UNICODE
    )
    if emoji_pattern.search(cleaned):
        return True

    # Require at least one alphanumeric character for regular words
    if not re.search(r'[a-zA-Z0-9]', cleaned):
        return False

    # Block HTML/XML tags
    if re.search(r'<[^>]+>', cleaned) or cleaned.startswith('<') or cleaned.endswith('>'):
        return False

    # Block code patterns (parentheses, brackets, semicolons combinations)
    if re.search(r'[();\[\]{}]+$', cleaned) or re.search(r'^[();\[\]{}]+', cleaned):
        return False
    if cleaned in ['();', ']);', '));', '});', '};', '];', ');']:
        return False

    # Block LaTeX commands
    if cleaned.startswith('\\') or cleaned in ['enumi', 'enumii', 'itemize', 'textbf', 'textit']:
        return False

    # Block technical/programming terms
    tech_terms = ['Unavailable', 'undefined', 'null', 'NULL', 'NaN', 'void', 'PROMOTED',
                  'onclick', 'href', 'src', 'div', 'span', 'blockquote', 'tbody', 'thead']
    if cleaned in tech_terms:
        return False

    # Require mostly ASCII letters (filter non-English words)
    ascii_letters = sum(1 for c in cleaned if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in cleaned if c.isalpha())
    if total_letters > 0 and ascii_letters / total_letters < 0.8:
        return False

    # Block very short tokens (likely fragments) unless common words
    if len(cleaned) <= 2 and cleaned.lower() not in ['i', 'a', 'an', 'am', 'is', 'be', 'do', 'go', 'no', 'so', 'up', 'we', 'us', 'me', 'my', 'ok', 'hi']:
        return False

    return True


def analyze_downstream(model, sae, feature_id, emotion_name, top_k=50):
    """Downstream: Which tokens does this feature promote/suppress?

    Includes deduplication and filtering to remove:
    - Duplicate tokens (stem-based: sad/sadness/saddened -> one entry)
    - Invalid tokens (BOM, emoji, subword fragments)
    """
    print(f"\n  Analyzing downstream for {emotion_name} (F{feature_id})...")

    feature_vec = sae.W_dec[feature_id]
    logits = feature_vec @ model.W_U

    results = []

    # Promoted tokens (fetch more to compensate for filtering)
    top_vals, top_ids = torch.topk(logits, top_k * 5)  # Fetch more for stem dedup
    seen_stems = set()
    promoted_count = 0
    for i in range(len(top_ids)):
        if promoted_count >= top_k:
            break
        token = model.to_string(top_ids[i])
        if not _is_valid_token(token):
            continue  # Skip invalid tokens
        token_clean = token.strip().lstrip('\u2581')
        stem = _simple_stem(token_clean)
        if stem in seen_stems:
            continue  # Skip morphological variants
        seen_stems.add(stem)
        promoted_count += 1
        results.append({
            "emotion": emotion_name,
            "feature_id": feature_id,
            "rank": promoted_count,
            "token": token,
            "logit": top_vals[i].item(),
            "direction": "promoted"
        })

    # Suppressed tokens (same logic)
    bot_vals, bot_ids = torch.topk(logits, top_k * 5, largest=False)
    seen_stems = set()
    suppressed_count = 0
    for i in range(len(bot_ids)):
        if suppressed_count >= top_k:
            break
        token = model.to_string(bot_ids[i])
        if not _is_valid_token(token):
            continue  # Skip invalid tokens
        token_clean = token.strip().lstrip('\u2581')
        stem = _simple_stem(token_clean)
        if stem in seen_stems:
            continue  # Skip morphological variants
        seen_stems.add(stem)
        suppressed_count += 1
        results.append({
            "emotion": emotion_name,
            "feature_id": feature_id,
            "rank": suppressed_count,
            "token": token,
            "logit": bot_vals[i].item(),
            "direction": "suppressed"
        })

    return results


# =============================================================================
# TOKEN-LEVEL EVIDENCE ANALYSIS
# =============================================================================

def analyze_token_drivers(model, sae, feature_id, feature_layer, prompt, device):
    """
    Analyze which input tokens drive the target SAE feature.

    For each token position, compute the SAE feature activation at that position.
    High activation = this token position contributes to the feature.

    Returns:
        List of dicts with {position, token, activation} for each position
    """
    hook_name = f"blocks.{feature_layer}.hook_resid_post"

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(prompt)

        # Get residual stream at feature layer: [batch, seq, d_model]
        resid = cache[hook_name][0]  # [seq, d_model]

        # Encode each position through SAE
        # sae.encode expects [batch, d_model] or [d_model]
        activations = []
        for pos in range(resid.shape[0]):
            pos_resid = resid[pos].to(device)
            act = sae.encode(pos_resid)[feature_id].item()
            activations.append(act)

    # Build results with token strings
    results = []
    for pos in range(len(activations)):
        token_id = tokens[0, pos].item()
        token_str = model.to_string(token_id)
        results.append({
            "position": pos,
            "token": token_str,
            "activation": activations[pos],
        })

    return results


def analyze_attention_sources(model, top_component, prompt, device):
    """
    Analyze what tokens a top causal attention component attends to.

    For an attention head, get the attention pattern at the last position
    to see which input tokens it's pulling information from.

    Args:
        model: HookedTransformer
        top_component: dict with {name, layer, type, head_idx}
        prompt: input text

    Returns:
        List of dicts with {source_position, token, attention_weight}
    """
    if top_component["type"] not in ["Attn", "Head"]:
        return []  # Only for attention components

    layer = top_component["layer"]
    head_idx = top_component.get("head_idx", -1)

    # For layer-level Attn, we average across all heads
    # For specific Head, we use that head's pattern

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(prompt)

        # Get attention pattern: [batch, n_heads, seq_q, seq_k]
        pattern_name = f"blocks.{layer}.attn.hook_pattern"
        if pattern_name not in cache:
            return []

        pattern = cache[pattern_name][0]  # [n_heads, seq_q, seq_k]

        # Get pattern for last query position
        if head_idx >= 0:
            # Specific head
            attn_weights = pattern[head_idx, -1, :]  # [seq_k]
        else:
            # Average across all heads
            attn_weights = pattern[:, -1, :].mean(dim=0)  # [seq_k]

    # Build results
    results = []
    for pos in range(attn_weights.shape[0]):
        token_id = tokens[0, pos].item()
        token_str = model.to_string(token_id)
        results.append({
            "source_position": pos,
            "token": token_str,
            "attention_weight": attn_weights[pos].item(),
        })

    return results


def analyze_token_evidence(model, sae, feature_info, prompts, top_components, device, top_k=10):
    """
    Analyze token-level evidence for emotion circuits.

    Only analyzes the EXAMPLE_PROMPT for each emotion (the displayed prompt).
    This ensures the salient tokens shown match the input prompt in the visualization.

    1. Salient Tokens: Which input tokens have high feature activation?
    2. Attention Sources: What tokens do top attention components attend to?

    Args:
        feature_info: dict with {emotion, layer, feature_id}
        prompts: list of prompt dicts (not used - we use EXAMPLE_PROMPTS)
        top_components: list of top causal components (sorted by effect)
        top_k: number of top tokens to report

    Returns:
        dict with 'drivers' and 'attention' DataFrames
    """
    emotion = feature_info["emotion"]
    feature_layer = feature_info["layer"]
    feature_id = feature_info["feature_id"]

    print(f"\n  Token Evidence Analysis for {emotion}...")

    # Use the selected example prompt (not the full prompts list)
    prompt = EXAMPLE_PROMPTS.get(emotion, "")
    if not prompt:
        print(f"    Warning: No example prompt for {emotion}")
        return {"drivers": pd.DataFrame(), "attention": pd.DataFrame()}

    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
    print(f"    Analyzing: \"{prompt_short}\"")

    all_drivers = []
    all_attention = []

    # 1. Analyze token drivers (salient tokens)
    drivers = analyze_token_drivers(model, sae, feature_id, feature_layer, prompt, device)

    # Sort by activation (descending) and take top_k
    drivers.sort(key=lambda x: x["activation"], reverse=True)
    for rank, d in enumerate(drivers[:top_k], 1):
        all_drivers.append({
            "emotion": emotion,
            "feature_layer": feature_layer,
            "feature_id": feature_id,
            "prompt": prompt,
            "rank": rank,
            "position": d["position"],
            "token": d["token"],
            "activation": d["activation"],
        })

    # 2. Analyze attention sources for top attention components
    attn_components = [c for c in top_components if c["type"] in ["Attn", "Head"]][:3]

    for comp in attn_components:
        attn_sources = analyze_attention_sources(model, comp, prompt, device)

        # Sort by attention weight (descending) and take top_k
        attn_sources.sort(key=lambda x: x["attention_weight"], reverse=True)
        for rank, a in enumerate(attn_sources[:top_k], 1):
            all_attention.append({
                "emotion": emotion,
                "feature_layer": feature_layer,
                "feature_id": feature_id,
                "component": comp["name"],
                "prompt": prompt,
                "rank": rank,
                "source_position": a["source_position"],
                "token": a["token"],
                "attention_weight": a["attention_weight"],
            })

    # Print summary
    if all_drivers:
        print(f"    Top salient tokens:")
        for d in all_drivers[:5]:
            print(f"      '{d['token'].strip()}': act={d['activation']:.3f}")

    if all_attention:
        attn_df = pd.DataFrame(all_attention)
        for comp_name in attn_df["component"].unique():
            comp_data = attn_df[attn_df["component"] == comp_name]
            top_tokens = comp_data.nsmallest(3, "rank")
            print(f"    {comp_name} attends to:")
            for _, row in top_tokens.iterrows():
                print(f"      '{row['token'].strip()}': weight={row['attention_weight']:.3f}")

    return {
        "drivers": pd.DataFrame(all_drivers) if all_drivers else pd.DataFrame(),
        "attention": pd.DataFrame(all_attention) if all_attention else pd.DataFrame(),
    }


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
    print("  CAUSAL CIRCUIT ANALYSIS (Activation Patching)")
    print("=" * 70)
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  SAE: {GEMMA_SCOPE_RELEASE}")
    print(f"  Method: Two-step Causal Flow")
    print(f"    1. Attribution: Activation Patching")
    print(f"    2. Validation: Intervention Experiments")
    print(f"  Device: {args.device}")

    # Load validated features
    print("\n  Loading validated features...")
    features = load_validated_features(output_dir)

    print("\n  Features to analyze:")
    for emotion, info in features.items():
        print(f"    {emotion}: Layer {info['layer']}, Feature {info['feature_id']}")

    # Load model
    model = load_model(args.device)

    # Load prompts
    n_prompts = 3 if args.test else args.n_prompts
    emotions_to_load = list(features.keys())  # Only load for emotions we're analyzing
    print(f"\n  Loading {n_prompts} prompts for: {', '.join(emotions_to_load)}...")

    all_prompts = get_prompts_with_labels(
        n_samples=n_prompts,
        emotions=[e.lower() for e in emotions_to_load],
        split="validation"
    )

    prompts_by_emotion = {e: [] for e in emotions_to_load}
    for p in all_prompts:
        emotion = p["emotion"].capitalize()
        if emotion in prompts_by_emotion:
            prompts_by_emotion[emotion].append(p)

    # ====================
    # STEP 1: CAUSAL ATTRIBUTION
    # ====================
    print("\n" + "=" * 70)
    print("  STEP 1: CAUSAL ATTRIBUTION (Activation Patching)")
    print("=" * 70)

    all_upstream = []
    top_components_by_emotion = {}

    for emotion, info in features.items():
        sae = load_sae(info["layer"], args.width, args.device)

        upstream_results = causal_attribution_upstream(
            model, sae,
            {"emotion": emotion, **info},
            prompts_by_emotion.get(emotion, []),
            args.device
        )
        all_upstream.extend(upstream_results)

        # Identify top components (new format with component_type and component_name)
        df = pd.DataFrame(upstream_results)
        agg = df.groupby(["component_name", "component_type", "upstream_layer", "head_idx"]).agg({
            "causal_effect": "mean",
        }).reset_index()

        comps = []
        for _, row in agg.iterrows():
            comps.append({
                "name": row["component_name"],
                "layer": int(row["upstream_layer"]),
                "type": row["component_type"],
                "effect": row["causal_effect"],
                "head_idx": int(row["head_idx"]),
            })
        comps.sort(key=lambda x: abs(x["effect"]), reverse=True)
        top_components_by_emotion[emotion] = comps

        # Print top 3
        print(f"\n  {emotion} Top Causal Components:")
        for c in comps[:3]:
            print(f"    {c['name']}: Δ = {c['effect']:+.3f}")

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    upstream_df = pd.DataFrame(all_upstream)
    upstream_path = os.path.join(output_dir, "circuit_upstream.csv")
    upstream_df.to_csv(upstream_path, index=False)
    print(f"\nSaved: {upstream_path}")

    # ====================
    # STEP 2: INTERVENTION EXPERIMENTS
    # ====================
    print("\n" + "=" * 70)
    print("  STEP 2: INTERVENTION VALIDATION")
    print("=" * 70)

    all_intervention = []

    for emotion, info in features.items():
        sae = load_sae(info["layer"], args.width, args.device)

        intervention_results = intervention_experiment(
            model, sae,
            {"emotion": emotion, **info},
            prompts_by_emotion.get(emotion, []),
            top_components_by_emotion[emotion],
            args.device
        )
        all_intervention.extend(intervention_results)

        # Print validation summary
        df = pd.DataFrame(intervention_results)
        combined = df[df["intervention_type"] == "combined_ablation"]
        if len(combined) > 0:
            avg_act_change = combined["activation_change_pct"].mean()
            avg_logit_change = combined["logit_change"].mean()
            print(f"\n  {emotion} Intervention Validation (Top-3 Ablation):")
            print(f"    Feature Activation: {avg_act_change:+.1f}%")
            print(f"    Emotion Logit: {avg_logit_change:+.3f}")

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    intervention_df = pd.DataFrame(all_intervention)
    intervention_path = os.path.join(output_dir, "circuit_intervention.csv")
    intervention_df.to_csv(intervention_path, index=False)
    print(f"\nSaved: {intervention_path}")

    # ====================
    # DOWNSTREAM ANALYSIS
    # ====================
    print("\n" + "=" * 70)
    print("  DOWNSTREAM ANALYSIS")
    print("=" * 70)

    all_downstream = []

    for emotion, info in features.items():
        sae = load_sae(info["layer"], args.width, args.device)
        downstream_results = analyze_downstream(model, sae, info["feature_id"], emotion)
        all_downstream.extend(downstream_results)

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    downstream_df = pd.DataFrame(all_downstream)
    downstream_path = os.path.join(output_dir, "circuit_downstream.csv")
    downstream_df.to_csv(downstream_path, index=False)
    print(f"Saved: {downstream_path}")

    # ====================
    # TOKEN EVIDENCE ANALYSIS
    # ====================
    print("\n" + "=" * 70)
    print("  TOKEN EVIDENCE ANALYSIS")
    print("=" * 70)

    all_token_drivers = []
    all_attention_sources = []

    for emotion, info in features.items():
        sae = load_sae(info["layer"], args.width, args.device)

        token_evidence = analyze_token_evidence(
            model, sae,
            {"emotion": emotion, **info},
            prompts_by_emotion.get(emotion, []),
            top_components_by_emotion.get(emotion, []),
            args.device
        )

        if not token_evidence["drivers"].empty:
            all_token_drivers.append(token_evidence["drivers"])
        if not token_evidence["attention"].empty:
            all_attention_sources.append(token_evidence["attention"])

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Save token drivers
    if all_token_drivers:
        drivers_df = pd.concat(all_token_drivers, ignore_index=True)
        drivers_path = os.path.join(output_dir, "circuit_token_drivers.csv")
        drivers_df.to_csv(drivers_path, index=False)
        print(f"Saved: {drivers_path}")

    # Save attention sources
    if all_attention_sources:
        attention_df = pd.concat(all_attention_sources, ignore_index=True)
        attention_path = os.path.join(output_dir, "circuit_attention_sources.csv")
        attention_df.to_csv(attention_path, index=False)
        print(f"Saved: {attention_path}")

    # ====================
    # SUMMARY
    # ====================
    print("\n" + "=" * 70)
    print("  CAUSAL CIRCUIT SUMMARY")
    print("=" * 70)

    for emotion in EMOTIONS:
        if emotion not in features:
            continue

        info = features[emotion]
        print(f"\n  {emotion} (L{info['layer']}:F{info['feature_id']}):")

        # Top causal components
        comps = top_components_by_emotion.get(emotion, [])[:3]
        print("    Top Causal Contributors:")
        for c in comps:
            print(f"      {c['name']}: Δ = {c['effect']:+.3f}")

        # Intervention validation
        int_data = intervention_df[
            (intervention_df["emotion"] == emotion) &
            (intervention_df["intervention_type"] == "combined_ablation")
        ]
        if len(int_data) > 0:
            avg_act = int_data["activation_change_pct"].mean()
            avg_logit = int_data["logit_change"].mean()
            print(f"    Intervention Validation:")
            print(f"      Ablating top-3 → Activation: {avg_act:+.1f}%, Logit: {avg_logit:+.3f}")

    print("\n" + "=" * 70)
    print("  Done! Run viz/plot_circuit.py to generate figures.")
    print("=" * 70)


if __name__ == "__main__":
    main()
