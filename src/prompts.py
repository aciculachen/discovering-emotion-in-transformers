"""
GoEmotions Dataset Loader for Emotion Feature Discovery

Loads Google's GoEmotions dataset from HuggingFace for emotion analysis.
This is a large-scale dataset of 58k Reddit comments with fine-grained emotion labels.

Source: https://huggingface.co/datasets/google-research-datasets/go_emotions
Publication: Demszky et al. (2020), ACL 2020

Supported Emotions: joy, sadness, anger, fear (and 23 others)
"""

import random
from typing import List, Dict, Optional

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")


# GoEmotions label mapping (simplified config)
GOEMOTIONS_LABELS = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

# Reverse mapping: name -> id
EMOTION_TO_ID = {v: k for k, v in GOEMOTIONS_LABELS.items()}

# Default emotions for our analysis
DEFAULT_EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust"]


def load_goemotions_dataset(
    emotions: List[str] = DEFAULT_EMOTIONS,
    n_samples: Optional[int] = None,
    random_seed: int = 42,
    split: str = "train"
) -> List[Dict]:
    """
    Load GoEmotions dataset and filter for specified emotions.

    Args:
        emotions: List of emotion labels to include (lowercase).
        n_samples: Number of samples per emotion. If None, use all.
        random_seed: Random seed for sampling reproducibility.
        split: Dataset split to use ("train", "validation", "test").

    Returns:
        List of dicts with keys: {text, emotion}
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "Please install the datasets library: pip install datasets"
        )

    print(f"Loading GoEmotions dataset ({split} split)...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)

    # Get emotion IDs for target emotions
    target_ids = {EMOTION_TO_ID[e.lower()]: e.capitalize() for e in emotions}

    prompts = {e: [] for e in emotions}

    for item in dataset:
        labels = item["labels"]
        text = item["text"].strip()

        if not text:
            continue

        # Strict filtering: sample must have EXACTLY ONE label total
        # AND that label must be one of our target emotions
        # This ensures no contamination from other emotions
        if len(labels) == 1 and labels[0] in target_ids:
            emotion = target_ids[labels[0]]
            prompts[emotion.lower()].append({
                "text": text,
                "emotion": emotion,
            })

    # Sample if needed
    result = []
    random.seed(random_seed)

    for emotion in emotions:
        emotion_prompts = prompts[emotion.lower()]

        if n_samples is not None and len(emotion_prompts) > n_samples:
            emotion_prompts = random.sample(emotion_prompts, n_samples)

        result.extend(emotion_prompts)

    print(f"Loaded {len(result)} prompts:")
    for emotion in emotions:
        count = sum(1 for p in result if p["emotion"].lower() == emotion.lower())
        print(f"  {emotion.capitalize()}: {count}")

    return result


def get_prompts_with_labels(
    use_full: bool = False,
    n_samples: int = 100,
    emotions: List[str] = DEFAULT_EMOTIONS,
    split: str = "train"
) -> List[Dict]:
    """
    Get prompts for emotion analysis.

    Args:
        use_full: If True, use all available samples; else use n_samples per emotion.
        n_samples: Number of samples per emotion when use_full=False.
        emotions: List of emotions to include.
        split: Dataset split ("train", "validation", "test").

    Returns:
        List of dicts with keys: {text, emotion}
    """
    if use_full:
        return load_goemotions_dataset(emotions=emotions, n_samples=None, split=split)
    else:
        return load_goemotions_dataset(emotions=emotions, n_samples=n_samples, split=split)


def get_dataset_stats(emotions: List[str] = DEFAULT_EMOTIONS) -> Dict:
    """Get statistics about the GoEmotions dataset for target emotions."""
    if not HF_AVAILABLE:
        return {"error": "datasets library not installed"}

    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")

    target_ids = {EMOTION_TO_ID[e.lower()]: e.capitalize() for e in emotions}

    counts = {e: 0 for e in emotions}
    pure_single_label_counts = {e: 0 for e in emotions}

    for item in dataset:
        labels = item["labels"]
        matching = [target_ids[lid] for lid in labels if lid in target_ids]

        for m in matching:
            counts[m.lower()] += 1

        # Pure single-label: exactly one label total AND it's a target emotion
        if len(labels) == 1 and labels[0] in target_ids:
            pure_single_label_counts[target_ids[labels[0]].lower()] += 1

    return {
        "total_samples": len(dataset),
        "all_matches": counts,
        "pure_single_label": pure_single_label_counts,
    }


if __name__ == "__main__":
    print("=== GoEmotions Dataset Statistics ===\n")

    try:
        stats = get_dataset_stats()
        print(f"Total samples in train split: {stats['total_samples']}")

        print("\nSamples per emotion (including multi-label):")
        for emotion, count in stats['all_matches'].items():
            print(f"  {emotion.capitalize()}: {count}")

        print("\nPure single-label samples (used for training):")
        for emotion, count in stats['pure_single_label'].items():
            print(f"  {emotion.capitalize()}: {count}")

        print("\n=== Sample Prompts (3 per emotion) ===")
        prompts = load_goemotions_dataset(n_samples=3)
        for p in prompts:
            print(f"[{p['emotion']}] {p['text'][:70]}...")

    except Exception as e:
        print(f"Error: {e}")
