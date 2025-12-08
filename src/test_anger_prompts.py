"""
Test multiple Anger prompts on F11903 to find the best salient tokens.
Usage: python test_anger_prompts.py --device cuda
"""

import argparse
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

MODEL_NAME = "gemma-2-2b"
GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"

# F11903 configuration
FEATURE_LAYER = 23
FEATURE_ID = 11903

# Anger prompts to test
TEST_PROMPTS = [
    "How stupid are you, if you dont realize it's staged",
    "How dare you disrespect [NAME] like that!",
    "What's the fucking point of this post?",
    "Seriously, quit flagging this post.",
    "Why is it arrogant to think that is wrong?",
    "I will lose my shit on you if you eat in my car.",
    "What a load of old shite.",
    "Dude wtf that was definitely off Brooklyn",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def analyze_salient_tokens(model, sae, prompt, device):
    """Get salient tokens for F11903."""
    hook_name = f"blocks.{FEATURE_LAYER}.hook_resid_post"

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(prompt)
        resid = cache[hook_name][0]

        results = []
        for pos in range(resid.shape[0]):
            pos_resid = resid[pos].to(device)
            act = sae.encode(pos_resid)[FEATURE_ID].item()
            token_id = tokens[0, pos].item()
            token_str = model.to_string(token_id)
            results.append({
                "position": pos,
                "token": token_str.strip().replace('\u2581', ''),
                "activation": act
            })

    return results


def main():
    args = parse_args()
    device = args.device

    print("=" * 70)
    print(f"  Testing Anger Prompts on L{FEATURE_LAYER}:F{FEATURE_ID}")
    print("=" * 70)

    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

    print(f"Loading SAE...")
    sae_id = f"layer_{FEATURE_LAYER}/width_16k/canonical"
    sae, _, _ = SAE.from_pretrained(
        release=GEMMA_SCOPE_RELEASE,
        sae_id=sae_id,
        device=device
    )

    print("\n" + "=" * 70)
    results_summary = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}] \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")

        salient = analyze_salient_tokens(model, sae, prompt, device)

        # Filter valid tokens (activation > 0.1, length > 1, not special)
        valid = [s for s in salient if s["activation"] > 0.1
                 and len(s["token"]) > 1
                 and s["token"] not in ['<bos>', '<eos>', '<pad>']]

        # Sort by activation
        valid.sort(key=lambda x: x["activation"], reverse=True)

        if valid:
            top5 = valid[:5]
            tokens_str = ", ".join([f"{t['token']}({t['activation']:.1f})" for t in top5])
            print(f"    Salient: {tokens_str}")
            results_summary.append({
                "prompt": prompt,
                "valid_count": len(valid),
                "top_tokens": [t["token"] for t in top5],
                "top_activation": top5[0]["activation"] if top5 else 0
            })
        else:
            print(f"    Salient: (none)")
            results_summary.append({
                "prompt": prompt,
                "valid_count": 0,
                "top_tokens": [],
                "top_activation": 0
            })

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY - Ranked by salient token quality")
    print("=" * 70)

    # Sort by valid_count then top_activation
    results_summary.sort(key=lambda x: (x["valid_count"], x["top_activation"]), reverse=True)

    for i, r in enumerate(results_summary, 1):
        tokens = ", ".join(r["top_tokens"][:3]) if r["top_tokens"] else "(none)"
        print(f"\n  {i}. \"{r['prompt'][:45]}...\"")
        print(f"     Valid tokens: {r['valid_count']}, Top: {tokens}")

    best = results_summary[0]
    print("\n" + "=" * 70)
    print(f"  RECOMMENDED: \"{best['prompt']}\"")
    print(f"  Salient tokens: {', '.join(best['top_tokens'][:5])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
