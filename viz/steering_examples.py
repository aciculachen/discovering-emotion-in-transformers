"""
Generate Steering Examples Table (LaTeX)

Creates a table showing example prompts with steering effects at different α values.
Shows how steering changes the probability of emotion-related words.

Usage:
    python steering_examples.py

Output:
    results/tables/steering_examples.tex
    (Also prints formatted examples to console)
"""

import os
import argparse
import pandas as pd
import numpy as np

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Disgust"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Steering Examples")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(script_dir, "..", "results")
    parser.add_argument("--results_dir", type=str, default=default_results,
                        help="Results directory")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(default_results, "tables"),
                        help="Output directory for LaTeX table")
    parser.add_argument("--examples_per_emotion", type=int, default=2,
                        help="Number of example prompts per emotion")
    return parser.parse_args()


def truncate_prompt(prompt, max_len=50):
    """Truncate prompt for display, adding ellipsis if needed."""
    if len(prompt) > max_len:
        return prompt[:max_len-3] + "..."
    return prompt


def escape_latex(text):
    """Escape special LaTeX characters."""
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '#': r'\#',
        '$': r'\$',
        '%': r'\%',
        '&': r'\&',
        '_': r'\_',
        '^': r'\^{}',
        '~': r'\~{}',
        '[': r'{[}',
        ']': r'{]}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def get_best_examples(steering_df, n_per_emotion=2):
    """
    Select the best steering examples for each emotion.
    Picks prompts with strongest steering effect (highest Δlog_prob at α=4).

    Returns:
        dict: {emotion: [{"prompt": str, "alpha_effects": {α: Δlog_prob}, ...}, ...]}
    """
    examples = {}

    for emotion in EMOTIONS:
        emo_df = steering_df[steering_df["emotion"] == emotion]
        if len(emo_df) == 0:
            continue

        # Get unique prompts and their α=4 effect
        prompts_effects = []
        for prompt in emo_df["prompt"].unique():
            prompt_df = emo_df[emo_df["prompt"] == prompt]

            # Get effect at α=4
            alpha4 = prompt_df[prompt_df["alpha"] == 4]
            if len(alpha4) == 0:
                continue

            effect_at_4 = alpha4["delta_log_prob"].values[0]

            # Get all alpha effects for this prompt
            alpha_effects = {}
            for _, row in prompt_df.iterrows():
                alpha_effects[row["alpha"]] = row["delta_log_prob"]

            prompts_effects.append({
                "prompt": prompt,
                "effect_at_4": effect_at_4,
                "alpha_effects": alpha_effects,
                "layer": prompt_df["layer"].iloc[0],
                "feature_id": prompt_df["feature_id"].iloc[0],
            })

        # Sort by effect at α=4 (descending) and take top N
        prompts_effects.sort(key=lambda x: x["effect_at_4"], reverse=True)
        examples[emotion] = prompts_effects[:n_per_emotion]

    return examples


def print_examples(examples):
    """Print examples to console in a readable format."""
    print("\n" + "=" * 80)
    print("STEERING EXAMPLES")
    print("=" * 80)

    for emotion in EMOTIONS:
        if emotion not in examples:
            continue

        print(f"\n{'─' * 80}")
        print(f"  {emotion.upper()}")
        print(f"{'─' * 80}")

        for i, ex in enumerate(examples[emotion]):
            prompt = truncate_prompt(ex["prompt"], 60)
            print(f"\n  Example {i+1}: \"{prompt}\"")
            print(f"  Feature: L{ex['layer']} #{ex['feature_id']}")
            print()

            # Print α values in a nice table
            alphas = sorted(ex["alpha_effects"].keys())
            header = "    α:       " + "  ".join([f"{a:>6}" for a in alphas])
            values = "    Δlog-p:  " + "  ".join([f"{ex['alpha_effects'][a]:>+6.3f}" for a in alphas])
            print(header)
            print(values)

            # Visual bar representation
            print()
            print("    Effect:  ", end="")
            for a in alphas:
                val = ex["alpha_effects"][a]
                if val > 0.1:
                    bar = "▓▓▓"
                elif val > 0.05:
                    bar = "▓▓░"
                elif val > 0:
                    bar = "▓░░"
                elif val > -0.05:
                    bar = "░░░"
                else:
                    bar = "░░░"
                print(f"  {bar:>6}", end="")
            print()


def generate_latex_table(examples, output_path):
    """Generate LaTeX table from examples."""

    latex = r"""\begin{table}[t]
\centering
\caption{Steering Examples: Effect of Feature Manipulation on Emotion Word Probability. For each emotion, we show example prompts and the change in log-probability ($\Delta$log-prob) of emotion-related words at different steering strengths ($\alpha$). Positive values indicate increased probability.}
\label{tab:steering_examples}
\small
\begin{tabular}{llcccccc}
\toprule
Emotion & Prompt (truncated) & $\alpha$=-2 & $\alpha$=-1 & $\alpha$=0 & $\alpha$=1 & $\alpha$=2 & $\alpha$=4 \\
\midrule
"""

    alphas_order = [-2, -1, 0, 1, 2, 4]

    for emotion in EMOTIONS:
        if emotion not in examples:
            continue

        for i, ex in enumerate(examples[emotion]):
            # Emotion column (only for first example)
            if i == 0:
                emo_cell = f"\\multirow{{{len(examples[emotion])}}}{{*}}{{{emotion}}}"
            else:
                emo_cell = ""

            # Truncate and escape prompt
            prompt = truncate_prompt(ex["prompt"], 35)
            prompt_escaped = escape_latex(prompt)

            # Alpha values
            alpha_vals = []
            for a in alphas_order:
                if a in ex["alpha_effects"]:
                    val = ex["alpha_effects"][a]
                    if a == 0:
                        alpha_vals.append("0")
                    else:
                        alpha_vals.append(f"{val:+.3f}")
                else:
                    alpha_vals.append("--")

            latex += f"{emo_cell} & {prompt_escaped} & {' & '.join(alpha_vals)} \\\\\n"

        latex += r"\addlinespace" + "\n"

    # Remove last addlinespace
    latex = latex.rsplit(r"\addlinespace", 1)[0]

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)

    print(f"\nSaved: {output_path}")


def generate_ascii_examples(examples):
    """Generate ASCII art visualization of steering effects."""

    print("\n" + "=" * 80)
    print("STEERING EFFECT VISUALIZATION (ASCII)")
    print("=" * 80)

    for emotion in EMOTIONS:
        if emotion not in examples:
            continue

        print(f"\n┌{'─' * 78}┐")
        print(f"│ {emotion.upper():<76} │")
        print(f"├{'─' * 78}┤")

        for ex in examples[emotion]:
            prompt = truncate_prompt(ex["prompt"], 50)
            print(f"│ Prompt: \"{prompt}\"")
            print(f"│")
            print(f"│   α     Δlog-prob   Effect")
            print(f"│  ───────────────────────────────────────────")

            for alpha in sorted(ex["alpha_effects"].keys()):
                val = ex["alpha_effects"][alpha]

                # Create bar visualization
                bar_len = int(abs(val) * 50)  # Scale factor
                bar_len = min(bar_len, 30)  # Cap at 30 chars

                if val > 0:
                    bar = "│" + "█" * bar_len + " " * (30 - bar_len)
                    direction = "→ more emotion"
                elif val < 0:
                    bar = " " * (30 - bar_len) + "█" * bar_len + "│"
                    direction = "← less emotion"
                else:
                    bar = " " * 14 + "│" + " " * 15
                    direction = "(baseline)"

                print(f"│  {alpha:>3}   {val:>+8.4f}   {bar}  {direction}")

            print(f"│")

        print(f"└{'─' * 78}┘")


def main():
    args = parse_args()

    # Load data
    print("Loading steering results...")
    steering_df = pd.read_csv(os.path.join(args.results_dir, "steering_results.csv"))

    # Get best examples
    examples = get_best_examples(steering_df, args.examples_per_emotion)

    # Print to console
    print_examples(examples)

    # Generate ASCII visualization
    generate_ascii_examples(examples)

    # Generate LaTeX table
    output_path = os.path.join(args.output_dir, "steering_examples.tex")
    generate_latex_table(examples, output_path)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
