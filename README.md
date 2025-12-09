# Discovering Human-Interpretable Emotion in Transformer Models

**CSE 6521 - Final Project**

## Overview

This project investigates emotion representations in the Gemma 2 2B language model using Gemma Scope Sparse Autoencoders (SAE). We discover, validate, and analyze emotion-selective features for five basic emotions: Joy, Sadness, Anger, Fear, and Disgust.

## Key Findings

1. **Emotion-Selective Features Exist**
   - Discovered 60 candidate features (12 per emotion)
   - 35 features (58.3%) passed dual validation tests
   - Features concentrated in layers 22-25

2. **Causal Circuit Analysis**
   - Identified upstream causal contributors (MLP and attention heads)
   - Mapped downstream effects on output vocabulary (promoted tokens)

3. **Steering Verification**
   - Steering features causally affects output emotion probability
   - Positive α increases emotion word probability
   - Negative α decreases emotion word probability

## Project Structure

```
submission/
├── run_all.sh                # Full pipeline (experiments + visualization)
├── run_all_exp.sh            # Run experiments only (GPU required)
├── run_all_viz.sh            # Run visualization only (CPU)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/                      # Source code
│   ├── prompts.py            # GoEmotions dataset loader
│   ├── discovery.py          # Feature discovery
│   ├── validation.py         # Confound & cross-emotion validation
│   ├── steering.py           # Steering experiments
│   ├── steering_generation.py # Text generation with steering
│   ├── circuit.py            # Circuit analysis (main)
│   ├── anger_circuit.py      # Anger circuit analysis
│   ├── fear_circuit.py       # Fear circuit analysis
│   └── disgust_circuit.py    # Disgust circuit analysis
│
├── viz/                      # Visualization scripts
│   ├── theme.py              # Unified theme (colors, fonts)
│   ├── plot_circuit.py       # Circuit diagrams
│   ├── plot_heatmap_layer.py # Emotion x Layer heatmap
│   ├── plot_heatmap_topn.py  # Top-N features heatmap
│   ├── plot_steering_curves.py        # Steering effect curves
│   ├── plot_steering_cross_emotion.py # Cross-emotion heatmap
│   ├── plot_validation_passrate.py    # Validation pass rates
│   ├── plot_intervention.py  # Ablation results
│   └── steering_examples.py  # Steering examples table (LaTeX)
│
└── results/                  # Pre-computed results
    ├── figures/              # Generated figures (PNG/PDF)
    ├── tables/               # LaTeX tables
    └── *.csv                 # Data files
```

## Installation

```bash
# Clone or extract the submission
cd submission

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Reproduction

### Option 1: Visualization Only (CPU)

Uses pre-computed CSV files to generate figures:

```bash
./run_all_viz.sh
```

This will generate:
- `results/figures/heatmap_*.png` - Feature selectivity heatmaps
- `results/figures/circuit_*.png` - Circuit diagrams
- `results/figures/steering_*.png` - Steering effect plots
- `results/tables/steering_examples.tex` - Steering examples table

### Option 2: Full Reproduction (GPU)

Runs all experiments from scratch:

```bash
./run_all.sh           # Default: CUDA
./run_all.sh mps       # Apple Silicon
./run_all.sh cpu       # CPU only (slow)
```

**Requirements:**
- GPU with 8GB+ VRAM (NVIDIA CUDA or Apple Silicon MPS)
- ~3GB disk space for model weights

### Run Individual Steps

```bash
# Experiments only (GPU required)
./run_all_exp.sh

# Visualization only (CPU)
./run_all_viz.sh

# Or run individual Python scripts:
python src/discovery.py --device cuda --full
python src/validation.py --device cuda --full
python src/steering.py --device cuda
python src/circuit.py --device cuda
```

## Output Files

### Data Files (CSV)

| File | Description |
|------|-------------|
| `feature_lists.csv` | Discovered features (60 total) |
| `validated_features.csv` | Features passing both tests (35 total) |
| `validation_confound.csv` | Confound validation results |
| `validation_cross_emotion.csv` | Cross-emotion validation results |
| `steering_results.csv` | Steering experiment results |
| `steering_controls.csv` | Cross-emotion steering controls |
| `circuit_upstream.csv` | Causal attribution data |
| `circuit_downstream.csv` | Promoted/suppressed tokens |
| `circuit_intervention.csv` | Ablation experiment results |

### Figures

| File | Description |
|------|-------------|
| `heatmap_emotion_layer.png` | Emotion x Layer selectivity |
| `heatmap_emotion_topn.png` | Top-4 features per emotion |
| `validation_passrate.png` | Validation pass rates |
| `steering_curves.png` | Steering effect (α vs Δlog-prob) |
| `steering_cross_emotion.png` | Cross-emotion specificity matrix |
| `circuit_*.png` | Circuit diagrams (5 emotions) |
| `circuit_intervention.png` | Ablation validation results |

## Technical Details

- **Model**: Gemma 2 2B (google/gemma-2-2b)
- **SAE**: Gemma Scope 16k (`gemma-scope-2b-pt-res-canonical`)
- **SAE Layers**: 18-25
- **Dataset**: GoEmotions (single-label filtered)
- **Emotions**: Joy, Sadness, Anger, Fear, Disgust

## Acknowledgments

- TransformerLens library by Neel Nanda
- SAE-Lens library by Joseph Bloom
- Gemma Scope by Google DeepMind
- GoEmotions dataset by Google Research
