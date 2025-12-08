#!/bin/bash
#
# Generate Visualizations from Experiment Results
#
# Usage:
#   ./run_all_viz.sh
#
# Requires: results/*.csv from run_all_exp.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Generating Visualizations"
echo "  Started: $(date)"
echo "============================================================"

# Check if results exist
if [ ! -f "results/feature_lists.csv" ]; then
    echo "Error: results/feature_lists.csv not found"
    echo "Run ./run_all_exp.sh first to generate results"
    exit 1
fi

echo ""
echo "[1/8] Heatmap: Emotion × Layer..."
python viz/plot_heatmap_layer.py

echo ""
echo "[2/8] Heatmap: Emotion × Top-N Features..."
python viz/plot_heatmap_topn.py

echo ""
echo "[3/8] Validation Pass Rates..."
python viz/plot_validation_passrate.py

echo ""
echo "[4/8] Steering Effect Curves..."
python viz/plot_steering_curves.py

echo ""
echo "[5/8] Steering Cross-Emotion Heatmap..."
python viz/plot_steering_cross_emotion.py

echo ""
echo "[6/8] Circuit Diagrams..."
python viz/plot_circuit.py

echo ""
echo "[7/8] Intervention Analysis..."
python viz/plot_intervention.py

echo ""
echo "[8/8] Steering Examples Table..."
python viz/steering_examples.py

echo ""
echo "============================================================"
echo "  Visualizations Complete!"
echo "  Finished: $(date)"
echo "  Figures: $SCRIPT_DIR/results/figures/"
echo "  Tables:  $SCRIPT_DIR/results/tables/"
echo "============================================================"
