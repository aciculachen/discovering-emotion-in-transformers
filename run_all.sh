#!/bin/bash
#
# Run Full Emotion Analysis Pipeline (5 Emotions)
# Combines run_all_exp.sh + run_all_viz.sh
#
# Usage:
#   ./run_all.sh              # Default: cuda device
#   ./run_all.sh mps          # Use MPS (Apple Silicon)
#   ./run_all.sh cpu          # Use CPU
#
# Or run separately:
#   ./run_all_exp.sh cuda     # Run experiments only
#   ./run_all_viz.sh          # Generate visualizations only

set -e  # Exit on error

DEVICE="${1:-cuda}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Full Pipeline: Experiments + Visualizations"
echo "============================================================"

# Run experiments
./run_all_exp.sh "$DEVICE"

# Generate visualizations
./run_all_viz.sh

echo ""
echo "============================================================"
echo "  Full Pipeline Complete!"
echo "============================================================"
