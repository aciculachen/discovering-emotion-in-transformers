#!/bin/bash
#
# Run Experiments (5 Emotions)
# Joy, Sadness, Anger, Fear, Disgust
#
# Usage:
#   ./run_all_exp.sh              # Default: cuda device
#   ./run_all_exp.sh mps          # Use MPS (Apple Silicon)
#   ./run_all_exp.sh cpu          # Use CPU
#

set -e  # Exit on error

DEVICE="${1:-cuda}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Emotion Analysis Experiments (5 Emotions)"
echo "  Device: $DEVICE"
echo "  Started: $(date)"
echo "============================================================"

# 1. Discovery (~40 min)
echo ""
echo "[1/4] Running Feature Discovery..."
echo "============================================================"
python src/discovery.py --device "$DEVICE" --full
echo "[1/4] Discovery completed: $(date)"

# 2. Validation (~15 min)
echo ""
echo "[2/4] Running Validation..."
echo "============================================================"
python src/validation.py --device "$DEVICE" --full
echo "[2/4] Validation completed: $(date)"

# 3. Steering (~35 min)
echo ""
echo "[3/4] Running Steering Experiments..."
echo "============================================================"
python src/steering.py --device "$DEVICE"
echo "[3/4] Steering completed: $(date)"

# 4. Circuit Analysis (~40 min)
echo ""
echo "[4/4] Running Circuit Analysis..."
echo "============================================================"
python src/circuit.py --device "$DEVICE"
echo "[4/4] Circuit completed: $(date)"

echo ""
echo "============================================================"
echo "  Experiments Complete!"
echo "  Finished: $(date)"
echo "  Results: $SCRIPT_DIR/results/"
echo ""
echo "  Next: Run ./run_all_viz.sh to generate figures"
echo "============================================================"
