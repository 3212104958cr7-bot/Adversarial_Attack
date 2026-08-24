#!/usr/bin/env bash
# Generates the I-RAVEN dataset into data/i-raven/, matching the folder/file
# layout PredRNet's data/raven.py loader expects:
#   data/i-raven/<config_folder>/RAVEN_<k>_<train|val|test>.npz
#
# The generator (SRAN repo) is Python 2.7 code (uses `print "..."` statements)
# -- it will NOT run under Python 3 as-is. This script creates a throwaway
# conda env for it. Not run automatically; review and run by hand:
#   bash data/prepare_iraven.sh [num_samples_per_config]
#
# Output size: ~3.5 GB for the default 10,000 samples/config x 7 configs.
set -euo pipefail

NUM_SAMPLES="${1:-10000}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/data/_sran_src"
OUT_DIR="$PROJECT_ROOT/data/i-raven"

if [ ! -d "$SRC_DIR" ]; then
    git clone --depth 1 https://github.com/husheng12345/SRAN.git "$SRC_DIR"
fi

if command -v conda >/dev/null 2>&1; then
    conda create -y -n iraven-gen python=2.7 2>&1 | tail -5
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate iraven-gen
    pip install -r "$SRC_DIR/I-RAVEN/requirements.txt"
else
    echo "conda not found -- create/activate a Python 2.7 environment yourself," \
         "then: pip install -r $SRC_DIR/I-RAVEN/requirements.txt" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
python "$SRC_DIR/I-RAVEN/main.py" \
    --num-samples "$NUM_SAMPLES" \
    --save-dir "$OUT_DIR" \
    --seed 1234 --val 2 --test 2

conda deactivate

echo "Done. Dataset written to $OUT_DIR"
echo "Configs: $(ls "$OUT_DIR")"
