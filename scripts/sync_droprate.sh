#!/usr/bin/env bash
# Pull the drop-rate ladder (job 14214329 on Della, 6307716 on Spock) and re-read Figure 2.
#
# The ladder answers the referee question "is 5% simply too small?" and its read-out was
# pre-registered before submission (see slurm/SilentReLU_flipflop_droprate_della.slurm):
#   an arm counts as an improvement only if its mean exceeds 571 active units (scale-free) AND
#   567 (absolute 4e-2) at 40,000 iterations -- the drop_rate=0.05 reference plus 3 reference SD
#   under BOTH criteria. If no arm clears the bar, that IS the result.
#
# Usage: bash scripts_sync_droprate.sh [della|spock]
set -euo pipefail
HOST="${1:-della}"
REPO="$HOME/Documents/GitHub/trainRNNbrain"
CELL="data/trained_RNNs/NBitFlipFlop_std_droprate"

echo "=== pulling $CELL from $HOST ==="
mkdir -p "$REPO/$CELL"
rsync -az --info=stats1 "$HOST:trainRNNbrain/$CELL/" "$REPO/$CELL/"

echo
echo "=== cells on disk ==="
for d in "$REPO/$CELL"/*/; do
  [ -d "$d" ] && printf '  %-58s %s nets\n' "$(basename "$d")" "$(find "$d" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
done

echo
echo "=== re-reading Figure 2 ==="
cd "$REPO"
PYTHONPATH=. python3 trainRNNbrain/experiments_and_analysis/fig_paper_F2.py

echo
echo "=== syncing the figure into the manuscript and rebuilding ==="
cd "$HOME/Documents/Projects/Dead_ReLU_paper"
./sync_figures.sh fig_paper_F2.png
latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 && echo "main.pdf rebuilt"
