#!/bin/bash
set -e -o pipefail
shopt -s nullglob

FILES=( *.slurm )

((${#FILES[@]})) || { echo "No .slurm files found."; exit 1; }

for f in "${FILES[@]}"; do
  echo "Submitting $f"
  sbatch "$f"
done