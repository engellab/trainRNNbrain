#!/bin/bash
set -e -o pipefail
shopt -s nullglob

EXCLUDE=(BareTraining.slurm SparsityOnly.slurm FRMagnitudeOnlyOnly.slurm BareTrainingMemoryAntiAngle.slurm SparsityAndFRMagnitudeMemoryAntiAngle.slurm SparsityAndFRMagnitude.slurm SparsityAndFRMagnitudeAlphaBetaSweep.slurm Dropout.slurm)  # <-- add files here (exact names)

declare -A SKIP
for x in "${EXCLUDE[@]}"; do SKIP["$x"]=1; done

FILES=( *.slurm )
((${#FILES[@]})) || { echo "No .slurm files found."; exit 1; }

for f in "${FILES[@]}"; do
  [[ ${SKIP["$f"]+x} ]] && { echo "Skipping $f"; continue; }
  echo "Submitting $f"
  sbatch "$f"
done
