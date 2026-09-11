#!/bin/bash
# AUTOMATIC READ-OUT of the W_inp init-scale sweep (T5) (SPOCK). Submit with
#   sbatch --dependency=afterany:<training array job id> SilentReLU_flipflop_sigmoid_analysis_spock.slurm
# so it runs once every training task has ended (finished or failed). CPU only. Writes the table of
# flipflop_winp_silence.py (row-norm sweep vs the ReLU ksweep baseline at k=3, N=500/1000) into its log
# and the figure into the repo's img/internal_figures on Spock. Exists so the result is readable in
# the morning without anyone's laptop being awake.

#SBATCH --job-name=FFwinpA
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/usr/people/pt1290/trainRNNbrain_logs/%x.%j.out
#SBATCH --error=/usr/people/pt1290/trainRNNbrain_logs/%x.%j.err

module purge
module load anacondapy/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -eo pipefail
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
REPO="${REPO:-$HOME/trainRNNbrain}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "code: $(git rev-parse --short HEAD)   $(date)"
ls data/trained_RNNs/NBitFlipFlop_std_winp/*/ || { echo "no winp runs found" >&2; exit 1; }
python trainRNNbrain/experiments_and_analysis/flipflop_winp_silence.py 2>&1 | grep -v Warning
