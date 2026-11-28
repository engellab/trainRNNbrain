#!/bin/bash
# READ OF THE MATCHED MAZE CONTROLS ON DELLA (CPU), chained after the training jobs so that the result JSON exists
# whether or not anyone is connected. Rules: docs/project_trajectory/2026-11-25_1237_rules-maze-matched-controls.md;
# chain: docs/project_trajectory/2026-11-25_*_maze-matched-controls-submitted.md. REPO is a clean detached clone of
# wp/maze-matched-controls; OUT holds Maze_suite_mac (the original controls, copied from the Mac) and the independent
# recount script oracle_matched.py. Without MATCHED=1 the original controls alone are read (dry run).
#   MATCHED=1 sbatch --export=ALL --dependency=afterany:<N=1,000 job>:<N=2,000 continuation> slurm/maze_matched_read_della.slurm
#
#SBATCH --job-name=MazeMatchedRead
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%j.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%j.err

module purge
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -eo pipefail

REPO="${REPO:-$HOME/trainRNNbrain_mctrl_read}"
D="$HOME/trainRNNbrain/data"
OUT="$D/trained_RNNs_rl_maze_mctrl_read"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
  echo "FATAL: $REPO has uncommitted changes." >&2; exit 1
fi
echo "code: $REPO @ $(git -C "$REPO" rev-parse --short HEAD); $(date)"

TAG=dry
M=(); MO=()
if [ "${MATCHED:-0}" = 1 ]; then
  TAG=matched
  M=(--matched1000 "$D/trained_RNNs_rl_maze_mctrl_N1000/Maze_interventions"
     --matched2000 "$D/trained_RNNs_rl_maze_mctrl_N2000/Maze_interventions")
  MO=("${M[1]}" "${M[3]}")
fi
A1="$D/trained_RNNs_rl_maze_entsp_N1000/Maze_interventions"
A2="$D/trained_RNNs_rl_maze_entsp_N2000/Maze_interventions"
FZ="$D/trained_RNNs_rl_maze_fz/Maze_interventions"
python "$REPO/trainRNNbrain/experiments_and_analysis/maze_matched_controls_read.py" "$OUT/reads_$TAG.json" \
  --suite "$OUT/Maze_suite_mac" --arms1000 "$A1" --arms2000 "$A2" --fz "$FZ" "${M[@]}" | tee "$OUT/verdict_$TAG.txt"
python "$OUT/oracle_matched.py" "$OUT/reads_$TAG.json" "$OUT/Maze_suite_mac" "$A1" "$A2" "$FZ" "${MO[@]}" \
  | tee "$OUT/oracle_$TAG.txt"
