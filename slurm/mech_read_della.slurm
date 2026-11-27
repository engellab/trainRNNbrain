#!/bin/bash
# MECHANISM TESTS: every read and verdict, CPU only (DELLA). Rules: 2026-11-25_1311_rules-mechanism-noise-decay-and-switch.md
#
# Reads the networks of parts A and B, writes JSON and text to $SAVE_TO/CDDM_mech_reads/ and trains nothing.
# Submitted after part B with --dependency=afterany (slurm/mech_submit_reads_della.sh), from the clone ~/trainRNNbrain_mechana.
#   0      mech_read.py read, grad; mech_recount.py; mech_read.py verdict (without the removal table)
#   1-9    mech_removal.py, arm (none, ent0.1, rate0.1) x seed
#   10-21  mech_optimizer.py weights and adam, the twelve networks of part A
#   30     mech_optimizer.py adam on the CDDM networks that existed before (controls, AdamW, noise-free; 3BFF runs on the Mac)
#   31-42  mech_optimizer.py weights on the networks that existed before
#
#SBATCH --job-name=MechRead
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -eo pipefail
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg
REPO="${REPO:-$HOME/trainRNNbrain_mechana}"
export PYTHONPATH="$REPO"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
OUT="$SAVE_TO/CDDM_mech_reads"
mkdir -p "$OUT"
EA="$REPO/trainRNNbrain/experiments_and_analysis"
echo "code: $(git -C "$REPO" rev-parse --short HEAD), status: $(git -C "$REPO" status --porcelain | wc -l) changed files"
SEEDS=(1153179695 2873231781 2425511753)
ARMS=(none ent0.1 rate0.1)
CELLS=("noise=default_wd=1e-6" "noise=default_wd=0" "noise=zero_wd=1e-6" "noise=zero_wd=0")
ID=$SLURM_ARRAY_TASK_ID

one() {
  # the single finished run folder matching glob "$1"
  local d=($1)
  [ "${#d[@]}" -eq 1 ] && [ -d "${d[0]}" ] || { echo "FATAL: ${#d[@]} folders match $1" >&2; exit 1; }
  echo "${d[0]}"
}

if [ "$ID" -eq 0 ]; then
  python "$EA/mech_read.py" read "$OUT/reads.json"
  python "$EA/mech_recount.py" "$SAVE_TO" "$OUT/recount.json" --reads "$OUT/reads.json" | tee "$OUT/recount.txt"
  python "$EA/mech_read.py" grad "$OUT/grad.json" | tee "$OUT/grad.txt"
  python "$EA/mech_read.py" verdict "$OUT/reads.json" --removal "$OUT" --fig "$OUT/mech_noise_decay_switch" | tee "$OUT/verdict.txt"
elif [ "$ID" -le 9 ]; then
  ARM=${ARMS[$(((ID - 1) / 3))]}; S=${SEEDS[$(((ID - 1) % 3))]}
  RUN=$(one "$SAVE_TO/CDDM_mech_switch/EqType=h_N=1000_arm=${ARM}/*seed=${S};*")
  PAR=$(one "$SAVE_TO/CDDM_mech_noise_decay/EqType=h_N=1000_${CELLS[0]}/*seed=${S};*")
  python "$EA/mech_removal.py" "$RUN" "$OUT/removal_${ARM}_s${S}.json" --switch "$PAR" | tee "$OUT/removal_${ARM}_s${S}.txt"
elif [ "$ID" -le 21 ]; then
  K=$((ID - 10)); C=${CELLS[$((K / 3))]}; S=${SEEDS[$((K % 3))]}
  RUN=$(one "$SAVE_TO/CDDM_mech_noise_decay/EqType=h_N=1000_${C}/*seed=${S};*")
  python "$EA/mech_optimizer.py" adam "$OUT/adam_${C}_s${S}.json" "$RUN" | tee "$OUT/adam_${C}_s${S}.txt"
  python "$EA/mech_optimizer.py" weights "$OUT/weights_${C}_s${S}.json" "$RUN" "$SNAP_TO/CDDM_mech_noise_decay/EqType=h_N=1000_${C}/s${S}" | tee "$OUT/weights_${C}_s${S}.txt"
elif [ "$ID" -eq 30 ]; then
  python "$EA/mech_optimizer.py" adam "$OUT/adam_old_cddm.json" "$SAVE_TO"/CDDM_paper_grid/EqType=h_N={1000,2000,4000}_arm=control/*/ \
    "$SAVE_TO"/CDDM_b5_optimizer/*opt=adamw*/*/ "$SAVE_TO"/CDDM_absorb_noise/*/*/ | tee "$OUT/adam_old_cddm.txt"
elif [ "$ID" -ge 31 ] && [ "$ID" -le 33 ]; then
  S=(1423687489 2657899762 2670295995); S=${S[$((ID - 31))]}; T=(25000 26000 24000); T=${T[$((ID - 31))]}
  RUN=$(one "$SAVE_TO/CDDM_c1_snapshots/EqType=h_N=1000_arm=control/*seed=${S};*")
  python "$EA/mech_optimizer.py" weights "$OUT/weights_c1control_s${S}.json" "$RUN" "$SNAP_TO/CDDM_c1_snapshots/EqType=h_N=1000_arm=control/s${S}" --base $T 50000 69000 | tee "$OUT/weights_c1control_s${S}.txt"
elif [ "$ID" -ge 34 ] && [ "$ID" -le 39 ]; then
  K=$((ID - 34)); S=(1423687489 2657899762 2670295995); S=${S[$((K % 3))]}; NZ=(default zero); NZ=${NZ[$((K / 3))]}
  RUN=$(one "$SAVE_TO/CDDM_absorb_noise/EqType=h_N=1000_noise=${NZ}/*seed=${S};*")
  python "$EA/mech_optimizer.py" weights "$OUT/weights_absorb_${NZ}_s${S}.json" "$RUN" "$SNAP_TO/CDDM_absorb_noise/EqType=h_N=1000_noise=${NZ}/s${S}" | tee "$OUT/weights_absorb_${NZ}_s${S}.txt"
elif [ "$ID" -ge 40 ] && [ "$ID" -le 42 ]; then
  S=(2670297180 1542248578 2851654641); S=${S[$((ID - 40))]}; T=(67000 50000 58000); T=${T[$((ID - 40))]}
  RUN=$(one "$SAVE_TO/NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/*seed=${S};*")
  python "$EA/mech_optimizer.py" weights "$OUT/weights_3bff_s${S}.json" "$RUN" "$SNAP_TO/NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/s${S}" --base $T | tee "$OUT/weights_3bff_s${S}.txt"
else
  echo "FATAL: no task $ID" >&2; exit 1
fi
