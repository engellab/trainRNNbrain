#!/bin/bash
# THE ENTROPY SET POINT (lambda_ent = 0.1) ACROSS SIZES: submission on Della.
# Rules: docs/project_trajectory/2026-11-25_1241_rules-entropy-setpoint-sizes.md (committed before any run).
# Run from the Mac:
#   ssh della 'bash -s pilot <PIN>' < slurm/entropy_sizes_submit_della.sh
#   ssh della 'bash -s run <PIN> <cell>...' < slurm/entropy_sizes_submit_della.sh
#   ssh della 'bash -s finish <PIN>' < ...   (after run: oracle and verdict once every read job has ended)
#       cells: ff500 ff2000 cddm500 cddm2000 cddm4000 cddm4000_75k maze500
# <PIN> is the short commit of the clone ~/trainRNNbrain_entsizes. Every job is 23 h or less (gpu-short).
# Every training job runs behind the GPU-hour guard of entropy_sizes_job.slurm. After each cell's last job, a CPU job reads
# the cell (entropy_sizes_read.py) into $READS, next to the networks; `finish` adds the independent check and the verdict.
# First segments are array tasks 4-6 (ent0.1) of the set-point launchers, the continuations (afterok on the whole array,
# because the continuation launcher sorts the parents that exist) are SilentReLU_ff_topup200k_della.slurm.
set -euo pipefail
MODE=$1; PIN=$2; shift 2
export REPO=$HOME/trainRNNbrain_entsizes
export SAVE_TO=/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs
DATA=$HOME/trainRNNbrain/data
[ "$(git -C "$REPO" rev-parse --short HEAD)" = "$PIN" ] || { echo "FATAL: $REPO is not at $PIN" >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "FATAL: $REPO is dirty" >&2; exit 1; }
[ "$(readlink -f "$DATA/trained_RNNs")" = "$(readlink -f "$SAVE_TO")" ] || { echo "FATAL: two data trees" >&2; exit 1; }
cd "$REPO"
EX=--exclude=della-i14g[1-20]
JOB=slurm/entropy_sizes_job.slurm
READS=/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/entropy_sizes_reads; ANA=$HOME/trainRNNbrain_entsizes_ana
CTRL=$HOME/entsizes_maze_ctrl                      # the Mac's Maze_suite N=500_act=relu and _cont, s1-3 (light files)
RECORDED=$HOME/entsizes_recorded                   # remedies_catalogue.json and g6_reads.json, from the Mac
FF=slurm/entropy_setpoint_della.slurm; CD=slurm/entropy_setpoint_cddm_della.slurm
TOP=slurm/SilentReLU_ff_topup200k_della.slurm; MZ=slurm/rl_maze_interventions_della.slurm

first() {  # first <launcher> <N> <iters> <time> <constraint> <job name>: first segment of three networks; prints the job id
  N=$2 ITERS_OVERRIDE=$3 LAUNCHER=$(basename "$1") sbatch --parsable --chdir="$REPO" --export=ALL --time="$4" \
    --constraint="$5" $EX --job-name="$6" --array=4-6 $JOB
}
cont() {   # cont <parent root> <N> <target> <time> <constraint> <job name> <parent job>: continue the three to <target>
  PARENTS="$SAVE_TO/$1/EqType=h_N=$2_arm=ent0.1/*/" TARGET=$3 KEEP=1 LAUNCHER=$(basename $TOP) sbatch --parsable \
    --chdir="$REPO" --export=ALL --time="$4" --constraint="$5" $EX --job-name="$6" --dependency="afterok:$7" --kill-on-invalid-dep=yes --array=1-3 $JOB
}
cpu() {    # cpu <job name> <dependency> <python arguments>: a CPU job in the analysis clone; prints the job id
  sbatch --parsable --job-name="$1" --dependency="$2" --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
    --chdir="$ANA" --output="$HOME/trainRNNbrain_logs/%x.%j.out" --error="$HOME/trainRNNbrain_logs/%x.%j.err" --wrap \
    "module load anaconda3/2024.10; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate torch-env; export PYTHONPATH=$ANA CUDA_VISIBLE_DEVICES=; python -u $3"
}
R=trainRNNbrain/experiments_and_analysis/entropy_sizes_read.py
readcell() {  # readcell <cell> <last training job>: read the cell when its last job has ended, however it ended
  cpu "EntSizeRead_$1" "afterany:$2" "$R read $READS --cell $1"
}
fresh() {  # fresh <path>: refuse to write where a run already exists
  [ ! -e "$1" ] || { echo "FATAL: $1 exists" >&2; exit 1; }
}

if [ "$MODE" = pilot ]; then
  export SWEEP=entropy_sizes_pilot
  for N in 500 2000 4000; do
    if [ "$N" = 4000 ]; then IT=500; C="nomig&gpu80"; else IT=2000; C=nomig; fi
    for L in $FF $CD; do
      J=$(N=$N ITERS_OVERRIDE=$IT sbatch --parsable --chdir="$REPO" --export=ALL --time=02:30:00 --constraint="$C" $EX \
          --mem-per-gpu=64G --job-name=EntSizePilot --array=4 $L)
      echo "pilot $L N=$N iters=$IT job $J"
    done
  done
  exit 0
fi

if [ "$MODE" = finish ]; then
  J=$(squeue -h -u "$USER" -o "%A %j" | awk '$2 ~ /^EntSizeRead_/ {printf ":%s", $1}')
  O=$(cpu EntSizeOracle "afterany$J" "$HOME/entsizes_scripts/oracle_sizes.py $READS")
  OM=$(cpu EntSizeOracleMz "afterany$J" "$HOME/entsizes_scripts/oracle_sizes.py maze $READS/maze500.json $DATA/trained_RNNs_rl_maze_entsp_N500/Maze_interventions $CTRL")
  V=$(cpu EntSizeVerdict "afterany$J" "$R verdict $READS --recorded $RECORDED --fig $READS/entropy_sizes")
  echo "oracle $O, maze oracle $OM, verdict $V, after reads$J"
  exit 0
fi

for CELL in "$@"; do
  FFR=NBitFlipFlop_entropy_setpoint; CDR=CDDM_entropy_setpoint
  case "$CELL" in
    ff500)
      fresh "$SAVE_TO/$FFR/EqType=h_N=500_arm=ent0.1"
      J1=$(first $FF 500 75000 08:00:00 nomig EntSizeFF500)
      J2=$(cont $FFR 500 200000 12:00:00 nomig EntSizeFF500c "$J1")
      echo "$CELL: $J1 (75,000), $J2 (200,000), read $(readcell "$CELL" "$J2")" ;;
    ff2000)
      fresh "$SAVE_TO/$FFR/EqType=h_N=2000_arm=ent0.1"
      J1=$(first $FF 2000 40000 16:00:00 nomig EntSizeFF2000)
      J2=$(cont $FFR 2000 75000 14:00:00 nomig EntSizeFF2000c1 "$J1")
      J3=$(cont ${FFR}_c75000 2000 137500 23:00:00 nomig EntSizeFF2000c2 "$J2")
      J4=$(cont ${FFR}_c137500 2000 200000 23:00:00 nomig EntSizeFF2000c3 "$J3")
      echo "$CELL: $J1 (40,000), $J2 (75,000), $J3 (137,500), $J4 (200,000), read $(readcell "$CELL" "$J4")" ;;
    cddm500)
      fresh "$SAVE_TO/$CDR/EqType=h_N=500_arm=ent0.1"
      J1=$(first $CD 500 100000 09:00:00 nomig EntSizeCD500)
      echo "$CELL: $J1 (100,000), read $(readcell "$CELL" "$J1")" ;;
    cddm2000)
      fresh "$SAVE_TO/$CDR/EqType=h_N=2000_arm=ent0.1"
      J1=$(first $CD 2000 100000 21:00:00 nomig EntSizeCD2000)
      echo "$CELL: $J1 (100,000), read $(readcell "$CELL" "$J1")" ;;
    cddm4000|cddm4000_75k)
      fresh "$SAVE_TO/$CDR/EqType=h_N=4000_arm=ent0.1"
      if [ "$CELL" = cddm4000 ]; then TGT=100000; else TGT=75000; fi
      J1=$(first $CD 4000 50000 23:00:00 "nomig&gpu80" EntSizeCD4000)
      J2=$(cont $CDR 4000 $TGT 23:00:00 "nomig&gpu80" EntSizeCD4000c "$J1")
      echo "$CELL: $J1 (50,000), $J2 ($TGT), read $(readcell cddm4000 "$J2")" ;;
    maze500)
      OUT=$DATA/trained_RNNs_rl_maze_entsp_N500; fresh "$OUT"
      J1=$(N_OVERRIDE=500 SAVE_TO=$OUT LAUNCHER=$(basename $MZ) sbatch --parsable --chdir="$REPO" --export=ALL --time=15:00:00 --constraint=nomig $EX \
           --job-name=EntSizeMz500 --array=130-132 $JOB)
      J2=$(N_OVERRIDE=500 SAVE_TO=$OUT PARENT_ROOT=$OUT/Maze_interventions EXTRA_UPDATES=2000 INIT_ADAM=true LAUNCHER=$(basename $MZ) \
           sbatch --parsable --chdir="$REPO" --export=ALL --time=15:00:00 --constraint=nomig $EX --job-name=EntSizeMz500c \
           --dependency="aftercorr:$J1" --kill-on-invalid-dep=yes --array=130-132 $JOB)
      echo "$CELL: $J1 (updates 1-2,000), $J2 (to 4,000) -> $OUT, read $(cpu EntSizeRead_maze500 "afterany:$J2" "$R maze $READS --arm $OUT/Maze_interventions --suite $CTRL")" ;;
    *) echo "FATAL: unknown cell $CELL" >&2; exit 1 ;;
  esac
done
