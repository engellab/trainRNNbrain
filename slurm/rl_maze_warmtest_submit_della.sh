#!/bin/bash
# Check and submit the Maze warm-start test on Della (rules: docs/project_trajectory/2026-11-17_2249_rules-maze-warm-start-test.md;
# launcher slurm/rl_maze_warmtest_della.slurm). Run from the Mac, after the launcher is pushed to origin/main:
#   ssh della 'bash -s' < slurm/rl_maze_warmtest_submit_della.sh
# Every step checks its result and the script stops at the first failure. Nothing existing is changed: the
# checkout ~/trainRNNbrain_mazefinite must already be at the pin (it is not created or moved; only the gitignored
# launcher is written into it), `git fetch` in ~/trainRNNbrain_mazelong only updates remote-tracking refs, the
# parents are read only, the launcher checks write under a new /tmp folder, and the outputs go to the new tree
# trained_RNNs_rl_maze_warmtest (the script refuses if it already holds a cell).
set -euo pipefail
PIN=8c6b36e9ea20c449aad550ac1b544d318b8c0631          # wp/maze-finite
REPO=$HOME/trainRNNbrain_mazefinite
LAUNCHER=slurm/rl_maze_warmtest_della.slurm
DATA=$HOME/trainRNNbrain/data
SAVE_TO=$DATA/trained_RNNs_rl_maze_warmtest
LONG_ROOT=$DATA/trained_RNNs_rl_maze_long/Maze_suite
RESUMED_ROOT=$DATA/trained_RNNs_rl_maze_long_resumed/Maze_suite
PARENTS=("$RESUMED_ROOT/N=500_act=relu_resume_cont10/s1" "$RESUMED_ROOT/N=500_act=relu_resume_cont10/s2"
         "$RESUMED_ROOT/N=500_act=relu_resume_cont10/s3" "$LONG_ROOT/N=1000_act=relu_cont10/s1"
         "$RESUMED_ROOT/N=1000_act=relu_resume_cont10/s2" "$LONG_ROOT/N=1000_act=relu_cont10/s3"
         "$LONG_ROOT/N=1000_act=relu_cont10/s4" "$LONG_ROOT/N=1000_act=relu_cont10/s5"
         "$RESUMED_ROOT/N=1000_act=relu_resume_cont10/s6")    # indices 4-9, 26-28

# 1. the code: the existing checkout at the pin, and the launcher from main (gitignored there)
[ -d "$REPO" ] || { echo "FATAL: $REPO does not exist." >&2; exit 1; }
[ "$(git -C "$REPO" rev-parse HEAD)" = "$PIN" ] || { echo "FATAL: $REPO is not at $PIN." >&2; exit 1; }
git -C "$HOME/trainRNNbrain_mazelong" fetch -q origin main
git -C "$REPO" show origin/main:"$LAUNCHER" > "$REPO/$LAUNCHER"
grep -q "^PIN=$PIN$" "$REPO/$LAUNCHER" || { echo "FATAL: the launcher on main does not pin $PIN." >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "FATAL: $REPO is not clean." >&2; exit 1; }
echo "code: $REPO @ $(git -C "$REPO" rev-parse HEAD); launcher from origin/main $(git -C "$REPO" rev-parse --short origin/main), md5 $(md5sum "$REPO/$LAUNCHER" | cut -d' ' -f1)"

module purge; module load anaconda3/2024.10
set +u   # conda's activation scripts read unset variables
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch-env
set -u

# 2. read-only check of the nine parents: weights, records ending at update 20,000, and a finite Adam state
#    (the warm arm loads it; three earlier stops left NaN moments). Prints Adam's step count per parameter.
python - "${PARENTS[@]}" <<'EOF'
import glob, json, sys, torch
bad = []
for d in sys.argv[1:]:
    one = lambda pat: sorted(glob.glob(f"{d}/{pat}"))
    if not one("*Params_Maze.npz") or not one("*AdamState_Maze.pt") or not one("*Progress.json"):
        bad.append(f"{d}: a file is missing"); continue
    last = json.load(open(one("*Progress.json")[0]))[-1]["update"]
    st = torch.load(one("*AdamState_Maze.pt")[0], map_location="cpu", weights_only=False)["state"]
    steps = [int(v["step"]) for v in st.values()]
    finite = all(bool(torch.isfinite(v[k]).all()) for v in st.values() for k in ("exp_avg", "exp_avg_sq"))
    print(f"{d.split('data/')[1]}: last update {last}, Adam steps {steps}, moments finite {finite}")
    if last != 20000 or not finite:
        bad.append(f"{d}: last update {last}, finite {finite}")
if bad:
    sys.exit("FATAL: " + "; ".join(bad))
EOF
if ls -d "$SAVE_TO"/Maze_suite/*/ >/dev/null 2>&1; then echo "FATAL: $SAVE_TO/Maze_suite already holds cells." >&2; exit 1; fi

# 3. launcher check on the login node: srun stubbed with tiny episodes, two updates per stage, output under a
#    new temporary folder (under $TMPDIR, else /tmp); index 7 (long tree) warm 11 -> warm 12 and cold 11, index 4 (resumed tree) warm 11
SMOKE=$(mktemp -d "${TMPDIR:-/tmp}/pt1290_mazewarm_smoke_XXXXXX"); mkdir "$SMOKE/bin"
printf '#!/bin/bash\nexec "$@" trainer.n_episodes=4 trainer.T=6 trainer.probe_episodes=4 hydra.run.dir=%s/hydra\n' "$SMOKE" > "$SMOKE/bin/srun"
chmod +x "$SMOKE/bin/srun"
for chk in "7 warm 11 20001-20002" "7 warm 12 20003-20004" "7 cold 11 20001-20002" "4 warm 11 20001-20002"; do
  set -- $chk
  ( cd "$REPO" && env -u INIT_ADAM PATH="$SMOKE/bin:$PATH" SLURM_ARRAY_TASK_ID=$1 ARM=$2 STAGE=$3 SAVE_TO="$SMOKE/tree" \
      LONG_ROOT="$LONG_ROOT" RESUMED_ROOT="$RESUMED_ROOT" EXTRA_UPDATES=2 bash "$LAUNCHER" > "$SMOKE/check_$1_$2_$3.log" 2>&1 ) \
    || { tail -20 "$SMOKE/check_$1_$2_$3.log"; exit 1; }
  grep -q "update clock: this run records updates $4" "$SMOKE/check_$1_$2_$3.log" || { echo "FATAL: index $1 $2 $3 clock is not $4" >&2; exit 1; }
  if [ "$2" = warm ]; then
    grep -q "restored the adam optimizer state" "$SMOKE/check_$1_$2_$3.log" || { echo "FATAL: index $1 $2 $3 did not restore Adam" >&2; exit 1; }
  else
    ! grep -q "restored the adam optimizer state" "$SMOKE/check_$1_$2_$3.log" || { echo "FATAL: index $1 cold restored Adam" >&2; exit 1; }
  fi
  grep -E "^continuation|update clock|restored|^done" "$SMOKE/check_$1_$2_$3.log"
done
# the warm stage's step count continues the parent's (2 updates x 8 optimizer steps, less skipped ones) and its rate
# is the constant one; the cold stage's restarts at 16 and ramps
python - "$LONG_ROOT/N=1000_act=relu_cont10/s1" "$SMOKE/tree/Maze_suite" <<'EOF'
import glob, json, sys, torch
parent, tree = sys.argv[1:]
step = lambda d: [int(v["step"]) for v in torch.load(glob.glob(f"{d}/*AdamState_Maze.pt")[0], map_location="cpu", weights_only=False)["state"].values()]
prog = lambda d: json.load(open(glob.glob(f"{d}/*Progress.json")[0]))
base = 1e-3 * (100 / 1000) ** (1 / 3)
ok = True
for cell, start in (("N=1000_act=relu_warm_cont11", step(parent)), ("N=1000_act=relu_warm_cont12", None), ("N=1000_act=relu_cold_cont11", [0, 0, 0])):
    d = f"{tree}/{cell}/s1"
    p = prog(d)
    if start is None:
        start = step(f"{tree}/N=1000_act=relu_warm_cont11/s1")
    skipped = sum(int(r["skipped"] + r["skipped_grad"]) for r in p)
    want = [s + 16 - skipped for s in start]
    lrs = [r["lr"] for r in p]
    want_lr = [base / 50, 2 * base / 50] if "cold" in cell else [base, base]
    good = step(d) == want and all(abs(a - b) < 1e-12 for a, b in zip(lrs, want_lr))
    ok &= good
    print(f"{cell}: steps {start} -> {step(d)} (expected {want}), lr {lrs} (expected {want_lr}) {'ok' if good else 'MISMATCH'}")
if not ok:
    sys.exit("FATAL: a launcher check's step count or rate is not as expected")
EOF

# 4. submit: warm stage 11 and cold stage 11 for the nine networks; warm stage 12 waits task by task (aftercorr)
#    on warm stage 11 of the same index
export REPO SAVE_TO LONG_ROOT RESUMED_ROOT EXTRA_UPDATES=2000
unset INIT_ADAM
cd "$REPO"
W11=$(ARM=warm STAGE=11 sbatch --parsable --chdir="$REPO" --array=4-9,26-28 --time=12:00:00 "$LAUNCHER"); W11=${W11%%;*}
echo "warm stage 11 -> $W11"
W12=$(ARM=warm STAGE=12 sbatch --parsable --chdir="$REPO" --array=4-9,26-28 --time=12:00:00 --dependency=aftercorr:"$W11" "$LAUNCHER"); W12=${W12%%;*}
echo "warm stage 12 -> $W12 (aftercorr:$W11)"
C11=$(ARM=cold STAGE=11 sbatch --parsable --chdir="$REPO" --array=4-9,26-28 --time=12:00:00 "$LAUNCHER"); C11=${C11%%;*}
echo "cold stage 11 -> $C11"
squeue -u "$USER" -n MazeWarm -o "%.20i %.10T %.24r %.12j" | head -40
