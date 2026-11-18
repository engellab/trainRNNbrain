#!/bin/bash
# Check and submit the long warm Maze continuation (Phase B) on Della (rules:
# docs/project_trajectory/2026-11-18_0745_rules-maze-long-warm-continuation.md; launcher
# slurm/rl_maze_warmlong_della.slurm). Run from the Mac, after the launcher is pushed to origin/wp/maze-plateau and
# after every warm stage 12 of the warm-start test has finished (and R1 and R4 of that test have passed):
#   ssh della 'bash -s' < slurm/rl_maze_warmlong_submit_della.sh
# Every step checks its result and the script stops at the first failure. Nothing existing is changed: the
# checkout ~/trainRNNbrain_mazefinite must already be at the pin (only the gitignored launcher is written into it),
# `git fetch` in ~/trainRNNbrain_mazelong only updates remote-tracking refs, the parents are read only, the launcher
# checks write under a new /tmp folder, and the outputs go to the new tree trained_RNNs_rl_maze_warmlong (the
# script refuses if it already holds a cell).
set -euo pipefail
PIN=8c6b36e9ea20c449aad550ac1b544d318b8c0631          # wp/maze-finite
BRANCH=wp/maze-plateau
REPO=$HOME/trainRNNbrain_mazefinite
LAUNCHER=slurm/rl_maze_warmlong_della.slurm
DATA=$HOME/trainRNNbrain/data
SAVE_TO=$DATA/trained_RNNs_rl_maze_warmlong
WARMTEST_ROOT=$DATA/trained_RNNs_rl_maze_warmtest/Maze_suite
CELLS=(N=500_act=relu_warm_cont12/s1 N=500_act=relu_warm_cont12/s2 N=500_act=relu_warm_cont12/s3
       N=1000_act=relu_warm_cont12/s1 N=1000_act=relu_warm_cont12/s2 N=1000_act=relu_warm_cont12/s3
       N=1000_act=relu_warm_cont12/s4 N=1000_act=relu_warm_cont12/s5 N=1000_act=relu_warm_cont12/s6)   # 4-9, 26-28

# 1. the code: the existing checkout at the pin, and the launcher from the task branch (gitignored there)
[ -d "$REPO" ] || { echo "FATAL: $REPO does not exist." >&2; exit 1; }
[ "$(git -C "$REPO" rev-parse HEAD)" = "$PIN" ] || { echo "FATAL: $REPO is not at $PIN." >&2; exit 1; }
git -C "$HOME/trainRNNbrain_mazelong" fetch -q origin "$BRANCH:refs/remotes/origin/$BRANCH"
git -C "$REPO" show "origin/$BRANCH:$LAUNCHER" > "$REPO/$LAUNCHER"
grep -q "^PIN=$PIN$" "$REPO/$LAUNCHER" || { echo "FATAL: the launcher on $BRANCH does not pin $PIN." >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "FATAL: $REPO is not clean." >&2; exit 1; }
echo "code: $REPO @ $(git -C "$REPO" rev-parse HEAD); launcher from origin/$BRANCH $(git -C "$REPO" rev-parse --short "origin/$BRANCH"), md5 $(md5sum "$REPO/$LAUNCHER" | cut -d' ' -f1)"
W12_JOB=17851662                                       # warm stage 12 of the warm-start test
if squeue -h -j "$W12_JOB" -o "%i" 2>/dev/null | grep -q .; then
  echo "FATAL: warm stage 12 ($W12_JOB) still has tasks queued or running; wait for all nine to finish." >&2; exit 1
fi

module purge; module load anaconda3/2024.10
set +u   # conda's activation scripts read unset variables
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch-env
set -u

# 2. read-only check of the nine parents (warm stage 12): weights, records ending at update 24,000, a finite Adam
#    state; prints Adam's step count per parameter
python - "${CELLS[@]/#/$WARMTEST_ROOT/}" <<'EOF'
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
    if last != 24000 or not finite:
        bad.append(f"{d}: last update {last}, finite {finite}")
if bad:
    sys.exit("FATAL: " + "; ".join(bad))
EOF
if ls -d "$SAVE_TO"/Maze_suite/*/ >/dev/null 2>&1; then echo "FATAL: $SAVE_TO/Maze_suite already holds cells." >&2; exit 1; fi

# 3. launcher check on the login node: srun stubbed with tiny episodes, two updates per stage, output under a new
#    temporary folder; index 7 stage 13 -> stage 14, index 4 stage 13
SMOKE=$(mktemp -d "${TMPDIR:-/tmp}/pt1290_mazewarmlong_smoke_XXXXXX"); mkdir "$SMOKE/bin"
printf '#!/bin/bash\nexec "$@" trainer.n_episodes=4 trainer.T=6 trainer.probe_episodes=4 hydra.run.dir=%s/hydra\n' "$SMOKE" > "$SMOKE/bin/srun"
chmod +x "$SMOKE/bin/srun"
for chk in "7 13 24001-24002" "7 14 24003-24004" "4 13 24001-24002"; do
  set -- $chk
  ( cd "$REPO" && env -u INIT_ADAM -u RECOVER PATH="$SMOKE/bin:$PATH" SLURM_ARRAY_TASK_ID=$1 STAGE=$2 SAVE_TO="$SMOKE/tree" \
      WARMTEST_ROOT="$WARMTEST_ROOT" EXTRA_UPDATES=2 bash "$LAUNCHER" > "$SMOKE/check_$1_$2.log" 2>&1 ) \
    || { tail -20 "$SMOKE/check_$1_$2.log"; exit 1; }
  grep -q "update clock: this run records updates $3" "$SMOKE/check_$1_$2.log" || { echo "FATAL: index $1 stage $2 clock is not $3" >&2; exit 1; }
  grep -q "restored the adam optimizer state" "$SMOKE/check_$1_$2.log" || { echo "FATAL: index $1 stage $2 did not restore Adam" >&2; exit 1; }
  grep -E "^continuation|update clock|restored|^done" "$SMOKE/check_$1_$2.log"
done
# the step count continues the real parent's (2 updates x 8 optimizer steps, less skipped ones) at the constant rate
python - "$WARMTEST_ROOT/N=1000_act=relu_warm_cont12/s1" "$SMOKE/tree/Maze_suite" <<'EOF'
import glob, json, sys, torch
parent, tree = sys.argv[1:]
step = lambda d: [int(v["step"]) for v in torch.load(glob.glob(f"{d}/*AdamState_Maze.pt")[0], map_location="cpu", weights_only=False)["state"].values()]
prog = lambda d: json.load(open(glob.glob(f"{d}/*Progress.json")[0]))
base = 1e-3 * (100 / 1000) ** (1 / 3)
ok, start = True, step(parent)
for cell in ("N=1000_act=relu_warm_cont13", "N=1000_act=relu_warm_cont14"):
    d = f"{tree}/{cell}/s1"
    p = prog(d)
    skipped = sum(int(r["skipped"] + r.get("skipped_grad", 0)) for r in p)
    want = [s + 16 - skipped for s in start]
    lrs = [r["lr"] for r in p]
    good = step(d) == want and all(abs(a - base) < 1e-12 for a in lrs)
    ok &= good
    print(f"{cell}: steps {start} -> {step(d)} (expected {want}), lr {lrs} (expected {base}) {'ok' if good else 'MISMATCH'}")
    start = step(d)
if not ok:
    sys.exit("FATAL: a launcher check's step count or rate is not as expected")
EOF

# 4. submit: stage 13 for the nine networks; stage 14 waits task by task (aftercorr) on stage 13 of the same index
export REPO SAVE_TO WARMTEST_ROOT EXTRA_UPDATES=5000
unset INIT_ADAM RECOVER
cd "$REPO"
S13=$(STAGE=13 sbatch --parsable --chdir="$REPO" --array=4-9,26-28 --time=23:00:00 "$LAUNCHER"); S13=${S13%%;*}
echo "stage 13 -> $S13"
S14=$(STAGE=14 sbatch --parsable --chdir="$REPO" --array=4-9,26-28 --time=23:00:00 --dependency=aftercorr:"$S13" "$LAUNCHER"); S14=${S14%%;*}
echo "stage 14 -> $S14 (aftercorr:$S13)"
squeue -u "$USER" -n MazeWarmLong -o "%.20i %.10T %.24r %.12j" | head -30
