#!/bin/bash
# Deploy and submit the three resumed Maze networks on Della (notebook entry
# docs/project_trajectory/2026-11-14_1605_maze-nonfinite-recovery.md). Run from the Mac:
#   ssh della 'bash -s' < slurm/rl_maze_resume_submit_della.sh
# Every step checks its result and the script stops at the first failure. Nothing existing is
# changed: a new worktree ~/trainRNNbrain_mazefinite is added (the script refuses if it exists), the
# outputs go to the new tree trained_RNNs_rl_maze_long_resumed, and ~/trainRNNbrain_mazelong (with
# its MazeLong jobs) is left as it is; `git fetch` there only updates remote-tracking refs.
set -euo pipefail
PIN=8c6b36e9ea20c449aad550ac1b544d318b8c0631          # wp/maze-finite
REPO=$HOME/trainRNNbrain_mazefinite
LAUNCHER=slurm/rl_maze_resume_della.slurm
DATA=$HOME/trainRNNbrain/data
SAVE_TO=$DATA/trained_RNNs_rl_maze_long_resumed
OLD_ROOT=$DATA/trained_RNNs_rl_maze_long/Maze_suite

# 1. the code: a new detached worktree at the pin, and the launcher from main (gitignored there)
[ -e "$REPO" ] && { echo "FATAL: $REPO exists; not touching it." >&2; exit 1; }
git -C "$HOME/trainRNNbrain_mazelong" fetch -q origin wp/maze-finite main
git -C "$HOME/trainRNNbrain_mazelong" worktree add -q --detach "$REPO" "$PIN"
git -C "$REPO" show origin/main:"$LAUNCHER" > "$REPO/$LAUNCHER"
grep -q "^PIN=$PIN$" "$REPO/$LAUNCHER" || { echo "FATAL: the launcher on main does not pin $PIN." >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "FATAL: $REPO is not clean." >&2; exit 1; }
echo "code: $REPO @ $(git -C "$REPO" rev-parse HEAD); launcher md5 $(md5sum "$REPO/$LAUNCHER" | cut -d' ' -f1)"

module purge; module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch-env

# 2. read-only check of the inference in the entry: were Adam's moments NaN at the stops?
for f in "$OLD_ROOT/N=1000_act=relu_cont5/s6/s6_AdamState_Maze.pt" "$OLD_ROOT/N=500_act=relu_cont6/s2/s2_AdamState_Maze.pt" \
         "$OLD_ROOT/N=1000_act=relu_cont3/s2/s2_AdamState_Maze.pt" "$OLD_ROOT/N=500_act=relu_cont7/s3/s3_AdamState_Maze.pt"; do
  python - "$f" <<'EOF'
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state"]
nan = [(k, round(float(torch.isnan(v["exp_avg"]).float().mean()), 3), round(float(torch.isnan(v["exp_avg_sq"]).float().mean()), 3)) for k, v in s.items()]
print(sys.argv[1].split("Maze_suite/")[1], "share of NaN entries (param, exp_avg, exp_avg_sq):", nan)
EOF
done

# 3. the parents exist, and no resumed folder exists yet
for p in N=1000_act=relu_cont/s2 N=1000_act=relu_cont4/s6 N=500_act=relu_cont5/s2 N=500_act=relu_cont6/s3; do
  ls "$OLD_ROOT/$p"/*Params_Maze.npz >/dev/null || { echo "FATAL: no parent $OLD_ROOT/$p" >&2; exit 1; }
done
if ls -d "$SAVE_TO"/Maze_suite/*_resume_* >/dev/null 2>&1; then echo "FATAL: resumed folders already exist in $SAVE_TO" >&2; exit 1; fi

# 4. launcher check on the login node: index 8 stage 3, srun stubbed with tiny overrides, output under /tmp
SMOKE=/tmp/pt1290_mazefinite_smoke; mkdir -p "$SMOKE/bin"
printf '#!/bin/bash\nexec "$@" trainer.n_episodes=4 trainer.T=6 trainer.probe_episodes=4 hydra.run.dir=%s/hydra\n' "$SMOKE" > "$SMOKE/bin/srun"
chmod +x "$SMOKE/bin/srun"
for chk in "8 3" "6 7"; do
  set -- $chk
  ( cd "$REPO" && PATH="$SMOKE/bin:$PATH" SLURM_ARRAY_TASK_ID=$1 STAGE=$2 SAVE_TO="$SMOKE/tree" OLD_ROOT="$OLD_ROOT" \
      EXTRA_UPDATES=2 INIT_ADAM=false bash "$LAUNCHER" > "$SMOKE/check_$1.log" 2>&1 ) || { tail -20 "$SMOKE/check_$1.log"; exit 1; }
  grep -E "^code:|update clock|^done" "$SMOKE/check_$1.log"
done

# 5. submit: one chain per network, each stage one task waiting (aftercorr) on the previous stage
export REPO SAVE_TO OLD_ROOT EXTRA_UPDATES=2000 INIT_ADAM=false
cd "$REPO"
for spec in "8 3" "28 5" "5 6" "6 7"; do
  set -- $spec; idx=$1; first=$2; dep=()
  for st in $(seq "$first" 10); do
    id=$(STAGE=$st sbatch --parsable --chdir="$REPO" --array="$idx" --time=12:00:00 ${dep[@]+"${dep[@]}"} "$LAUNCHER")
    id=${id%%;*}
    echo "index $idx stage $st -> $id"
    dep=(--dependency=aftercorr:"$id")
  done
done
squeue -u "$USER" -n MazeResume -o "%.18i %.10T %.20r %.12j" | head -30
