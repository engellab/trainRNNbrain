## 2026-11-05 12:50 — Uniform-settings cells: Della did not answer in 4 hours, submission pending (issue #6)

Follow-up to `2026-11-05_0903_uniform-settings-3bff-wd-cddm-rescale.md`. Della was polled from the
Mac every 10 minutes from 08:54 to 12:50 (24 tries, `ssh -o BatchMode=yes -o ConnectTimeout=15
della`); every try timed out at the connection, so neither array was submitted. The 3BFF weight-decay
cells (wd 1e-5 and 1e-4, 6 runs) and the CDDM rescaling cell (3 runs) still have to be submitted
with the commands in that entry's "Submission" block, unchanged: a detached worktree
`~/trainRNNbrain_wd` at 99530cf, then `TASK=ff ... --array=4-9 slurm/b0_wd.slurm` and
`TASK=cddm ... --array=9-11 slurm/SilentReLU_target_ladder_della.slurm`, each with
`REPO=$HOME/trainRNNbrain_wd --chdir=$HOME/trainRNNbrain_wd --time=08:00:00 --constraint=nomig`.
The pre-registered read-out in that entry stands as written.
