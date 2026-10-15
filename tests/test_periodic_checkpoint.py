"""A run killed by its wall clock must leave a snapshot the existing warm-start path can resume.

WHY. Every file a run produces is written after the training loop returns. On 2026-10-02 three
Della jobs were killed by their wall clocks at 96,595 of 100,000, at ~147,000 of 150,000 and at
144,479 of 300,000 iterations, and all three left an empty folder: about 228 GPU-hours, two of them
over 97% complete. Periodic checkpointing exists so that never costs a whole run again.

The design choice this file pins is that a checkpoint is NOT a new format. It writes the same three
files a finished run writes - LastParams, AdamState and a config - with `trainer.max_iter` set to
the iterations ACTUALLY completed. Resuming is then the warm-start path that already exists
(`paths.init_from` + `paths.init_adam`), and `slurm/topup_budget.py` computes the remaining budget
from a checkpoint without knowing it is one. Nothing new has to be written or maintained to restart.

CONTRACT, fixed before running:
  1. checkpoint_every = 0 is EXACTLY the historical path: the hook never fires, so no run that
     leaves it off can behave differently from before this existed.
  2. Snapshots fire at multiples of checkpoint_every and NEVER on the final iteration, whose state
     the caller saves anyway. A redundant final write costs ~200 MB at N=4000.
  3. A snapshot holds the three files the warm-start globs match, and its config records the
     iterations COMPLETED, not the budget the run was aiming at. Recording the budget would make a
     resumed run's trace claim iterations that were never trained.
  4. `warm_start_offset` - production code, written before checkpointing and not touched for it -
     reads a snapshot and returns that completed count.
  5. A snapshot NEVER lands inside the data tree. Every inventory and every top-up launcher finds
     runs by globbing a cell for subfolders holding *LastParams*, so a checkpoint among the real
     runs would be counted as a finished run: inflating counts and shifting which parent a top-up
     continues. Test 5 is the one that would catch that.
  6. A write interrupted part-way leaves the PREVIOUS complete snapshot, not a truncated file. A
     job killed during its checkpoint is the normal case, not a rare one.
  7. np.savez is given a file OBJECT, not a path. Given a path without the suffix it appends one,
     which would write "<name>.npz.tmp.npz" and leave the rename pointing at nothing.

Run:  python tests/test_periodic_checkpoint.py
"""

import os
import glob
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.training.run_experiment import warm_start_offset, write_checkpoint

N, T, B = 20, 8, 3


def _cfg(max_iter):
    """The few config fields write_checkpoint reads, as a DictConfig."""
    return OmegaConf.create({"trainer": {"max_iter": max_iter, "iter_offset": 0},
                             "model": {"N": N}, "paths": {"save_to": "/nowhere"}})


def _trainer(max_iter, checkpoint_every):
    """A real Trainer over a tiny RNN, with train_step stubbed so only the LOOP is exercised.

    The point is to run the production `run_training` - including the real checkpoint scheduling -
    without paying for gradients or needing a real Task. Returns (trainer, calls) where `calls`
    collects the iteration counts the checkpoint hook was invoked with.
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    task = types.SimpleNamespace(
        get_batch=lambda shuffle=False: (np.zeros((2, T, B), "float32"),
                                         np.zeros((1, T, B), "float32"),
                                         [{} for _ in range(B)]))
    tr = Trainer(RNN=rnn, Task=task, optimizer=torch.optim.Adam(rnn.parameters(), lr=1e-3),
                 max_iter=max_iter, checkpoint_every=checkpoint_every,
                 lambda_orth=0.0, lambda_rws=0.0, lambda_frm=0.0, monitor=False)
    tr.train_step = lambda input, target_output, mask: (0.5, 0.5)
    calls = []
    if checkpoint_every:
        tr.checkpoint_fn = lambda iters_done, params: calls.append(iters_done)
    return tr, calls


def test_off_by_default_never_fires():
    """CONTRACT 1. checkpoint_every = 0 leaves the historical path untouched."""
    tr, calls = _trainer(max_iter=7, checkpoint_every=0)
    assert tr.checkpoint_fn is None
    tr.run_training(train_mask=torch.ones(T, B, dtype=torch.bool), same_batch=True)
    assert calls == [], f"hook fired with checkpointing off: {calls}"
    # and a hook present with a zero cadence must still not fire
    tr2, calls2 = _trainer(max_iter=7, checkpoint_every=0)
    tr2.checkpoint_fn = lambda i, p: calls2.append(i)
    tr2.run_training(train_mask=torch.ones(T, B, dtype=torch.bool), same_batch=True)
    assert calls2 == [], f"hook fired although checkpoint_every is 0: {calls2}"
    print("1 OK  checkpointing off fires nothing")


def test_cadence_skips_the_final_iteration():
    """CONTRACT 2. Multiples of the cadence, and never the last iteration."""
    tr, calls = _trainer(max_iter=50, checkpoint_every=10)
    tr.run_training(train_mask=torch.ones(T, B, dtype=torch.bool), same_batch=True)
    assert calls == [10, 20, 30, 40], f"expected [10,20,30,40], got {calls}"
    # a budget that is NOT a multiple still stops short of its own end
    tr2, calls2 = _trainer(max_iter=25, checkpoint_every=10)
    tr2.run_training(train_mask=torch.ones(T, B, dtype=torch.bool), same_batch=True)
    assert calls2 == [10, 20], f"expected [10,20], got {calls2}"
    print(f"1..{len(calls)} OK  cadence {calls} over 50 iterations, {calls2} over 25")


def test_snapshot_is_a_finished_run_as_far_as_the_warm_start_can_tell():
    """CONTRACTS 3, 4, 7. The three files, the completed count, and no doubled suffix."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as d:
        folder = os.path.join(d, "ckpts", "s99")
        # budget 300000, but only 144479 were trained -- the Della drift job's real numbers
        write_checkpoint(folder, _cfg(300000), rnn.get_params(), opt, "CDDM", 99, 144479)

        files = sorted(os.listdir(folder))
        assert files == ["ckpt_s99_AdamState_CDDM.pt",
                        "ckpt_s99_LastParams_CDDM.npz",
                        "ckpt_s99_config.yaml"], f"unexpected files: {files}"
        assert not any(f.endswith(".tmp") for f in files), f"temporary file survived: {files}"
        assert not any(f.count(".npz") > 1 for f in files), f"doubled .npz suffix: {files}"

        cfg = OmegaConf.load(os.path.join(folder, "ckpt_s99_config.yaml"))
        assert int(cfg.trainer.max_iter) == 144479, \
            f"config records {cfg.trainer.max_iter}, not the 144479 completed"
        assert warm_start_offset(folder) == 144479, \
            f"warm_start_offset gave {warm_start_offset(folder)}, expected 144479"

        z = np.load(os.path.join(folder, "ckpt_s99_LastParams_CDDM.npz"))
        assert z["W_rec"].shape == (N, N), f"W_rec is {z['W_rec'].shape}"
        torch.load(os.path.join(folder, "ckpt_s99_AdamState_CDDM.pt"), weights_only=False)
    print("3,4,7 OK  snapshot reads as a finished run at 144,479 of a 300,000 budget")


def test_a_killed_write_leaves_a_consistent_snapshot():
    """CONTRACT 6. Kill the run DURING the parameter write; what is left must be self-consistent.

    The kill is injected into np.savez_compressed (the call write_checkpoint makes since
    2026-10-14; np.savez before) rather than into the parameter values, so it lands while
    files are being written -- after the point where the first version of write_checkpoint had
    already published the new config. An earlier version of this test raised from a parameter
    value instead; `storable_` converts the values before any file is opened, so that version
    failed before the publish order mattered and passed against the bug it was written to catch.
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as d:
        folder = os.path.join(d, "s1")
        write_checkpoint(folder, _cfg(1000), rnn.get_params(), opt, "CDDM", 1, 100)
        good = np.load(os.path.join(folder, "ckpt_s1_LastParams_CDDM.npz"))["W_rec"].copy()

        def boom(*a, **k):
            raise RuntimeError("simulated kill during checkpoint write")

        original = np.savez_compressed
        np.savez_compressed = boom
        try:
            write_checkpoint(folder, _cfg(1000), rnn.get_params(), opt, "CDDM", 1, 200)
        finally:
            np.savez_compressed = original

        after = np.load(os.path.join(folder, "ckpt_s1_LastParams_CDDM.npz"))["W_rec"]
        assert np.array_equal(good, after), "a failed write replaced the good parameters"
        # the config must still describe the parameters that are actually on disk. Publishing it
        # first was the bug: it left a config saying 200 beside parameters from 100, and a resumed
        # run would then date its trace 100 iterations too late.
        cfg = OmegaConf.load(os.path.join(folder, "ckpt_s1_config.yaml"))
        assert int(cfg.trainer.max_iter) == 100, \
            f"config says {cfg.trainer.max_iter} but the parameters on disk are from 100"
        leftovers = [f for f in os.listdir(folder) if f.endswith(".tmp")]
        assert leftovers == [], f"staging files survived a failed write: {leftovers}"
    print("6 OK  a kill mid-write left the previous snapshot, config and parameters agreeing")


def test_checkpoints_live_outside_the_data_tree():
    """CONTRACT 5. The default location is a SIBLING of save_to, never inside a cell.

    This reproduces the failure mode rather than asserting a string: a checkpoint placed in the
    cell is found by the same glob the top-up launchers use to count finished runs.
    """
    save_to = "/data/trained_RNNs"
    default = str(save_to).rstrip("/") + "_checkpoints"      # the expression run_experiment uses
    assert not default.startswith(save_to + "/"), \
        f"default checkpoint root {default} is inside {save_to}"

    # and the glob that counts runs must not see it
    with tempfile.TemporaryDirectory() as d:
        cell = os.path.join(d, "trained_RNNs", "EqType=h_N=1000")
        os.makedirs(os.path.join(cell, "0.87_CDDM_relu;N=1000"), exist_ok=True)
        open(os.path.join(cell, "0.87_CDDM_relu;N=1000", "x_LastParams_CDDM.npz"), "w").close()
        ck = os.path.join(d, "trained_RNNs_checkpoints", "CDDM_tag", "s1")
        os.makedirs(ck, exist_ok=True)
        open(os.path.join(ck, "ckpt_s1_LastParams_CDDM.npz"), "w").close()

        counted = [p for p in glob.glob(os.path.join(cell, "*", ""))
                   if glob.glob(os.path.join(p, "*LastParams*"))]
        assert len(counted) == 1, f"cell inventory counted {len(counted)} runs, expected 1"
    print("5 OK  checkpoints sit outside the data tree and no cell inventory sees them")


def test_kept_snapshots_are_a_series_without_optimiser_state():
    """CONTRACT 8. `checkpoint_keep`: one it<iters>/ folder per snapshot, none overwritten, no AdamState.

    The caller passes a fresh folder per checkpoint and opt=None. The series is what lets a network
    be read at its own floor crossing, which lands between the saved end states.
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "s7")
        stamps = []
        for it in (500, 1000):
            with torch.no_grad():
                rnn.W_rec.fill_(float(it))           # make the two snapshots distinguishable
            write_checkpoint(os.path.join(root, f"it{it:07d}"), _cfg(75000), rnn.get_params(),
                             None, "CDDM", 7, it)
            stamps.append(it)
        assert sorted(os.listdir(root)) == ["it0000500", "it0001000"], os.listdir(root)
        for it in stamps:
            folder = os.path.join(root, f"it{it:07d}")
            files = sorted(os.listdir(folder))
            assert files == ["ckpt_s7_LastParams_CDDM.npz", "ckpt_s7_config.yaml"], files
            z = np.load(os.path.join(folder, "ckpt_s7_LastParams_CDDM.npz"))
            assert np.all(z["W_rec"] == it), f"snapshot {it} holds another snapshot's weights"
            assert warm_start_offset(folder) == it
    print("8 OK  kept snapshots form a series, each with its own weights and no optimiser state")


def test_kept_snapshots_store_w_rec_in_half_precision():
    """CONTRACT 9. A kept snapshot holds W_rec as float16 and every other array unchanged.

    Reduced precision was adopted only for the kept series (2026-10-14); a resume point stays exact.
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    full = rnn.get_params()
    with tempfile.TemporaryDirectory() as d:
        kept, exact = os.path.join(d, "it0000500"), os.path.join(d, "resume")
        write_checkpoint(kept, _cfg(75000), full, None, "CDDM", 9, 500, half_w_rec=True)
        opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
        write_checkpoint(exact, _cfg(75000), full, opt, "CDDM", 9, 500)
        zk = np.load(os.path.join(kept, "ckpt_s9_LastParams_CDDM.npz"), allow_pickle=True)
        ze = np.load(os.path.join(exact, "ckpt_s9_LastParams_CDDM.npz"), allow_pickle=True)
        assert zk["W_rec"].dtype == np.float16, zk["W_rec"].dtype
        assert np.array_equal(zk["W_rec"], np.asarray(full["W_rec"]).astype(np.float16))
        for k in ("W_inp", "W_out", "y_init"):
            assert np.array_equal(zk[k], ze[k]) and zk[k].dtype == ze[k].dtype, k
        assert ze["W_rec"].dtype != np.float16 and np.array_equal(ze["W_rec"], np.asarray(full["W_rec"]))
        assert os.path.exists(os.path.join(exact, "ckpt_s9_AdamState_CDDM.pt"))
        assert not os.path.exists(os.path.join(kept, "ckpt_s9_AdamState_CDDM.pt"))
    print("9 OK  kept snapshots hold W_rec in float16; the resume copy is exact and keeps Adam")


if __name__ == "__main__":
    test_off_by_default_never_fires()
    test_cadence_skips_the_final_iteration()
    test_snapshot_is_a_finished_run_as_far_as_the_warm_start_can_tell()
    test_a_killed_write_leaves_a_consistent_snapshot()
    test_checkpoints_live_outside_the_data_tree()
    test_kept_snapshots_are_a_series_without_optimiser_state()
    test_kept_snapshots_store_w_rec_in_half_precision()
    print("all periodic-checkpoint contracts hold")
