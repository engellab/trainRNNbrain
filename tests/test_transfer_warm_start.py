"""A transfer to a new task copies the trained network, keeps a fresh read-out, and starts its own clock.

WHY. The plasticity test (B4, docs/notes/agent-bplan.md) moves trained 3-bit flip-flop networks
to NBitFlipFlopHyper at k = 3: the same 3 inputs, 7 outputs instead of 3. Only the read-out is new,
so the input and recurrent weights must arrive EXACTLY as the source left them - otherwise "silent
units came back" could mean "the copy perturbed them". The ordinary warm start (`paths.init_from`)
stops on the read-out's shape, by design, and must keep doing so.

The transfer source goes under its own key, `paths.transfer_from`, never `paths.init_from`.
`common.warm_start_chain` follows `paths.init_from` to join losses along a chain of continuations;
had the source been recorded there, the floor of the new task would be fitted through the old
task's loss as well (and a kept-snapshot source, which has no TrainLosses.json, would make
`noisy_loss_series` return nothing at all).

CONTRACT, fixed before running:
  1. `load_warm_start(..., keep_fresh=("W_out",))` copies W_rec, W_inp and y_init exactly and leaves
     W_out equal to a fresh network of the same seed and the NEW shape.
  2. The ordinary warm start (keep_fresh empty) still copies every parameter, W_out included, and
     still raises on a shape mismatch. A misspelt keep_fresh name raises instead of copying all.
  3. End to end, through run_experiment.py and the source's own saved config, with lr = 0 so any
     difference is the copy's: the transfer run's saved W_rec / W_inp / y_init equal the source's
     bit for bit, its W_out equals the fresh draw, its trace starts at iteration 0, and its saved
     config holds transfer_from = the source, init_from = null, iter_offset = 0, the hyper task.
  4. The chain of a transfer run has length 1, so its noisy loss is its own alone.
  5. A transfer from a CONTINUATION's config (iter_offset and init_from inherited) fails loudly
     unless both are reset: with init_from inherited it is an ordinary warm start and stops on
     W_out's shape; with only iter_offset inherited it stops on the clock. Same for init_adam.
  6. A continuation of a transfer run (init_from = the transfer run, transfer_from inherited) is an
     ordinary warm start of the 7-output network: W_out carries over, the clock continues, and its
     chain is [transfer run, continuation] - it stops at the transfer, never reaching the source.
  7. A kept snapshot works as a source as-is (it holds a config and LastParams, no TrainLosses).

Run:  python tests/test_transfer_warm_start.py
"""

import glob
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omegaconf import OmegaConf

from trainRNNbrain.experiments_and_analysis import common
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.training.run_experiment import load_warm_start, write_checkpoint
from trainRNNbrain.utils import filter_kwargs

N = 20
RUN = os.path.join(REPO, "trainRNNbrain", "training", "run_experiment.py")
HYPER = ["task.taskname=NBitFlipFlopHyper",
         "task._target_=trainRNNbrain.tasks.TaskNBitFlipFlopHyper.TaskNBitFlipFlopHyper",
         "task.n_outputs=7"]


def _rnn(n_outputs, seed):
    """A small standard-architecture RNN_torch (ReLU, no Dale, 3 inputs) with the given read-out."""
    return RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                     io_nonnegativity=False, self_connections=True, equation_type="h",
                     n_inputs=3, n_outputs=n_outputs, seed=seed)


def _trained_like(rnn):
    """Overwrite every weight with values no initialisation produces, so a copy is unmistakable."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for _, p in rnn.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 3.0 + 1.0)
        rnn.y_init = torch.linspace(0.1, 2.0, N)
    return rnn


def _run(args, expect_ok=True):
    """Run run_experiment.py with the given Hydra arguments on CPU.

    Args:
        args: list of str, command-line arguments after the script path.
        expect_ok: bool, whether a zero exit status is expected.
    Returns:
        str: the combined stdout and stderr.
    """
    env = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, RUN, *args], capture_output=True, text=True, env=env,
                       cwd=REPO)
    out = p.stdout + p.stderr
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"run_experiment failed ({p.returncode}):\n{out[-3000:]}")
    if not expect_ok and p.returncode == 0:
        raise AssertionError(f"run_experiment succeeded but should have failed:\n{out[-2000:]}")
    return out


def _only(pattern):
    """The single path matching a glob pattern."""
    hits = sorted(glob.glob(pattern))
    assert len(hits) == 1, f"expected one match for {pattern}, got {hits}"
    return hits[0]


def _params(run_dir):
    """The LastParams arrays of a run or snapshot folder, as a dict."""
    z = np.load(_only(os.path.join(run_dir, "*LastParams*.npz")), allow_pickle=True)
    return {k: z[k] for k in z.files}


def _cfg(run_dir):
    """The saved config of a run or snapshot folder."""
    return OmegaConf.load(_only(os.path.join(run_dir, "*_config.yaml")))


def _from(run_dir, *overrides):
    """Hydra arguments that rebuild a run from its own saved config, as the launchers do."""
    c = _only(os.path.join(run_dir, "*_config.yaml"))
    return ["--config-path", os.path.dirname(c),
            f"--config-name={os.path.basename(c)[:-len('.yaml')]}", *overrides]


def _transfer_args(src, root, tag, *extra):
    """The transfer overrides the B4 launcher passes, plus test-specific extras."""
    return _from(src, *HYPER, "trainer.max_iter=6", "++trainer.iter_offset=0",
                 "++paths.init_from=null", f"++paths.transfer_from='{src}'",
                 "++paths.init_adam=false", "trainer.lr=0", "seed=4242",
                 f"paths.save_to={root}/data", f"experiment_tag={tag}", *extra)


def test_keep_fresh_copies_all_but_the_readout():
    """CONTRACT 1."""
    src = _trained_like(_rnn(3, seed=1))
    with tempfile.TemporaryDirectory() as d:
        write_checkpoint(d, OmegaConf.create({"trainer": {"max_iter": 1}}), src.get_params(), None,
                         "NBitFlipFlop", 1, 1)
        dst = _rnn(7, seed=2)
        load_warm_start(dst, d, keep_fresh=("W_out",))
    fresh = _rnn(7, seed=2)
    for nm in ("W_rec", "W_inp"):
        assert torch.equal(getattr(dst, nm), getattr(src, nm)), f"{nm} not copied exactly"
    assert torch.equal(dst.y_init, src.y_init.float()), "y_init not copied"
    assert dst.W_out.shape == (7, N), f"W_out is {tuple(dst.W_out.shape)}"
    assert torch.equal(dst.W_out, fresh.W_out), "W_out is not this network's own fresh draw"
    print("1 OK  W_rec, W_inp, y_init copied exactly; W_out (7, N) is the fresh draw")


def test_ordinary_warm_start_is_unchanged():
    """CONTRACT 2."""
    src = _trained_like(_rnn(3, seed=1))
    with tempfile.TemporaryDirectory() as d:
        write_checkpoint(d, OmegaConf.create({"trainer": {"max_iter": 1}}), src.get_params(), None,
                         "NBitFlipFlop", 1, 1)
        same = _rnn(3, seed=2)
        load_warm_start(same, d)
        for nm, p in src.named_parameters():
            assert torch.equal(getattr(same, nm), p), f"ordinary warm start did not copy {nm}"
        try:
            load_warm_start(_rnn(7, seed=2), d)
        except ValueError as e:
            assert "W_out" in str(e), e
        else:
            raise AssertionError("an ordinary warm start into a 7-output model did not raise")
        try:
            load_warm_start(_rnn(7, seed=2), d, keep_fresh=("Wout",))
        except KeyError as e:
            assert "Wout" in str(e), e
        else:
            raise AssertionError("a misspelt keep_fresh name was accepted")
    print("2 OK  ordinary warm start copies W_out too and still raises on its shape; typo raises")


def test_end_to_end():
    """CONTRACTS 3-7, through run_experiment.py on CPU at N = 20."""
    with tempfile.TemporaryDirectory() as root:
        # the source: a 3-bit flip-flop run with kept snapshots, then a continuation of it, so the
        # continuation's config carries iter_offset and init_from like the claim-1 200k runs
        _run(["model=rnn_relu_standard", "trainer=trainer_ptrack_freshbatch", "task=NBitFlipFlop",
              "task.n_inputs=3", "task.n_outputs=3", "task.batch_size=16", "model.equation_type=h",
              f"model.N={N}", "trainer.max_iter=30", "trainer.track_every=10",
              "trainer.store_participation_every=10", "trainer.checkpoint_every=10",
              "+trainer.checkpoint_keep=true", f"paths.checkpoint_to={root}/snaps",
              f"paths.save_to={root}/data", "++light_outputs=true", "seed=11",
              "trainer.trainer_tag=''", "experiment_tag=src"])
        young = _only(f"{root}/data/NBitFlipFlop_src/*/")
        _run(_from(young, "trainer.max_iter=20", "++trainer.iter_offset=auto",
                   f"++paths.init_from='{young}'", "++paths.init_adam=true", "seed=12",
                   "trainer.checkpoint_every=0", f"paths.save_to={root}/data",
                   "experiment_tag=src_c50"))
        old = _only(f"{root}/data/NBitFlipFlop_src_c50/*/")
        assert int(_cfg(old).trainer.iter_offset) == 30 and _cfg(old).paths.init_from

        # CONTRACT 5: a transfer that inherits the continuation's init_from is an ordinary warm
        # start and stops on W_out; one that inherits only its iter_offset stops on the clock
        args = [a for a in _transfer_args(old, root, "bad") if not a.startswith(
            ("++paths.init_from", "++trainer.iter_offset"))]
        out = _run(args, expect_ok=False)
        assert "'W_out' is (3, 20) in the checkpoint but (7, 20)" in out, out[-1500:]
        args = [a for a in _transfer_args(old, root, "bad") if not a.startswith("++trainer.iter_offset")]
        out = _run(args, expect_ok=False)
        assert "trainer.iter_offset is 30" in out, out[-1500:]
        out = _run(_transfer_args(old, root, "bad", "++paths.init_adam=true"), expect_ok=False)
        assert "paths.init_adam must be false" in out, out[-1500:]
        assert not glob.glob(f"{root}/data/NBitFlipFlopHyper_bad/*/"), "a failed transfer saved a run"
        print("5 OK  inherited init_from, iter_offset or init_adam each stop the transfer")

        # CONTRACT 3: the transfer, lr = 0 so the saved parameters ARE the copied ones
        _run(_transfer_args(old, root, "xfer"))
        xfer = _only(f"{root}/data/NBitFlipFlopHyper_xfer/*/")
        s, t, c = _params(old), _params(xfer), _cfg(xfer)
        for nm in ("W_rec", "W_inp", "y_init", "bias"):
            assert np.array_equal(s[nm], t[nm]), f"{nm} differs between source and transfer"
        rnn_seed = int(c.seed) + (65537 ** 3) % 7309          # run_experiment's seed for net i = 0
        model = OmegaConf.merge(c.model, {"n_inputs": 3, "n_outputs": 7})
        kw = {k: v for k, v in filter_kwargs(RNN_torch, OmegaConf.to_container(model)).items()
              if k != "_target_"}
        fresh = RNN_torch(**kw, seed=rnn_seed)
        assert t["W_out"].shape == (7, N), t["W_out"].shape
        assert np.array_equal(t["W_out"], fresh.W_out.detach().numpy()), "W_out is not the fresh draw"
        assert c.paths.transfer_from.rstrip("/") == old.rstrip("/"), c.paths.transfer_from
        assert c.paths.init_from is None, c.paths.init_from
        assert int(c.trainer.iter_offset) == 0 and int(c.trainer.max_iter) == 6
        assert c.task.taskname == "NBitFlipFlopHyper" and int(c.task.n_outputs) == 7
        with open(_only(os.path.join(xfer, "*ParticipationTrace.pkl")), "rb") as fh:
            tr = pickle.load(fh)
        assert tr["iters"][0] == 0 and tr["iters"][-1] < 6, tr["iters"]
        print("3 OK  transfer copies W_rec/W_inp/y_init/bias bit for bit, fresh (7, N) W_out, "
              "clock and trace from 0, transfer_from recorded")

        # CONTRACT 4: the chain stops at the transfer run
        chain = common.warm_start_chain(xfer)
        assert chain == [xfer.rstrip("/")], chain
        it, L = common.noisy_loss_series(xfer)
        assert len(L) == 6 and it[0] == 1, (len(L), it[:3])
        print("4 OK  warm_start_chain of a transfer run is the run alone; 6 losses, its own")

        # CONTRACT 6: continuing the transfer run is an ordinary warm start of the 7-output net
        _run(_from(xfer, "trainer.max_iter=4", "++trainer.iter_offset=auto",
                   f"++paths.init_from='{xfer}'", "++paths.init_adam=true", "seed=13",
                   f"paths.save_to={root}/data", "experiment_tag=xfer_c10"))
        top = _only(f"{root}/data/NBitFlipFlopHyper_xfer_c10/*/")
        u, ct = _params(top), _cfg(top)
        for nm in ("W_rec", "W_inp", "W_out"):
            assert np.array_equal(u[nm], t[nm]), f"continuation did not carry {nm} over"
        assert int(ct.trainer.iter_offset) == 6, ct.trainer.iter_offset
        assert common.warm_start_chain(top) == [xfer.rstrip("/"), top.rstrip("/")]
        assert len(common.noisy_loss_series(top)[1]) == 10
        print("6 OK  a continuation of a transfer carries W_out, continues the clock at 6, and its "
              "chain stops at the transfer run")

        # CONTRACT 7: a kept snapshot of the young source works as-is
        snap = _only(f"{root}/snaps/NBitFlipFlop_src/s11/it0000020/")
        _run(_transfer_args(snap, root, "xfer_snap"))
        xs = _only(f"{root}/data/NBitFlipFlopHyper_xfer_snap/*/")
        for nm in ("W_rec", "W_inp", "y_init"):
            assert np.array_equal(_params(snap)[nm], _params(xs)[nm]), f"{nm} differs (snapshot)"
        assert not np.array_equal(_params(snap)["W_rec"], s["W_rec"]), "snapshot equals the 50-it net"
        assert common.warm_start_chain(xs) == [xs.rstrip("/")]
        print("7 OK  a kept snapshot transfers exactly and its transfer's chain is the run alone")


if __name__ == "__main__":
    test_keep_fresh_copies_all_but_the_readout()
    test_ordinary_warm_start_is_unchanged()
    test_end_to_end()
    print("all transfer contracts hold")
