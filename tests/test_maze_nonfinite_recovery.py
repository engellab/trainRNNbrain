"""Checks for TrainerPPO's recovery from non-finite values (14 Nov 2026, branch wp/maze-finite; notebook
entry docs/project_trajectory/2026-11-14_1605_maze-nonfinite-recovery.md), on the small test maze:

1. an optimizer step whose loss is finite but whose gradient is not is skipped (skipped_grad), and the
   weights and Adam's moments stay as they were; without the guard the same step writes NaN into
   every weight and both moments (torch's clip_grad_norm_ multiplies every gradient by NaN);
2. weights that write NaN after an update are never kept as a snapshot, and the run restores the
   last verified one instead (the N = 1,000 seed 2 stop: its snapshot was the unstable update);
3. a poisoned optimizer state (NaN moments, the N = 1,000 seed 6 and N = 500 seed 2 stops) is
   replaced by the snapshot's own moments on a restore, the learning rate is the current one, and
   the run ends with finite weights and moments, the event counted in the progress record;
4. a restored snapshot is not restored again until a newer one is verified; the next older one is;
   with none left the run stops.
Run: python tests/test_maze_nonfinite_recovery.py   (CPU, a few seconds)
"""
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_rl_maze import make_task, make_trainer


def nan_gradient_hook(grad):
    """A tensor hook that turns a gradient into NaN (a finite loss with a non-finite gradient)."""
    return grad * float("nan")


def all_finite(trainer):
    """Whether every weight and every floating-point optimizer state tensor of `trainer` is finite."""
    tensors = list(trainer.RNN.parameters())
    for st in trainer.optimizer.state.values():
        tensors += [v for v in st.values() if torch.is_tensor(v) and v.is_floating_point()]
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def small_run(**hp):
    """A TrainerPPO on the test maze set up for run_training: N = 32, no critic, 4 episodes of 10
    decisions, probes off but the first, checkpoint_every 2; `hp` overrides."""
    torch.manual_seed(5)
    args = dict(critic=False, n_episodes=4, T=10, n_updates=6, probe_every=1000, probe_episodes=4,
                checkpoint_every=2, warmup_updates=0)
    args.update(hp)
    return make_trainer(make_task(), **args)


def test_nonfinite_gradient_is_skipped():
    """Check 1: a NaN gradient under a finite loss skips the step; the unguarded step would poison
    every weight and both Adam moments."""
    trainer = small_run(n_epochs=1, n_minibatches=2)
    out = trainer.rollout(4, 10)
    adv, ret = trainer.mc_advantages(out["rewards"], out["alive"], 0.99)
    trainer.ppo_update(out, adv, ret)                       # Adam's moments exist from here on
    before = {k: v.detach().clone() for k, v in trainer.RNN.state_dict().items()}
    moments = {id(p): {k: v.clone() for k, v in st.items()} for p, st in trainer.optimizer.state.items()}
    handle = trainer.RNN.W_rec.register_hook(nan_gradient_hook)
    stats = trainer.ppo_update(out, adv, ret)
    assert stats["skipped"] == 0 and stats["skipped_grad"] == 2, stats
    assert all(torch.equal(v, before[k]) for k, v in trainer.RNN.state_dict().items())
    for p, st in trainer.optimizer.state.items():
        assert all(torch.equal(v, moments[id(p)][k]) for k, v in st.items())
    # what 1a80f5e did with this minibatch: clip, then step
    trainer.optimizer.zero_grad()
    states, means, _ = trainer.sequence(out["u"], out["noise"])
    loss = -(trainer.log_prob(out["actions"], means[:, :10]) * adv).mean()
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.RNN.parameters(), trainer.max_grad_norm)
    trainer.optimizer.step()
    handle.remove()
    assert all(bool(torch.isnan(p).all()) for p in trainer.RNN.parameters())
    assert all(bool(torch.isnan(st["exp_avg"]).all()) for st in trainer.optimizer.state.values())


def test_unstable_weights_never_become_a_snapshot():
    """Check 2: the weights after update 2 are made non-finite (as if that update had diverged);
    update 2 is a checkpoint, but its state is not kept, and update 3 restores the start state."""
    trainer = small_run()
    real_update = trainer.ppo_update
    calls = []

    def diverging_update(batch, adv, ret):
        """ppo_update, then NaN into W_rec after the second call."""
        stats = real_update(batch, adv, ret)
        calls.append(1)
        if len(calls) == 2:
            with torch.no_grad():
                trainer.RNN.W_rec[0, 0] = float("nan")
        return stats

    trainer.ppo_update = diverging_update
    rnn, progress, probes, _ = trainer.run_training()
    rec = progress[2]                                        # update 3
    assert rec["restored"] == trainer.rollout_tries and rec["snapshot_restores"] == 1
    assert rec["snapshot_from"] == 0, rec
    assert all(r["snapshot_restores"] == 0 for i, r in enumerate(progress) if i != 2)
    assert 2 not in [s["update"] for s in trainer.snapshots]
    assert all(torch.isfinite(p).all() for p in rnn.parameters())


def test_poisoned_moments_are_replaced_on_restore():
    """Check 3: at the checkpoint after update 4 the weights and Adam's moments are made NaN (the
    state of the seed 6 and N = 500 seed 2 runs); update 5 restores the verified snapshot of update 2
    with its own moments and the current learning rate, and the run ends finite."""
    trainer = small_run(warmup_updates=10)
    lr_base = trainer.lr

    def poison(updates_done, params):
        """checkpoint_fn: after update 4, NaN into W_rec and into both moments of every parameter."""
        if updates_done == 4:
            with torch.no_grad():
                trainer.RNN.W_rec[0, 0] = float("nan")
                for st in trainer.optimizer.state.values():
                    st["exp_avg"].fill_(float("nan"))
                    st["exp_avg_sq"].fill_(float("nan"))

    trainer.checkpoint_fn = poison
    rnn, progress, probes, _ = trainer.run_training()
    rec = progress[4]                                        # update 5
    assert rec["snapshot_restores"] == 1 and rec["snapshot_from"] == 2, rec
    assert abs(rec["lr"] - lr_base * 5 / 10) < 1e-12
    assert all(abs(g["lr"] - lr_base * 6 / 10) < 1e-12 for g in trainer.optimizer.param_groups)
    assert all_finite(trainer)
    assert [s["update"] for s in trainer.snapshots] == [0, 2]   # update 4's candidate was not kept


def test_used_snapshots_are_skipped_then_the_run_stops():
    """Check 4: snapshots of updates 0 (finite), 1 and 2 (NaN weights): with the current weights NaN
    the newest two fail and update 0 is restored; asked again, no unused snapshot is left and the
    run stops; once the marks are cleared (a newer snapshot verified), update 0 is available again."""
    trainer = small_run()
    good = trainer.capture_state(0)
    with torch.no_grad():
        trainer.RNN.W_rec[0, 0] = float("nan")
    trainer.snapshots = [good, trainer.capture_state(1), trainer.capture_state(2)]
    batch, discarded, restores, snapshot_from = trainer.safe_rollout(4, 10)
    assert (discarded, restores, snapshot_from) == (3 * trainer.rollout_tries, 3, 0)
    assert torch.isfinite(batch["logp"]).all()
    with torch.no_grad():
        trainer.RNN.W_rec[0, 0] = float("nan")
    try:
        trainer.safe_rollout(4, 10)
    except RuntimeError as err:
        assert "stopped" in str(err)
    else:
        raise AssertionError("safe_rollout did not stop with every snapshot used")
    for snap in trainer.snapshots:
        snap["used"] = False
    with torch.no_grad():
        trainer.RNN.W_rec[0, 0] = float("nan")
    assert trainer.safe_rollout(4, 10)[3] == 0


def main():
    """Run every check and print one line each."""
    for fn in (test_nonfinite_gradient_is_skipped, test_unstable_weights_never_become_a_snapshot,
               test_poisoned_moments_are_replaced_on_restore, test_used_snapshots_are_skipped_then_the_run_stops):
        fn()
        print(f"ok  {fn.__name__}")


if __name__ == "__main__":
    main()
