"""Checks for TaskCDDMHyper (trainRNNbrain/tasks/TaskCDDMHyper.py), the new task of the CDDM transfer
test (G18, round 18, 7 Nov 2026): its inputs and condition list equal TaskCDDM's for the same grid;
output channels 0-1 equal TaskCDDM's 2-output target; channels 2-4 equal sign(motion), sign(colour) and
their product in the decision window and 0 outside it, computed here from the condition dicts with
np.sign, independently of the task's own code path; a wrong output count is refused; and
configs/task/CDDMHyper.yaml composes to configs/task/CDDM.yaml with only taskname, _target_ and
n_outputs changed.
Run: python tests/test_cddm_hyper.py   (CPU, seconds)
"""
import os
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from trainRNNbrain.tasks.TaskCDDM import TaskCDDM
from trainRNNbrain.tasks.TaskCDDMHyper import TaskCDDMHyper

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = dict(n_steps=30, n_inputs=6, cue_on=0, cue_off=30, stim_on=10, stim_off=30, dec_on=20, dec_off=30,
            coherences=[-1.0, -0.25, 0.0, 0.125, 1.0])


def test_targets_against_independent_construction():
    """Inputs, conditions and channels 0-1 equal TaskCDDM's; channels 2-4 equal the signs and their
    product from the condition dicts inside [dec_on - 1, dec_off) and 0 elsewhere."""
    hyper = TaskCDDMHyper(n_outputs=5, **GRID)
    plain = TaskCDDM(n_outputs=2, **GRID)
    inp_h, tgt_h, cond_h = hyper.get_batch()
    inp_p, tgt_p, cond_p = plain.get_batch()
    n_trials = 2 * len(GRID["coherences"]) ** 2
    assert inp_h.shape == (6, 30, n_trials) and tgt_h.shape == (5, 30, n_trials)
    assert np.array_equal(inp_h, inp_p) and cond_h == cond_p
    assert np.array_equal(tgt_h[:2], tgt_p)
    window = np.zeros(30, dtype=bool)
    window[GRID["dec_on"] - 1:GRID["dec_off"]] = True
    for b, cond in enumerate(cond_h):
        m, c = np.sign(cond["motion_coh"]), np.sign(cond["color_coh"])
        for ch, val in ((2, m), (3, c), (4, m * c)):
            assert np.all(tgt_h[ch, window, b] == val), (b, ch, cond)
            assert np.all(tgt_h[ch, ~window, b] == 0), (b, ch, cond)
    # the inputs carry the coherences the conditions name (motion on rows 2-3, colour on rows 4-5)
    for b, cond in enumerate(cond_h):
        t = GRID["stim_on"]
        assert np.isclose(inp_h[2, t, b] - inp_h[3, t, b], cond["motion_coh"])
        assert np.isclose(inp_h[4, t, b] - inp_h[5, t, b], cond["color_coh"])


def test_output_count_refused():
    """n_outputs other than 5 raises ValueError."""
    try:
        TaskCDDMHyper(n_outputs=2, **GRID)
    except ValueError:
        return
    raise AssertionError("n_outputs=2 was accepted")


def test_config_is_cddm_with_three_keys_swapped():
    """task=CDDMHyper composes to task=CDDM except taskname, _target_ and n_outputs."""
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        a = OmegaConf.to_container(compose("base", overrides=["task=CDDM"]).task)
        b = OmegaConf.to_container(compose("base", overrides=["task=CDDMHyper"]).task)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert diff == ["_target_", "n_outputs", "taskname"], diff
    assert b["n_outputs"] == 5 and b["_target_"] == "trainRNNbrain.tasks.TaskCDDMHyper.TaskCDDMHyper"


def main():
    """Run every check and print one line each."""
    for fn in (test_targets_against_independent_construction, test_output_count_refused,
               test_config_is_cddm_with_three_keys_swapped):
        fn()
        print(f"ok: {fn.__name__}")


if __name__ == "__main__":
    main()
