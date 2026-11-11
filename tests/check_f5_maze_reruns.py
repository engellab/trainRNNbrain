"""Check that every Maze read-out dropout and rescaling network drawn in Fig. 5a,b sits at its fz_reruns.json value.

Runs fig_paper_F5_tasks.py with paperstyle.save wrapped, takes the finished figure, and reads the data of
every marker artist drawn in the Maze colour in panels a (fold change) and b bottom (success rate). For each
of the six rerun arms it predicts, from data/round19/fz_reruns.json alone and the caption's drawing rule, the
values that must be drawn: the mean over the seeds read at update 2,000 (fold change = active / 430, the
controls' mean; success), a cross at the value of each such seed without a learned point, and one cross at 0
per stopped seed in panel a. Pass rule (second version, set before its run; the first version wrongly
expected a marker per learned network, which the figure does not draw): every predicted value is found
among the drawn Maze marker values to 1e-9, panel a holds at least as many Maze markers at 0 as there are
stopped seeds, and no pre-correction mean (entry 2026-11-09_1513 values) is drawn unless it equals a
predicted value.

Run (repository root): CUDA_VISIBLE_DEVICES= PYTHONPATH=.:trainRNNbrain/experiments_and_analysis \
    python tests/check_f5_maze_reruns.py   (round 22, dead-relu-paper#228)
"""
import contextlib
import io
import json
import runpy
import sys

import matplotlib.colors as mc
import numpy as np

import paperstyle as ps

ARMS = ("dropout0.05_beta4", "dropout0.2_beta4", "rescale_tgt4", "rescale", "rescale_mild", "rescale_few")
OLD_MEAN_ACTIVE = ((455 + 373 + 420) / 3, (413 + 461 + 436) / 3, (352 + 197) / 2, (314 + 321) / 2,
                   (255 + 231 + 234) / 3)   # pre-correction means (entry 2026-11-09_1513)


def drawn_values(ax, colour):
    """All y values of marker artists in `ax` whose marker colour (face or edge) is `colour`.

    Args:
        ax: matplotlib axes; colour: str, a hex colour.
    Returns:
        list of float.
    """
    out = []
    target = mc.to_rgb(colour)
    for ln in ax.get_lines():
        if ln.get_marker() in (None, "None", "", " "):
            continue
        cols = [ln.get_markeredgecolor(), ln.get_markerfacecolor(), ln.get_color()]
        if any(c not in (None, "none") and np.allclose(mc.to_rgb(c), target) for c in cols):
            out += [float(v) for v in np.atleast_1d(ln.get_ydata())]
    return out


def main():
    """Run the figure script, compare drawn values with predictions, print a per-network table and PASS/FAIL."""
    keep = {}
    ps.save = lambda fig, name, *a, **k: keep.setdefault("fig", fig)
    sys.argv = ["fig_paper_F5_tasks.py"]
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path("trainRNNbrain/experiments_and_analysis/fig_paper_F5_tasks.py", run_name="__main__")
    fig = keep["fig"]
    ax_a = next(a for a in fig.axes if a.get_ylabel() == "fold change")
    ax_s = next(a for a in fig.axes if a.get_ylabel() == "success rate")
    fold_drawn = drawn_values(ax_a, ps.TASK_MAZE)
    succ_drawn = drawn_values(ax_s, ps.TASK_MAZE)
    fz = {a["key"]: a for a in json.load(open("data/round19/fz_reruns.json"))["arms"]}
    ok = True
    pred_f, pred_s, n_stop = [], [], 0
    for key in ARMS:
        read = [s["at2000"] for s in fz[key]["seeds"] if s["at2000"]["reached"]]
        n_stop += len(fz[key]["seeds"]) - len(read)
        mf, ms = float(np.mean([p["active"] for p in read])) / 430.0, float(np.mean([p["success"] for p in read]))
        want = [("mean", mf, ms)] + [(f"cross s{s['seed']}", s["at2000"]["active"] / 430.0, s["at2000"]["success"])
                                     for s in fz[key]["seeds"] if s["at2000"]["reached"] and not s["at2000"]["learned"]]
        for tag, f, sv in want:
            hf = any(abs(v - f) < 1e-9 for v in fold_drawn)
            hs = any(abs(v - sv) < 1e-9 for v in succ_drawn)
            ok &= hf and hs
            pred_f.append(f)
            pred_s.append(sv)
            print(f"{key:>18} {tag:>9}: fold {f:.4f} drawn {hf}; success {sv:.4f} drawn {hs}")
        print(f"{key:>18}   stopped: {len(fz[key]['seeds']) - len(read)}")
    zeros = sum(abs(v) < 1e-12 for v in fold_drawn)
    print(f"Maze markers at 0 in panel a: {zeros}; stopped seeds of these arms: {n_stop}")
    ok &= zeros >= n_stop
    stale = [m for m in OLD_MEAN_ACTIVE if any(abs(v - m / 430.0) < 1e-9 for v in fold_drawn)
             and not any(abs(p - m / 430.0) < 1e-9 for p in pred_f)]
    print("pre-correction means still drawn in panel a:", stale)
    ok &= not stale
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
