"""Does dropout keep units alive? 3-bit flip-flop, N=1000, 150k iterations.

Pre-registered read-out of `slurm/SilentReLU_flipflop_dropout_spock.slurm` (job 6201944, the FIRST
run in which dropout actually reached the gradient; every earlier dropout run is void). Eight
dropout nets = {none, rws, frm, both} x {mute, dead}, one seed each, against the no-dropout
reference cells at the SAME k, N and iteration (3 seeds each):
    none -> NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=500000
    rws  -> NBitFlipFlop_std_pen/EqType=h_k=3_N=1000_pen=rws
    frm  -> NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=frm_iters=400000
    both -> NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=both_iters=400000
Reference runs trained past 150k are read BACKWARD to 150k, never at their own end.

Both silence criteria are reported (scale-free and the flip-flop-calibrated absolute 4e-2), plus
`loss_clean_train`, the noise-free task loss probed with dropout OFF, so the dropout nets' loss is
their no-dropout performance and is directly comparable to the references.

Decision rule, fixed before the numbers were seen: "dropout keeps units alive" := the dropout net's
live count exceeds its reference mean by more than 3 reference sd under BOTH criteria.

Usage: python flipflop_dropout_readout.py [<trained_RNNs root>]
"""
import glob
import os
import pickle
import sys

import numpy as np

from trainRNNbrain.experiments_and_analysis.common import DATA_DIR, SILENT_FLIPFLOP, active_count

READ_AT = 150000
DROP_SUB = "NBitFlipFlop_std_dropout"
REFS = {  # penalty arm -> reference cell folder (relative to the trained_RNNs root)
    "none": "NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=500000",
    "rws":  "NBitFlipFlop_std_pen/EqType=h_k=3_N=1000_pen=rws",
    "frm":  "NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=frm_iters=400000",
    "both": "NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=both_iters=400000",
}


def read_net(net_dir):
    """Read one trained network at READ_AT iterations.

    Args:
        net_dir: per-network folder holding *_ParticipationTrace.pkl.
    Returns:
        (live_scalefree, live_abs4e-2, clean_loss, r2) as (int, int, float, float), or None if the
        trace does not reach READ_AT. r2 is the folder-name prefix (validation r^2 at the end).
    """
    f = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
    if not f:
        return None
    with open(f[0], "rb") as fh:
        tr = pickle.load(fh)
    pit = np.asarray(tr["participation_iters"])
    j = np.flatnonzero(pit <= READ_AT)
    if not j.size or pit[j[-1]] < READ_AT - 200:
        return None
    p = np.asarray(tr["participation"][j[-1]], dtype=float)

    it = np.asarray(tr["iters"])
    loss = np.asarray(tr["metrics"]["loss_clean_train"], dtype=float)
    i = np.flatnonzero(it[:len(loss)] <= READ_AT)
    # median of the last 50 probes: the clean loss is deterministic but still ripples batch to batch
    clean = float(np.median(loss[i[-50:]])) if i.size else float("nan")
    try:
        r2 = float(os.path.basename(f[0]).split("_")[0])
    except ValueError:
        r2 = float("nan")
    return active_count(p, "scalefree"), active_count(p, SILENT_FLIPFLOP), clean, r2


def collect(cell_dir):
    """Read every network in a cell folder. Returns an array (n_nets, 4) of read_net rows."""
    rows = [read_net(d) for d in sorted(glob.glob(os.path.join(cell_dir, "*", "")))]
    return np.array([r for r in rows if r is not None], dtype=float)


def main(root):
    """Print the dropout table with each arm's no-dropout reference and the 3-sd verdict."""
    refs = {pen: collect(os.path.join(root, sub)) for pen, sub in REFS.items()}
    print(f"read at {READ_AT} iterations; live counts out of N=1000\n")
    print(f"{'pen':5} {'dropout':8} {'n':>2}  {'live_scalefree':>18}  {'live_abs4e-2':>18}  "
          f"{'clean_loss':>18}  {'r2':>6}")
    for pen in ("none", "rws", "frm", "both"):
        a = refs[pen]
        print(f"{pen:5} {'-- none':8} {len(a):>2}  "
              f"{a[:, 0].mean():8.0f} ± {a[:, 0].std():<7.0f} {a[:, 1].mean():8.0f} ± {a[:, 1].std():<7.0f} "
              f"{a[:, 2].mean():8.5f} ± {a[:, 2].std():<7.5f} {a[:, 3].mean():6.3f}")
        for kind in ("mute", "dead"):
            cell = os.path.join(root, DROP_SUB, f"EqType=h_k=3_N=1000_pen={pen}_do={kind}")
            b = collect(cell)
            if not len(b):
                print(f"{pen:5} {kind:8}  -  (no net read at {READ_AT})")
                continue
            v = b[0]
            marks = []
            for col in (0, 1):
                sd = a[:, col].std()
                marks.append("+" if v[col] > a[:, col].mean() + 3 * sd else
                             "-" if v[col] < a[:, col].mean() - 3 * sd else ".")
            verdict = "KEEPS ALIVE" if marks == ["+", "+"] else "".join(marks)
            print(f"{pen:5} {kind:8} {len(b):>2}  {v[0]:8.0f} {'':9} {v[1]:8.0f} {'':9} "
                  f"{v[2]:8.5f} {'':9} {v[3]:6.3f}   {verdict}")
        print()
    print("verdict column: '+/-/.' = above / below / within 3 reference sd, "
          "[scale-free, absolute 4e-2]; KEEPS ALIVE = '+' under BOTH criteria (pre-registered).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
