"""Audit every sweep the presentation draws against the standard architecture.

THE STANDARD, as of 2026-10-01: self_connections=True, bias fixed at 0 (bias_range [0, 0]),
dale=False, io_nonnegativity=False, gamma=0, and at least 50,000 training iterations. Anything a
panel shows that departs from it has to say so on the slide, and anything that departs without a
reason to is re-run.

A key ABSENT from a config means the sweep predates it and the model's own default applied, so the
default is what is reported - with the key marked, because "absent" and "set to the default" are not
the same evidence.
"""
import glob
import os

from omegaconf import OmegaConf

# the model's defaults, for keys a config predates (RNN_torch signature)
DEFAULTS = {"self_connections": False, "dale": False, "io_nonnegativity": False, "gamma": 0}
STANDARD = {"self_connections": True, "bias_fixed": True, "dale": False,
            "io_nonnegativity": False, "gamma": 0}

SWEEPS = [
    ("slides 7b/8/10/11, F1c", "CDDM_std_g0_drift"),
    ("slide 8 arms", "CDDM_std_g0_activations"),
    ("slide 10 arms", "CDDM_std_g0_weightdecay"),
    ("slides 12/13", "CDDM_std_g0"),
    ("slide 12 arms", "CDDM_std_g0_metabolic"),
    ("slide 14", "CDDM_fb2792_g0_noise"),
    ("slide 9, F1c", "NBitFlipFlop_std_ksweep"),
    ("F1c N=4000", "NBitFlipFlop_std_bigN"),
    ("F2 flip-flop", "NBitFlipFlop_paper_grid"),
    ("F2 CDDM", "CDDM_paper_grid"),
    ("F2 DMTS", "DMTS_paper_grid"),
    ("F2 duplication", "NBitFlipFlop_ff_revive"),
    ("F2 dropout sizes", "NBitFlipFlop_dropout_sizes"),
]


def read_cell(cell):
    """The five standard knobs plus max_iter for one cell, and which keys were absent.

    Args:
        cell: a cell directory holding one run folder per seed.
    Returns:
        (dict of knob -> value, set of absent keys, max_iter) or None if no config is readable.
    """
    cfgs = glob.glob(os.path.join(cell, "*", "*_config.yaml"))
    if not cfgs:
        return None
    c = OmegaConf.load(cfgs[0])
    m = c.get("model", {}) or {}
    absent = {k for k in DEFAULTS if k not in m}
    br = m.get("bias_range", [0.0, 0.0])
    got = {k: (m[k] if k in m else DEFAULTS[k]) for k in DEFAULTS}
    got["bias_fixed"] = (float(br[0]) == 0.0 and float(br[1]) == 0.0)
    return got, absent, int(c.trainer.max_iter)


if __name__ == "__main__":
    print(f"{'panel':24s} {'sweep':30s} {'deviations from the standard':50s} {'iters'}")
    for panel, sweep in SWEEPS:
        cells = [c for c in sorted(glob.glob(f"data/trained_RNNs/{sweep}/*")) if os.path.isdir(c)]
        seen, iters = {}, set()
        for cell in cells:
            got = read_cell(cell)
            if not got:
                continue
            vals, absent, mi = got
            iters.add(mi)
            key = tuple(sorted(vals.items()))
            seen.setdefault(key, (vals, absent))
        if not seen:
            print(f"{panel:24s} {sweep:30s} {'(no configs on disk)':50s}")
            continue
        for vals, absent in seen.values():
            dev = [f"{k}={vals[k]}" + ("*" if k in absent else "")
                   for k in STANDARD if vals[k] != STANDARD[k]]
            lo = min(iters)
            flag = "" if lo >= 50_000 else f"  <-- below 50k"
            print(f"{panel:24s} {sweep:30s} {(', '.join(dev) or 'none'):50s} "
                  f"{sorted(iters)}{flag}")
    print("\n* the key is ABSENT from the config: the sweep predates it and the model default applied")
