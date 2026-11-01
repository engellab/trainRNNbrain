#!/usr/bin/env python3
"""
The per-unit rules and settings that were trained on 3BFF but NOT continued to 200,000 iterations
(dead-relu-paper#90 review, item 5): the second block of Supplementary Table 3.

Every cell folder under data/trained_RNNs/NBitFlipFlop_*/ (the original sweeps, not their
'_c200000' continuations) whose saved config is a 3-bit flip-flop (task.n_inputs == 3) with a
per-unit rule switched on (read_remedies.classify is not "control"), and which has no continuation in
the per-unit-rule catalogue (remedies_catalogue.json), is one row. Sweeps that vary no per-unit
rule (activations, input scale, optimiser, size series of the default networks) are not walked, and
cells with the sparsity penalty alone are left out: that penalty acts on weights and is a standard
setting of Results 3 (Methods M08), not a per-unit rule.

For each cell: networks found, training length (from the saved config), and the mean active count
(common.load_run, the paper's active rule) at the participation probe nearest the measurement
point: 40,000 iterations for cells trained to 40,000, 75,000 iterations for longer ones. Fold
change: that mean over the mean of the control networks at the same N and the same iteration:
  40,000: Figure 5's control networks at 40,000 iterations (their original sweep NBitFlipFlop_ff_revive
          arm none, N = 500-2,000; NBitFlipFlop_paper_grid arm control, N = 4,000); for cells with the
          cubic saturation gamma = 0.1, the gamma = 0.1 controls of NBitFlipFlop_ff_revive_g01;
  75,000: Figure 5's control networks at 75,000 iterations (the catalogue's control cells).
No held-out R^2 is computed here (none was scored for these cells).

Usage:  python table_rule_screen.py
Output: LaTeX rows on stdout and a check list.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, load_run
from fig_paper_F5_drive import load_catalogue
from read_remedies import cfg_of, classify, sel

# Sweeps whose cells vary something other than a per-unit rule, or that are not 3BFF experiments.
SKIP_SWEEPS = ("std_ksweep", "std_winp", "std_activations", "std_sigmoid", "std_switch", "std_bigN",
               "std_inpcap", "n100", "b5_optimizer", "size_series", "c1_snapshots",
               # warm-started from rate-penalty networks to test switching the sparsity penalty
               # on or off (N = 2,000, 50,000 more iterations): not a rule trained from scratch
               "std_switch")
# What each kind of cell is, in words (matched on the cell folder name, first match wins).
KIND = (("tgt=median_norm", "synaptic scaling"),
        ("op=synaptogenesis", "synaptogenesis (new synapses from active units)"),
        ("op=disinhibit", "disinhibition (inhibitory inputs weakened)"),
        ("mode=orth", "replacement by a row orthogonal to the active rows"),
        ("mode=mix", "replacement by a mix of donor rows"),
        ("mode=bias_kick", "bias offset (bias kick)"),
        ("DETUNED_SELFWEIGHT", "prune-and-duplicate, earlier code (self-weight error)"),
        ("arm=prune_copy", "prune-and-duplicate without the replacement cap"),
        ("cn=0", "prune-and-duplicate, copy control"),
        ("mode=random", "revival method: random redraw"),
        ("mode=zero_out", "revival method: redraw, outgoing zeroed"),
        ("do=dead", "read-out dropout, whole-pass variant"),
        ("do=mute", "read-out dropout"),
        ("arm=mute", "read-out dropout"),
        ("arm=duplication", "prune-and-duplicate"),
        ("pen=both", "rate + sparsity"),
        ("pen=frm", "rate penalty"))


def kind_of(cell, sweep):
    """The rule of a cell in words, from its folder name.

    Args:
        cell: the cell folder name; sweep: its sweep folder name.
    Returns:
        str; the copy-control label only for the copy_iid / copy_permute sweeps.
    """
    for key, label in KIND:
        if key in cell and (key != "cn=0" or sweep in ("copy_iid", "copy_permute")):
            if key.startswith("do=mute") and "pen=" in cell and "pen=none" not in cell:
                return label + " with " + ("rate + sparsity" if "pen=both" in cell else
                                           "rate penalty" if "pen=frm" in cell else "sparsity penalty")
            return label
    return "other"


def cell_runs(cell_dir):
    """Every run folder of a cell that has a participation trace and a loss, as load_run dicts."""
    return [r for r in (load_run(d) for d in sorted(glob.glob(os.path.join(cell_dir, "*", ""))))
            if r is not None]


def mean_active_at(runs, it):
    """Mean active count over runs at the probe nearest iteration `it`, the probes, and divergences.

    A network whose participation is non-finite at that probe has diverged; load_run counts it as
    0 active units. Such networks are counted and left out of the mean.

    Args:
        runs: load_run dicts; it: iteration.
    Returns:
        (mean active count over networks that did not diverge, or nan when all did; list of probe
        iterations; number of diverged networks).
    """
    vals, probes = [], []
    for r in runs:
        j = int(np.argmin(np.abs(r["it_act"] - it)))
        vals.append(r["active"][j])
        probes.append(float(r["it_act"][j]))
    vals = np.asarray(vals, float)
    ok = vals > 0
    return (float(vals[ok].mean()) if ok.any() else float("nan")), probes, int((~ok).sum())


def controls_40k():
    """Mean active count of the control networks at 40,000 iterations, by (N, gamma)."""
    out = {}
    for N in (500, 1000, 2000):
        out[(N, 0.0)] = mean_active_at(cell_runs(os.path.join(
            DATA_DIR, "NBitFlipFlop_ff_revive", f"EqType=h_k=3_N={N}_pen=none_arm=none")), 40000)[0]
    out[(4000, 0.0)] = mean_active_at(cell_runs(os.path.join(
        DATA_DIR, "NBitFlipFlop_paper_grid", "EqType=h_N=4000_arm=control")), 40000)[0]
    for N in (500, 1000, 2000, 4000):
        out[(N, 0.1)] = mean_active_at(cell_runs(os.path.join(
            DATA_DIR, "NBitFlipFlop_ff_revive_g01", f"EqType=h_k=3_N={N}_pen=none_arm=none")), 40000)[0]
    return out


def collect():
    """Every row of the block: cells with a per-unit rule on 3BFF that the catalogue does not hold.

    Returns:
        list of dicts (sweep, cell, kind, N, n, trained, it, active, fold), sorted by kind and cell.
    """
    cat = load_catalogue()
    continued = {c["cell"].replace("_c200000", "") for c in cat}
    ctrl75 = {c["N"]: np.mean(c["active_75k"]) for c in cat if c["role"] == "control"}
    ctrl40 = controls_40k()
    rows = []
    for cell_dir in sorted(glob.glob(os.path.join(DATA_DIR, "NBitFlipFlop_*", "*", ""))):
        sweep_dir, cell = os.path.split(cell_dir.rstrip("/"))
        sweep = os.path.basename(sweep_dir).replace("NBitFlipFlop_", "")
        if sweep.endswith("_c200000") or any(sweep.startswith(s) for s in SKIP_SWEEPS):
            continue
        if f"NBitFlipFlop_{sweep}/{cell}" in continued:
            continue
        runs = cell_runs(cell_dir)
        if not runs:
            continue
        cfg = cfg_of(runs[0]["run_dir"])
        if cfg is None or sel(cfg, "task.n_inputs") != 3 or classify(cfg) == "control":
            continue
        frm, rws = float(sel(cfg, "trainer.lambda_frm") or 0), float(sel(cfg, "trainer.lambda_rws") or 0)
        drop = bool(sel(cfg, "trainer.dropout") or False)
        if rws > 0 and frm == 0 and not drop:
            continue                       # the sparsity penalty alone: a Results 3 setting
        N, gamma = int(sel(cfg, "model.N")), float(sel(cfg, "model.gamma") or 0.0)
        trained = int(sel(cfg, "trainer.max_iter"))
        it = 40000 if trained <= 40000 else 75000
        act, probes, n_div = mean_active_at(runs, it)
        base = ctrl40[(N, gamma)] if it == 40000 else ctrl75[N]
        assert max(abs(p - it) for p in probes) <= 1000, f"probe far from {it} in {cell_dir}"
        rows.append(dict(sweep=sweep, cell=cell, kind=kind_of(cell, sweep), N=N, n=len(runs),
                         trained=trained, it=it, active=act, fold=act / base, gamma=gamma,
                         diverged=n_div))
    rows.sort(key=lambda r: (r["kind"], r["sweep"], r["N"], r["cell"]))
    return rows


def tex_cell(cell):
    """A cell folder name for the table: the setting part only, in typewriter type, escaped."""
    s = cell.replace("EqType=h_", "").replace("k=3_", "")
    return "\\texttt{" + s.replace("_", "\\_\\allowbreak{}") + "}"


def print_tex(rows):
    """Print the LaTeX rows: kind, sweep, cell, N, networks, trained, active, fold.

    Args:
        rows: collect() output.
    Returns:
        None.
    """
    print("% rows printed by table_rule_screen.py")
    for r in rows:
        n = f"{r['N']:,}".replace(",", "{,}")
        act = "--" if not np.isfinite(r["active"]) else f"{r['active']:,.0f}".replace(",", "{,}")
        fold = "--" if not np.isfinite(r["fold"]) else f"{r['fold']:.2f}"
        nets = f"{r['n']}" + (f" ({r['diverged']} div.)" if r["diverged"] else "")
        tr = f"{r['trained']:,}".replace(",", "{,}")
        g = "$^\\gamma$" if r["gamma"] else ""
        sweep = "\\texttt{" + r["sweep"].replace("_", "\\_") + "}"
        print(f"\\raggedright {r['kind']}{g} & \\raggedright {sweep} {tex_cell(r['cell'])} & {n} & "
              f"{nets} & {tr} & {act} & {fold} \\\\")


def print_check(rows):
    """Print counts and fold ranges per kind of cell.

    Args:
        rows: collect() output.
    Returns:
        None.
    """
    print("\n% check list")
    kinds = sorted({r["kind"] for r in rows})
    for k in kinds:
        rr = [r for r in rows if r["kind"] == k]
        f = [r["fold"] for r in rr if np.isfinite(r["fold"])]
        print(f"% {k}: {len(rr)} cells, {sum(r['n'] for r in rr)} networks "
              f"({sum(r['diverged'] for r in rr)} diverged), trained "
              f"{sorted({r['trained'] for r in rr})}, fold {min(f):.2f}-{max(f):.2f}")
    print(f"% total {len(rows)} cells")


if __name__ == "__main__":
    out = collect()
    print_tex(out)
    print_check(out)
