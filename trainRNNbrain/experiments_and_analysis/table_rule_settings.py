#!/usr/bin/env python3
"""
Supplementary table of every per-unit-rule setting trained on 3BFF (dead-relu-paper#90).

One row per catalogue cell: each setting of the five per-unit rules that was trained at N = 1,000,
plus the prune-and-duplicate variant trained at N = 500 and 2,000, and the rate penalty at
lambda_frm = 0.03 (the combination test's rate-penalty-alone cell). For each cell the table gives
the networks trained, the mean active count, fold change and held-out R^2 change at 75,000
iterations (the same-size read-out), and the floor crossings with their fold change.

Definitions (as in Supplementary Table 3 since dead-relu-paper#103, fig_ed_remedies.py):
  fold change  the cell's mean active count over the mean of the size-series control networks
               (Figure 1's default 3BFF networks, catalogue role "control") at the same N and the
               same measurement point;
  R^2 change   the cell's mean held-out R^2 (36 noise draws, heldout_r2.floor_r2) minus the same
               control networks' mean, over networks with R^2 > 0; a network with R^2 <= 0 (weights
               inside a loss burst) is left out of the mean and counted in its own column.
The rate penalty at lambda_frm = 0.1 has no 36-draw score at 75,000 iterations (no catalogue
value); its R^2 change comes from the removal analysis's intact scores (a3_remedy_ablation.py, 9
noise draws), against the control networks scored by that same analysis, and is marked.
The lambda_frm = 0.03 cell comes from the combination-test read (read_b3.json, 36 draws).

Data: data/remedy_reads_31oct/ (remedies_catalogue.json, read_b3.json, a3_N*.json); provenance in
fig_paper_F5_drive.py and notebook 2026-10-31_1732 / _1750.

Usage:  python table_rule_settings.py
Output: the LaTeX rows on stdout, then a plain-text check list (per-rule ranges).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_paper_F5_drive import READS, load_a3, load_b3, load_catalogue

# The setting each rule shows in Figure 5 (its size-series cell at N = 1,000).
SHOWN = {
    "frm": "NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm=frm0.1",
    "synnoise": "NBitFlipFlop_synnoise_c200000/EqType=h_k=3_N=1000_sw=1.0",
    "rescale": "NBitFlipFlop_target_ladder_c200000/EqType=h_k=3_N=1000_tgt=10.0",
    "duplicate": "NBitFlipFlop_paper_grid_c200000/EqType=h_N=1000_arm=duplication",
    "dropout": "NBitFlipFlop_dropout_sizes_c200000/EqType=h_k=3_N=1000_pen=none_do=mute_rate=0.20_beta=4",
}
ORDER = ("frm", "synnoise", "rescale", "duplicate", "dropout")
RULE_NAME = {"frm": "Rate penalty", "synnoise": "Noisy synapses", "rescale": "Rescaling",
             "duplicate": "Prune-and-duplicate", "dropout": "Read-out dropout"}


def sig(cell):
    """The rule parameters of a catalogue cell as a dict (empty for the noisy-synapse cells).

    Args:
        cell: catalogue cell dict.
    Returns:
        dict parsed from the cell's 'signature' JSON, or {} when the signature is not JSON.
    """
    try:
        return json.loads(cell["signature"])
    except (json.JSONDecodeError, TypeError):
        return {}


def short_name(cell):
    """The sweep folder of a cell without the '_c200000' top-up suffix, for the table's cell column.

    Args:
        cell: catalogue cell dict.
    Returns:
        str, e.g. 'rescale_revive' for 'NBitFlipFlop_rescale_revive_c200000/...'.
    """
    sweep = cell["cell"].split("/")[0].replace("NBitFlipFlop_", "")
    for suffix in ("_c200000", "_N500", "_N2000"):
        sweep = sweep.replace(suffix, "")
    return sweep


def describe(fam, s, shown, sweep=""):
    """The parameters in which a setting differs from the rule's shown setting, in Methods symbols.

    Args:
        fam: rule family key; s: sig() of the cell; shown: sig() of the shown cell;
        sweep: short_name() of the cell (the norm_mode sweep predates commit 4287801, before
            which the 'median_active' row-length reference grew the input row as well).
    Returns:
        LaTeX string ('shown in Fig. 5' for the shown setting itself is set by the caller).
    """
    if fam == "dropout":
        return f"$q = {s['drop_rate']:g}$, $\\beta = {s['dropout_beta']:g}$"
    if fam == "duplicate":
        parts = [f"$s = {s['copy_noise']:g}$"]
        if not s.get("copy_norm_preserve", False):
            parts.append("length not kept")
        if s["max_replace_frac"] != shown["max_replace_frac"]:
            parts.append(f"{s['max_replace_frac']:g}$N$ per check")
        if s["maturity"] != shown["maturity"]:
            parts.append(f"wait {s['maturity']:,}" if s["maturity"] else "no wait")
        return ", ".join(parts)
    if fam == "rescale":
        if float(s.get("rescale_target_frac", 0.0)) <= 0:
            # first form of the rule (Trainer.py, rescale_target_frac <= 0): no target, budget 8,
            # outgoing weights redrawn at every step while the unit is inactive
            return (f"first form$^*$, $\\alpha = {s['rescale_alpha']:g}$, "
                    + ("length kept" if s.get("rescale_normalize", True) else "length not kept"))
        parts = [f"$m = {s['rescale_target_frac']:g}$"]
        if s.get("rescale_alpha") != shown["rescale_alpha"]:
            parts.append(f"$\\alpha = {s['rescale_alpha']:g}$")
        if not s.get("rescale_normalize", True):
            parts.append("length not kept")
        if not s.get("rescale_active_only", False):
            parts.append("all sources")
        if s.get("rescale_norm_mode", "self") == "median_active":
            parts.append("length to active median" + (", input row too" if sweep == "norm_mode"
                                                       else ""))
        if s.get("rescale_protect_frac", 0.2) != shown["rescale_protect_frac"]:
            parts.append(f"{s['rescale_protect_frac']:g}$N$ at once")
        if s.get("rescale_refractory", 0):
            parts.append(f"hold {s['rescale_refractory']:,}")
        return ", ".join(parts)
    raise ValueError(fam)


def mean_r2_change(r2, base):
    """Mean R^2 change over networks with R^2 > 0, and how many networks had R^2 <= 0.

    Args:
        r2: list of held-out R^2 values; base: the control networks' mean R^2.
    Returns:
        (float or None, int).
    """
    r2 = np.asarray(r2, float)
    ok = np.isfinite(r2) & (r2 > 0)
    return (float(r2[ok].mean() - base) if ok.any() else None), int((~ok).sum())


def row(cell, ctrl, fam, shown_sig, a3=None):
    """One table row for a catalogue cell.

    Args:
        cell: catalogue cell dict; ctrl: the size-series control cell at the same N; fam: family;
        shown_sig: sig() of the rule's shown cell; a3: load_a3() (only for the lambda 0.1 cell).
    Returns:
        dict with name, folder (the catalogue cell, for the source comment), N, setting, n_runs,
        n_75k, active_75k (mean), fold_75k, dr2_75k, n_bad (R^2 <= 0 at 75k), r2_note ("" or
        "a3"), n_cross, fold_floor (or None), shown.
    """
    shown = cell["cell"] == SHOWN.get(fam)
    if fam == "synnoise":
        setting = f"$\\sigma_w = {float(cell['signature'].split('=')[1]):g}$"
    elif fam == "frm":
        setting = "$\\lambda_{\\mathrm{frm}} = 0.1$"
    else:
        setting = describe(fam, sig(cell), shown_sig, short_name(cell))
    a75 = np.asarray(cell["active_75k"], float)
    fold75 = a75.mean() / np.mean(ctrl["active_75k"])
    note = ""
    if cell["r2_75k"]:
        dr2, nbad = mean_r2_change(cell["r2_75k"], np.mean(ctrl["r2_75k"]))
    else:
        # no 36-draw score at 75k: the removal analysis's 9-draw intact scores, rule and control
        nets = [r for r in a3[cell["N"]] if r["cell"].rstrip("/").endswith(cell["cell"].split("/", 1)[1])]
        assert len(nets) == cell["n_runs"], f"a3 has {len(nets)} networks of {cell['cell']}"
        base = np.mean([r["r2_full"] for r in a3[cell["N"]] if r["family"] == "control"])
        dr2, nbad = mean_r2_change([r["r2_full"] for r in nets], base)
        note = "a3"
    fold_floor = (np.mean(cell["active"]) / np.mean(ctrl["active"])) if cell["active"] else None
    return dict(name=short_name(cell), folder=cell["cell"], N=cell["N"], setting=setting,
                n_runs=cell["n_runs"],
                n_75k=len(a75), active_75k=a75.mean(), fold_75k=fold75, dr2_75k=dr2, n_bad=nbad,
                r2_note=note, n_cross=cell["n_read"], fold_floor=fold_floor, shown=shown)


def b3_rate_row(b3, ctrl):
    """The rate penalty alone at lambda_frm = 0.03 (the combination test's 'rate' cell), N = 1,000.

    Args:
        b3: load_b3(); ctrl: the N = 1,000 size-series control cell.
    Returns:
        a row() dict.
    """
    runs = b3["runs"]["rate"]
    a75 = np.asarray([r["fixed"]["active"] for r in runs], float)
    dr2, nbad = mean_r2_change([r["fixed"]["r2"] for r in runs], np.mean(ctrl["r2_75k"]))
    floor = [r["floor"]["active"] for r in runs if r["floor"] is not None]
    return dict(name="combination test", folder="combination test, read_b3.json runs 'rate'",
                N=1000, setting="$\\lambda_{\\mathrm{frm}} = 0.03$, combination test",
                n_runs=len(runs), n_75k=len(a75), active_75k=a75.mean(),
                fold_75k=a75.mean() / np.mean(ctrl["active_75k"]), dr2_75k=dr2, n_bad=nbad,
                r2_note="", n_cross=len(floor),
                fold_floor=(np.mean(floor) / np.mean(ctrl["active"])) if floor else None,
                shown=False)


def collect():
    """Every row of the table, grouped by rule, the shown setting first in each group.

    Returns:
        dict family -> list of row() dicts.
    """
    cat = load_catalogue()
    a3, b3 = load_a3(), load_b3()
    ctrl = {c["N"]: c for c in cat if c["role"] == "control"}
    by_name = {c["cell"]: c for c in cat}
    out = {}
    for fam in ORDER:
        shown_sig = sig(by_name[SHOWN[fam]])
        # every setting at N = 1,000; at other sizes only settings other than the shown one
        cells = [c for c in cat if c["family"] == fam and c["role"] in ("topup", "b2")
                 and (c["N"] == 1000 or c["signature"] != by_name[SHOWN[fam]]["signature"])]
        rows = [row(c, ctrl[c["N"]], fam, shown_sig, a3) for c in cells]
        if fam == "frm":
            rows.append(b3_rate_row(b3, ctrl[1000]))
            # lambda_frm = 0.01, the combination-test pilot's other value: trained to 200,000
            # iterations (Della jobs 15598507-08) but read only under an earlier floor read-out
            # (notebook 2026-10-16_1053) and never at 75,000 iterations; its weights are on Della.
            rows.append(dict(name="combination-test pilot", N=1000,
                             folder="combination-test pilot, Della jobs 15598507-08",
                             setting="$\\lambda_{\\mathrm{frm}} = 0.01$, combination-test pilot",
                             n_runs=3, n_75k=None,
                             active_75k=None, fold_75k=None, dr2_75k=None, n_bad=0, r2_note="",
                             n_cross=None, fold_floor=None, shown=False))
        # the shown setting, then the other settings of its sweep, then the other sweeps
        shown_name = short_name(by_name[SHOWN[fam]])
        rows.sort(key=lambda r: (not r["shown"], r["name"] != shown_name, r["N"] != 1000, r["name"],
                                 r["setting"]))
        out[fam] = rows
    return out


def tex_signed(v):
    """A float with sign and 3 decimals and a true minus sign; '--' for None."""
    return "--" if v is None else f"{v:+.3f}".replace("-", "$-$")


def sweep_numbers(rows):
    """Number the sweeps 1, 2, ... in the order they first appear in the table.

    The printed table identifies a sweep (a set of networks trained together) by this number; the
    folder names go into source comments.

    Args:
        rows: collect() output.
    Returns:
        dict sweep name -> int.
    """
    out = {}
    for fam in ORDER:
        for r in rows[fam]:
            out.setdefault(r["name"], len(out) + 1)
    return out


def print_tex(rows):
    """Print the LaTeX rows of the table, each after a source comment naming its cell folder.

    Args:
        rows: collect() output.
    Returns:
        None.
    """
    num = sweep_numbers(rows)
    print("% rows printed by table_rule_settings.py")
    for fam in ORDER:
        # '\\*': no page break between a rule's heading and its first row
        print(f"\\multicolumn{{8}}{{l}}{{\\emph{{{RULE_NAME[fam]}}}}} \\\\*")
        for r in rows[fam]:
            setting = r["setting"] + (" (shown)" if r["shown"] else "")
            n = f"{r['N']:,}".replace(",", "{,}")
            dr2 = tex_signed(r["dr2_75k"]) + ("$^\\P$" if r["r2_note"] == "a3" else "") + \
                (f"$^\\dagger$" if r["n_bad"] else "")
            ff = "--" if r["fold_floor"] is None else f"{r['fold_floor']:.2f}"
            setting = "\\raggedright " + setting       # wrapped lines are not justified
            print(f"% sweep {num[r['name']]}: {r['folder']}")
            if r["active_75k"] is None:                  # not read (lambda_frm = 0.01)
                print(f"{num[r['name']]} & {setting} & 1{{,}}000 & -- & -- & -- & -- & not read$^\\S$ \\\\")
                continue
            act = f"{r['active_75k']:,.0f}".replace(",", "{,}")
            print(f"{num[r['name']]} & {setting} & {n} & "
                  f"{r['n_75k']} & {act} & "
                  f"{r['fold_75k']:.2f} & {dr2} & "
                  f"{r['n_cross']}/{r['n_runs']}, {ff} \\\\")
        if fam != ORDER[-1]:
            print("\\addlinespace")


def print_check(rows):
    """Print, per rule, the number of settings at N = 1,000 and the range of fold and R^2 change.

    Args:
        rows: collect() output.
    Returns:
        None.
    """
    print("\n% check list")
    print("% sweep numbers: " + ", ".join(f"{k} {v}" for v, k in sweep_numbers(rows).items()))
    for fam in ORDER:
        r1 = [r for r in rows[fam] if r["N"] == 1000 and r["active_75k"] is not None]
        f = [r["fold_75k"] for r in r1]
        d = [r["dr2_75k"] for r in r1 if r["dr2_75k"] is not None]
        low = [(r["name"], r["setting"], round(r["fold_75k"], 2)) for r in r1 if r["fold_75k"] < 1]
        print(f"% {fam:9s} settings at N=1000: {len(r1)}; fold at 75k {min(f):.2f}-{max(f):.2f}; "
              f"R2 change {min(d):+.3f} to {max(d):+.3f}; below 1: {low}")
        for r in r1:
            if r["n_bad"]:
                print(f"%    {r['name']} {r['setting']}: {r['n_bad']} network(s) with R2 <= 0 at 75k")


if __name__ == "__main__":
    out = collect()
    print_tex(out)
    print_check(out)
