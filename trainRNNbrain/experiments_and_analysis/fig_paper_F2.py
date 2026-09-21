#!/usr/bin/env python3
"""
Manuscript Figure 2 - DROPOUT. The cheapest remedy: it keeps about a hundred more units alive, it
costs nothing measurable, and it is nowhere near enough.

Dropout is the first remedy the paper offers because it is the one a reader would try first, it is
free, and its failure to do more is what motivates the penalty of Figure 3. The figure therefore
has to be honest about a small effect rather than dress it up, and it has to say precisely what
"dropout" means here, because the variant used is not standard dropout:

  (a) THE TWO KINDS, AS CIRCUITS       `mute` masks the unit's read-out weight only - the unit goes
                                       on driving its neighbours but the task loss cannot see it.
                                       `dead` removes it from the recurrent dynamics as well. Drawn
                                       as the same three-unit circuit three times so the difference
                                       is a cut edge, not a paragraph.
  (b) THE SAMPLING RULE, MEASURED      units are not dropped uniformly: p_drop is proportional to a
                                       softmax of participation, so the busiest units are preferred.
                                       Read off a real trained network rather than asserted. Two
                                       numbers matter and both are on the panel: exactly 50 of 1000
                                       units go per iteration (drop_rate x N, because the softmax
                                       weights sum to one), and the targeting is MILD - the 50
                                       busiest units take 14.5% of the drops against 5% under
                                       uniform sampling. A weak intervention, and a weak effect.
  (c) WHAT IT DOES ALONG TRAINING      live units vs iteration, 7 seeds per arm. Dropout lifts the
                                       whole curve and does not flatten it: both arms are still
                                       silencing at the same rate at 150k, so this is an offset,
                                       not a cure.
  (d) WHAT IT COSTS                    live units against noise-free task loss, one point per seed,
                                       with the equivalence test. `dead` is positively equivalent
                                       to no dropout (TOST p = 1.2e-05); `mute` costs 3.1%.
  (e) IS 5% JUST TOO LITTLE?           the drop-rate ladder (0.05 -> 0.40) and a sharper-targeting
                                       arm, submitted 2026-09-20 precisely so this figure can answer
                                       the obvious referee question. Panel renders a placeholder
                                       until those runs land.

CRITERION AND READ-OUT as Figure 1: scale-free participation, matched compute, every seed drawn.

Usage:  python fig_paper_F2.py
Output: img/internal_figures/fig_paper_F2.png
"""

import glob
import os
import pickle
import sys

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, SILENT_REL
from flipflop_diversity import load_net
from flipflop_dropout_readout import collect, welch
from trainRNNbrain.training.training_utils import prepare_task_arguments

DROP = f"{DATA_DIR}/NBitFlipFlop_std_dropout"
RATE = f"{DATA_DIR}/NBitFlipFlop_std_droprate"
CELL = "EqType=h_k=3_N=1000_pen=none_do={kind}"
N_UNITS = 1000
READ_AT = 150_000
SAMPLE_AT = 50_000           # iteration at which panel (b) reads the participation vector
SAMPLER_CACHE = "data/fig_paper_F2_sampler.npz"
DROP_RATE, BETA = 0.05, 1.0

ARMS = [("none", "no dropout", ps.BASE),
        ("mute", "dropout: mute", ps.COND_COL["mute"]),
        ("dead", "dropout: dead", ps.COND_COL["dead"])]

# The rate ladder submitted as job 6307716 (see slurm/SilentReLU_flipflop_droprate_spock.slurm).
# Read at 40k, where the standard setting already shows its full effect.
LADDER_READ_AT = 40_000
LADDER = [("0.05", "1", "5%\n(standard)"), ("0.10", "1", "10%"),
          ("0.20", "1", "20%"), ("0.40", "1", "40%")]
SHARP = ("0.05", "4", "5%, sharper\ntargeting (β=4)")


def spearman(a, b):
    """Spearman rank correlation of two 1-D arrays, without a scipy dependency.

    Args:
        a, b: equal-length 1-D arrays.
    Returns:
        the rank correlation as a float.
    """
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def trace_rows(cell_glob):
    """Every (iters, participation matrix) pair under a cell folder.

    Args:
        cell_glob: path of a cell folder holding one sub-folder per network.
    Returns:
        list of (iters, P) with P of shape (n_probes, N).
    """
    out = []
    for f in sorted(glob.glob(os.path.join(cell_glob, "*", "*ParticipationTrace.pkl"))):
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        it, P = np.asarray(d.get("participation_iters", [])), np.asarray(d.get("participation", []))
        if len(it) and P.ndim == 2:
            out.append((it, P))
    return out


def live_curve(it, P):
    """Active-unit count at every probe of one trace.

    Args:
        it: (n_probes,) iterations; P: (n_probes, N) participation.
    Returns:
        (it, counts) with counts an int array of the same length.
    """
    q = np.quantile(P, 0.95, axis=1, keepdims=True)
    return it, (P >= SILENT_REL * q).sum(axis=1)


def live_at(cell_glob, iteration):
    """Active units per seed at one iteration.

    Args:
        cell_glob: cell folder; iteration: read-out iteration.
    Returns:
        (n_seeds,) int array; empty if the cell is missing.
    """
    out = []
    for it, P in trace_rows(cell_glob):
        j = int(np.argmin(np.abs(it - iteration)))
        if abs(it[j] - iteration) > 2000:
            continue
        p = P[j]
        out.append(int((p >= SILENT_REL * np.quantile(p, 0.95)).sum()))
    return np.array(out)


def tost(a, b, margin_frac=0.05):
    """Two one-sided tests for equivalence of two means within +-margin_frac of b's mean.

    "p = 0.95 so dropout is free" is absence of evidence, not evidence of equivalence. TOST asks the
    question the paper actually means: is the difference small enough to be uninteresting? The
    margin is fixed at 5% of the reference loss, chosen because the rate penalty of Figure 3 costs
    7%, so the bar is "cheaper than the remedy we recommend".

    Args:
        a, b: 1-D samples (a = dropout arm, b = reference); margin_frac: equivalence margin as a
            fraction of mean(b).
    Returns:
        (p_tost, diff_frac, lo_frac, hi_frac): the larger of the two one-sided p-values, the
        relative difference, and its 95% CI, all as fractions of mean(b).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    delta = margin_frac * b.mean()
    try:
        from scipy import stats
        p = max(stats.t.sf((d + delta) / se, df), stats.t.cdf((d - delta) / se, df))
        crit = stats.t.ppf(0.975, df)
    except ImportError:
        from math import erfc, sqrt
        p = max(erfc((d + delta) / se / sqrt(2)) / 2, erfc(-(d - delta) / se / sqrt(2)) / 2)
        crit = 1.96
    return float(p), d / b.mean(), (d - crit * se) / b.mean(), (d + crit * se) / b.mean()


def panel_a(ax):
    """Panel (a): `mute` and `dead` drawn as the same circuit with different edges cut."""
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    cases = [("mute", "mute", ps.COND_COL["mute"]),
             ("dead", "dead", ps.COND_COL["dead"])]
    w = 0.40
    for ci, (title, kind, col) in enumerate(cases):
        x0 = 0.05 + ci * (w + 0.10)
        ax.text(x0 + w / 2, 0.955, title, ha="center", fontsize=6.8, color=col, fontweight="bold")

        # three units in a row, recurrently connected, feeding one read-out
        ux = [x0 + 0.055, x0 + w / 2, x0 + w - 0.055]
        uy = 0.60
        dropped = 1                                   # the middle unit is the one sampled out
        for i, x in enumerate(ux):
            gone = (kind == "dead" and i == dropped)
            ax.scatter(x, uy, s=95, zorder=5,
                       color="none" if gone else (ps.FAINT if i != dropped else col),
                       edgecolor=ps.FAINT if gone else (ps.MUTED if i != dropped else col),
                       linewidth=0.9, linestyle=":" if gone else "-")
            ax.text(x, uy, f"$r_{i + 1}$", ha="center", va="center", fontsize=5.4, zorder=6,
                    color=ps.FAINT if gone else ("white" if i == dropped else ps.MUTED))

        # Recurrent edges between neighbours. Endpoints are the unit CENTRES and the shrink clears
        # the glyph; rad then puts the forward arc above and the return arc below. Offsetting the
        # endpoints vertically instead pinches the pair into a bowtie over the units.
        for i in range(2):
            cut = (kind == "dead" and dropped in (i, i + 1))
            col = ps.FAINT if cut else ps.MUTED
            ls = ":" if cut else "-"
            ps.arrow(ax, (ux[i], uy), (ux[i + 1], uy), col=col, rad=-0.62, style="-|>",
                     lw=0.7, ls=ls, shrink=6.5, mutation_scale=6)
            ps.arrow(ax, (ux[i + 1], uy), (ux[i], uy), col=col, rad=-0.62, style="-|>",
                     lw=0.7, ls=ls, shrink=6.5, mutation_scale=6)

        # the read-out
        ry = 0.215
        ps.box(ax, x0 + w / 2 - 0.058, ry - 0.045, 0.116, 0.09, "read-out", col=ps.MUTED,
               face="#f2f1ec", lw=0.6, fs=5.4)
        for i, x in enumerate(ux):
            cut = (kind in ("mute", "dead") and i == dropped)
            ps.arrow(ax, (x, uy - 0.075), (x0 + w / 2 + (i - 1) * 0.036, ry + 0.05),
                     col=ps.FAINT if cut else ps.MUTED, lw=0.7, ls=":" if cut else "-")
            if cut:
                mx = (x + x0 + w / 2 + (i - 1) * 0.036) / 2
                my = (uy - 0.075 + ry + 0.05) / 2
                ax.plot([mx - 0.016, mx + 0.016], [my - 0.022, my + 0.022], lw=1.0, color=ps.BAD,
                        zorder=7)
                ax.plot([mx - 0.016, mx + 0.016], [my + 0.022, my - 0.022], lw=1.0, color=ps.BAD,
                        zorder=7)




def sampler_cache(refresh=False, n_nets=4, n_trials=256):
    """Both participation definitions, per unit, for several trained dropout networks.

    THIS IS THE POINT OF PANEL (b), so it is worth being exact about. The dropout sampler calls
    `Trainer.get_participation_`, which reads the RAW states - for equation_type "h" those are
    PRE-ACTIVATIONS - and returns q_0.9(|x|) + std(|x|). The participation the trace logs, and that
    every "active unit" count in this paper uses, is `participation_from_states_`: the same formula
    applied to the ReLU'd RATE. Trainer's own docstring says the two are "deliberately kept
    distinct". They are not interchangeable, and reading p_drop off the logged vector - which is
    what the first version of this panel did - measures the wrong thing.

    The difference is not cosmetic. A unit held far BELOW threshold on every trial has a large
    negative pre-activation, hence a large |x|, hence a high sampled participation - while its rate
    is identically zero. The sampler cannot tell that unit apart from a genuinely busy one.

    Args:
        refresh: re-simulate even if the cache exists; n_nets: networks to average over;
        n_trials: batch size for the noise-free probe.
    Returns:
        dict with per-net arrays: 'v_drop' (sampler's participation), 'v_rate' (the logged one),
        'live' (bool mask under the scale-free rule) and 'p_drop'.
    """
    if os.path.exists(SAMPLER_CACHE) and not refresh:
        z = np.load(SAMPLER_CACHE)
        return {k: z[k] for k in z.files}
    folders = sorted(glob.glob(os.path.join(DROP, CELL.format(kind="dead"), "*", "")))[:n_nets]
    vd, vr, lv, pd_ = [], [], [], []
    for folder in folders:
        cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
        cfg.task.batch_size = n_trials
        task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
        inputs, _, _ = task.get_batch()
        net, _ = load_net(folder)
        net.clear_history()
        net.y = net.y_init
        net.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
        h = np.asarray(net.get_history(), float)
        X = h.reshape(h.shape[0], -1)
        R = np.maximum(X, 0.0)
        v_drop = np.quantile(np.abs(X), 0.9, axis=1) + np.abs(X).std(axis=1)
        v_rate = R.std(axis=1) + np.quantile(R, 0.9, axis=1)
        e = np.exp(BETA * (v_drop - v_drop.max()))
        vd.append(v_drop)
        vr.append(v_rate)
        lv.append(v_rate >= SILENT_REL * np.quantile(v_rate, 0.95))
        pd_.append(np.clip(DROP_RATE * N_UNITS * e / e.sum(), 0.0, 0.999))
    out = {"v_drop": np.array(vd), "v_rate": np.array(vr), "live": np.array(lv),
           "p_drop": np.array(pd_)}
    np.savez_compressed(SAMPLER_CACHE, **out)
    return out


def panel_b(ax, refresh=False):
    """Panel (b): what the sampler thinks 'busy' means, against what busy actually is.

    Returns:
        dict of the numbers quoted on the panel and in the caption.
    """
    c = sampler_cache(refresh=refresh)
    if not len(c.get("v_drop", [])):
        ax.text(0.5, 0.5, "no dropout networks on disk", ha="center", transform=ax.transAxes)
        return {}
    vd, vr, live, pdr = c["v_drop"][0], c["v_rate"][0], c["live"][0], c["p_drop"][0]

    ax.scatter(vd[~live], np.maximum(vr[~live], 1e-6), s=2.4, color=ps.FAINT, alpha=0.55,
               edgecolor="none", zorder=3, label=f"silent ({(~live).sum()})")
    ax.scatter(vd[live], np.maximum(vr[live], 1e-6), s=2.4, color=ps.COND_COL["dead"], alpha=0.6,
               edgecolor="none", zorder=4, label=f"active ({live.sum()})")

    rho = spearman(vd, vr)
    wasted = np.array([p[~l].sum() / p.sum() for p, l in zip(c["p_drop"], c["live"])])
    dropped = np.array([p.sum() for p in c["p_drop"]])

    ax.set(xscale="log", yscale="log", xlabel="sampler score", ylabel="participation  $p_i$")
    ax.text(0.03, 0.96, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=6.2, color=ps.INK,
            va="top")
    ax.legend(loc="lower right", fontsize=5.8)
    ps.ygrid(ax)
    return {"spearman_rho": float(rho), "wasted_share": float(wasted.mean()),
            "wasted_sd": float(wasted.std(ddof=1)), "dropped": float(dropped.mean()),
            "n_nets": int(len(c["p_drop"]))}


def panel_c(ax):
    """Panel (c): active units along training, every seed. Returns per-arm final counts."""
    finals = {}
    for kind, label, col in ARMS:
        rows = trace_rows(os.path.join(DROP, CELL.format(kind=kind)))
        if not rows:
            continue
        for it, P in rows:
            t, c = live_curve(it, P)
            m = t <= READ_AT
            ax.plot(t[m], c[m], lw=0.5, color=col, alpha=0.32, zorder=3)
        grid = np.linspace(2000, READ_AT, 120)
        stack = []
        for it, P in rows:
            t, c = live_curve(it, P)
            stack.append(np.interp(grid, t, c))
        mu = np.mean(stack, axis=0)
        ax.plot(grid, mu, lw=1.5, color=col, zorder=5,
                label=f"{label}  (n={len(rows)})")
        finals[kind] = live_at(os.path.join(DROP, CELL.format(kind=kind)), READ_AT)
        nudge = {"none": -12, "mute": 14, "dead": -6}.get(kind, 0)
        ax.text(READ_AT * 1.06, mu[-1] + nudge, f"{mu[-1]:.0f}", fontsize=5.8, color=col,
                va="center")
    ax.set(xscale="log", xlabel="training iteration", ylabel="active units",
           xlim=(2e3, 2.3e5), ylim=(150, 1000))
    ax.legend(loc="lower left", fontsize=5.9)

    ps.ygrid(ax)
    return finals


def panel_d(ax):
    """Panel (d): active units against noise-free task loss, one point per seed. Returns the TOSTs."""
    data, verdicts = {}, {}
    for kind, label, col in ARMS:
        rows = collect(os.path.join(DROP, CELL.format(kind=kind)))
        if not len(rows):
            continue
        data[kind] = rows
        ax.scatter(rows[:, 2], rows[:, 0], s=13, color=col, alpha=0.9, edgecolor="none", zorder=4,
                   label=f"{label}  (n={len(rows)})")
        ax.errorbar(rows[:, 2].mean(), rows[:, 0].mean(),
                    xerr=rows[:, 2].std(ddof=1), yerr=rows[:, 0].std(ddof=1),
                    fmt="o", ms=4.5, color=col, mec="white", mew=0.6, lw=1.0, zorder=6, capsize=1.5)
    if "none" in data:
        ref = data["none"]
        for kind, _, _ in ARMS[1:]:
            if kind in data:
                verdicts[kind] = tost(data[kind][:, 2], ref[:, 2])
        ax.axvline(ref[:, 2].mean(), color=ps.BASE, lw=0.7, ls=":", zorder=2)
        ax.axvspan(ref[:, 2].mean() * 0.95, ref[:, 2].mean() * 1.05, color="#f2f1ec", zorder=0)
        ax.text(ref[:, 2].mean(), 1.0, "  ±5%", transform=ax.get_xaxis_transform(),
                fontsize=5.4, color=ps.MUTED, va="top", ha="left")
    ax.set(xlabel="noise-free task loss", ylabel="active units")
    ax.legend(loc="upper left", fontsize=5.9)

    ps.ygrid(ax)
    return verdicts


def panel_e(ax):
    """Panel (e): the drop-rate ladder. Renders a placeholder while the runs are still training."""
    xs, groups, cols, labels = [], [], [], []
    ref = live_at(os.path.join(DROP, CELL.format(kind="dead")), LADDER_READ_AT)
    base = live_at(os.path.join(DROP, CELL.format(kind="none")), LADDER_READ_AT)
    for i, (rate, beta, label) in enumerate(LADDER):
        if rate == "0.05" and beta == "1":
            g = ref
        else:
            g = live_at(os.path.join(RATE, f"EqType=h_k=3_N=1000_pen=none_do=dead_rate={rate}_beta={beta}"),
                        LADDER_READ_AT)
        xs.append(i)
        groups.append(g)
        cols.append(ps.COND_COL["dead"])
        labels.append(label)
    rate, beta, label = SHARP
    xs.append(len(LADDER) + 0.45)
    groups.append(live_at(os.path.join(RATE, f"EqType=h_k=3_N=1000_pen=none_do=dead_rate={rate}_beta={beta}"),
                          LADDER_READ_AT))
    cols.append(ps.SLOTS[4])
    labels.append(label)

    res = ps.strip(ax, xs, groups, cols, rng=np.random.default_rng(3))
    if len(base):
        ax.axhline(base.mean(), color=ps.BASE, lw=0.8, ls="--", zorder=2)
        ax.text(-0.45, base.mean() + 4, "no dropout", fontsize=5.6, color=ps.BASE,
                va="bottom", ha="left")
    if len(ref) > 1:
        bar = ref.mean() + 3 * ref.std(ddof=1)
        ax.axhline(bar, color=ps.BAD, lw=0.8, ls="-.", zorder=2)
        ax.text(-0.45, bar + 4, "pre-registered bar", fontsize=5.6, color=ps.BAD,
                va="bottom", ha="left")
    for x, (m, sd, n) in zip(xs, res):
        if n:
            ax.text(x, m + 34, f"{m:.0f}", ha="center", fontsize=5.8, color=ps.INK)
        else:
            ax.text(x, (base.mean() if len(base) else 400) + 60, "training", ha="center",
                    fontsize=5.4, color=ps.FAINT, rotation=90)
    ax.set(xticks=xs, xticklabels=labels, ylabel="active units",
           xlabel="fraction dropped per iteration",
           xlim=(-0.6, xs[-1] + 0.7))
    ax.tick_params(axis="x", labelsize=5.6)
    ps.ygrid(ax)
    return list(zip(labels, res))


def main():
    """Assemble Figure 2 and write it. Returns the output path."""
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 158 * ps.MM))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.92, 1.0], hspace=0.46, wspace=0.34)

    ax_a = fig.add_subplot(gs[0, :2])
    panel_a(ax_a)
    ps.panel_letter(ax_a, "a", dx=-0.015, dy=1.0)

    ax_b = fig.add_subplot(gs[0, 2])
    info_b = panel_b(ax_b)
    ps.panel_letter(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    finals = panel_c(ax_c)
    ps.panel_letter(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    verdicts = panel_d(ax_d)
    ps.panel_letter(ax_d, "d")

    ax_e = fig.add_subplot(gs[1, 2])
    ladder = panel_e(ax_e)
    ps.panel_letter(ax_e, "e")

    out = ps.save(fig, "fig_paper_F2")

    print("\n--- numbers quoted in the caption ---")
    for k, v in info_b.items():
        print(f"  {k}: {v:.4g}")
    for kind, c in finals.items():
        if len(c):
            print(f"  live at 150k, {kind:5}: {c.mean():.1f} ± {c.std(ddof=1):.1f} (n={len(c)})")
    for kind, (p, d, lo, hi) in verdicts.items():
        print(f"  loss {kind:5}: {d:+.2%} [{lo:+.2%}, {hi:+.2%}]  TOST p={p:.3g}")
    for label, (m, sd, n) in ladder:
        lab = label.replace("\n", " ")
        print(f"  ladder {lab:26} {m:7.1f} ± {sd:5.1f} (n={n})" if n else
              f"  ladder {lab:26} still training")
    return out


if __name__ == "__main__":
    main()
