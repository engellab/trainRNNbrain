#!/usr/bin/env python3
"""
Build the cache behind manuscript Figure 2: four measures for every network of the three remedies.

RUNS ON THE CLUSTER, where the trained networks live; the figure script reads the cache it writes.
The four arms are matched on everything but the intervention -- gamma = 0, N = 1000, 3-bit
flip-flop, 40,000 iterations, lr 1e-3, weight decay 1e-6, sigma_rec = sigma_inp = 0.05, batch 1024.
Cells trained with a different gamma are deliberately NOT included: gamma is cubic saturation in
the dynamics, so it changes the base network and a cross-gamma comparison is not like for like.
(That rules out NBitFlipFlop_ff_revive_g01_fix, the only other corrected duplication sweep.)

⚠️ THE DUPLICATION CELL IS THE CORRECTED CONSTRUCTION. Before 2026-09-24 duplication zeroed the
2x2 weight block spanning donor and copy, which left the copy with no self-connection and the donor
with half of its own, so a single duplication moved the output by up to 8.9e-03 against an output
scale of 0.27. Cells carrying that bug are named `__DETUNED_SELFWEIGHT` on disk and are excluded.

THE FOUR MEASURES, per network:
  r2            three of them, because one is not enough. `r2` is recomputed WITH the network's
                own noise, the quantity stored in the folder name, and exists so the rebuild can be
                checked against it -- it is the GATE, not the read-out. `r2_common` is the read-out:
                sigma_w = 0 with the recurrent and input noise every arm shares, averaged over
                COMMON_DRAWS draws, so the arms are compared under one condition rather than each
                under its own. `r2_clean` is the fully noise-free pass.
  active units  scale-free rule, p_i >= 0.05 * q_95(p), on p_i = std(r_i) + q_0.9(|r_i|).
  dimensionality  participation ratio (sum ev)^2 / sum ev^2 of the noise-free rate covariance over
                the ACTIVE units: the number of directions the population uses, 1 if every unit does
                the same thing. The count of components reaching 95% of the variance is recorded
                beside it, because participation ratio and a variance threshold can disagree.
  weight distribution  histogram of log10|W_rec| over the nonzero entries, plus how lognormal
                those magnitudes are and which way they err: the width of ln|W| (sigma_log), its
                distance to the best-fit normal against a same-size normal reference (ks, ks_ref),
                and its skewness and excess kurtosis, both 0 for an exact lognormal. Recorded for
                the whole matrix and for the incoming rows of the active units only, since
                duplication and rescale both act on incoming rows.

⚠️ GATE. Every network is rebuilt from ITS OWN saved config and scored; anything whose recomputed r2
misses the stored value by more than R2_TOL is dropped, because a wrong forward pass still yields a
plausible-looking dimensionality and a perfectly plausible weight histogram.

Usage (on the cluster, under a repo new enough for every feature these runs used):
    python f2_remedies_cache.py                     # writes ~/fig_paper_F2_cache.npz
    python f2_remedies_cache.py OUT.npz
    F2_REPO=~/other_worktree python f2_remedies_cache.py
Then copy the file to data/fig_paper_F2_cache.npz beside the figure script.
"""

import glob
import os
import re
import sys

import hydra
import numpy as np
import torch
from scipy import stats
from omegaconf import OmegaConf

# The repo the networks are REBUILT from must have every model feature they were trained with.
# `_cperturb` predates `sigma_w` and raises TypeError on the whole synaptic-noise arm, so the
# default is a worktree new enough to carry it. Older cells still pass the r2 gate from here,
# which is the check that the task class has not moved under them.
REPO = os.environ.get("F2_REPO", "/home/pt1290/trainRNNbrain_sizeser")
sys.path.insert(0, REPO)
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask

D = os.environ.get("F2_DATA", "/home/pt1290/trainRNNbrain/data/trained_RNNs")
R2_TOL = 0.03
R2_KEY_GATE = "r2_common"   # the field TASK_MIN_R2 is applied to
COMMON_DRAWS = 8             # noise draws averaged for the common test condition
TRIALS = 128                 # 300 timesteps x 128 trials = 38400 samples against 1000 units, far
                             # above what a covariance over at most 1000 units needs. 256 trials
                             # holds two 1000x76800 rate matrices at once and is OOM-killed.
SILENT_REL = 0.05            # the scale-free silence rule used everywhere in this project
ZERO_TOL = 1e-12
# log10|W| bin edges. The range has to cover the WIDEST arm, not the control: at [-8, 0.5] the
# target-based rescale cells lost up to 4.2% of their weights off the right edge, which is the
# tail that panel (e) exists to show. The lower edge matches ZERO_TOL, so nothing falls off the
# left; the check below asserts the loss is negligible rather than trusting the range.
LOG_BINS = np.linspace(-12.0, 4.0, 321)

# WHICH CELLS GO IN IS DISCOVERED, NOT LISTED. A hand-written cell list went stale three times in
# one day: it held rescale's first form and missed four later sweeps of the same rule, then missed
# the target ladder that set its operating point. Every cell under SWEEP_GLOB is now read, its saved
# config classified, and the ones that match on task, size, budget and gamma are kept. A new sweep
# joins the figure by finishing, not by being remembered.
SWEEP_GLOB = ("NBitFlipFlop_*", "CDDM_*", "DMTS_*")
SIZES = (500, 1000, 2000, 4000)

# EACH TASK HAS ITS OWN BUDGET and they are not interchangeable: DMTS needs 150,000 iterations to be
# solved at all (at 40,000 the unpenalised network does not escape) and CDDM 100,000. Matching
# iterations ACROSS tasks would compare a converged flip-flop with an unconverged DMTS, so the
# budget is per task and every comparison in the figure stays inside one task.
ITERS = {"NBitFlipFlop": 40000, "CDDM": 100000, "DMTS": 150000}

# ⚠️ DMTS AT A 7 TAU DELAY IS NOT ALWAYS SOLVED. An unpenalised network either escapes to r2 ~ 0.999
# or sits at ~0.42 forever, and an unsolved network's active-unit count is not comparable with a
# solved one's - it is a different dynamical object, not a worse version of the same one. Networks
# below this are dropped and the SOLVE RATE is printed per cell, so the loss is visible.
TASK_MIN_R2 = {"DMTS": 0.8}
EXCLUDE = ("__DETUNED_SELFWEIGHT", "RETRACTED")   # superseded constructions, named on disk


def classify(cfg):
    """Which arm a saved config belongs to, or None if it is not one of the four.

    The arms are defined by what the config switches on, so a cell cannot be filed under the wrong
    one by its folder name. Anything that is a different intervention - the non-copy revival rules,
    synaptic scaling, a loss penalty - returns None and is left for the figures that cover it.

    Args:
        cfg: the OmegaConf config saved beside a trained network.
    Returns:
        one of "control", "mute", "duplicate", "rescale", "synnoise", or None.
    """
    m, t = cfg.model, cfg.trainer
    if float(getattr(m, "sigma_w", 0.0) or 0.0) > 0.0:
        return "synnoise"
    if any(float(getattr(t, k, 0.0) or 0.0) > 0.0 for k in ("lambda_met", "lambda_orth")):
        return None                                   # metabolic / orthogonality: not this figure
    lam_frm = float(getattr(t, "lambda_frm", 0.0) or 0.0)
    lam_rws = float(getattr(t, "lambda_rws", 0.0) or 0.0)
    if lam_frm > 0.0 or lam_rws > 0.0:
        # The penalty arm of this figure is the PAIR. The rate penalty alone is satisfied by
        # transients and weight sparsity alone does not raise any rate, so the two are only a remedy
        # together; Figures 3-5 take them apart and this one carries the combination.
        return "both" if (lam_frm > 0.0 and lam_rws > 0.0) else None
    if bool(getattr(t, "synaptic_scaling", False)):
        return None                                   # its own family, measured elsewhere
    if bool(getattr(t, "dropout", False)):
        return "mute" if str(t.dropout_args.dropout_kind) == "mute" else None
    if bool(getattr(t, "prune_reinit", False)):
        a = t.prune_args
        mode = str(a.get("reinit_mode", ""))
        if mode == "rescale":
            # `reinit_mode: rescale` is also how synaptogenesis and disinhibition are configured;
            # what separates them is `revive_op`, so filing on the mode alone mixes three rules.
            return "rescale" if str(a.get("revive_op", "rescale")) == "rescale" else None
        if mode == "copy":
            # copy_noise is duplication's own knob and stays in; what is excluded is the copy
            # DECOMPOSITION, which replaces the donor's weights with iid draws or a permutation and
            # asks what a copy inherits rather than what duplication does.
            if not bool(a.get("copy_iid", False)) and not bool(a.get("copy_permute", False)):
                return "duplicate"
        return None                                   # orth, mix, bias_kick, random, zero_out
    return "control"


def discover(root):
    """Every matched cell on disk, classified by arm.

    Args:
        root: the trained-RNN directory to scan.
    Returns:
        list of (arm, task, N, cell path relative to root), sorted, one entry per cell folder.
    """
    out = []
    cells = [c for g in SWEEP_GLOB for c in glob.glob(os.path.join(root, g, "*"))]
    for cell in sorted(cells):
        if not os.path.isdir(cell) or any(x in cell for x in EXCLUDE):
            continue
        cfgs = sorted(glob.glob(os.path.join(cell, "*", "*_config.yaml")))
        if not cfgs:
            continue
        try:
            cfg = OmegaConf.load(cfgs[0])
        except Exception:
            continue
        m, t = cfg.model, cfg.trainer
        task = str(cfg.task.get("taskname", ""))
        if task not in ITERS or str(m.equation_type) != "h" or float(m.gamma) != 0.0:
            continue
        if int(t.max_iter) != ITERS[task] or int(m.N) not in SIZES:
            continue
        if task == "NBitFlipFlop" and int(cfg.task.n_inputs) != 3:
            continue
        arm = classify(cfg)
        if arm is not None:
            out.append((arm, task, int(m.N), os.path.relpath(cell, root)))
    return out


def participation_ratio(x):
    """Number of directions a population uses: (sum ev)^2 / sum ev^2 of its covariance.

    Args:
        x: array (n_units, n_samples) of firing rates.
    Returns:
        float, between 1 and n_units.
    """
    ev = np.linalg.eigvalsh(np.cov(np.asarray(x, dtype=np.float64)))
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def n_comp_95(x):
    """How many principal components carry 95% of a population's variance.

    Args:
        x: array (n_units, n_samples) of firing rates.
    Returns:
        int.
    """
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.asarray(x, dtype=np.float64))))[::-1]
    ev = ev[ev > 1e-12]
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)


def weight_shape(w, rng=np.random.default_rng(0)):
    """How lognormal a set of weight magnitudes is, and which way it errs.

    A fold-range says only how far the extremes sit apart, which two very different distributions
    can share. What the comparison needs is whether |W| is lognormal at all - the shape cortical
    synaptic strengths take - and, where it is not, whether the weights are too CONCENTRATED or too
    SPREAD relative to the untreated network.

    Both come from ln|W|, which is normal exactly when |W| is lognormal:
      sigma_log   its standard deviation: the width of the lognormal, and the axis on which "too
                  concentrated" and "too spread" are read against the control's value.
      ks          the Kolmogorov-Smirnov distance from ln|W| to the best-fit normal. The parameters
                  are fitted on the same data, so this is a DISTANCE, not a calibrated test, and
                  with ~10^6 weights any real distribution would reject at any p-value. `ks_ref`
                  gives it a scale: the same distance computed on a same-size sample drawn from
                  that fitted normal. A ks near ks_ref means "lognormal as far as this can tell".
      skew_log    asymmetry of ln|W|, 0 for an exact lognormal. Its SIGN says which side the
                  departure is on: negative means a heavy tail of very small weights.
      kurt_log    excess kurtosis of ln|W|, 0 for an exact lognormal. Positive means the mass is
                  peaked with heavy tails, negative that it is flatter than a lognormal.
      spread      log10 of the q99/q01 magnitude ratio, kept as a robust width for reference.

    Args:
        w: array of weights of any shape; zeros and structural zeros are dropped.
        rng: generator for the same-size normal reference sample.
    Returns:
        dict of the above plus 'hist' and 'n_nonzero', or None if fewer than 100 nonzeros.
    """
    m = np.abs(np.asarray(w, dtype=np.float64).ravel())
    m = m[m > ZERO_TOL]
    if m.size < 100:
        return None
    lg = np.log(m)
    mu, sd = lg.mean(), lg.std(ddof=1)
    z = (lg - mu) / sd
    ks = float(stats.kstest(lg, "norm", args=(mu, sd)).statistic)
    ref = rng.normal(mu, sd, m.size)
    ks_ref = float(stats.kstest(ref, "norm", args=(ref.mean(), ref.std(ddof=1))).statistic)
    return dict(hist=np.histogram(np.log10(m), bins=LOG_BINS)[0].astype(np.int64),
                n_nonzero=int(m.size),
                sigma_log=float(sd),
                ks=ks,
                ks_ref=ks_ref,
                skew_log=float((z ** 3).mean()),
                kurt_log=float((z ** 4).mean() - 3.0),
                spread=float(np.log10(np.quantile(m, 0.99) / np.quantile(m, 0.01))))


def load_net(net_dir):
    """Rebuild a trained network from the config saved beside its weights.

    Rebuilding from a reconstructed config instead of the saved one substitutes the wrong noise
    levels and drops the trained bias, and then fails the r2 gate on every network.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        (RNN_torch, task, training mask, stored r2 read off the folder name, weight dict).
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(net_dir, "*_config.yaml"))[0])
    raw = dict(np.load(glob.glob(os.path.join(net_dir, "*LastParams*.npz"))[0], allow_pickle=True))
    d = {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v)
         for k, v in raw.items()}
    m = cfg.model
    # activation_args comes from the CONFIG, not the npz. Different sweeps saved it differently -
    # the cperturb runs stored the dict, the CDDM/flip-flop penalty runs stored only its KEYS as a
    # string array, which dict() turns into a ValueError - and the config is the authoritative
    # record either way.
    rnn = RNN_torch(N=int(m.N), activation_args=OmegaConf.to_container(m.activation_args),
                    equation_type=str(m.equation_type), dale=bool(m.dale),
                    io_nonnegativity=bool(m.io_nonnegativity),
                    self_connections=bool(m.self_connections), bias_range=list(m.bias_range),
                    gamma=float(m.gamma), dt=float(m.dt), tau=float(m.tau),
                    sigma_rec=float(m.sigma_rec), sigma_inp=float(m.sigma_inp),
                    sigma_out=float(m.sigma_out),
                    # ⚠️ sigma_w IS PART OF THE TRAINED CONDITION. The synaptic-noise arm redraws
                    # W_rec around its mean at every timestep; rebuilding without it scores the
                    # network under dynamics it never trained in. Left out, the sw=2.0 and sw=3.0
                    # cells recomputed at r2 0.66-0.83 against a stored 0.92-0.93 and failed the
                    # gate, and the quieter cells passed while still being scored wrongly.
                    sigma_w=float(getattr(m, "sigma_w", 0.0)),
                    n_inputs=int(cfg.task.n_inputs),
                    n_outputs=int(cfg.task.n_outputs), seed=0)
    with torch.no_grad():
        for k in ("W_rec", "W_inp", "W_out"):
            getattr(rnn, k).copy_(torch.tensor(np.asarray(d[k], dtype=np.float32)))
        yi = torch.tensor(np.asarray(d["y_init"], dtype=np.float32))
        if isinstance(rnn.y_init, torch.nn.Parameter):
            rnn.y_init.copy_(yi)
        else:
            rnn.y_init = yi
        if d.get("bias") is not None and np.asarray(d["bias"]).size == rnn.N:
            rnn.bias.copy_(torch.tensor(np.asarray(d["bias"], dtype=np.float32)))
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    stored = float(re.match(r"(-?[0-9.]+|nan)_", os.path.basename(net_dir)).group(1))
    return rnn, task, mask, stored, d


def analyse(net_dir):
    """All four measures for one trained network.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        dict of scalars plus the two weight histograms.
    """
    rnn, task, mask, stored, d = load_net(net_dir)
    n_from_cfg = int(rnn.N)
    bi, bt, _ = task.get_batch()
    bi = torch.tensor(bi[:, :, :TRIALS], dtype=torch.float32)
    bt = torch.tensor(bt[:, :, :TRIALS], dtype=torch.float32)
    with torch.no_grad():
        states, out = rnn(bi, w_noise=True)
        r2_noisy = float(Trainer.r2_score(out, bt, mask))
        r_noisy = torch.relu(states).numpy().reshape(rnn.N, -1)   # float32, on purpose
        srec, sinp, sw = float(rnn.sigma_rec), float(rnn.sigma_inp), float(rnn.sigma_w)

        # THE COMMON TEST CONDITION. Scoring each network in its OWN trained condition is not a
        # comparison: the synaptic-noise arm is then the only one measured with its wiring
        # fluctuating, and its stored score is one draw, which is untrustworthy once sigma_w is
        # large. Every arm is therefore also scored with sigma_w = 0 and the recurrent and input
        # noise left at the values every arm shares, averaged over COMMON_DRAWS draws.
        rnn.sigma_w = 0.0
        draws = []
        for _ in range(COMMON_DRAWS):
            _, out_k = rnn(bi, w_noise=True)
            draws.append(float(Trainer.r2_score(out_k, bt, mask)))
        rnn.sigma_w = sw

        rnn.sigma_rec = rnn.sigma_inp = rnn.sigma_w = 0.0
        states_c, out_c = rnn(bi, w_noise=False)
        rnn.sigma_rec, rnn.sigma_inp, rnn.sigma_w = srec, sinp, sw
        r2_clean = float(Trainer.r2_score(out_c, bt, mask))
        r_clean = torch.relu(states_c).numpy().reshape(rnn.N, -1)

    # active units on the NOISY run, which is the condition the network trained in and the one every
    # other active-unit count in this project is measured under
    p = r_noisy.std(axis=1) + np.quantile(np.abs(r_noisy), 0.9, axis=1)
    live = p >= SILENT_REL * np.quantile(p, 0.95)

    W = np.asarray(d["W_rec"], dtype=np.float64)
    whole = weight_shape(W)
    rows = weight_shape(W[live]) if live.sum() > 5 else None
    inp = weight_shape(np.asarray(d["W_inp"], dtype=np.float64))

    if whole is not None:
        outside = 1.0 - whole["hist"].sum() / whole["n_nonzero"]
        assert outside < 1e-4, (f"{outside:.2%} of |W_rec| falls outside the histogram range "
                               f"{LOG_BINS[0]:.0f}..{LOG_BINS[-1]:.0f}; widen LOG_BINS")
    out_d = dict(N_cfg=n_from_cfg, stored=stored, r2=r2_noisy, r2_clean=r2_clean,
                 r2_common=float(np.mean(draws)), r2_common_sd=float(np.std(draws, ddof=1)),
                 n_active=int(live.sum()),
                 dims=float("nan"), dims95=float("nan"),
                 w_inp_sigma=float("nan") if inp is None else inp["sigma_log"])
    if live.sum() >= 2:
        out_d["dims"] = participation_ratio(r_clean[live])
        out_d["dims95"] = float(n_comp_95(r_clean[live]))
    for tag, sh in (("w", whole), ("wact", rows)):
        for k in ("hist", "n_nonzero", "sigma_log", "ks", "ks_ref", "skew_log",
                  "kurt_log", "spread"):
            out_d[f"{tag}_{k}"] = (np.zeros(len(LOG_BINS) - 1, np.int64) if k == "hist" else
                                   float("nan")) if sh is None else sh[k]
    return out_d


def main(out_path):
    """Score every network of every cell and write the cache. Returns the output path."""
    cells = discover(D)
    print(f"discovered {len(cells)} matched cells under {D}/{SWEEP_GLOB}")
    for task in sorted({t for _, t, _, _ in cells}):
        print(f"  {task}:")
        for arm in sorted({a for a, t, _, _ in cells if t == task}):
            per_n = {n: sum(1 for a2, t2, m, _ in cells if a2 == arm and t2 == task and m == n)
                     for n in SIZES}
            print(f"    {arm:>10s}: " + ", ".join(f"N={n}: {k}" for n, k in per_n.items() if k))
    recs, fields = [], None
    for arm, task, n_units, pat in cells:
        for nd in sorted(glob.glob(os.path.join(D, pat, "*"))):
            if not os.path.isdir(nd):
                continue
            try:
                r = analyse(nd)
            except Exception as e:
                print(f"  SKIP {arm:>10s} {os.path.basename(nd)[:12]}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            if r["N_cfg"] != n_units:
                print(f"  SKIP {arm:>10s} {os.path.basename(nd)[:12]}: cell says N={n_units} but "
                      f"the saved config says N={r['N_cfg']}", flush=True)
                continue
            gate = abs(r["stored"] - r["r2"]) < R2_TOL
            print(f"  {task[:5]:>5s} {arm:>10s} N={n_units:5d} stored {r['stored']:7.4f} recomp {r['r2']:7.4f} "
                  f"common {r['r2_common']:6.3f}±{r['r2_common_sd']:.3f} clean {r['r2_clean']:6.3f} "
                  f"{'PASS' if gate else 'FAIL':>4s}  active {r['n_active']:4d}  "
                  f"dims {r['dims']:6.2f}  sigma_log {r['w_sigma_log']:5.2f}", flush=True)
            if not gate:
                continue
            if r[R2_KEY_GATE] < TASK_MIN_R2.get(task, -np.inf):
                print(f"  UNSOLVED {task} {arm} N={n_units} r2={r[R2_KEY_GATE]:.3f}", flush=True)
                continue
            r.update(arm=arm, task=task, N=n_units, cell=pat)
            recs.append(r)
            fields = fields or sorted(r)
    if not recs:
        raise SystemExit("no network passed the gate")
    cache = {k: np.array([r[k] for r in recs]) for k in fields}
    cache["log_bins"] = LOG_BINS
    np.savez_compressed(out_path, **cache)
    print(f"\nwrote {out_path}: {len(recs)} networks, "
          f"{ {a: sum(r['arm'] == a for r in recs) for a in dict.fromkeys(r['arm'] for r in recs)} }")
    return out_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/fig_paper_F2_cache.npz"))
