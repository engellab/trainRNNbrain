"""Where the trained networks sit relative to a lognormal target, for weights and for rates.

The claim this figure makes: the WEIGHTS are already lognormal in both arms and barely differ
between them; the entire departure is in the RATE distribution, where `none` carries a ~30% atom
at exactly zero and `frm+rws` has no zeros but a spread crushed to 0.8 decades. Cortex sits
between them -- no zero atom, ~2 decades of spread.
"""
import glob
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "data/trained_RNNs/CDDM_ptrack_g0_nodale"
ARMS = {"none": "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
        "frm+rws": "EqType=h_N=1000_LmbdRWS=0.05_LmbdFR=0.2"}
COL = {"none": "#C44E52", "frm+rws": "#4C72B0"}


def gather(arm_dir):
    """Pool |W_inp|, |W_rec| and final participation over every net in one arm."""
    wi, wr, pp = [], [], []
    for f in sorted(glob.glob(f"{ROOT}/{arm_dir}/*/*LastParams_*.json")):
        d = json.load(open(f))
        wi.append(np.abs(np.asarray(d["W_inp"], float)).ravel())
        wr.append(np.abs(np.asarray(d["W_rec"], float)).ravel())
    for f in sorted(glob.glob(f"{ROOT}/{arm_dir}/*/*ParticipationTrace.pkl")):
        pp.append(np.asarray(pickle.load(open(f, "rb"))["participation"][-1], float))
    return np.concatenate(wi), np.concatenate(wr), np.concatenate(pp)


data = {a: gather(d) for a, d in ARMS.items()}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
titles = ["|W_inp|", "|W_rec|", "participation (firing rates)"]

for k, (ax, title) in enumerate(zip(axes, titles)):
    for arm, col in COL.items():
        x = data[arm][k]
        nz = x[x > 1e-12]
        zero_frac = (x <= 1e-12).mean()
        ax.hist(np.log10(nz), bins=70, density=True, histtype="step", lw=1.8,
                color=col, label=f"{arm}  (zero: {zero_frac:.0%})")
        if zero_frac > 0.01:      # draw the atom at zero explicitly; a log axis hides it
            ax.axvline(np.log10(nz).min() - 0.35, color=col, lw=4, alpha=0.5)
            ax.text(np.log10(nz).min() - 0.3, ax.get_ylim()[1] * 0.85,
                    f"{zero_frac:.0%}\nat 0", color=col, fontsize=8, va="top")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("log10 magnitude")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("density")
# Best-fit normal in log10 for each arm, dashed: if the data are lognormal the step histogram
# tracks the dashed curve. No arbitrary reference band -- the earlier version drew one at the
# axis midpoint, which carries no information.
for k, ax in enumerate(axes):
    for arm, col in COL.items():
        x = data[arm][k]
        lg = np.log10(x[x > 1e-12])
        gx = np.linspace(lg.min(), lg.max(), 300)
        ax.plot(gx, np.exp(-0.5 * ((gx - lg.mean()) / lg.std()) ** 2) /
                (lg.std() * np.sqrt(2 * np.pi)), ls="--", lw=1.0, color=col, alpha=0.65)
axes[0].text(0.02, 0.98, "dashed = best-fit lognormal", transform=axes[0].transAxes,
             fontsize=7, va="top", color="0.35")
axes[2].text(0.02, 0.70, "cortex: no atom at 0,\n~2 decades spread", transform=axes[2].transAxes,
             fontsize=7, va="top", color="0.35")

fig.suptitle("Input weights are a MIXTURE of two lognormals (context vs stimulus channels);\nthe rate distribution is where the arms actually differ",
             fontsize=12)
fig.tight_layout()
fig.savefig("img/internal_figures/fig_lognormal_gap.png", dpi=150)
print("wrote img/internal_figures/fig_lognormal_gap.png")

print("\nsummary (pooled over 5 nets per arm):")
for arm in ARMS:
    for k, t in enumerate(titles):
        x = data[arm][k]
        nz = x[x > 1e-12]
        lg = np.log10(nz)
        print(f"  {arm:8s} {t:28s} zero={np.mean(x<=1e-12):6.1%}  "
              f"spread(q01-q99)={np.quantile(lg,0.99)-np.quantile(lg,0.01):4.1f} decades")
