"""Two figures per Yang rule: three example trials as channel-by-time images, and the same trials as
polar ring snapshots, each with a description of the task.

<rule>.png       three trials side by side; top row the 85 input channels as an image (fixation,
                 modality-1 ring, modality-2 ring, rule vector), bottom row the 33 output targets
                 (fixation, response ring). Go signal in red, unscored grace shaded; the response is sustained to step 300.
<rule>_structure.png  (contextdm1 for now; `--structure <rule>`) three trials, one row each: the
                 trial structure as step functions over time (fixation input, stimulus on each
                 ring, rule, go signal, fixation output, response present) with three snapshot
                 times marked by arrows, and at exactly those times ring 1, ring 2 and the
                 response target as polar bar plots (one bar per unit at its preferred direction,
                 length = activation). Snapshot times are chosen per task to be the informative
                 ones: for contextdm1 (A) fixation before the stimulus, (B) stimulus on, before the
                 go, (C) mid-response.
Output: img/internal_figures/yang_tasks/ and an index of the descriptions.

Usage: python fig_yang_tasks.py [--structure contextdm1]
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from trainRNNbrain.experiments_and_analysis.common import IMG_DIR
from trainRNNbrain.tasks.TaskYang import RULES, TaskYang

DESC = {
    "fdgo": "Go. Fixate; a stimulus appears on ring 1 and stays on. When the fixation input goes off, "
            "respond toward the stimulus direction (bump on the response ring, fixation output down).",
    "reactgo": "Reaction-time Go. No separate go cue: the stimulus onset IS the go signal; respond toward "
               "it immediately (1-tau grace unscored).",
    "delaygo": "Delay Go. A brief stimulus on ring 1, then a delay with no stimulus; when fixation goes "
               "off, respond toward the REMEMBERED direction.",
    "fdanti": "Anti. As Go, but respond toward the OPPOSITE direction (stimulus + 180 deg).",
    "reactanti": "Reaction-time Anti. As RT Go, but respond opposite to the stimulus.",
    "delayanti": "Delay Anti. As Delay Go, but respond opposite to the remembered direction.",
    "dm1": "Decision-making, modality 1. Two stimuli at different directions on ring 1 with strengths 1+c "
           "and 1-c (|c| in 0.04..0.32); stimuli stay on. Respond toward the STRONGER one after fixation off.",
    "dm2": "Decision-making, modality 2. As DM1 on ring 2.",
    "contextdm1": "Context-dependent DM, attend modality 1 (Mante's task). BOTH rings show the same two "
                  "directions, each ring with its own evidence c1, c2. Respond by ring 1's evidence; ring 2 is a distractor.",
    "contextdm2": "Context-dependent DM, attend modality 2. As above, respond by ring 2's evidence; ring 1 is the distractor.",
    "multidm": "Multisensory DM. Both rings, same two directions, evidence c1 and c2. Respond toward the "
               "direction favoured by the SUM c1 + c2.",
    "delaydm1": "Delayed DM1. Stimuli on ring 1 turn OFF before a delay; respond from memory when fixation goes off.",
    "delaydm2": "Delayed DM2. As Delayed DM1 on ring 2.",
    "contextdelaydm1": "Delayed context DM, attend 1. Both rings shown, then off, delay; respond by ring 1's evidence.",
    "contextdelaydm2": "Delayed context DM, attend 2. Both rings shown, then off, delay; respond by ring 2's evidence.",
    "multidelaydm": "Delayed multisensory DM. Both rings, then off, delay; respond by the summed evidence.",
    "dms": "Delayed match-to-sample. Sample on ring 1, delay, test on ring 1 (test onset = go). If the test "
           "MATCHES the sample direction, respond toward it; otherwise keep fixating (no response).",
    "dnms": "Delayed NON-match-to-sample. As DMS, but respond only when the test does NOT match.",
    "dmc": "Delayed match-to-category. Directions in the upper half-circle are category A, lower half B. "
           "Respond toward the test if sample and test are in the SAME category; else keep fixating.",
    "dnmc": "Delayed NON-match-to-category. Respond only when the categories DIFFER.",
}


SNAPSHOTS = {   # per rule: (label, function of the trial dict -> step) x 3; the informative moments
    "contextdm1": [("A: fixation, no stimulus", lambda c: c["t_fix"] // 2),
                   ("B: both rings on, before go", lambda c: (c["t_fix"] + c["t_go"]) // 2),
                   ("C: response (stimuli still on)", lambda c: (c["t_go"] + c["t_end"]) // 2)],
}


def structure(task, rule, path, n_trials=3):
    """Trial structure over time with arrows at the snapshot times, and the rings at those times.

    Args:
        task: TaskYang; rule: rule name (needs an entry in SNAPSHOTS); path: output file.
    """
    X, Y, C = task.task_batch(rule, n=400)
    if rule in ("contextdm1", "contextdm2"):
        # pick trials that show the independence of the two rings' evidence: a congruent one, a
        # conflict trial with a strong distractor, and a conflict trial with weak attended evidence
        c1 = np.array([c["sub"]["coh1"] for c in C]); c2 = np.array([c["sub"]["coh2"] for c in C])
        att, dis = (c1, c2) if rule == "contextdm1" else (c2, c1)
        pick = [int(np.flatnonzero((np.sign(att) == np.sign(dis)) & (np.abs(att) >= 0.16))[0]),
                int(np.flatnonzero((np.sign(att) != np.sign(dis)) & (np.abs(dis) >= 0.16))[0]),
                int(np.flatnonzero((np.sign(att) != np.sign(dis)) & (np.abs(att) <= 0.08) & (np.abs(dis) >= 0.16))[0])]
        kinds = ["congruent: both rings favour the same option", "CONFLICT: distractor ring strongly favours the other option",
                 "CONFLICT: weak attended evidence, strong distractor"]
    else:
        pick, kinds = list(range(n_trials)), [""] * n_trials
    X, Y, C = X[..., pick], Y[..., pick], [C[i] for i in pick]
    R, T = task.n_ring, task.n_steps
    snaps = SNAPSHOTS[rule]
    fig = plt.figure(figsize=(26, 5.6 * n_trials))
    gs = fig.add_gridspec(n_trials, 10, width_ratios=[3.6] + [1] * 9, wspace=0.3, hspace=0.5)
    for b in range(n_trials):
        c = C[b]["sub"]
        ax = fig.add_subplot(gs[b, 0])
        rows = [("fixation input", X[0, :, b] > 0.5),
                ("stimulus on ring 1", X[task.i_mod1, :, b].max(axis=0) > 0.05),
                ("stimulus on ring 2", X[task.i_mod2, :, b].max(axis=0) > 0.05),
                ("rule input (contextdm1)", X[task.i_rule + RULES.index(rule), :, b] > 0.5),
                ("go signal (fixation off)", np.arange(T) >= c["t_go"]),
                ("target: fixation output", Y[0, :, b] > 0.5),
                ("target: response bump", Y[1:, :, b].max(axis=0) > 0.5)]
        for k, (name, tr) in enumerate(rows):
            y0 = (len(rows) - 1 - k) * 1.4
            ax.step(np.arange(T), y0 + tr.astype(float), where="post", color="k", lw=1.2)
            ax.text(-4, y0 + 0.5, name, ha="right", va="center", fontsize=8)
        ax.axvspan(c["t_go"], c["t_go"] + task.grace, color="C3", alpha=0.12, lw=0)
        ax.axvline(c["t_go"], color="C3", lw=1.0)
        ymax = len(rows) * 1.4
        for (label, f), col in zip(snaps, ["C0", "C2", "C4"]):
            tt = f(c)
            ax.axvline(tt, color=col, lw=1.0, ls=":")
            ax.annotate(label.split(":")[0], xy=(tt, ymax), xytext=(tt, ymax + 0.9), ha="center", fontsize=9,
                        color=col, arrowprops=dict(arrowstyle="->", color=col))
        ax.set_xlim(0, T); ax.set_ylim(-0.3, ymax + 1.6); ax.set_yticks([])
        ax.set_xlabel("step (tau = 10 steps)", fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(f"trial {b + 1} ({kinds[b]}): go at {c['t_go']}; dir1 {np.degrees(c['dir1'] % (2*np.pi)):.0f} deg, "
                     f"dir2 {np.degrees(c['dir2'] % (2*np.pi)):.0f} deg; evidence ring 1 c1 = {c['coh1']:+.2f}, "
                     f"ring 2 c2 = {c['coh2']:+.2f}  ->  respond to "
                     f"{'dir1' if c['coh1'] > 0 else 'dir2'} ({np.degrees(c['resp_dir']):.0f} deg)", fontsize=9, loc="left")
        for m, ((label, f), col) in enumerate(zip(snaps, ["C0", "C2", "C4"])):
            tt = f(c)
            for k, (name, vec, vmax, colr) in enumerate([("ring 1 input", X[task.i_mod1, tt, b], 1.6, "C0"),
                                                         ("ring 2 input", X[task.i_mod2, tt, b], 1.6, "C1"),
                                                         ("response target", Y[1:, tt, b], 0.9, "C3")]):
                pax = fig.add_subplot(gs[b, 1 + 3 * m + k], projection="polar")
                pax.bar(task.pref, vec, width=2 * np.pi / R * 0.8, color=colr, alpha=0.85)
                pax.set_ylim(0, vmax); pax.set_yticks([])
                pax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
                pax.set_xticklabels(["0", "90", "180", "270"], fontsize=6)
                pax.plot([c["dir1"], c["dir1"]], [0, vmax], color="0.3", lw=0.6, ls="--")
                pax.plot([c["dir2"], c["dir2"]], [0, vmax], color="0.3", lw=0.6, ls="--")
                if k == 2:
                    pax.plot([c["resp_dir"]], [vmax * 0.95], marker="*", color="k", ms=9)
                # the two bar heights at the stimulus directions, so 1+c vs 1-c is legible at small c
                cc = c["coh1"] if k == 0 else c["coh2"] if k == 1 else None
                note = (f"dir1 {1 + cc:.2f}  dir2 {1 - cc:.2f}" if (cc is not None and vec.max() > 0.05)
                        else ("bump at " + f"{np.degrees(c['resp_dir']):.0f} deg" if (k == 2 and vec.max() > 0.5) else "empty"))
                pax.set_title((f"{label} (t={tt})\n" if k == 0 else "\n") + f"{name}\n{note}", fontsize=8,
                              color=col if k == 0 else "k")
                if k == 0:
                    first = pax.get_position()
                if k == 2:
                    last = pax.get_position()
                    fig.add_artist(plt.Line2D([first.x0, last.x1], [first.y0 - 0.012] * 2, color=col, lw=2.5,
                                              transform=fig.transFigure))
    fig.suptitle(f"{rule}: trial structure and ring snapshots.  " + textwrap.fill(DESC[rule], 170)
                 + "\nThe two directions are the two CHOICE OPTIONS, shared by both rings (as in Mante: motion and colour both speak about left vs right); "
                   "each ring's evidence c (sign and size) is drawn independently, so half the trials are conflict trials. "
                   "Dashed radii = the two options; star = correct response. B: both rings on, strengths 1+c and 1-c printed; C: response bump at the option ring 1 favours.",
                 fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    """Draw and save the per-rule figures, and the structure figure for the requested rule."""
    out = os.path.join(IMG_DIR, "yang_tasks")
    os.makedirs(out, exist_ok=True)
    task = TaskYang(n_steps=300, n_inputs=85, n_outputs=33, rules=RULES, batch_size=1024, seed=1)
    R = task.n_ring
    for rule in RULES:
        X, Y, C = task.task_batch(rule, n=3)
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), gridspec_kw={"height_ratios": [85, 40]})
        for b in range(3):
            c = C[b]["sub"]
            ax, ay = axes[0, b], axes[1, b]
            ax.imshow(X[:, :, b], aspect="auto", cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
            ax.set_yticks([0, 1 + R // 2, 1 + R + R // 2, 1 + 2 * R + 10])
            ax.set_yticklabels(["fixation in", "ring 1 (mod 1)", "ring 2 (mod 2)", "rule (20)"], fontsize=8)
            for y in (0.5, 0.5 + R, 0.5 + 2 * R):
                ax.axhline(y, color="0.6", lw=0.5)
            ay.imshow(Y[:, :, b], aspect="auto", cmap="Greys", vmin=0, vmax=0.9, interpolation="nearest")
            ay.set_yticks([0, 1 + R // 2]); ay.set_yticklabels(["fixation out", "response ring"], fontsize=8)
            ay.axhline(0.5, color="0.6", lw=0.5)
            for a in (ax, ay):
                a.axvline(c["t_go"], color="C3", lw=1.0)
                a.axvspan(c["t_go"], c["t_go"] + task.grace, color="C3", alpha=0.12, lw=0)
                a.set_xlim(0, task.n_steps)
            info = {k: v for k, v in c.items() if k not in ("t_fix", "t_go", "t_end", "resp_dir")}
            info = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items())
            ax.set_title(f"trial {b + 1}: go at {c['t_go']}, end {c['t_end']}\n{info}", fontsize=8)
            ay.set_xlabel("step (tau = 10 steps)", fontsize=8)
        fig.suptitle(f"{rule}\n" + textwrap.fill(DESC[rule], 150), fontsize=10)
        fig.text(0.5, 0.005, "inputs: fixation on until the go signal (red); outputs: fixation 0.85 until go, then 0.05 + response bump "
                 "if the trial responds, sustained to the end of the trial (step 300); red band = unscored grace", ha="center", fontsize=8, color="0.3")
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        fig.savefig(os.path.join(out, f"{rule}.png"), dpi=110)
        plt.close(fig)
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("# Yang task family: example trials\n\n" + "\n".join(f"- **{r}** ([figure]({r}.png)): {DESC[r]}" for r in RULES) + "\n")
    import sys
    rule = sys.argv[sys.argv.index("--structure") + 1] if "--structure" in sys.argv else "contextdm1"
    structure(task, rule, os.path.join(out, f"{rule}_structure.png"))
    print("saved figures to", os.path.abspath(out))


if __name__ == "__main__":
    main()
