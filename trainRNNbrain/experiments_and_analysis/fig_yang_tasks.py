"""One figure per Yang rule: three example trials, every input and output channel, with a description.

For each of the 20 rules: three trials side by side; top row the 85 input channels as an image
(fixation, modality-1 ring, modality-2 ring, rule vector), bottom row the 33 output targets
(fixation, response ring). The go signal is marked in red, the unscored grace period shaded.
Output: img/internal_figures/yang_tasks/<rule>.png (20 files) and an index of the descriptions.

Usage: python fig_yang_tasks.py
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


def main():
    """Draw and save the 20 figures."""
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
                a.axvline(c["t_end"], color="0.4", lw=0.8, ls="--")
                a.set_xlim(0, task.n_steps)
            info = {k: v for k, v in c.items() if k not in ("t_fix", "t_go", "t_end", "resp_dir")}
            info = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items())
            ax.set_title(f"trial {b + 1}: go at {c['t_go']}, end {c['t_end']}\n{info}", fontsize=8)
            ay.set_xlabel("step (tau = 10 steps)", fontsize=8)
        fig.suptitle(f"{rule}\n" + textwrap.fill(DESC[rule], 150), fontsize=10)
        fig.text(0.5, 0.005, "inputs: fixation on until the go signal (red); outputs: fixation 0.85 until go, then 0.05 + response bump "
                 "if the trial responds; red band = unscored grace; dashed = trial end (unscored after)", ha="center", fontsize=8, color="0.3")
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        fig.savefig(os.path.join(out, f"{rule}.png"), dpi=110)
        plt.close(fig)
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("# Yang task family: example trials\n\n" + "\n".join(f"- **{r}** ([figure]({r}.png)): {DESC[r]}" for r in RULES) + "\n")
    print("saved 20 figures to", os.path.abspath(out))


if __name__ == "__main__":
    main()
