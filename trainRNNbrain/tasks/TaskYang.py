"""The 20 cognitive tasks of Yang, Joglekar, Song, Newsome & Wang (Nat. Neurosci. 2019), ported to
this repo's Task interface, for training one network on many tasks with a rule input.

Shared input/output space (the point of the family: a stimulus direction means the same thing in
every task, so units shared between tasks share a computation):
  input  = [fixation (1) | modality-1 ring (n_eachring) | modality-2 ring (n_eachring) | rule (20)]
  output = [fixation (1) | response ring (n_eachring)]
A stimulus at direction theta with strength s is the bump s * 0.8 * exp(-d^2 / 2 sigma^2) over the
ring (d = circular distance). The rule channel of the trial's task is 1 for the whole trial; all 20
rule channels are always present, so a network trained on ONE rule has the same architecture as
one trained on all 20 (the single-task reference of the multi-task experiment).

Timing: this repo's step is tau/10 (dt=1, tau=10), so Yang's 100 ms tau is 10 steps and a 3 s trial
is T=300. Epoch lengths are drawn PER TRIAL (the Trainer supports per-trial scoring masks), from
ranges that keep every trial inside T; unused steps after a trial are zero input and unscored.
Targets: fixation output 0.85 while fixating, 0.05 in the response epoch of a trial that responds;
response ring 0.05 baseline + 0.8 bump at the response direction, 0.05 elsewhere / before. The
first `grace` steps of the response epoch are unscored, as in the paper. One deviation: the paper
weights the response epoch x5 in the loss; here every scored entry has weight 1.

The 20 rules (names as in the paper's code):
  fdgo reactgo delaygo fdanti reactanti delayanti          Go family (mod-1 stimulus)
  dm1 dm2 contextdm1 contextdm2 multidm                     decision-making, stimulus on till the end
  delaydm1 delaydm2 contextdelaydm1 contextdelaydm2 multidelaydm   same, stimulus off before the go
  dms dnms dmc dnmc                                         (non-)match to sample / category
contextdm1 is Mante's context-dependent decision (attend modality 1, ignore 2) in ring coding —
the "CDDM" of the multi-task experiment.
"""
import numpy as np

from trainRNNbrain.tasks.TaskBase import Task

RULES = ["fdgo", "reactgo", "delaygo", "fdanti", "reactanti", "delayanti",
         "dm1", "dm2", "contextdm1", "contextdm2", "multidm",
         "delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm",
         "dms", "dnms", "dmc", "dnmc"]


class TaskYang(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, rules, n_per_task, n_eachring=32,
                 sigma_tuning=np.pi / 8, coherences=(0.04, 0.08, 0.16, 0.32), grace=10, seed=None):
        """Set up the ring geometry and the rule subset.

        Args:
            n_steps: trial length T (must hold the longest epoch sequence: 300 at the defaults);
            n_inputs: must be 1 + 2*n_eachring + 20; n_outputs: must be 1 + n_eachring;
            rules: list of rule names (subset of RULES) the network is trained on;
            n_per_task: trials per rule in a batch (batch = n_per_task * len(rules));
            n_eachring: units per ring; sigma_tuning: bump width (rad); coherences: |c| set for
            the DM families (strengths 1 +- c); grace: unscored steps after the go signal;
            seed: seeds the trial generator.
        """
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.rules = list(rules)
        for r in self.rules:
            if r not in RULES:
                raise ValueError(f"unknown rule {r}; choose from {RULES}")
        self.subtask_names = self.rules                       # the composite-task API
        self.n_per_task = int(n_per_task)
        self.n_ring = int(n_eachring)
        self.sigma = float(sigma_tuning)
        self.cohs = np.asarray(coherences, dtype=float)
        self.grace = int(grace)
        if n_inputs != 1 + 2 * self.n_ring + len(RULES) or n_outputs != 1 + self.n_ring:
            raise ValueError(f"TaskYang: n_inputs must be {1 + 2 * self.n_ring + len(RULES)} and "
                             f"n_outputs {1 + self.n_ring}; got {n_inputs}, {n_outputs}")
        self.pref = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_ring)   # preferred directions
        self.i_mod1 = slice(1, 1 + self.n_ring)
        self.i_mod2 = slice(1 + self.n_ring, 1 + 2 * self.n_ring)
        self.i_rule = 1 + 2 * self.n_ring

    # ----------------------------------------------------------------- ring code
    def bump(self, theta, strength=1.0):
        """Ring activation for a direction: strength * 0.8 * exp(-d^2 / 2 sigma^2), (n_ring,)."""
        d = np.angle(np.exp(1j * (self.pref - theta)))
        return strength * 0.8 * np.exp(-0.5 * (d / self.sigma) ** 2)

    def dur(self, lo, hi):
        """One epoch length in steps, uniform on [lo, hi]."""
        return int(self.rng.integers(lo, hi + 1))

    def coh(self):
        """A signed coherence from the configured set."""
        return float(self.rng.choice(self.cohs) * self.rng.choice([-1.0, 1.0]))

    # ----------------------------------------------------------------- one trial
    def trial(self, rule):
        """Inputs, targets and the scoring mask of one trial of `rule`.

        Every trial is fix -> (stim -> delay -> [test]) -> response. Directions are uniform on the
        circle; DM stimuli are two directions >= pi/2 apart with strengths 1 +- c.

        Returns:
            (X (n_inputs, T), Y (n_outputs, T), scored (T,) bool, condition dict).
        """
        T = self.n_steps
        X = np.zeros((self.n_inputs, T))
        Y = np.zeros((self.n_outputs, T))
        Y[1:, :] = 0.05
        X[self.i_rule + RULES.index(rule), :] = 1.0
        cond = {"rule": rule}
        t_fix = self.dur(30, 50)
        resp_dir, respond = None, True
        th1 = float(self.rng.uniform(0, 2 * np.pi))

        if rule in ("fdgo", "reactgo", "delaygo", "fdanti", "reactanti", "delayanti"):
            resp_dir = th1 if "go" in rule else th1 + np.pi
            cond.update(stim_dir=th1)
            if rule.startswith("react"):                       # go signal = stimulus onset
                t_go = t_fix
                t_end = t_go + 40
                X[self.i_mod1, t_go:t_end] += self.bump(th1)[:, None]
            elif rule.startswith("fd"):                        # stimulus on until the end
                t_go = t_fix + self.dur(30, 100)
                t_end = t_go + 40
                X[self.i_mod1, t_fix:t_end] += self.bump(th1)[:, None]
            else:                                              # brief stimulus, delay, go
                t_s = t_fix + self.dur(30, 50)
                t_go = t_s + self.dur(30, 100)
                t_end = t_go + 40
                X[self.i_mod1, t_fix:t_s] += self.bump(th1)[:, None]

        elif rule in ("dm1", "dm2", "contextdm1", "contextdm2", "multidm",
                      "delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm"):
            th2 = th1 + float(self.rng.uniform(np.pi / 2, 3 * np.pi / 2))
            c1 = self.coh()                                    # modality-1 evidence for th1 over th2
            c2 = self.coh()                                    # modality-2 evidence
            if rule.startswith("multi") and c1 + c2 == 0:
                c2 = -c2 if self.rng.random() < 0.5 else c2 * 0.5  # never exactly balanced
            delayed = "delay" in rule
            t_s = t_fix + self.dur(30, 100)
            t_go = t_s + (self.dur(30, 100) if delayed else 0)
            t_end = t_go + 40
            t_off = t_s if delayed else t_end                  # stimulus off at delay start, or never
            use1 = rule in ("dm1", "delaydm1", "contextdm1", "contextdelaydm1", "multidm", "multidelaydm")
            use2 = rule in ("dm2", "delaydm2", "contextdm2", "contextdelaydm2", "multidm", "multidelaydm")
            show1 = use1 or rule.startswith("context")         # context tasks show BOTH modalities
            show2 = use2 or rule.startswith("context")
            if show1:
                X[self.i_mod1, t_fix:t_off] += (self.bump(th1, 1 + c1) + self.bump(th2, 1 - c1))[:, None]
            if show2:
                X[self.i_mod2, t_fix:t_off] += (self.bump(th1, 1 + c2) + self.bump(th2, 1 - c2))[:, None]
            if rule.startswith("multi"):
                ev = c1 + c2
            elif rule in ("dm1", "delaydm1", "contextdm1", "contextdelaydm1"):
                ev = c1
            else:
                ev = c2
            resp_dir = th1 if ev > 0 else th2
            cond.update(dir1=th1, dir2=th2, coh1=c1, coh2=c2)

        else:                                                  # dms dnms dmc dnmc
            t_s = t_fix + self.dur(30, 50)                     # sample off
            t_go = t_s + self.dur(30, 100)                     # test on = go signal
            t_end = t_go + 40
            cat = lambda th: int((th % (2 * np.pi)) < np.pi)   # category = which half-circle
            if rule in ("dms", "dnms"):
                match = self.rng.random() < 0.5
                th2 = th1 if match else th1 + float(self.rng.uniform(np.pi / 4, 7 * np.pi / 4))
                same = match
            else:
                match = self.rng.random() < 0.5
                th2 = float(self.rng.uniform(0, 2 * np.pi))
                while cat(th2) != (cat(th1) if match else 1 - cat(th1)):
                    th2 = float(self.rng.uniform(0, 2 * np.pi))
                same = match
            X[self.i_mod1, t_fix:t_s] += self.bump(th1)[:, None]
            X[self.i_mod1, t_go:t_end] += self.bump(th2)[:, None]
            respond = same if rule in ("dms", "dmc") else not same
            resp_dir = th2
            cond.update(sample_dir=th1, test_dir=th2, match=bool(same))

        # fixation input on until the go signal; targets; scoring
        X[0, :t_go] = 1.0
        Y[0, :t_go] = 0.85
        if respond:
            Y[0, t_go:t_end] = 0.05
            Y[1:, t_go:t_end] = 0.05 + self.bump(resp_dir)[:, None]
        else:
            Y[0, t_go:t_end] = 0.85
        scored = np.zeros(T, dtype=bool)
        scored[:t_end] = True
        scored[t_go:t_go + self.grace] = False
        cond.update(t_fix=t_fix, t_go=t_go, t_end=t_end, respond=bool(respond),
                    resp_dir=(None if not respond else float(resp_dir % (2 * np.pi))))
        return X, Y, scored, cond

    # ----------------------------------------------------------------- batches
    def batch_of(self, rule, n, task_idx):
        """`n` trials of one rule, stacked on the last axis, with composite-style conditions."""
        Xs, Ys, conds = [], [], []
        for _ in range(n):
            X, Y, scored, c = self.trial(rule)
            Xs.append(X); Ys.append(Y)
            conds.append({"task": rule, "task_idx": task_idx, "scored": scored, "sub": c})
        return np.stack(Xs, axis=2), np.stack(Ys, axis=2), conds

    def get_batch(self, shuffle=False):
        """A mixed batch: n_per_task trials of every rule in `rules`, grouped by rule.

        Returns:
            (inputs (n_inputs, T, B), targets (n_outputs, T, B), conditions), B = n_per_task * n_rules;
            each condition holds "task", "task_idx", "scored" (T,) bool and "sub" (the trial dict).
        """
        parts = [self.batch_of(r, self.n_per_task, i) for i, r in enumerate(self.rules)]
        inputs = np.concatenate([p[0] for p in parts], axis=2)
        targets = np.concatenate([p[1] for p in parts], axis=2)
        conds = sum([p[2] for p in parts], [])
        if shuffle:
            perm = self.rng.permutation(inputs.shape[-1])
            inputs, targets = inputs[..., perm], targets[..., perm]
            conds = [conds[k] for k in perm]
        return inputs, targets, conds

    def task_batch(self, i, n=256):
        """A batch of ONE rule (index or name) for per-task read-outs."""
        if isinstance(i, str):
            i = self.rules.index(i)
        return self.batch_of(self.rules[i], n, i)

    @staticmethod
    def batch_mask(conditions):
        """Per-trial scoring mask of a batch, (T, B) bool, from the conditions get_batch returned."""
        return np.stack([c["scored"] for c in conditions], axis=1)
