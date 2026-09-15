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

Timing: T = 300 steps with tau = 10 steps, as every other task here (Yang's 100 ms tau is 10 steps,
so T is a 3 s trial). Fixation, stimulus and delay lengths are drawn PER TRIAL (the Trainer supports
per-trial scoring masks); the response epoch then runs to the END of the trial, so every trial uses
and scores all T steps (Pavel, 2026-09-15: no unused tail). The response therefore lasts 5-27 tau
depending on the trial, and must be sustained.
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
    def __init__(self, n_steps, n_inputs, n_outputs, rules, batch_size=1024, n_eachring=32,
                 sigma_tuning=np.pi / 8, coherences=(0.04, 0.08, 0.16, 0.32), grace=10, seed=None):
        """Set up the ring geometry and the rule subset.

        Args:
            n_steps: trial length T (must hold the longest epoch sequence: 300 at the defaults);
            n_inputs: must be 1 + 2*n_eachring + 20; n_outputs: must be 1 + n_eachring;
            rules: list of rule names (subset of RULES) the network is trained on;
            batch_size: trials per batch, split near-evenly over the rules (random assignment);
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
        self.batch_size = int(batch_size)
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
    def bumps(self, theta, strength=1.0):
        """Ring activations for directions theta (n,): strength * 0.8 * exp(-d^2 / 2 sigma^2), (n_ring, n)."""
        d = np.angle(np.exp(1j * (self.pref[:, None] - np.asarray(theta)[None, :])))
        return np.asarray(strength) * 0.8 * np.exp(-0.5 * (d / self.sigma) ** 2)

    def cohs_(self, n):
        """n signed coherences from the configured set."""
        return self.rng.choice(self.cohs, n) * self.rng.choice([-1.0, 1.0], n)

    # ----------------------------------------------------------------- one rule, n trials
    def batch_of(self, rule, n, task_idx):
        """`n` trials of one rule, generated in one vectorised pass.

        Every trial is fix -> (stim -> delay -> [test]) -> response-until-T, epoch lengths drawn per trial.
        Directions are uniform on the circle; DM stimuli are two directions >= pi/2 apart with
        strengths 1 +- c.

        Returns:
            (X (n_inputs, T, n), Y (n_outputs, T, n), conditions) with composite-style conditions
            {"task", "task_idx", "scored" (T,) bool, "sub": the trial's epochs and directions}.
        """
        T, rng = self.n_steps, self.rng
        X = np.zeros((self.n_inputs, T, n), dtype=np.float32)   # float32: the trainer casts anyway,
        Y = np.zeros((self.n_outputs, T, n), dtype=np.float32)  # and a 1024-trial batch is 100 MB
        Y[1:] = 0.05
        X[self.i_rule + RULES.index(rule)] = 1.0
        t = np.arange(T)[:, None]
        bc = lambda a: np.broadcast_to(np.asarray(a), (n,))[None, :]
        win = lambda a, b: (t >= bc(a)) & (t < bc(b))     # (T, n) bool from per-trial bounds
        t_fix = rng.integers(30, 51, n)
        th1 = rng.uniform(0, 2 * np.pi, n)
        respond = np.ones(n, dtype=bool)
        sub = {}

        if rule in ("fdgo", "reactgo", "delaygo", "fdanti", "reactanti", "delayanti"):
            resp_dir = th1 if "go" in rule else th1 + np.pi
            if rule.startswith("react"):                       # go signal = stimulus onset
                t_go = t_fix
                t_end = np.full(n, T)
                on = win(t_go, t_end)
            elif rule.startswith("fd"):                        # stimulus on until the end
                t_go = t_fix + rng.integers(30, 101, n)
                t_end = np.full(n, T)
                on = win(t_fix, t_end)
            else:                                              # brief stimulus, delay, go
                t_s = t_fix + rng.integers(30, 51, n)
                t_go = t_s + rng.integers(30, 101, n)
                t_end = np.full(n, T)
                on = win(t_fix, t_s)
            X[self.i_mod1] += self.bumps(th1)[:, None, :] * on[None]
            sub = dict(stim_dir=th1)

        elif rule in ("dm1", "dm2", "contextdm1", "contextdm2", "multidm",
                      "delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm"):
            th2 = th1 + rng.uniform(np.pi / 2, 3 * np.pi / 2, n)
            c1, c2 = self.cohs_(n), self.cohs_(n)              # evidence for th1 over th2, per modality
            if rule.startswith("multi"):
                tie = c1 + c2 == 0
                c2[tie] = -c2[tie]                             # never exactly balanced
            delayed = "delay" in rule
            t_s = t_fix + rng.integers(30, 101, n)
            t_go = t_s + (rng.integers(30, 101, n) if delayed else 0)
            t_end = np.full(n, T)
            on = win(t_fix, t_s if delayed else t_end)        # off at delay start, or never
            use1 = rule in ("dm1", "delaydm1", "contextdm1", "contextdelaydm1", "multidm", "multidelaydm")
            use2 = rule in ("dm2", "delaydm2", "contextdm2", "contextdelaydm2", "multidm", "multidelaydm")
            if use1 or rule.startswith("context"):             # context tasks show BOTH modalities
                X[self.i_mod1] += (self.bumps(th1, 1 + c1) + self.bumps(th2, 1 - c1))[:, None, :] * on[None]
            if use2 or rule.startswith("context"):
                X[self.i_mod2] += (self.bumps(th1, 1 + c2) + self.bumps(th2, 1 - c2))[:, None, :] * on[None]
            ev = c1 + c2 if rule.startswith("multi") else (c1 if use1 else c2)
            resp_dir = np.where(ev > 0, th1, th2)
            sub = dict(dir1=th1, dir2=th2, coh1=c1, coh2=c2)

        else:                                                  # dms dnms dmc dnmc
            t_s = t_fix + rng.integers(30, 51, n)              # sample off
            t_go = t_s + rng.integers(30, 101, n)              # test on = go signal
            t_end = np.full(n, T)
            match = rng.random(n) < 0.5
            if rule in ("dms", "dnms"):
                th2 = np.where(match, th1, th1 + rng.uniform(np.pi / 4, 7 * np.pi / 4, n))
            else:                                              # category = half-circle
                cat1 = (th1 % (2 * np.pi)) < np.pi
                target_cat = np.where(match, cat1, ~cat1)
                th2 = rng.uniform(0, np.pi, n) + np.pi * (~target_cat)
            respond = match if rule in ("dms", "dmc") else ~match
            resp_dir = th2
            X[self.i_mod1] += (self.bumps(th1)[:, None, :] * win(t_fix, t_s)[None]
                               + self.bumps(th2)[:, None, :] * win(t_go, t_end)[None])
            sub = dict(sample_dir=th1, test_dir=th2, match=match)

        fixwin = win(0, t_go)
        respwin = win(t_go, t_end)
        X[0] = fixwin
        Y[0] = 0.85 * fixwin + np.where(respond[None, :], 0.05, 0.85) * respwin
        Y[1:] += self.bumps(resp_dir)[:, None, :] * (respwin & respond[None, :])[None]
        scored = win(0, t_end) & ~win(t_go, t_go + self.grace)
        conds = [{"task": rule, "task_idx": task_idx, "scored": scored[:, b],
                  "sub": {**{k: (bool(v[b]) if v.dtype == bool else float(v[b])) for k, v in sub.items()},
                          "t_fix": int(t_fix[b]), "t_go": int(t_go[b]), "t_end": int(t_end[b]),
                          "respond": bool(respond[b]),
                          "resp_dir": float(resp_dir[b] % (2 * np.pi)) if respond[b] else None}}
                 for b in range(n)]
        return X, Y, conds

    # ----------------------------------------------------------------- batches
    def get_batch(self, shuffle=False):
        """A mixed batch of `batch_size` trials, rules assigned near-evenly at random, grouped by rule.

        Returns:
            (inputs (n_inputs, T, B), targets (n_outputs, T, B), conditions); each condition holds
            "task", "task_idx", "scored" (T,) bool and "sub" (the trial dict).
        """
        counts = np.bincount(self.rng.permutation(np.arange(self.batch_size) % len(self.rules)),
                             minlength=len(self.rules))
        parts = [self.batch_of(r, int(c), i) for i, (r, c) in enumerate(zip(self.rules, counts)) if c]
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
