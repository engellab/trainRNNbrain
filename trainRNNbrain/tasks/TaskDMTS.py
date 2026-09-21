from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskDMTS(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on_sample, stim_off_sample,
                 stim_on_match, stim_off_match,
                 dec_on, dec_off,
                 random_window, n_stim=None, tonic=False, num_rep=64, seed=None):
        """Delayed match to sample: two pulses separated by a delay; match if they were the same.

        CHANNEL LAYOUT (n_stim = 2, tonic = True, n_inputs = 4 -- the 2026-09-21 design):
            0 .. n_stim-1   stimulus identity, one-hot, a +1 pulse at the sample and at the match
            n_stim          tonic drive, constant 1.0 for the whole trial (only if tonic)
            n_inputs - 1    decision cue, 1.0 from dec_on to dec_off

        `n_stim` is SEPARATE from `n_inputs` because they used to be welded together: the stimulus
        count was inferred as n_inputs - 1, which forced one channel per identity and made the batch
        4x4 = 16 pairs with only the 4 diagonal ones a match -- a 25/75 class imbalance in which
        always answering "non-match" scores 75%. With n_stim = 2 the batch is 2x2 = 4 pairs, 50/50.

        The one-hot identity code is deliberate. Four stimuli cannot be placed equidistantly in two
        dimensions, so a distributed code would make some non-match pairs more similar than others;
        one-hot keeps every pair equally discriminable. With n_stim = 2 that is moot, but the
        parameter keeps larger stimulus sets available on the same footing.

        ⚠️ The tonic channel is mathematically a BIAS: a constant input reaches unit i as
        W_inp[i, n_stim] at every timestep. It sets an operating point and cannot carry memory (it
        is identical on every trial), and because every measure in this project is a count of
        non-silent units, a constant drive is a confound on the main measure. It is a flag so that
        `tonic: false` tests exactly that for the cost of one re-run. For reference, a trainable
        bias in [-1, 1] was worth +5.0 active units against a bias fixed at 0.

        Args:
            n_steps, n_inputs, n_outputs: trial length and channel counts.
            stim_on_sample/off, stim_on_match/off, dec_on/dec_off: epoch boundaries in steps.
            random_window: sample and match onsets are jittered by +/- this many steps.
            n_stim: number of stimulus identities; defaults to the legacy n_inputs - 1.
            tonic: if True, channel n_stim is held at 1.0 for the whole trial.
            num_rep: repeats of the full n_stim^2 condition set, so the batch is
                num_rep * n_stim^2 trials. It was hard-coded at 64, which silently tied the
                batch size to the stimulus count: 1024 trials at 4 stimuli but 256 at 2.
            seed: RNG seed for the jitter.
        """
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.n_stim = (n_inputs - 1) if n_stim is None else int(n_stim)
        self.tonic = bool(tonic)
        self.num_rep = int(num_rep)
        self.stim_on_sample = stim_on_sample
        self.stim_off_sample = stim_off_sample
        self.stim_on_match = stim_on_match
        self.stim_off_match = stim_off_match
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.random_window = random_window

    def generate_input_target_stream(self, num_sample_channel, num_match_channel):
        if self.random_window == 0:
            random_offset_1 = random_offset_2 = 0
        else:
            random_offset_1 = self.rng.integers(-self.random_window, self.random_window)
            random_offset_2 = self.rng.integers(-self.random_window, self.random_window)
        input_stream = np.zeros([self.n_inputs, self.n_steps])
        input_stream[num_sample_channel, self.stim_on_sample + random_offset_1:self.stim_off_sample + random_offset_1] = 1.0
        input_stream[num_match_channel, self.stim_on_match + random_offset_2:self.stim_off_match + random_offset_2] = 1.0
        # decision cue on the LAST channel. It was hard-wired to channel 2, which with n_inputs=3
        # (the default) IS the last channel, but with more stimuli (DMTS_long: 4 + cue) it collided
        # with stimulus 2 - the cue was still time-locked and unambiguous, so those runs stand,
        # but the cue now has its own line (fixed 2026-09-15).
        if self.tonic:
            input_stream[self.n_stim, :] = 1.0
        input_stream[self.n_inputs - 1, self.dec_on:self.dec_off] = 1.0

        condition = {"num_sample_channel" : num_sample_channel,
                     "num_match_channel" : num_match_channel,
                     "sample_on" : self.stim_on_sample + random_offset_1,
                     "sample_off" : self.stim_off_sample + random_offset_1,
                     "match_on" : self.stim_on_match + random_offset_2,
                     "match_off": self.stim_off_match + random_offset_2,
                     "dec_on" : self.dec_on,
                     "dec_off" : self.dec_off}

        # Target stream
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        if self.n_outputs == 2:
            if (num_sample_channel == num_match_channel):
                target_stream[0, self.dec_on: self.dec_off] = 1
            elif (num_sample_channel != num_match_channel):
                target_stream[1, self.dec_on: self.dec_off] = 1
        else:
            if (num_sample_channel == num_match_channel):
                target_stream[0, self.dec_on: self.dec_off] = 1

        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False, num_rep=None):
        """Every (sample, match) pair repeated num_rep times.

        Args:
            shuffle: permute trial order; num_rep: repeats per condition, defaulting to the
            configured self.num_rep so batch size lives in config, not a signature default.
        Returns:
            (inputs, targets, conditions); inputs is (n_inputs, n_steps, num_rep * n_stim**2).
        """
        num_rep = self.num_rep if num_rep is None else num_rep

        # batch size = 256 for two inputs
        inputs = []
        targets = []
        conditions = []

        for i in range(num_rep):
            for num_sample_channel in range(self.n_stim):
                for num_match_channel in range(self.n_stim):
                    correct_choice = 1 if (num_sample_channel == num_match_channel) else -1
                    input_stream, target_stream, condition = self.generate_input_target_stream(num_sample_channel, num_match_channel)
                    inputs.append(deepcopy(input_stream))
                    targets.append(deepcopy(target_stream))
                    conditions.append(deepcopy(condition))

        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        if shuffle:
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
        return inputs, targets, conditions
