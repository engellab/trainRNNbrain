import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskNBitFlipFlop(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 mu, n_flip_steps,
                 batch_size=256, seed=None):
        '''
        for tanh neurons only
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.mu = mu
        self.n_refractory = self.n_flip = n_flip_steps
        self.lmbd = self.mu / self.n_steps
        self.batch_size = batch_size

    def generate_flipflop_times(self):
        inds = []
        last_ind = 0
        while last_ind < self.n_steps:
            r = self.rng.random()
            ind = last_ind + self.n_refractory + int(-(1 / self.lmbd) * np.log(r))
            if (ind < self.n_steps): inds.append(ind)
            last_ind = ind
        return inds

    def generate_input_target_stream(self):
        """One trial: pulse trains on each bit, and the sign of each bit's most recent pulse.

        The target is a forward-fill of the pulse signs, computed with a running maximum over event
        positions rather than a per-timestep scan. The scan it replaces was O(n_steps * n_events) per
        channel because it tested `i in inds_flips` on a Python list at every timestep; with
        same_batch=False a batch is drawn EVERY iteration, so that cost sat directly on the training
        loop (measured 0.057 s per batch at k=8, ~40% on top of the GPU step).

        Returns:
            (input_stream, target_stream, condition) - arrays of shape (n_inputs, n_steps) and
            (n_outputs, n_steps), and a dict mapping each bit to its flip and flop indices.
        """
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        pos_grid = np.arange(self.n_steps)
        condition = {}
        for n in range(self.n_inputs):
            inds = np.asarray(self.generate_flipflop_times(), dtype=int)
            # self.rng, not np.random: the signs were previously drawn from the GLOBAL numpy stream
            # while the event times came from self.rng, so seeding the task did not reproduce a
            # trial. Same 50/50 distribution, so the data are unchanged - only reproducibility is.
            signs = np.where(self.rng.random(len(inds)) < 0.5, -1.0, 1.0)

            for ind, s in zip(inds, signs):
                input_stream[n, ind: ind + self.n_flip] = s

            # Forward-fill: carry each pulse's sign until the next pulse. Positions of events are
            # running-maximised, so every timestep points at the most recent event at or before it;
            # before the first event the pointer is 0 and events[0] is 0, giving a 0 target there.
            events = np.zeros(self.n_steps)
            if inds.size:
                events[inds] = signs
            target_stream[n] = events[np.maximum.accumulate(np.where(events != 0, pos_grid, 0))]

            condition[n] = {"inds_flips": inds[signs > 0].tolist(),
                            "inds_flops": inds[signs < 0].tolist()}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        """A whole batch of trials, generated in one vectorised pass.

        Every channel of every trial is an independent pulse train, so the batch is built by drawing
        all inter-event gaps at once rather than by looping trial-by-trial and channel-by-channel.
        This matters because with same_batch=False a batch is drawn on EVERY training iteration:
        the per-trial version cost 0.038 s at k=8 with 256 trials, which sat directly on the training
        loop, and it scales linearly with batch_size.

        The generative process is identical to the per-trial path: gaps of
        `n_refractory + int(-(1/lmbd) * log(U))`, cumulative-summed, truncated at n_steps, with
        i.i.d. +-1 signs. `MAX_EV` bounds the events per channel; the minimum possible gap is
        n_refractory, so n_steps // n_refractory + 2 can never be exceeded.

        Args:
            shuffle: permute trials before returning (kept for interface compatibility; the trials
                are i.i.d. so it is a no-op statistically).
        Returns:
            (inputs, targets, conditions) with inputs/targets of shape
            (n_channels, n_steps, batch_size) and conditions a list of per-trial dicts.
        """
        B, C, T = self.batch_size, self.n_inputs, self.n_steps
        n_ch = B * C
        max_ev = T // self.n_refractory + 2

        gaps = self.n_refractory + (-(1.0 / self.lmbd) * np.log(self.rng.random((n_ch, max_ev)))).astype(int)
        inds = np.cumsum(gaps, axis=1)
        valid = inds < T
        signs = np.where(self.rng.random((n_ch, max_ev)) < 0.5, -1.0, 1.0)

        rows = np.repeat(np.arange(n_ch), max_ev)[valid.ravel()]
        cols = inds.ravel()[valid.ravel()]
        vals = signs.ravel()[valid.ravel()]

        # Input: each event is a pulse of width n_flip. Scatter once per offset - n_flip passes,
        # rather than one Python-level slice assignment per event.
        inp = np.zeros((n_ch, T))
        for off in range(self.n_flip):
            c = cols + off
            m = c < T
            inp[rows[m], c[m]] = vals[m]

        # Target: forward-fill each event's sign until the next event, via a running maximum over
        # event positions along time.
        ev = np.zeros((n_ch, T))
        ev[rows, cols] = vals
        pos = np.where(ev != 0, np.arange(T)[None, :], 0)
        tgt = np.take_along_axis(ev, np.maximum.accumulate(pos, axis=1), axis=1)

        inputs = inp.reshape(B, C, T).transpose(1, 2, 0)
        targets = tgt.reshape(B, C, T).transpose(1, 2, 0)

        # `rows` is sorted (np.repeat is row-major and the validity mask preserves order), so each
        # channel's events are a contiguous slice and searchsorted finds the boundaries in one pass.
        # Masking per channel instead would be O(B*C*n_events) - 230M element comparisons at B=1024,
        # k=8, which dominated everything else here.
        bounds = np.searchsorted(rows, np.arange(n_ch + 1))
        conditions = []
        for b in range(B):
            cond = {}
            for c in range(C):
                j = b * C + c
                s, i = vals[bounds[j]:bounds[j + 1]], cols[bounds[j]:bounds[j + 1]]
                cond[c] = {"inds_flips": i[s > 0].tolist(), "inds_flops": i[s < 0].tolist()}
            conditions.append(cond)

        if shuffle:
            perm = self.rng.permutation(np.arange(inputs.shape[-1]))
            inputs, targets = inputs[..., perm], targets[..., perm]
            conditions = [conditions[i] for i in perm]
        return inputs, targets, conditions