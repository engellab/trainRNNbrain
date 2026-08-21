from copy import deepcopy
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
        inputs = []
        targets = []
        conditions = []
        for i in range(self.batch_size):
            input_stream, target_stream, condition = self.generate_input_target_stream()
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