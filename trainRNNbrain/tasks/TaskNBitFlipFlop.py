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
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        condition = {}
        for n in range(self.n_inputs):
            inds_flips_and_flops = self.generate_flipflop_times()
            mask = [0 if np.random.rand() < 0.5 else 1 for i in range(len(inds_flips_and_flops))]
            inds_flips = []
            inds_flops = []
            for i in range(len(inds_flips_and_flops)):
                if mask[i] == 0:
                    inds_flops.append(inds_flips_and_flops[i])
                elif mask[i] == 1.0:
                    inds_flips.append(inds_flips_and_flops[i])
            for ind in inds_flips:
                input_stream[n, ind: ind + self.n_refractory] = 1.0
            for ind in inds_flops:
                input_stream[n, ind: ind + self.n_refractory] = -1.0

            last_flip_ind = 0
            last_flop_ind = 0
            for i in range(self.n_steps):
                if i in inds_flips:
                    last_flip_ind = i
                elif i in inds_flops:
                    last_flop_ind = i
                if last_flop_ind < last_flip_ind:
                    target_stream[n, i] = 1.0
                elif last_flop_ind > last_flip_ind:
                    target_stream[n, i] = -1.0
            condition[n] = {"inds_flips": inds_flips, "inds_flops": inds_flops}
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