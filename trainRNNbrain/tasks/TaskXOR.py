from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskXOR(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on, stim_off, dec_on, dec_off, n_rep=64, seed=None):
        '''
        :param n_steps: number of steps in the trial
        '''
        Task.__init__(self, n_steps, n_inputs=n_inputs, n_outputs=n_outputs, seed=seed)
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.n_rep = n_rep

    def generate_input_target_stream(self, logical_values):
        '''
        '''
        # Cue input stream
        v1 = logical_values[0]
        v2 = logical_values[1]
        input_stream = np.zeros((self.n_inputs, self.n_steps))

        ind1 = 0 if v1 == 0 else 1
        ind2 = 2 if v2 == 0 else 3

        input_stream[ind1, self.stim_on:self.stim_off] = 1
        input_stream[ind2, self.stim_on:self.stim_off] = 1

        out_ind = 1 if v1 == v2 else 0
        # Target stream
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        target_stream[out_ind, self.dec_on:self.dec_off] = 1
        condition = {"v1": v1, "v2": v2, "match" : bool(out_ind)}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        '''
        '''
        inputs = []
        targets = []
        conditions = []
        for i in range(self.n_rep):
            for logical_values in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                input_stream, target_stream, condition = self.generate_input_target_stream(logical_values)
                inputs.append(deepcopy(input_stream))
                targets.append(deepcopy(target_stream))
                conditions.append(deepcopy(condition))

        # batch_size should be a last dimension
        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        if shuffle:
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
        return inputs, targets, conditions
