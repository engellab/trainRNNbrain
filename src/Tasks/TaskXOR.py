from copy import deepcopy
import numpy as np
from rnn_coach.src.Tasks.TaskBase import Task

class TaskXOR(Task):
    def __init__(self, n_steps, task_params):
        '''
        :param n_steps: number of steps in the trial
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=4, n_outputs=2, task_params=task_params)
        self.task_params = task_params
        self.n_steps = n_steps
        self.n_inputs = 4
        self.n_outputs = 2
        self.stim_on = self.task_params["stim_on"]
        self.stim_off = self.task_params["stim_off"]
        self.dec_on = self.task_params["dec_on"]
        self.dec_off = self.task_params["dec_off"]

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

    def get_batch(self, shuffle=False, num_rep=64):
        '''
        '''
        inputs = []
        targets = []
        conditions = []
        for i in range(num_rep):
            for logical_values in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                input_stream, target_stream, condition = self.generate_input_target_stream(logical_values)
                inputs.append(deepcopy(input_stream))
                targets.append(deepcopy(target_stream))

        # batch_size should be a last dimension
        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        if shuffle:
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
        return inputs, targets, conditions

