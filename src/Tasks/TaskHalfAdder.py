from copy import deepcopy
import numpy as np
from rnn_coach.src.Tasks.TaskBase import Task

class TaskHalfAdder(Task):
    def __init__(self, n_steps, task_params):
        '''
        :param n_steps: number of steps in the trial
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=2, n_outputs=1, task_params=task_params)
        self.task_params = task_params
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
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

        input_stream[0, self.stim_on:self.stim_off] = v1
        input_stream[1, self.stim_on:self.stim_off] = v2

        # Target stream
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        if self.n_oututs == 1:
            target_stream[0, self.dec_on:self.dec_off] = (v1 + v2) % 2
        elif self.n_outputs == 2:
            target_stream[int((v1 + v2) % 2), self.dec_on:self.dec_off] = 1
        condition = {"v1": v1, "v2": v2, "output" : (v1 + v2) % 2}
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

