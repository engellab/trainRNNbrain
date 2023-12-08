from copy import deepcopy
import numpy as np
from src.Tasks.TaskBase import Task

class TaskModulo1Addition(Task):
    def __init__(self, n_steps, task_params):
        '''
        3 inputs: two for the two number to be added modulo 1, another one - constant input (bias)
        The output should be the modul0 1 addition of the two numbers
        the input to the first two channel belongs to (0, 1)
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=2, n_outputs=1, task_params=task_params)
        self.task_params = task_params
        self.n_steps = n_steps
        self.n_inputs = 3
        self.n_outputs = 1
        self.stim_on = self.task_params["stim_on"]
        self.stim_off = self.task_params["stim_off"]
        self.dec_on = self.task_params["dec_on"]
        self.dec_off = self.task_params["dec_off"]

    def generate_input_target_stream(self, inp_vals):
        '''
        '''
        # Cue input stream
        input_stream = np.zeros((self.n_inputs, self.n_steps))

        input_stream[0, self.stim_on:self.stim_off] = inp_vals[0]
        input_stream[1, self.stim_on:self.stim_off] = inp_vals[1]
        input_stream[-1, self.stim_on:self.stim_off] = 1

        # Target stream
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        target_stream[0, self.dec_on:self.dec_off] = (inp_vals[0] + inp_vals[1]) % 1.0
        condition = {"inp_vals": inp_vals, "out_val" : (inp_vals[0] + inp_vals[1]) % 1.0}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        '''
        better for it to be an odd batch size so that it includes zero as an input
        '''
        inputs = []
        targets = []
        conditions = []
        for inp_val1 in np.linspace(0, 1, 16):
            for inp_val2 in np.linspace(0, 1, 16):
                input_stream, target_stream, condition = self.generate_input_target_stream(np.array([inp_val1, inp_val2]))
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

