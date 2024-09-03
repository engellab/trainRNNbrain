from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskSquareNumber(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on, stim_off, dec_on, dec_off,
                 batch_size=256, seed=None):
        '''
        2 inputs: one for the number, another one - constant input (bias)
        The output should be the squared input in the first channel
        the input to the first channel belongs to (0, 1)
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=n_inputs, n_outputs=n_outputs, seed=seed)
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.batch_size = batch_size

    def generate_input_target_stream(self, inp_val):
        '''
        '''
        # Cue input stream
        input_stream = np.zeros((self.n_inputs, self.n_steps))

        input_stream[0, self.stim_on:self.stim_off] = inp_val
        input_stream[-1, self.stim_on:self.stim_off] = 1

        # Target stream
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        target_stream[0, self.dec_on:self.dec_off] = inp_val ** 2
        condition = {"inp_val": inp_val, "out_val" : inp_val ** 2}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        '''
        '''
        inputs = []
        targets = []
        conditions = []
        for inp_val in np.linspace(0, 1, self.batch_size):
            input_stream, target_stream, condition = self.generate_input_target_stream(inp_val)
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

