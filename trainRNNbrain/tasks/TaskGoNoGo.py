from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskGoNoGo(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on, stim_off, cue_on, cue_off, batch_size=256, seed=None):
        '''
        :param n_steps: number of steps in the trial
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=n_inputs, n_outputs=n_outputs, seed=seed)
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.cue_on = cue_on
        self.cue_off = cue_off
        self.batch_size = batch_size

    def generate_input_target_stream(self, input_value):
        '''
        '''
        # Cue input stream
        input_stream = np.zeros((self.n_inputs, self.n_steps))

        input_stream[0, self.stim_on:self.stim_off] = input_value
        input_stream[1, self.cue_on:self.cue_off] = 1
        input_stream[2, :] = 1

        output_value = 0 if input_value <= 0.5 else 1
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        target_stream[0, self.cue_on:self.cue_off] = output_value
        condition = {"input_value": input_value, "output_value": output_value}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        '''
        '''
        inputs = []
        targets = []
        conditions = []
        for input_value in np.linspace(0, 1, self.batch_size):
            input_stream, target_stream, condition = self.generate_input_target_stream(input_value)
            inputs.append(deepcopy(input_stream))
            targets.append(deepcopy(target_stream))
            conditions.append(np.copy(condition))

        # batch_size should be a last dimension
        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        if shuffle:
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
        return inputs, targets, conditions

