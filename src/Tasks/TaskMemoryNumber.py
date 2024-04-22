from copy import deepcopy
import numpy as np
from src.Tasks.TaskBase import Task

class TaskMemoryNumber(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on_range, stim_duration, recall_on, recall_off, Amp=2,
                 seed=None, batch_size=128):
        '''
        Given a one-channel input x in range (-2, 2) for a short period of time
        Output x after in the `recall' period signified by an additional input +1 in the second input channel.

        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed=seed)
        self.stim_on_range = stim_on_range
        self.stim_duration = stim_duration
        self.Amp = Amp
        self.recall_on = recall_on
        self.recall_off = recall_off
        self.batch_size = batch_size

    def generate_input_target_stream(self, number):
        stim_on = int(self.rng.uniform(*self.stim_on_range))
        duration = self.stim_duration
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        input_stream[0, stim_on: stim_on + duration] = np.maximum(number, 0)
        input_stream[1, stim_on: stim_on + duration] = np.maximum(-number, 0)
        input_stream[2, self.recall_on: self.recall_off] = 1

        target_stream[0, self.recall_on: self.recall_off] = np.maximum(number, 0)
        target_stream[1, self.recall_on: self.recall_off] = np.maximum(-number, 0)
        condition = {"number": number, "stim_on" : stim_on, "duration" : duration}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        numbers = np.linspace(-self.Amp, self.Amp, self.batch_size)
        for number in numbers:
            input_stream, target_stream, condition = self.generate_input_target_stream(number)
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
