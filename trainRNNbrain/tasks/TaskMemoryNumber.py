from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskMemoryNumber(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on, stim_duration, recall_on, recall_off,
                 random_window,
                 seed=None, batch_size=256):
        '''
        Given a one-channel input x in range (0, 1) for a short period of time
        Output x after in the `recall' period signified by an additional input +1 in the second input channel.

        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed=seed)
        self.stim_on = stim_on
        self.stim_duration = stim_duration
        self.recall_on = recall_on
        self.recall_off = recall_off
        self.batch_size = batch_size
        self.random_window = random_window

    def generate_input_target_stream(self, number):
        if self.random_window == 0:
            random_offset = 0
        else:
            random_offset = self.rng.integers(-self.random_window, self.random_window)
        stim_on = self.stim_on + random_offset
        duration = self.stim_duration
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        input_stream[0, stim_on: stim_on + duration] = number
        input_stream[1, self.recall_on: self.recall_off] = 1
        input_stream[2, :] = 1 # constant bias

        target_stream[0, self.recall_on: self.recall_off] = number
        condition = {"number": number, "stim_on" : stim_on, "duration" : duration}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        numbers = np.linspace(0, 1, self.batch_size)
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
