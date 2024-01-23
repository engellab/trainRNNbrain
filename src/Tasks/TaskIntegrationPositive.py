from copy import deepcopy
import numpy as np
from src.Tasks.TaskBase import Task


class TaskIntegrationPositive(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.w = task_params["w"]
        self.random_offset_range = task_params["random_offset_range"]
        # a tuple which defines the range for the inputs

    def generate_input_target_stream(self, InputDuration):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        if self.random_offset_range == 0 or self.random_offset_range is None:
            r = 0
        else:
            r = np.random.randint(self.random_offset_range)
        input_stream[0, r:r + InputDuration] = 1
        input_stream[1, :] = 1
        signal = input_stream[0, :]
        integrated_signal = np.cumsum(signal) * self.w
        for t, x in enumerate(integrated_signal):
            target_stream[0, t] = np.abs(x)

        condition = {"InputDuration" : InputDuration,
                     "integrated_signal": integrated_signal,
                     "signal" : signal,
                     "offset" : r}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False, batch_size = 200):
        inputs = []
        targets = []
        conditions = []
        for i in range(batch_size):
            t_max = batch_size
            InputDuration = i
            input_stream, target_stream, condition = self.generate_input_target_stream(InputDuration)
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