from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

'''
Task defined as follows: output whatever comes as an input (identity transformation)
'''

class TaskIdentity(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, batch_size=256, seed=None):
        '''
        for tanh neurons only
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.batch_size = batch_size

    def generate_input_target_stream(self, values):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        for i in range(values.shape[0]):
            input_stream[i, :] = values[i]
            target_stream[i, :] = values[i]
        condition = {"values" : values}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        for i in range(self.batch_size):
            values = np.random.rand(self.n_inputs)
            input_stream, target_stream, condition = self.generate_input_target_stream(values)
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