import numpy as np

'''
Generic class-template for task which should contain 'generate_input_target_stream' and 'get_batch' methods
'''


class Task():
    def __init__(self, n_steps, n_inputs, n_outputs, seed):
        '''
        :param n_inputs: number of input channels
        :param num_outputs: number of target output-time series.
        '''
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.seed = seed
        if not (self.seed is None):
            self.rng = np.random.default_rng(seed=self.seed)
        else:
            self.rng = np.random.default_rng()

    def generate_input_target_stream(self, **kwargs):
        '''
        input_stream should have a dimensionality n_inputs x n_steps
        target_stream should have a dimensionality n_outputs x n_steps
        :param kwargs:
        :return:
        '''
        raise NotImplementedError("This is a generic Task class!")

    def get_batch(self, **kwargs):
        raise NotImplementedError("This is a generic Task class!")