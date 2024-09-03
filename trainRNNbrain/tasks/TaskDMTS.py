from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskDMTS(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on_sample, stim_off_sample,
                 stim_on_match, stim_off_match,
                 dec_on, dec_off,
                 random_window, seed=None):
        '''
        Delayed match to sample task: if the two subsequent stimuli were the same - match (make the choice to the right), if not - to the left
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.stim_on_sample = stim_on_sample
        self.stim_off_sample = stim_off_sample
        self.stim_on_match = stim_on_match
        self.stim_off_match = stim_off_match
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.random_window = random_window

    def generate_input_target_stream(self, num_sample_channel, num_match_channel):
        if self.random_window == 0:
            random_offset_1 = random_offset_2 = 0
        else:
            random_offset_1 = self.rng.integers(-self.random_window, self.random_window)
            random_offset_2 = self.rng.integers(-self.random_window, self.random_window)
        input_stream = np.zeros([self.n_inputs, self.n_steps])
        input_stream[num_sample_channel, self.stim_on_sample + random_offset_1:self.stim_off_sample + random_offset_1] = 1.0
        input_stream[num_match_channel, self.stim_on_match + random_offset_2:self.stim_off_match + random_offset_2] = 1.0
        input_stream[2, self.dec_on:self.dec_off] = 1.0 # to signify the decision period

        condition = {"num_sample_channel" : num_sample_channel,
                     "num_match_channel" : num_match_channel,
                     "sample_on" : self.stim_on_sample + random_offset_1,
                     "sample_off" : self.stim_off_sample + random_offset_1,
                     "match_on" : self.stim_on_match + random_offset_2,
                     "match_off": self.stim_off_match + random_offset_2,
                     "dec_on" : self.dec_on,
                     "dec_off" : self.dec_off}

        # Target stream
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        if self.n_outputs == 2:
            if (num_sample_channel == num_match_channel):
                target_stream[0, self.dec_on: self.dec_off] = 1
            elif (num_sample_channel != num_match_channel):
                target_stream[1, self.dec_on: self.dec_off] = 1
        else:
            if (num_sample_channel == num_match_channel):
                target_stream[0, self.dec_on: self.dec_off] = 1

        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False, num_rep = 64):

        # batch size = 256 for two inputs
        inputs = []
        targets = []
        conditions = []

        for i in range(num_rep):
            for num_sample_channel in range(self.n_inputs - 1):
                for num_match_channel in range(self.n_inputs - 1):
                    correct_choice = 1 if (num_sample_channel == num_match_channel) else -1
                    input_stream, target_stream, condition = self.generate_input_target_stream(num_sample_channel, num_match_channel)
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
