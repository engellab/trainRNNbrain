from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskDelayDM(Task):
    '''Delayed decision making task: get the stimulus, wait and then make a decision after the dec_on cue comes in'''
    def __init__(self, n_steps, n_inputs, n_outputs, stim_on, stim_off, dec_on, dec_off, directions, seed=None):
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.directions = directions

    def generate_input_target_stream(self, direction):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        output_stream = np.zeros((self.n_outputs, self.n_steps))

        # add auditory cue to input
        if (direction != -1):  # no stim for catch trials
            input_stream[direction, self.stim_on:self.stim_off] = 1
            output_stream[direction, self.dec_on:self.dec_off] = 1
        condition = {"direction" : direction}
        # add go cue to input to channel 2 of input
        input_stream[2, self.dec_on:self.dec_off] = 1
        return input_stream, output_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []

        for d in self.directions:
            input, output, condition = self.generate_input_target_stream(d)
            inputs.append(deepcopy(input))
            targets.append(deepcopy(output))
            conditions.append(condition)

        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)

        if (shuffle):
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
        return inputs, targets, conditions