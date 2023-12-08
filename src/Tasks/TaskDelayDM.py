from copy import deepcopy
import numpy as np
from src.Tasks.TaskBase import Task

class TaskDelayDM(Task):
    '''Delayed decision making task: get the stimulus, wait and then make a decision after the cue comes in'''
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cue_on = self.task_params["cue_on"]
        self.cue_off = self.task_params["cue_off"]
        self.go_on = self.task_params["go_on"]

    def generate_input_target_stream(self, direction):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        output_stream = np.zeros((self.n_outputs, self.n_steps))

        # add auditory cue to input
        if (direction != -1):  # no stim for catch trials
            input_stream[direction, self.cue_on:self.cue_off + 1] = 1
            output_stream[direction, self.go_on:] = 1
        condition = {"direction" : direction}
        # add go cue to input to channel 2 of input
        input_stream[2, self.go_on:] = 1
        # input_stream[3, :] = 1
        return input_stream, output_stream, condition

    def get_batch(self, shuffle=False):
        directions = self.task_params["directions"]
        inputs = []
        targets = []
        conditions = []

        for d in directions:
            input, output, condition = self.generate_input_target_stream(d)
            inputs.append(deepcopy(input))
            targets.append(deepcopy(output))
            conditions.append(condition)

        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        # assert(inputs.shape[-1] == self.batch_size)
        # assert(targets.shape[-1] == self.batch_size)

        if (shuffle):
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
        return inputs, targets, conditions