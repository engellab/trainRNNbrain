from copy import deepcopy
import numpy as np
from rnn_coach.src.Tasks.TaskBase import Task

class TaskMemoryAngle(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''

        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.stim_on = task_params["stim_on"]
        self.stim_off = task_params["stim_off"]
        self.random_window = task_params["random_window"]
        self.recall_on = task_params["recall_on"]
        self.recall_off = task_params["recall_off"]

    def generate_input_target_stream(self, theta):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))

        random_offset = self.rng.integers(-self.random_window, self.random_window) if (self.random_window != 0) else 0
        stim_on = self.stim_on + random_offset
        stim_off = self.stim_off + random_offset
        num_angle_encoding_inps = (self.n_inputs - 1)
        num_angle_encoding_outs = (self.n_inputs - 1)
        arc = 2 * np.pi / num_angle_encoding_inps
        ind_channel = int(np.floor(theta / arc))
        v = theta % arc

        input_stream[ind_channel, stim_on: stim_off] = (1 - v/arc)
        input_stream[(ind_channel + 1) % num_angle_encoding_inps, stim_on: stim_off] = v/arc
        input_stream[-1, self.recall_on: self.recall_off] = 1

        # Supplying it with an explicit instruction to recall the theta + 180
        theta_hat = (theta) % (2 * np.pi)
        arc_hat = (2 * np.pi) / num_angle_encoding_outs
        ind_channel = int(np.floor(theta_hat / arc_hat))
        w = theta_hat % arc_hat

        target_stream[ind_channel, self.recall_on: self.recall_off] = 1 - w/arc_hat
        target_stream[(ind_channel + 1) % num_angle_encoding_outs, self.recall_on: self.recall_off] = w/arc_hat

        condition = {"theta": theta, "stim_on": stim_on, "stim_off" : stim_off,
                     "recall_on" : self.recall_on, "recall_off" : self.recall_off}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        thetas = 2 * np.pi * np.linspace(0, 1, 73)[:-1]
        for theta in thetas:
            input_stream, target_stream, condition = self.generate_input_target_stream(theta)
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

