from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskMemoryAntiAngle(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 stim_on, stim_off, random_window, recall_on, recall_off,
                 batch_size=72, seed=None):
        '''
        Given a four-channel input 2*cos(theta) and 2*sin(theta) specifying an angle theta (present only for a short period of time),
        Output 2*cos(theta+pi), 2*sin(theta+pi) in the recall period (signified by +1 provided in the third input-channel)
        This task is similar (but not exactly the same) to the task described in
        "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs"
        Laura Driscoll1, Krishna Shenoy, David Sussillo
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.random_window = random_window
        self.recall_on = recall_on
        self.recall_off = recall_off
        self.batch_size = batch_size

    def generate_input_target_stream(self, theta):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))

        random_offset = self.rng.integers(-self.random_window, self.random_window) if (self.random_window != 0) else 0
        stim_on = self.stim_on + random_offset
        stim_off = self.stim_off + random_offset
        num_angle_encoding_inps = (self.n_inputs - 2)
        num_angle_encoding_outs = (self.n_inputs - 2)
        arc = 2 * np.pi / num_angle_encoding_inps
        ind_channel = int(np.floor(theta / arc))
        v = theta % arc

        input_stream[ind_channel % num_angle_encoding_inps, stim_on: stim_off] = (1 - v/arc)
        input_stream[(ind_channel + 1) % num_angle_encoding_inps, stim_on: stim_off] = v/arc
        input_stream[-2, :] = 1
        input_stream[-1, self.recall_on: self.recall_off] = 1

        # Supplying it with an explicit instruction to recall the theta + 180
        theta_hat = (theta + np.pi) % (2 * np.pi)
        arc_hat = (2 * np.pi) / num_angle_encoding_outs
        ind_channel = int(np.floor(theta_hat / arc_hat))
        w = theta_hat % arc_hat

        target_stream[ind_channel % num_angle_encoding_outs, self.recall_on: self.recall_off] = 1 - w/arc_hat
        target_stream[(ind_channel + 1) % num_angle_encoding_outs, self.recall_on: self.recall_off] = w/arc_hat

        theta_encoding = [np.round(input_stream[i, stim_on], 2) for i in range(num_angle_encoding_inps)]
        Anti_theta_encoding = [np.round(target_stream[i, self.recall_on], 2) for i in range(num_angle_encoding_outs)]
        condition = {"Theta": int(np.round(360 * theta/(2 * np.pi), 1)),
                     "stim_on": stim_on, "stim_off" : stim_off,
                     "recall_on" : self.recall_on, "recall_off" : self.recall_off,
                     "Theta encoding": theta_encoding,
                     "Anti-Theta encoding": Anti_theta_encoding}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        thetas = 2 * np.pi * np.linspace(0, 1, self.batch_size - 1)[:-1]

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

