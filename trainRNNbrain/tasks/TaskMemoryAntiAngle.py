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
        num_angle_encoding_outs = self.n_outputs
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
        B = self.batch_size - 1
        thetas = 2 * np.pi * np.linspace(0, 1, B + 1)[:-1]

        nin, nout = self.n_inputs - 2, self.n_outputs
        stim_sl = slice(self.stim_on, self.stim_off)
        rec_sl = slice(self.recall_on, self.recall_off)

        inputs = np.zeros((B, self.n_inputs, self.n_steps))
        targets = np.zeros((B, self.n_outputs, self.n_steps))

        arc = 2 * np.pi / nin
        i = np.floor(thetas / arc).astype(int) % nin
        v = np.mod(thetas, arc)
        a0, a1 = 1 - v / arc, v / arc
        b = np.arange(B)

        inputs[b, i, stim_sl] = a0[:, None]
        inputs[b, (i + 1) % nin, stim_sl] = a1[:, None]
        inputs[:, -2, :] = 1
        inputs[:, -1, rec_sl] = 1

        theta_hat = np.mod(thetas + np.pi, 2 * np.pi)
        arc_hat = 2 * np.pi / nout
        j = np.floor(theta_hat / arc_hat).astype(int) % nout
        w = np.mod(theta_hat, arc_hat)
        c0, c1 = 1 - w / arc_hat, w / arc_hat

        targets[b, j, rec_sl] = c0[:, None]
        targets[b, (j + 1) % nout, rec_sl] = c1[:, None]

        inputs = np.transpose(inputs, (1, 2, 0))
        targets = np.transpose(targets, (1, 2, 0))

        conditions = []
        for k, theta in enumerate(thetas):
            theta_enc = [float(np.round(inputs[m, stim_sl.start, k], 2)) for m in range(nin)]
            anti_enc = [float(np.round(targets[m, rec_sl.start, k], 2)) for m in range(nout)]
            conditions.append({
                "Theta": float(np.round(360 * theta / (2 * np.pi), 1)),
                "stim_on": self.stim_on, "stim_off": self.stim_off,
                "recall_on": self.recall_on, "recall_off": self.recall_off,
                "Theta encoding": theta_enc,
                "Anti-Theta encoding": anti_enc,
            })

        if shuffle:
            perm = self.rng.permutation(B)
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[p] for p in perm]

        return inputs, targets, conditions


