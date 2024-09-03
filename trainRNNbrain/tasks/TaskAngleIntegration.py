from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskAngleIntegration(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 w, amp_range, mu_blocks, min_block_length,
                 batch_size=256, seed=None):
        '''
        Two channels representing stirring to the left and to the right.
        By default, if no input is present, the network outputs in a channel corresponding to 0 degrees.
        when the input comes (the inputs are mutually exclusive), the angle should be integrated and the new output
        channel should start to be active (corresponding to the integrated angle)
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        # the rate with which the angle is integrated per 10 ms of time :
        # say the right channel is active with strength A, and after 20 ms of constant input to the right channel,
        # the integrated angle should be A * w * (20/10) = 2Aw.
        self.w = w
        # a tuple which defines the range for the inputs
        self.Amp_range = amp_range
        # the number of blocks during the trial
        self.mu = mu_blocks
        self.lmbd = self.mu / self.n_steps
        self.n_min_block_length = min_block_length
        self.batch_size = batch_size

    def generate_switch_times(self):
        inds = [0]
        last_ind = 0
        while last_ind < self.n_steps:
            r = self.rng.random()
            ind = last_ind + self.n_min_block_length + int(-(1 / self.lmbd) * np.log(r))
            if (ind < self.n_steps): inds.append(ind)
            last_ind = ind
        return inds

    def generate_input_target_stream(self):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))

        # generate timings for blocks
        inds = self.generate_switch_times()
        n_blocks = len(inds)
        if self.Amp_range[0] == self.Amp_range[1]:
            amps = [self.Amp_range[0] for i in range(n_blocks)]
        else:
            amps = [self.Amp_range[0] + self.rng.random() * (self.Amp_range[1] - self.Amp_range[0]) for i in range(n_blocks)]

        for i, amp in enumerate(amps):
            t1 = inds[i]
            t2 = self.n_steps if (i == len(inds) - 1) else inds[i + 1]
            ind_inp_channel = 0 if amp >= 0 else 1
            input_stream[ind_inp_channel, t1:t2] = np.abs(amp)
        input_stream[-1, :] = 1
        signal = input_stream[0, :] - input_stream[1, :]
        integrated_theta = np.cumsum(signal) * self.w
        # converting integrated theta to outputs:

        arc = 2 * np.pi / self.n_outputs
        for t, theta in enumerate(integrated_theta):
            ind_channel = int(np.floor(theta / arc))
            v = theta % arc
            target_stream[ind_channel % self.n_outputs, t] = 1 - v/arc
            target_stream[(ind_channel + 1) % self.n_outputs, t] = v/arc

        condition = {"amps": amps, "block_starts": inds, "integrated_theta": integrated_theta, "signal" : signal}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        for i in range(self.batch_size):
            input_stream, target_stream, condition = self.generate_input_target_stream()
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


class TaskAngleIntegrationSimplified(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, w, batch_size=120, seed=None):
        '''
        Two channels representing stirring to the left and to the right.
        By default, if no input is present, the network outputs in a channel corresponding to 0 degrees.
        when the input comes (the inputs are mutually exclusive), the angle should be integrated and the new output
        channel should start to be active (corresponding to the integrated angle)
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed=seed)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # the rate with which the angle is integrated per 10 ms of time :
        # say the right channel is active with strength A, and after 20 ms of constant input to the right channel,
        # the integrated angle should be A * w * (20/10) = 2Aw.
        self.w = w
        # a tuple which defines the range for the inputs

        self.batch_size = batch_size

    def generate_input_target_stream(self, ind_inp_channel, InputDuration):

        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        input_stream[ind_inp_channel, 10:InputDuration+10] = 1
        input_stream[-1, :] = 1
        signal = input_stream[0, :] - input_stream[1, :]
        integrated_theta = np.cumsum(signal) * self.w
        # converting integrated theta to outputs:

        arc = 2 * np.pi / self.n_outputs
        for t, theta in enumerate(integrated_theta):
            ind_channel = int(np.floor(theta / arc))
            v = theta % arc
            target_stream[ind_channel % self.n_outputs, t] = 1 - v/arc
            target_stream[(ind_channel + 1) % self.n_outputs, t] = v/arc

        condition = {"ind_channel": ind_channel,
                     "InputDuration" : InputDuration,
                     "integrated_theta": integrated_theta,
                     "signal" : signal}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        for inp_ind_channel in [0, 1]:
            for i in range(self.batch_size//2):
                t_max = (self.n_steps//2-10)
                InputDuration = int((float(i) / float(self.batch_size//2)) * t_max)
                input_stream, target_stream, condition = self.generate_input_target_stream(inp_ind_channel, InputDuration)
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