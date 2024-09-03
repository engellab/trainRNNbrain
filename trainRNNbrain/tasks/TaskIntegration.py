from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskIntegration(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 w, amp_range, mu_blocks, refractory_period, min_block_length,
                 batch_size=256, seed=None):
        '''
        Two input channels representing velocities of moving to the left and to the right.
        By default, if no input is present, the network outputs in a channel corresponding to 0 coordinate.
        when the input comes (the inputs are mutually exclusive), the coordinate should be integrated
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        # the rate with which the angle is integrated per 10 ms of time :
        # say the right channel is active with strength A, and after 20 ms of constant input to the right channel,
        # the integrated x should be A * w * (20/10) = 2Aw.
        self.w = w
        # a tuple which defines the range for the inputs
        self.Amp_range = amp_range
        # the number of blocks during the trial
        self.mu = mu_blocks
        self.refractory_period = refractory_period
        self.lmbd = self.mu / self.n_steps
        self.n_min_block_length = min_block_length
        self.batch_size = batch_size

    def generate_switch_times(self):
        inds = [0]
        last_ind = 0
        while last_ind < self.n_steps:
            r = self.rng.random()
            ind = last_ind + self.n_min_block_length + int(-(1 / self.lmbd) * np.log(r)) + self.refractory_period
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
            t2 = self.n_steps if (i == len(inds) - 1) else (inds[i + 1] - self.refractory_period)
            ind_channel = 0 if amp >= 0 else 1
            input_stream[ind_channel, t1: t2] = np.abs(amp)
            input_stream[ind_channel, t2: (t2 + self.refractory_period)] = 0
        input_stream[-1, :] = 1
        signal = input_stream[0, :] - input_stream[1, :]
        integrated_signal = np.cumsum(signal) * self.w
        # converting integrated x to outputs:
        for t, x in enumerate(integrated_signal):
            ind_channel = 0 if x > 0 else 1
            target_stream[ind_channel, t] = np.abs(x)
        condition = {"amps": amps, "block_starts": inds, "integrated_signal": integrated_signal, "signal" : signal}
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


class TaskIntegrationSimplified(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, w, batch_size=256, seed=None):
        '''
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed=seed)
        self.w = w
        self.batch_size = batch_size
        # a tuple which defines the range for the inputs

    def generate_input_target_stream(self, ind_inp_channel, InputDuration):

        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        input_stream[ind_inp_channel, 10:InputDuration+10] = 1
        input_stream[-1, :] = 1
        signal = input_stream[0, :] - input_stream[1, :]
        integrated_signal = np.cumsum(signal) * self.w
        for t, x in enumerate(integrated_signal):
            ind_channel = 0 if x > 0 else 1
            target_stream[ind_channel, t] = np.abs(x)

        condition = {"ind_channel": ind_channel,
                     "InputDuration" : InputDuration,
                     "integrated_signal": integrated_signal,
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