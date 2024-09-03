from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskBlockDM(Task):
    def __init__(self, n_steps, n_inputs, n_outputs,
                 cue_on, cue_off,
                 stim_on, stim_off,
                 dec_on, dec_off, coherences, seed=None):

        Task.__init__(self, n_steps=n_steps, n_inputs=n_inputs, n_outputs=n_outputs, seed=seed)
        self.cue_on = cue_on
        self.cue_off = cue_off
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.coherences = coherences

    def generate_input_target_stream(self, block, coherence):
        '''
        generate an input and target for a single trial with the supplied coherences
        :param block: could be either True of False. If True - output zero for the entire duration of the trial
        if False - output standard decision-making result
        :param coherence: coherence of information in a channel, range: (-1, 1)
        :return: input_stream, target_stream
        input_stream - input time series (both context and sensory): n_inputs x num_steps
        target_stream - time sereis reflecting the correct decision: num_outputs x num_steps
        '''

        # given the context and coherences of signals
        # generate input array (n_inputs, n_steps)
        # and target array (ideal output of the Decision-making system)

        # Cue input stream
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        input_stream[0, self.cue_on:self.cue_off] = int(block) * np.ones(self.cue_off - self.cue_on)
        input_stream[1, self.stim_on - 1:self.stim_off] = coherence * np.ones([self.stim_off - self.stim_on + 1])

        # Target stream
        target_stream = np.zeros((1, self.n_steps))
        target_stream[0, self.dec_on - 1:self.dec_off] = 0 if block else np.sign(coherence)
        return input_stream, target_stream

    def get_batch(self, shuffle=False):
        '''
        coherences: list containing range of coherences for each channel (e.g. [-1, -0.5, -0.25,  0, 0.25, 0.5, 1]
        :return: array of inputs, array of targets, and the conditions (context, coherences and the correct choice)
        '''
        inputs = []
        targets = []
        conditions = []
        for block in [True, False]:
            for c in self.coherences:
                input_stream, target_stream = self.generate_input_target_stream(block, c)
                inputs.append(deepcopy(input_stream))
                targets.append(deepcopy(target_stream))

        # batch_size should be a last dimension
        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        if shuffle:
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
        return inputs, targets, conditions