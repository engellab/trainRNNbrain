from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskCDDMMante(Task):
    '''Use with the tanh activation function only, as in the original paper Mante et al. (2013)'''
    def __init__(self, n_steps, n_inputs, n_outputs,
                 cue_on, cue_off, stim_on, stim_off, dec_on, dec_off,
                 coherences, seed=None):
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.cue_on = cue_on
        self.cue_off = cue_off
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.coherences = coherences


    def generate_input_target_stream(self, context, motion_coh, color_coh):
        '''
        generate an input and target for a single trial with the supplied coherences
        :param context: could be either 'motion' or 'color' (see Mante et. all 2013 paper)
        :param motion_coh: coherence of information in motion channel, range: (0, 1)
        :param color_coh: coherence of information in color channel, range: (0, 1)
        :return: input_stream, target_stream
        input_stream - input time series (both context and sensory): n_inputs x num_steps
        target_stream - time sereis reflecting the correct decision: num_outputs x num_steps

        :param protocol_dict: a dictionary which provides the trial structure:
        cue_on, cue_off - defines the timespan when the contextual information is supplied
        stim_on, stim_off - defines the timespan when the sensory information is supplied
        dec_on, dec_off - defines the timespan when the decision has to be present in the target stream
        all the values should be less than n_steps
        '''

        # given the context and coherences of signals
        # generate input array (n_inputs, n_steps)
        # and target array (ideal output of the Decision-making system)

        # Cue input stream
        cue_input = np.zeros((self.n_inputs, self.n_steps))
        ind_ctxt = 0 if context == "motion" else 1
        cue_input[ind_ctxt, self.cue_on:self.cue_off] = np.ones(self.cue_off - self.cue_on)

        sensory_input = np.zeros((self.n_inputs, self.n_steps))
        # Motion input stream
        sensory_input[2, self.stim_on - 1:self.stim_off] = motion_coh * np.ones([self.stim_off - self.stim_on + 1])
        sensory_input[3, self.stim_on - 1:self.stim_off] = color_coh * np.ones([self.stim_off - self.stim_on + 1])
        input_stream = cue_input + sensory_input

        # Target stream
        if self.n_outputs == 1:
            target_stream = np.zeros((1, self.n_steps))
            target_stream[0, self.dec_on - 1:self.dec_off] = np.sign(motion_coh) if (context == 'motion') else np.sign(
                color_coh)
        elif self.n_outputs == 2:
            target_stream = np.zeros((2, self.n_steps))
            relevant_coh = motion_coh if (context == 'motion') else color_coh
            if relevant_coh == 0.0:
                pass
            else:
                decision = np.sign(relevant_coh)
                ind = 0 if (decision == 1.0) else 1
                target_stream[ind, self.dec_on - 1:self.dec_off] = 1
        return input_stream, target_stream

    def get_batch(self, shuffle=False):
        '''
        coherences: list containing range of coherences for each channel (e.g. [-1, -0.5, -0.25,  0, 0.25, 0.5, 1]
        :return: array of inputs, array of targets, and the conditions (context, coherences and the correct choice)
        '''
        inputs = []
        targets = []
        conditions = []
        for context in ["motion", "color"]:
            for c1 in self.coherences:
                for c2 in self.coherences:
                    relevant_coh = c1 if context == 'motion' else c2
                    irrelevant_coh = c2 if context == 'motion' else c1
                    motion_coh = c1 if context == 'motion' else c2
                    color_coh = c1 if context == 'color' else c2
                    coh_pair = (relevant_coh, irrelevant_coh)

                    correct_choice = 1 if ((context == "motion" and motion_coh > 0) or (
                            context == "color" and color_coh > 0)) else -1
                    conditions.append({'context': context,
                                       'motion_coh': motion_coh,
                                       'color_coh': color_coh,
                                       'correct_choice': correct_choice})
                    input_stream, target_stream = self.generate_input_target_stream(context, coh_pair[0], coh_pair[1])
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
