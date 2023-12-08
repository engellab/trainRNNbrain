from copy import deepcopy
import numpy as np
from src.Tasks.TaskBase import Task

class TaskCDDM(Task):

    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cue_on = self.task_params["cue_on"]
        self.cue_off = self.task_params["cue_off"]
        self.stim_on = self.task_params["stim_on"]
        self.stim_off = self.task_params["stim_off"]
        self.dec_on = self.task_params["dec_on"]
        self.dec_off = self.task_params["dec_off"]
        self.coherences = self.task_params["coherences"]

    def generate_input_target_stream(self, context, motion_coh, color_coh):
        '''
        generate an input and target for a single trial with the supplied coherences
        :param context: could be either 'motion' or 'color' (see Mante et. all 2013 paper, however the input structure is modified)
        :param motion_coh: coherence of information in motion channel, range: (-1, 1)
        :param color_coh: coherence of information in color channel, range: (-1, 1)
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

        # Transform coherence to signal
        motion_r = np.maximum(0, (1 + motion_coh) / 2)
        motion_l = np.maximum(0, (1 - motion_coh) / 2)
        color_r = np.maximum(0, (1 + color_coh) / 2)
        color_l = np.maximum(0, (1 - color_coh) / 2)

        # Cue input stream
        cue_input = np.zeros((self.n_inputs, self.n_steps))
        ind_ctxt = 0 if context == "motion" else 1
        cue_input[ind_ctxt, self.cue_on:self.cue_off] = np.ones(self.cue_off - self.cue_on)

        sensory_input = np.zeros((self.n_inputs, self.n_steps))
        # Motion input stream
        sensory_input[2, self.stim_on - 1:self.stim_off] = motion_r * np.ones([self.stim_off - self.stim_on + 1])
        sensory_input[3, self.stim_on - 1:self.stim_off] = motion_l * np.ones([self.stim_off - self.stim_on + 1])
        # Color input stream
        sensory_input[4, self.stim_on - 1:self.stim_off] = color_r * np.ones([self.stim_off - self.stim_on + 1])
        sensory_input[5, self.stim_on - 1:self.stim_off] = color_l * np.ones([self.stim_off - self.stim_on + 1])
        input_stream = cue_input + sensory_input

        # Target stream
        if self.n_outputs == 1:
            target_stream = np.zeros((1, self.n_steps))
            target_stream[0, self.dec_on - 1:self.dec_off] = np.sign(motion_coh) if (context == 'motion') else np.sign(
                color_coh)
        else:
            target_stream = np.zeros((self.n_outputs, self.n_steps))
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
        coherences = self.task_params["coherences"]
        inputs = []
        targets = []
        conditions = []
        for context in ["motion", "color"]:
            for c1 in coherences:
                for c2 in coherences:
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


class TaskCDDMplus(Task):

    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        the only difference with CDDM task is that the networks also has to output not only the decision, but the
        original sensory inputs as well

        Warning! for the network to perform CDDM one has to adjust the cost function so that the first two inputs
        are considered more important!
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cue_on = self.task_params["cue_on"]
        self.cue_off = self.task_params["cue_off"]
        self.stim_on = self.task_params["stim_on"]
        self.stim_off = self.task_params["stim_off"]
        self.dec_on = self.task_params["dec_on"]
        self.dec_off = self.task_params["dec_off"]
        self.coherences = self.task_params["coherences"]

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

        # Transform coherence to signal
        motion_r = (1 + motion_coh) / 2
        motion_l = 1 - motion_r
        color_r = (1 + color_coh) / 2
        color_l = 1 - color_r

        # Cue input stream
        cue_input = np.zeros((self.n_inputs, self.n_steps))
        ind_ctxt = 0 if context == "motion" else 1
        cue_input[ind_ctxt, self.cue_on:self.cue_off] = np.ones(self.cue_off - self.cue_on)

        sensory_input = np.zeros((self.n_inputs, self.n_steps))
        # Motion input stream
        sensory_input[2, self.stim_on - 1:self.stim_off] = motion_r * np.ones([self.stim_off - self.stim_on + 1])
        sensory_input[3, self.stim_on - 1:self.stim_off] = motion_l * np.ones([self.stim_off - self.stim_on + 1])
        # Color input stream
        sensory_input[4, self.stim_on - 1:self.stim_off] = color_r * np.ones([self.stim_off - self.stim_on + 1])
        sensory_input[5, self.stim_on - 1:self.stim_off] = color_l * np.ones([self.stim_off - self.stim_on + 1])
        input_stream = cue_input + sensory_input

        # Target stream
        target_stream = np.zeros((6, self.n_steps))
        relevant_coh = motion_coh if (context == 'motion') else color_coh
        if relevant_coh == 0.0:
            pass
        else:
            decision = np.sign(relevant_coh)
            ind = 0 if (decision == 1.0) else 1
            target_stream[ind, self.dec_on - 1:self.dec_off] = 1
        # extra outputs (the targets are set to the sensory inputs)
        target_stream[2, self.dec_on - 1:self.dec_off] = motion_r
        target_stream[3, self.dec_on - 1:self.dec_off] = motion_l
        target_stream[4, self.dec_on - 1:self.dec_off] = color_r
        target_stream[5, self.dec_on - 1:self.dec_off] = color_l
        return input_stream, target_stream

    def get_batch(self, shuffle=False):
        '''
        coherences: list containing range of coherences for each channel (e.g. [-1, -0.5, -0.25,  0, 0.25, 0.5, 1]
        :return: array of inputs, array of targets, and the conditions (context, coherences and the correct choice)
        '''
        coherences = self.task_params["coherences"]
        inputs = []
        targets = []
        conditions = []
        for context in ["motion", "color"]:
            for c1 in coherences:
                for c2 in coherences:
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

# if __name__ == '__main__':
#     n_steps = 750
#     n_inputs = 6
#     n_outputs = 2
#     task_params = dict()
#     task_params["coherences"] = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
#     task_params["cue_on"] = 0
#     task_params["cue_off"] = 750
#     task_params["stim_on"] = 250
#     task_params["stim_off"] = 750
#     task_params["dec_on"] = 500
#     task_params["dec_off"] = 750
#     task = TaskCDDM(n_steps, n_inputs, n_outputs, task_params)
#     inputs, targets, conditions = task.get_batch()
#     print(inputs.shape, targets.shape)
