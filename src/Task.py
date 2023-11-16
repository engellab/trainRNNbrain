from copy import deepcopy

import numpy as np

'''
Class which generates the input time-series and the correct output for the CDDM task for multiple coherences
'''


class Task():
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        :param n_inputs: number of input channels
        :param num_outputs: number of target output-time series.
        '''
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.task_params = task_params
        self.seed = task_params["seed"]
        if not (self.seed is None):
            self.rng = np.random.default_rng(seed=self.seed)
        else:
            self.rng = np.random.default_rng()

    def generate_input_target_stream(self, **kwargs):
        '''
        input_stream should have a dimensionality n_inputs x n_steps
        target_stream should have a dimensionality n_outputs x n_steps
        :param kwargs:
        :return:
        '''
        raise NotImplementedError("This is a generic Task class!")

    def get_batch(self, **kwargs):
        raise NotImplementedError("This is a generic Task class!")

class TaskIdentity(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        for tanh neurons only
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def generate_input_target_stream(self, values):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        for i in range(values.shape[0]):
            input_stream[i, :] = values[i]
            target_stream[i, :] = values[i]
        condition = {"values" : values}
        return input_stream, target_stream, condition

    def get_batch(self, batch_size=256, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        for i in range(batch_size):
            values = np.random.rand(self.n_inputs)
            input_stream, target_stream, condition = self.generate_input_target_stream(values)
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


class TaskCDDM(Task):

    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        :param n_steps: number of steps in the trial, default is 750
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
        :param n_steps: number of steps in the trial, default is 750
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


class TaskCDDMMante(Task):

    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        :param n_steps: number of steps in the trial, default is 750
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


class TaskDelayDM(Task):
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

class TaskDMTS(Task):

    def __init__(self, n_steps, n_inputs, task_params, n_outputs=2):
        '''
        :param n_steps: number of steps in the trial, default is 750
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.stim_on_sample = self.task_params["stim_on_sample"]
        self.stim_off_sample = self.task_params["stim_off_sample"]
        self.stim_on_match = self.task_params["stim_on_match"]
        self.stim_off_match = self.task_params["stim_off_match"]
        self.dec_on = self.task_params["dec_on"]
        self.dec_off = self.task_params["dec_off"]
        self.random_window = self.task_params["random_window"]

    def generate_input_target_stream(self, num_sample_channel, num_match_channel):
        if self.random_window == 0:
            random_offset_1 = random_offset_2 = 0
        else:
            random_offset_1 = self.rng.integers(-self.random_window, self.random_window)
            random_offset_2 = self.rng.integers(-self.random_window, self.random_window)
        input_stream = np.zeros([self.n_inputs, self.n_steps])
        input_stream[num_sample_channel, self.stim_on_sample + random_offset_1:self.stim_off_sample + random_offset_1] = 1.0
        input_stream[num_match_channel, self.stim_on_match + random_offset_2:self.stim_off_match + random_offset_2] = 1.0
        input_stream[2, self.dec_on:self.dec_off] = 1.0 # to signify the decision period

        condition = {"num_sample_channel" : num_sample_channel,
                     "num_match_channel" : num_match_channel,
                     "sample_on" : self.stim_on_sample + random_offset_1,
                     "sample_off" : self.stim_off_sample + random_offset_1,
                     "match_on" : self.stim_on_match + random_offset_2,
                     "match_off": self.stim_off_match + random_offset_2,
                     "dec_on" : self.dec_on,
                     "dec_off" : self.dec_off}

        # Target stream
        target_stream = np.zeros((2, self.n_steps))
        if (num_sample_channel == num_match_channel):
            target_stream[0, self.dec_on: self.dec_off] = 1
        elif (num_sample_channel != num_match_channel):
            target_stream[1, self.dec_on: self.dec_off] = 1

        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False, num_rep = 64):
        # batch size = 256 for two inputs
        inputs = []
        targets = []
        conditions = []
        for i in range(num_rep):
            for num_sample_channel in range(self.n_inputs - 1):
                for num_match_channel in range(self.n_inputs - 1):
                    correct_choice = 1 if (num_sample_channel == num_match_channel) else -1
                    input_stream, target_stream, condition = self.generate_input_target_stream(num_sample_channel, num_match_channel)
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


class TaskNBitFlipFlop(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        for tanh neurons only
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.mu = self.task_params["mu"]
        self.n_refractory = self.n_flip = self.task_params["n_flip_steps"]
        self.lmbd = self.mu / self.n_steps

    def generate_flipflop_times(self):
        inds = []
        last_ind = 0
        while last_ind < self.n_steps:
            r = self.rng.random()
            ind = last_ind + self.n_refractory + int(-(1 / self.lmbd) * np.log(r))
            if (ind < self.n_steps): inds.append(ind)
            last_ind = ind
        return inds

    def generate_input_target_stream(self):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        condition = {}
        for n in range(self.n_inputs):
            inds_flips_and_flops = self.generate_flipflop_times()
            mask = [0 if np.random.rand() < 0.5 else 1 for i in range(len(inds_flips_and_flops))]
            inds_flips = []
            inds_flops = []
            for i in range(len(inds_flips_and_flops)):
                if mask[i] == 0:
                    inds_flops.append(inds_flips_and_flops[i])
                elif mask[i] == 1.0:
                    inds_flips.append(inds_flips_and_flops[i])
            for ind in inds_flips:
                input_stream[n, ind: ind + self.n_refractory] = 1.0
            for ind in inds_flops:
                input_stream[n, ind: ind + self.n_refractory] = -1.0

            last_flip_ind = 0
            last_flop_ind = 0
            for i in range(self.n_steps):
                if i in inds_flips:
                    last_flip_ind = i
                elif i in inds_flops:
                    last_flop_ind = i
                if last_flop_ind < last_flip_ind:
                    target_stream[n, i] = 1.0
                elif last_flop_ind > last_flip_ind:
                    target_stream[n, i] = -1.0
            condition[n] = {"inds_flips": inds_flips, "inds_flops": inds_flops}
        return input_stream, target_stream, condition

    def get_batch(self, batch_size=256, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        for i in range(batch_size):
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


class TaskMemoryAntiNumber(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        Given an one-channel input x in range (-2, 2) for a short period of time
        Output -x after in the `recall' period signified by an additional input +1 in the second input channel.

        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.stim_on_range = task_params["stim_on_range"]
        self.stim_duration = task_params["stim_duration"]
        self.recall_on = task_params["recall_on"]
        self.recall_off = task_params["recall_off"]

    def generate_input_target_stream(self, number):
        stim_on = int(self.rng.uniform(*self.stim_on_range))
        duration = self.stim_duration
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        input_stream[0, stim_on: stim_on + duration] = np.maximum(number, 0)
        input_stream[1, stim_on: stim_on + duration] = np.maximum(-number, 0)
        input_stream[2, self.recall_on: self.recall_off] = 1

        target_stream[0, self.recall_on: self.recall_off] = np.maximum(-number, 0)
        target_stream[1, self.recall_on: self.recall_off] = np.maximum(number, 0)
        condition = {"number": number, "stim_on" : stim_on, "duration" : duration}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        numbers = np.linspace(-2, 2, 128)
        for number in numbers:
            input_stream, target_stream, condition = self.generate_input_target_stream(number)
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


class TaskMemoryAntiAngle(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        Given a four-channel input 2*cos(theta) and 2*sin(theta) specifying an angle theta (present only for a short period of time),
        Output 2*cos(theta+pi), 2*sin(theta+pi) in the recall period (signified by +1 provided in the third input-channel)
        This task is similar (but not exactly the same) to the task described in
        "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs"
        Laura Driscoll1, Krishna Shenoy, David Sussillo
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

        input_stream[ind_channel % num_angle_encoding_inps, stim_on: stim_off] = (1 - v/arc)
        input_stream[(ind_channel + 1) % num_angle_encoding_inps, stim_on: stim_off] = v/arc
        input_stream[-1, self.recall_on: self.recall_off] = 1

        # Supplying it with an explicit instruction to recall the theta + 180
        theta_hat = (theta + np.pi) % (2 * np.pi)
        arc_hat = (2 * np.pi) / num_angle_encoding_outs
        ind_channel = int(np.floor(theta_hat / arc_hat))
        w = theta_hat % arc_hat

        target_stream[ind_channel % num_angle_encoding_outs, self.recall_on: self.recall_off] = 1 - w/arc_hat
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


class TaskAngleIntegration(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        Two channels representing stirring to the left and to the right.
        By default, if no input is present, the network outputs in a channel corresponding to 0 degrees.
        when the input comes (the inputs are mutually exclusive), the angle should be integrated and the new output
        channel should start to be active (corresponding to the integrated angle)
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # the rate with which the angle is integrated per 10 ms of time :
        # say the right channel is active with strength A, and after 20 ms of constant input to the right channel,
        # the integrated angle should be A * w * (20/10) = 2Aw.
        self.w = task_params["w"]
        # a tuple which defines the range for the inputs
        self.Amp_range = task_params["amp_range"]
        # the number of blocks during the trial
        self.mu = task_params["mu_blocks"]
        self.lmbd = self.mu / self.n_steps
        self.n_min_block_length = task_params["min_block_length"]

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
            ind_channel = 0 if amp >= 0 else 1
            input_stream[ind_channel, t1:t2] = amp
        input_stream[-1, :] = 1
        signal = np.sum(input_stream, axis=0)
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

    def get_batch(self, shuffle=False, batch_size = 256):
        inputs = []
        targets = []
        conditions = []
        for i in range(batch_size):
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
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        Two channels representing stirring to the left and to the right.
        By default, if no input is present, the network outputs in a channel corresponding to 0 degrees.
        when the input comes (the inputs are mutually exclusive), the angle should be integrated and the new output
        channel should start to be active (corresponding to the integrated angle)
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # the rate with which the angle is integrated per 10 ms of time :
        # say the right channel is active with strength A, and after 20 ms of constant input to the right channel,
        # the integrated angle should be A * w * (20/10) = 2Aw.
        self.w = task_params["w"]
        # a tuple which defines the range for the inputs

    def generate_input_target_stream(self, ind_channel, InputDuration):

        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        input_stream[ind_channel, 10:InputDuration+10] = 1
        input_stream[-1, :] = 1
        signal = np.sum(input_stream, axis=0)
        integrated_theta = np.cumsum(signal) * self.w
        # converting integrated theta to outputs:

        arc = 2 * np.pi / self.n_outputs
        for t, theta in enumerate(integrated_theta):
            ind_channel = int(np.floor(theta / arc))
            v = theta % arc
            target_stream[ind_channel % self.n_outputs, t] = 1 - v/arc
            target_stream[(ind_channel + 1) % self.n_outputs, t] = v/arc

        condition = {"ind_channel": ind_channel, "integrated_theta": integrated_theta, "signal" : signal}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False, batch_size = 120):
        inputs = []
        targets = []
        conditions = []
        for ind_channel in [0, 1]:
            for i in range(batch_size//2):
                t_max = (self.n_steps//2-10)
                InputDuration = int((float(i) / float(batch_size//2)) * t_max)
                input_stream, target_stream, condition = self.generate_input_target_stream(ind_channel, InputDuration)
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


class TaskBlockDM(Task):

    def __init__(self, n_steps, task_params):
        '''
        :param n_steps: number of steps in the trial, default is 750
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=2, n_outputs=1, task_params=task_params)
        self.task_params = task_params
        self.n_steps = n_steps
        self.n_inputs = 2
        self.n_outputs = 1
        self.cue_on = self.task_params["cue_on"]
        self.cue_off = self.task_params["cue_off"]
        self.stim_on = self.task_params["stim_on"]
        self.stim_off = self.task_params["stim_off"]
        self.dec_on = self.task_params["dec_on"]
        self.dec_off = self.task_params["dec_off"]
        self.coherences = self.task_params["coherences"]

    def generate_input_target_stream(self, block, coherence):
        '''
        generate an input and target for a single trial with the supplied coherences
        :param block: could be either True of False. If True - output zero for the entire duration of the trial
        if False - output standard Decision Making result
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
        input_stream[0, self.cue_on:self.cue_off] = np.int(block) * np.ones(self.cue_off - self.cue_on)
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
        coherences = self.task_params["coherences"]
        inputs = []
        targets = []
        conditions = []
        for block in [True, False]:
            for c in coherences:
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


class TaskColorDiscriminationRGB(Task):
    def __init__(self, n_steps,n_inputs=3, n_outputs=12, task_params=None):

        '''
        Given a rgb color get lms representation and then get the hsv representation
        # bin hue into 12 colors:
        # red, red-orange, orange, orange-yellow
        # yellow, yellow-green, green, cyan
        # blue, blue-violet, violet, magenta
        '''
        import colorsys
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = 3
        self.n_outputs = 12
        self.color_on = task_params["color_on"]
        self.color_off = task_params["color_off"]
        self.colors = ["red", "vermillion", "orange", "amber",
                       "yellow", "chartreuse", "green", "cyan",
                       "blue", "indigo", "violet", "magenta"]
        self.rgb_to_hsv = colorsys.rgb_to_hsv
        self.hsv_to_rgb = colorsys.hsv_to_rgb

    def generate_input_target_stream(self, rgb):
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        input_stream = np.hstack([rgb.reshape(-1, 1)] * self.n_steps)
        hue = self.rgb_to_hsv(*rgb)[0]
        # bin hue into 12 colors
        color_ind = int(((hue + 1.0 / (2 * self.n_outputs)) % 1) // (1.0 / self.n_outputs))
        target_stream[color_ind, self.color_on: self.color_off] = 1.0
        condition = {"rgb": np.round(np.array(rgb), 4),
                     "color": self.colors[color_ind],
                     "hue": 360 * np.round(hue, 4),
                     "color_bin": color_ind}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        batch_size = 360
        for i in range(batch_size):
            hsv = (i * (1./batch_size), 1, 1)
            rgb = np.array(self.hsv_to_rgb(*hsv))
            input_stream, target_stream, condition = self.generate_input_target_stream(rgb)
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

if __name__ == '__main__':
    # n_steps = 750
    # n_inputs = 6
    # n_outputs = 2
    # task_params = dict()
    # task_params["coherences"] = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    # task_params["cue_on"] = 0
    # task_params["cue_off"] = 750
    # task_params["stim_on"] = 250
    # task_params["stim_off"] = 750
    # task_params["dec_on"] = 500
    # task_params["dec_off"] = 750
    # task = TaskCDDM(n_steps, n_inputs, n_outputs, task_params)
    # inputs, targets, conditions = task.get_batch()
    # print(inputs.shape, targets.shape)

    # n_steps = 750
    # n_inputs = 2
    # n_outputs = 2
    # task_params = dict()
    # task_params["n_steps"] = n_steps
    # task_params["n_inputs"] = n_inputs
    # task_params["n_outputs"] = n_outputs
    # task_params["stim_on_sample"] = 100
    # task_params["stim_off_sample"] = 200
    # task_params["stim_on_match"] = 300
    # task_params["stim_off_match"] = 400
    # task_params["dec_on"] = 500
    # task_params["dec_off"] = 750
    # task = TaskDMTS(n_steps, n_inputs, task_params)
    # inputs, targets, conditions = task.get_batch()
    # print(inputs.shape, targets.shape)
    #
    #
    # n_steps = 750
    # n_inputs = 2
    # n_outputs = 2
    # task_params = dict()
    # task_params["n_steps"] = n_steps
    # task_params["n_inputs"] = n_inputs
    # task_params["n_outputs"] = n_outputs
    # task_params["n_flip_steps"] = 20
    # task_params["mu"] = 7
    # task = TaskNBitFlipFlop(n_steps, n_inputs, n_outputs, task_params)
    # inputs, targets, conditions = task.get_batch()
    # print(inputs.shape, targets.shape)

    # n_steps = 320
    # n_inputs = 3
    # n_outputs = 2
    # task_params = dict()
    # task_params["stim_on"] = n_steps // 8
    # task_params["stim_off"] = 3 * n_steps//16
    # task_params["recall_on"] = 5 * n_steps//8
    # task_params["recall_off"] = n_steps
    # task = TaskMemoryAntiAngle(n_steps, n_inputs, n_outputs, task_params)
    # inputs, targets, conditions = task.get_batch()
    # print(inputs.shape, targets.shape)

    # n_steps = 320
    # n_inputs = 2
    # n_outputs = 4
    # task_params = dict()
    # task_params["stim_on_range"] = [n_steps // 8,  3 * n_steps // 16]
    # task_params["stim_duration"] = 15
    # task_params["recall_on"] = 5 * n_steps // 8
    # task_params["recall_off"] = n_steps
    # task_params["seed"] = 0
    # task = TaskMemoryAntiNumber(n_steps, n_inputs, n_outputs, task_params)
    # inputs, targets, conditions = task.get_batch()
    # print(inputs.shape, targets.shape)

    n_steps = 320
    n_inputs = 2
    n_outputs = 4
    task_params = dict()
    task_params["w"] = 0.1 / (2 * np.pi)
    task_params["amp_range"] = (-1, 1)
    task_params["mu_blocks"] = 8
    task_params["min_block_length"] = 10
    task_params["seed"] = 0
    task = TaskAngleIntegration(n_steps, n_inputs, n_outputs, task_params)
    inputs, targets, conditions = task.get_batch()
    print(inputs.shape, targets.shape)
