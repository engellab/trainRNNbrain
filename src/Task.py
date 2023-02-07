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
        :param shuffle: shuffle the final array
        :param generator_numpy: the random generator (for reproducibility, if using shuffle=True)
        :return: array of inputs, array of targets, and the conditions (context, coherences and the correct choice)
        '''
        coherences = self.task_params["coherences"]
        inputs = []
        targets = []
        conditions = []
        if self.rng is None:
            generator_numpy = np.random.default_rng()
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
            perm = self.rng.permutation(len(inputs))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
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

    def generate_input_target_stream(self, num_sample_channel, num_match_channel):
        input_stream = np.zeros([self.n_inputs, self.n_steps])
        input_stream[num_sample_channel, self.stim_on_sample:self.stim_off_sample] = 1.0
        input_stream[num_match_channel, self.stim_on_match:self.stim_off_match] = 1.0

        # Target stream
        target_stream = np.zeros((2, self.n_steps))
        if (num_sample_channel == num_match_channel):
            target_stream[0, self.dec_on: self.dec_off] = 1
        elif (num_sample_channel != num_match_channel):
            target_stream[1, self.dec_on: self.dec_off] = 1
        return input_stream, target_stream

    def get_batch(self, shuffle=False):
        # batch size = 256 for two inputs
        inputs = []
        targets = []
        conditions = []
        for num_sample_channel in range(self.n_inputs):
            for num_match_channel in range(self.n_inputs):
                correct_choice = 1 if (num_sample_channel == num_match_channel) else -1
                conditions.append({'num_sample_channel': num_sample_channel,
                                   'num_match_channel': num_match_channel,
                                   'correct_choice': correct_choice})
                input_stream, target_stream = self.generate_input_target_stream(num_sample_channel, num_match_channel)
                inputs.append(deepcopy(input_stream))
                targets.append(deepcopy(target_stream))

        inputs = 64 * inputs
        targets = 64 * targets
        conditions = 64 * conditions

        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        if shuffle:
            perm = self.rng.permutation(len(inputs))
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
            r = np.random.rand()
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

    def get_batch(self, batch_size=256, shuffle=False, generator_numpy=None):
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
            perm = self.rng.permutation(len(inputs))
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
        input_stream[0, stim_on: stim_on + duration] = number
        input_stream[1, self.recall_on: self.recall_off] = 1

        target_stream[0, self.recall_on: self.recall_off] = -number
        condition = {"number": number}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        numbers = np.linspace(-2, 2, 32)
        for number in numbers:
            input_stream, target_stream, condition = self.generate_input_target_stream(number)
            inputs.append(deepcopy(input_stream))
            targets.append(deepcopy(target_stream))
            conditions.append(deepcopy(condition))
        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        inputs = np.repeat(inputs, axis=2, repeats=11)
        targets = np.repeat(targets, axis=2, repeats=11)
        if shuffle:
            perm = self.rng.permutation(len(inputs))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
        return inputs, targets, conditions


class TaskMemoryAntiAngle(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, task_params):
        '''
        Given a two-channel input 2 cos(theta) and 2 sin(theta) specifying an angle theta (present only for a short period of time),
        Output Acos(theta+pi), Asin(theta+pi) in the recall period (signified by +1 provided in the third input-channel)
        This task is similar (but not exactly the same) to the task described in
        "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs"
        Laura Driscoll1, Krishna Shenoy, David Sussillo
        '''
        Task.__init__(self, n_steps, n_inputs, n_outputs, task_params)
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.stim_on_range = task_params["stim_on_range"]
        self.stim_duration = task_params["stim_duration"]
        self.recall_on = task_params["recall_on"]
        self.recall_off = task_params["recall_off"]

    def generate_input_target_stream(self, theta):
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        stim_on = int(self.rng.uniform(*self.stim_on_range))
        duration = self.stim_duration

        input_stream[0, stim_on: stim_on + duration] = 2 * np.cos(theta)
        input_stream[1, stim_on: stim_on + duration] = 2 * np.sin(theta)
        input_stream[2, self.recall_on: self.recall_off] = 1

        # Supplying it with an explicit instruction to recall the theta + 180
        target_stream[0, self.recall_on: self.recall_off] = 2 * np.cos(theta + np.pi)
        target_stream[1, self.recall_on: self.recall_off] = 2 * np.sin(theta + np.pi)
        condition = {"theta": theta}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        thetas = 2 * np.pi * np.linspace(0, 1, 37)[:-1]
        for theta in thetas:
            input_stream, target_stream, condition = self.generate_input_target_stream(theta)
            inputs.append(deepcopy(input_stream))
            targets.append(deepcopy(target_stream))
            conditions.append(deepcopy(condition))
        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        inputs = np.repeat(inputs, axis=2, repeats=11)
        targets = np.repeat(targets, axis=2, repeats=11)
        if shuffle:
            perm = self.rng.permutation(len(inputs))
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

    n_steps = 320
    n_inputs = 2
    n_outputs = 1
    task_params = dict()
    task_params["stim_on"] = n_steps // 8
    task_params["stim_off"] = 3 * n_steps // 16
    task_params["recall_on"] = 5 * n_steps // 8
    task_params["recall_off"] = n_steps
    task = TaskMemoryAntiNumber(n_steps, n_inputs, n_outputs, task_params)
    inputs, targets, conditions = task.get_batch()
    print(inputs.shape, targets.shape)
