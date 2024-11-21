from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskMemoryDM(Task):
    '''Delayed decision-making task: get the stimulus, wait and then make a decision after the dec_on cue comes in'''
    def __init__(self, n_steps=157,
                 n_inputs=3, n_outputs=2,
                 stim_on=7, stim_off=37, dec_on=110, dec_off=157,
                 random_window=7, seed=None):
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.dec_on = dec_on
        self.dec_off = dec_off
        self.random_window = random_window


    def generate_input_target_stream(self, inp_ind):
        out_ind = inp_ind if not (inp_ind is None) else 100000
        random_offset = 0 if self.random_window == 0 else self.rng.integers(-self.random_window, self.random_window)
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        output_stream = np.zeros((self.n_outputs, self.n_steps))
        stim_on = self.stim_on + random_offset
        stim_off = self.stim_off + random_offset

        if inp_ind is 100000:
            pass
        else:
            input_stream[inp_ind, stim_on:stim_off] = 1
            # add Go Cue to input to channel 2 of input
            input_stream[2, self.dec_on:self.dec_off] = 1
            output_stream[out_ind, self.dec_on:self.dec_off] = 1

        condition = {"inp_ind" : inp_ind, "out_ind": out_ind}
        return input_stream, output_stream, condition

    def get_batch(self, shuffle=False):
        inputs = []
        targets = []
        conditions = []
        for inp_ind in [0, 1, 100000]:
            input, output, condition = self.generate_input_target_stream(inp_ind)
            inputs.append(deepcopy(input))
            targets.append(deepcopy(output))
            conditions.append(condition)

        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)

        if (shuffle):
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
        return inputs, targets, conditions

# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#     task = TaskMemoryDM(random_window=0)
#     inputs, target, _ = task.get_batch()
#     print(inputs.shape)
#     print(target.shape)
#
#
#     for i in range(inputs.shape[0]):
#         fig = plt.figure(figsize=(10, 3))
#         plt.plot(inputs[0, :, i], color = 'r')
#         plt.plot(inputs[1, :, i], color='blue')
#         plt.plot(inputs[2, :, i], color='green')
#         plt.plot(target[0, :, i], color='orange', linestyle='--')
#         plt.plot(target[1, :, i], color='black', linestyle='--')
#         plt.grid(True)
#         plt.show()

