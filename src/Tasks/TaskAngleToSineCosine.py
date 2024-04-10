from copy import deepcopy
import numpy as np
from rnn_coach.src.Tasks.TaskBase import Task

class TaskAngleToSineCosine(Task):
    def __init__(self, n_steps, task_params):
        '''
         n_inputs encoding an angle + 1 constant input
         4 outputs: 2 reserved for cosine, 2 reserved for sine.
         the outputs can only be positive, hence cosine is represented by two output channels:
          one is activa when cosine is positive, another one when it is negative. Analogously for sine.
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=5, n_outputs=4, task_params=task_params)
        self.task_params = task_params
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.stim_on = self.task_params["stim_on"]
        self.stim_off = self.task_params["stim_off"]
        self.dec_on = self.task_params["dec_on"]
        self.dec_off = self.task_params["dec_off"]

    def generate_input_target_stream(self, angle):
        '''
        '''
        # Cue input stream
        angle = angle % (2*np.pi)
        input_stream = np.zeros((self.n_inputs, self.n_steps))
        num_angle_encoding_inps = (self.n_inputs - 1)
        num_angle_encoding_outs = (self.n_inputs - 1)

        arc = 2 * np.pi / num_angle_encoding_inps
        ind_channel = int(np.floor(angle / arc))
        v = angle % arc

        input_stream[ind_channel, self.stim_on: self.stim_off] = (1 - v / arc)
        input_stream[(ind_channel + 1) % num_angle_encoding_inps, self.stim_on: self.stim_off] = v / arc
        input_stream[-1, :] = 1 # constant input

        cosA = np.cos(angle)
        sinA = np.sin(angle)
        target_stream = np.zeros((self.n_outputs, self.n_steps))
        ind_cos = 0 if cosA > 0 else 1
        ind_sin = 2 if sinA > 0 else 3
        target_stream[ind_cos, :] = np.abs(cosA)
        target_stream[ind_sin, :] = np.abs(sinA)
        condition = {"angle": angle, "cosA": cosA, "sinA": sinA, "ind_cos": ind_cos, "ind_sin" : ind_sin}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        '''
        '''
        inputs = []
        targets = []
        conditions = []
        for angle in np.linspace(0, 2 * np.pi, 360):
            input_stream, target_stream, condition = self.generate_input_target_stream(angle)
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
