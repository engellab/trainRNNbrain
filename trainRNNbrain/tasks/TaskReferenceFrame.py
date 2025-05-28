from copy import deepcopy
import numpy as np
from trainRNNbrain.tasks.TaskBase import Task

class TaskReferenceFrame(Task):
    def __init__(self, n_steps, n_inputs_retina, n_inputs_head, n_outputs,
                 stim_on, stim_off, batch_size=12, seed=None):
        '''
        :param n_steps: number of steps in the trial
        '''
        Task.__init__(self, n_steps=n_steps, n_inputs=n_inputs_retina + n_inputs_head, n_outputs=n_outputs, seed=seed)
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.batch_size = batch_size
        self.n_inputs_retina = n_inputs_retina
        self.n_inputs_head = n_inputs_head

    def generate_input_target_stream(self, theta_retina, theta_head):
        '''
        theta goes from -np.pi/2 to np.pi/2
        '''
        # Cue input stream

        input_stream = np.zeros((self.n_inputs_retina + self.n_inputs_head, self.n_steps))

        arc_retina = 2 * np.pi / self.n_inputs_retina
        arc_head = 2 * np.pi / self.n_inputs_head

        ind_channel_retina = int(np.floor(theta_retina / arc_retina))
        v_retina = theta_retina % arc_retina
        ind_channel_head = int(np.floor(theta_head / arc_head))
        v_head = theta_head % arc_head

        input_stream[ind_channel_retina % self.n_inputs_retina, self.stim_on: self.stim_off] = 1 - v_retina / arc_retina
        input_stream[(ind_channel_retina + 1) % self.n_inputs_retina, self.stim_on: self.stim_off] = v_retina / arc_retina

        input_stream[ind_channel_head % self.n_inputs_head + self.n_inputs_retina, self.stim_on: self.stim_off] = 1 - v_head / arc_head
        input_stream[(ind_channel_head + 1) % self.n_inputs_head + self.n_inputs_retina, self.stim_on: self.stim_off] = v_head / arc_head

        theta_ego = (theta_retina + theta_head) % (2 * np.pi)

        target_stream = np.zeros((self.n_outputs, self.n_steps))
        arc_ego = 2 * np.pi / self.n_outputs
        v_ego = theta_ego % arc_ego
        ind_channel_ego = int(np.floor(theta_ego / arc_ego))
        target_stream[ind_channel_ego % self.n_outputs, self.stim_on: self.stim_off] = 1 - v_ego / arc_ego
        target_stream[(ind_channel_ego + 1) % self.n_outputs, self.stim_on: self.stim_off] = v_ego / arc_ego

        condition = {"theta_retina": theta_retina, "theta_head": theta_head, "theta_ego": theta_ego}
        return input_stream, target_stream, condition

    def get_batch(self, shuffle=False):
        '''
        '''
        inputs = []
        targets = []
        conditions = []
        for theta_retina in np.linspace(0, 2 * np.pi, self.batch_size)[:-1]:
            for theta_head in np.linspace(0, 2 * np.pi, self.batch_size)[:-1]:
                input_stream, target_stream, condition = self.generate_input_target_stream(theta_retina, theta_head)
                inputs.append(deepcopy(input_stream))
                targets.append(deepcopy(target_stream))
                conditions.append(deepcopy(condition))

        # batch_size should be a last dimension
        inputs = np.stack(inputs, axis=2)
        targets = np.stack(targets, axis=2)
        if shuffle:
            perm = self.rng.permutation(np.arange((inputs.shape[-1])))
            inputs = inputs[..., perm]
            targets = targets[..., perm]
            conditions = [conditions[index] for index in perm]
        return inputs, targets, conditions


if __name__ == '__main__':
    task = TaskReferenceFrame(n_steps=1, n_inputs_retina=7, n_inputs_head=7, n_outputs=7,
                 stim_on=0, stim_off=1, batch_size=33, seed=None)
    theta_retina = 5 * np.pi / 7
    theta_head = 2 * np.pi - 2 * np.pi / 7
    input_stream, target_stream, condition = task.generate_input_target_stream(theta_retina, theta_head)
    print(input_stream)
    print("_____")
    print(target_stream)
    print(condition)
