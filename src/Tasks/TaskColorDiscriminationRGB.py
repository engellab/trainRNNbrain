from copy import deepcopy
import numpy as np
from src.Tasks.TaskBase import Task

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
