import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../')
from src.generate_CDDM_trials import generate_all_trials
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from src.RNN_numpy import RNN_numpy
from matplotlib.widgets import Slider

def plot_random_trials(RNN, num_trials_plot, mask, sigma_rec, sigma_inp, data_gen_params, disp=True, generator_numpy=None):
    if (generator_numpy is None):
        generator_numpy = np.random.default_rng()
    n_steps = data_gen_params["n_steps"]
    # validation:
    inputs, targets, conditions = generate_all_trials(n_steps=data_gen_params["n_steps"],
                                                      coherences=data_gen_params["coherences_valid"],
                                                      protocol_dict=data_gen_params["protocol_dict"],
                                                      num_outputs=RNN.W_out.shape[0],
                                                      generator_numpy=generator_numpy)
    inds = np.random.choice(np.arange(inputs.shape[0]), num_trials_plot, replace=False)
    inputs = inputs[inds, :, :]
    if RNN.W_out.shape[0] == 1:
        targets = (targets[:, :, 0] - targets[:, :, 1])[:, :, np.newaxis]

    targets = targets[inds, :, :]
    list_of_inputs = [inputs[i, :, :] for i in range(inputs.shape[0])]

    # Checking the fit visually
    inds = generator_numpy.choice(list(range(len(list_of_inputs))), num_trials_plot, replace=False)
    figs = []
    trial_length_steps = (list_of_inputs[0].shape[0])
    for i in inds:
        RNN.clear_history()
        RNN.y = deepcopy(RNN.y_init)
        context = 'motion' if list_of_inputs[i][-1, :][0] == 1.0 else 'color'
        j = 2 if context == 'motion' else 4
        correct_choice = np.sign(list_of_inputs[i][n_steps-1, :][j] - list_of_inputs[i][n_steps-1, :][j+1])
        RNN.run(trial_length_steps, list_of_inputs[i], sigma_rec = sigma_rec, sigma_inp=sigma_inp, save_history=True)
        predicted_output = RNN.get_output(collapse=False)
        target_output = targets[i]

        fig_output = plt.figure(figsize=(12, 3))
        plt.suptitle(
            f"{context}, correct choice = {correct_choice}, inputs = {np.round(np.array(list_of_inputs[i][-1, :]), 3)}")
        if RNN.W_out.shape[0] == 1:
            plt.plot(predicted_output, color='r', label='predicted output')
            plt.plot(mask, target_output[mask], color='r', linestyle='--', label='target output')
        else:
            plt.plot(predicted_output[:, 0], color='r', label='predicted OutR')
            plt.plot(predicted_output[:, 1], color='b', label='predicted OutL')
            plt.plot(mask, target_output[mask, 0], color='r', linestyle='--', label='target OutR')
            plt.plot(mask, target_output[mask, 1], color='b', linestyle='--', label='target OutL')
        plt.grid(True)
        plt.xlabel("time step", fontsize=16)
        plt.ylabel("Activity", fontsize=16)
        plt.legend()
        if disp:
            plt.show()
        figs.append(fig_output)
    return figs


def plot_trials(conditions, outputs, targets, dt):

    def update(num_trial):
        i = int(num_trial)
        context = conditions[i]["context"]
        correct_choice = conditions[i]["correct_choice"]
        color_coh = conditions[i]["color_coh"]
        motion_coh = conditions[i]["motion_coh"]
        plt.suptitle(f"{context} context, correct choice = {correct_choice}, motion coherence = {np.round(motion_coh,3)}, color coherence = {np.round(color_coh,3)}")
        output = outputs[i, :, :]
        target = targets[i, :, :]
        line1.set_ydata(output[:, 0])
        line2.set_ydata(output[:, 1])
        line3.set_ydata(target[:, 0])
        line4.set_ydata(target[:, 1])
        fig.canvas.draw_idle()
        return None

    output_init = outputs[0, :, :]
    target_init = targets[0, :, :]
    plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize=(10, 4))
    line1, = ax.plot(np.arange(output_init.shape[0]) * dt, output_init[:, 0], color='r', label='RNN OutR')
    line2, = ax.plot(np.arange(output_init.shape[0]) * dt, output_init[:, 1], color='b', label='RNN OutL')
    line3, = ax.plot(np.arange(output_init.shape[0]) * dt, target_init[:, 0], linestyle='--', color='r',
                     label='target OutR')
    line4, = ax.plot(np.arange(output_init.shape[0]) * dt, target_init[:, 1], linestyle='--', color='b',
                     label='target OutL')
    ax.set_ylim([-0.2, 1.6])
    ax.set_ylabel("time, ms", fontsize=16)
    ax.legend(fontsize=16, loc=2)
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, bottom=0.2)
    axslider = fig.add_axes([0.1, 0.05, 0.85, 0.03])

    trial_select = Slider(ax=axslider, valmin=0, valmax=outputs.shape[0] - 1, valstep=1, label='trial')
    trial_select.on_changed(update)
    plt.show()
    plt.style.use("default")
    return None
