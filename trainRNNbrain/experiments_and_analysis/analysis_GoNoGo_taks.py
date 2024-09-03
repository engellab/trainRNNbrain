import numpy as np
from matplotlib import pyplot as plt
import json
import os
from trainRNNbrain.rnn_coach.RNN_numpy import RNN_numpy
from trainRNNbrain.rnn_coach.Tasks.TaskGoNoGo import TaskGoNoGo

def plot_trials(rnn, input_batch, target_batch, mask, sigma_rec=0.03, sigma_inp=0.03, labels=None):
    n_inputs = input_batch.shape[0]
    n_steps = input_batch.shape[1]
    batch_size = input_batch.shape[2]
    fig_output, axes = plt.subplots(batch_size, 1, figsize=(7, 8))
    rnn.clear_history()
    rnn.y = np.copy(rnn.y_init)
    rnn.run(input_timeseries=input_batch,
                 sigma_rec=sigma_rec,
                 sigma_inp=sigma_inp)
    predicted_output = rnn.get_output()
    colors = ["r", "b", "g", "c", "m", "y", 'k']
    n_outputs = rnn.W_out.shape[0]
    for k in range(batch_size):
        axes[k].plot(input_batch[0, :, k], color='blue', label='input value')
        for i in range(n_outputs):
            tag = labels[i] if not (labels is None) else ''
            axes[k].plot(predicted_output[i, :, k], color=colors[i], label=f'predicted {tag}')
            axes[k].plot(mask, target_batch[i, mask, k], color=colors[i], linestyle='--', label=f'target {tag}')
        axes[k].set_ylim([-0.1, 1.1])
        axes[k].spines.right.set_visible(False)
        axes[k].spines.top.set_visible(False)

    axes[0].legend(fontsize=12, frameon=False, bbox_to_anchor=(1.0, 1.0))
    axes[batch_size // 2].set_ylabel("Output", fontsize=12)
    axes[-1].set_xlabel("time step, ms", fontsize=12)
    fig_output.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    return fig_output



data_folder = '/Users/tolmach/Documents/GitHub/trainRNNbrain/data/trained_RNNs/GoNoGo_relu_constrained=True'
subfolder_name = '0.0225652_GoNoGo;relu;N=49;lmbdo=0.3;orth_inp_only=True;lmbdr=0.5;lr=0.005;maxiter=2000'
file_name = '0.0225652_params_GoNoGo.json'
config_file_name = '0.0225652_config.json'
file_path = os.path.join(data_folder, subfolder_name, file_name)

params = json.load(open(file_path, 'r'))
W_rec = np.array(params["W_rec"])
W_inp = np.array(params["W_inp"])
W_out = np.array(params["W_out"])
N = params["N"]
tau = params["tau"]
dt = params["dt"]
sigma_inp = 0.05
sigma_rec = 0.05

activation = lambda x: np.maximum(0.0, x)
rnn = RNN_numpy(N=N, tau=tau, dt=dt, W_inp=W_inp, W_rec=W_rec, W_out=W_out, activation=activation)

file_path = os.path.join(data_folder, subfolder_name, config_file_name)
config_dict = json.load(open(file_path, 'r'))

print(list(config_dict.keys()))
mask = config_dict["mask"]
n_steps = config_dict["n_steps"]
task_params = config_dict["task_params"]
task = TaskGoNoGo(n_steps=n_steps, task_params=task_params)

input_batch, target_batch, conditions = task.get_batch()

rnn.run(input_timeseries=input_batch, sigma_inp=sigma_inp, sigma_rec=sigma_rec)
trajectories = rnn.get_history()
outputs = rnn.get_output()


# PLOT TRIALS
batch_size = input_batch.shape[-1]
# rnd_inds = np.random.choice(np.arange(batch_size), size=10, replace=False)
inds = np.arange(batch_size)[::20]
input_batch_subsampled = input_batch[..., inds]
target_batch_subsampled = target_batch[..., inds]
fig = plot_trials(rnn=rnn,
                  input_batch=input_batch_subsampled, target_batch=target_batch_subsampled,
                  mask=mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
plt.savefig(os.path.join(data_folder, subfolder_name, "random_trials_2.pdf"))












