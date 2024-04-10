import torch
import numpy as np
import json
import os

def remove_silent_nodes(rnn_torch, task, net_params):
    input_batch, target_batch, conditions = task.get_batch()
    rnn_torch.sigma_rec = rnn_torch.sigma_inp = torch.tensor(0, device=rnn_torch.device)
    y, predicted_output_rnn = rnn_torch(torch.from_numpy(input_batch.astype("float32")).to(rnn_torch.device))
    Y = torch.hstack([y.detach()[:, :, i] for i in range(y.shape[-1])]).T
    Y_mean = torch.mean(torch.abs(Y), axis=0)
    inds_fr = (torch.where(Y_mean > 0)[0]).tolist()
    N_reduced = len(inds_fr)
    N = N_reduced
    W_rec = rnn_torch.W_rec.data.cpu().detach().numpy()[inds_fr, :]
    W_rec = W_rec[:, inds_fr]
    net_params["W_rec"] = np.copy(W_rec)
    W_out = net_params["W_out"][:, inds_fr]
    net_params["W_out"] = np.copy(W_out)
    W_inp = net_params["W_inp"][inds_fr, :]
    net_params["W_inp"] = np.copy(W_inp)
    net_params["bias_rec"] = None
    net_params["y_init"] = np.zeros(N_reduced)
    RNN_params = {"W_inp": np.array(net_params["W_inp"]),
                  "W_rec": np.array(net_params["W_rec"]),
                  "W_out": np.array(net_params["W_out"]),
                  "b_rec": np.array(net_params["bias_rec"]),
                  "y_init": np.zeros(N)}
    net_params["N"] = N_reduced
    rnn_torch.set_params(RNN_params)
    return rnn_torch, net_params


def set_paths(folder_name):
    from pathlib import Path
    home = str(Path.home())
    if home == '/home/pt1290':
        projects_folder = home
        RNN_configs_path = os.path.join(projects_folder, 'rnn_coach/data/configs')
        data_save_path = f'/../../../../scratch/gpfs/pt1290/rnn_coach/data/trained_RNNs/{folder_name}'
    elif home == '/Users/tolmach':
        projects_folder = home + '/Documents/GitHub'
        data_save_path = os.path.join(projects_folder, f'rnn_coach/data/trained_RNNs/{folder_name}')
        RNN_configs_path = os.path.join(projects_folder, 'rnn_coach/data/configs')
    else:
        pass
    return home, data_save_path, RNN_configs_path

