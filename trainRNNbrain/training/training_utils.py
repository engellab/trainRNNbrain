import torch
import numpy as np
import json
import os
from omegaconf import DictConfig, OmegaConf
import ast
from matplotlib import pyplot as plt

def prepare_task_arguments(cfg_task, dt):
    if "T" in list(cfg_task.keys()):
        n_steps = int(cfg_task.T / dt)
    elif "n_steps" in list(cfg_task.keys()):
        n_steps = cfg_task.n_steps

    conf = {"_target_": cfg_task._target_, "n_steps" : n_steps, "seed": cfg_task.seed}
    task_params = cfg_task.task_params
    for key in cfg_task.keys():
        if "T_" in key:
            exec(f"conf[key.split(\"T_\")[1]] = int(cfg_task.{key} / dt)")
        else:
            if key in task_params:
                conf[key] = cfg_task[key]
    return OmegaConf.create(conf)

def prepare_RNN_arguments(cfg_model, cfg_task):
    conf = dict()
    for key in cfg_model.keys():
        conf[key] = cfg_model[key]
    conf["input_size"] = cfg_task.n_inputs
    conf["output_size"] = cfg_task.n_outputs
    conf["seed"] = cfg_task.seed
    return OmegaConf.create(conf)


def get_training_mask(cfg_task, dt):
    Ts = cfg_task.mask_params
    mask_part_list = []
    for i in range(len(Ts)):
        tuple = ast.literal_eval(Ts[i])
        t1 = int(tuple[0]/dt)
        t2 = int(tuple[1]/dt)
        mask_part_list.append(t1 + np.arange(t2 - t1))
    return np.concatenate(mask_part_list)

def remove_silent_nodes(rnn_torch, task, net_params, thr=1e-10):
    input_batch, target_batch, conditions = task.get_batch()
    rnn_torch.sigma_rec = rnn_torch.sigma_inp = torch.tensor(0, device=rnn_torch.device)
    y, predicted_output_rnn = rnn_torch(torch.from_numpy(input_batch.astype("float32")).to(rnn_torch.device))
    Y_mean = torch.mean(torch.abs(y), dim=(1, 2))
    inds_fr = (torch.where(Y_mean > thr)[0]).tolist()
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
    net_params["activation_slope"] = float(rnn_torch.activation_slope.cpu().numpy())
    RNN_params = {"W_inp": np.array(net_params["W_inp"]),
                  "W_rec": np.array(net_params["W_rec"]),
                  "W_out": np.array(net_params["W_out"]),
                  "b_rec": np.array(net_params["bias_rec"]),
                  "activation_slope": net_params["activation_slope"],
                  "y_init": np.zeros(N)}
    net_params["N"] = N_reduced
    rnn_torch.set_params(RNN_params)
    return rnn_torch, net_params

def plot_train_val_losses(train_losses, val_losses):
    fig_trainloss, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(train_losses, color='r', label='train loss (log scale)')
    if len(val_losses) != 0:
        ax.plot(val_losses, color='b', label='valid loss (log scale)')

    ax.set_yscale("log")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True)

    ax.legend(frameon=False)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    return fig_trainloss

def plot_loss_breakdown(loss_monitor):
    fig_loss_breakdown, ax = plt.subplots(figsize=(8, 4))
    ax.set_yscale('log')
    for key in loss_monitor:
        ax.plot(loss_monitor[key], label=key)
    # ax.set_ylim([0, 1])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Value")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    plt.tight_layout()
    return fig_loss_breakdown

def get_trajectories(RNN_valid, input_batch_valid, target_batch_valid, conditions_valid):
    RNN_valid.clear_history()
    RNN_valid.run(input_timeseries=input_batch_valid, sigma_rec=0, sigma_inp=0)
    RNN_trajectories = RNN_valid.get_history()
    RNN_output = RNN_valid.get_output()
    trajecory_data = {}
    trajecory_data["inputs"] = input_batch_valid
    trajecory_data["trajectories"] = RNN_trajectories
    trajecory_data["outputs"] = RNN_output
    trajecory_data["targets"] = target_batch_valid
    trajecory_data["conditions"] = conditions_valid
    return trajecory_data



