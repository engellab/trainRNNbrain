import os
import sys

sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import json
from src.DataSaver import DataSaver
from src.DynamicSystemAnalyzer import DynamicSystemAnalyzer
from src.PerformanceAnalyzer import PerformanceAnalyzer
from src.RNN_numpy import RNN_numpy
from src.utils import get_project_root, numpify
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Task import *
from matplotlib import pyplot as plt
import torch
import time

disp = True
activation = "relu"
taskname = "DMTS"
train_config_file = f"train_config_{taskname}_{activation}.json"
config_dict = json.load(open(os.path.join(get_project_root(), "LA_data", "configs", train_config_file), mode="r"))

# defining RNN:
N = config_dict["N"]
activation_name = config_dict["activation"]
if activation_name == 'relu':
    activation = lambda x: torch.maximum(x, torch.tensor(0))
elif activation_name == 'tanh':
    activation = torch.tanh
elif activation_name == 'sigmoid':
    activation = lambda x: 1 / (1 + torch.exp(-x))
elif activation_name == 'softplus':
    activation = lambda x: torch.log(1 + torch.exp(5 * x))

dt = config_dict["dt"]
tau = config_dict["tau"]
constrained = config_dict["constrained"]
connectivity_density_rec = config_dict["connectivity_density_rec"]
spectral_rad = config_dict["sr"]
sigma_inp = config_dict["sigma_inp"]
sigma_rec = config_dict["sigma_rec"]
seed = config_dict["seed"]
rng = torch.Generator()
if not seed is None:
    rng.manual_seed(seed)
input_size = config_dict["num_inputs"]
output_size = config_dict["num_outputs"]

# Task:
n_steps = config_dict["n_steps"]
task_params = config_dict["task_params"]

# Trainer:
lambda_orth = config_dict["lambda_orth"]
lambda_r = config_dict["lambda_r"]
mask = np.array(config_dict["mask"])
max_iter = config_dict["max_iter"]
tol = config_dict["tol"]
lr = config_dict["lr"]
weight_decay = config_dict["weight_decay"]
same_batch = config_dict["same_batch"]

# General:
tag = config_dict["tag"]
timestr = time.strftime("%Y%m%d-%H%M%S")
data_folder = os.path.join(config_dict["data_folder"], timestr)

# # creating instances:
rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                      activation=activation, constrained=constrained,
                      sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                      connectivity_density_rec=connectivity_density_rec,
                      spectral_rad=spectral_rad,
                      random_generator=rng)
task = TaskDMTS(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn_torch.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
trainer = Trainer(RNN=rnn_torch, Task=task,
                  max_iter=max_iter, tol=tol,
                  optimizer=optimizer, criterion=criterion,
                  lambda_orth=lambda_orth, lambda_r=lambda_r)

datasaver = DataSaver(data_folder)

try:
    SLURM_JOB_ID = int(os.environ["SLURM_JOB_ID"])
    task_params["seed"] = SLURM_JOB_ID
    seed = SLURM_JOB_ID
except:
    SLURM_JOB_ID = None

rnn_trained, train_losses, val_losses, best_net_params = trainer.run_training(train_mask=mask, same_batch=same_batch)

fig_trainloss = plt.figure(figsize=(10, 3))
plt.plot(train_losses, color='r', label='train loss (log scale)')
plt.plot(val_losses, color='b', label='valid loss (log scale)')
plt.yscale("log")
plt.grid(True)
plt.legend(fontsize=16)
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_trainloss, "train&valid_loss")

# validate
RNN_valid = RNN_numpy(N=best_net_params["N"],
                      dt=best_net_params["dt"],
                      tau=best_net_params["tau"],
                      activation=numpify(activation),
                      W_inp=best_net_params["W_inp"],
                      W_rec=best_net_params["W_rec"],
                      W_out=best_net_params["W_out"],
                      bias_rec=best_net_params["bias_rec"],
                      y_init=best_net_params["y_init"])

analyzer = PerformanceAnalyzer(RNN_valid)
score_function = lambda x, y: np.mean((x - y) ** 2)
input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid,
                                      mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
print(f"MSE validation: {np.round(score, 5)}")
if not (datasaver is None): datasaver.save_data(best_net_params, f"params_{taskname}_{np.round(score, 5)}.pkl")
if not (datasaver is None): datasaver.save_data(config_dict, "config.json")

print(f"Plotting random trials")
inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
inputs = input_batch_valid[..., inds]
targets = target_batch_valid[..., inds]

fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_trials, "random_trials")

dsa = DynamicSystemAnalyzer(RNN_valid)
dsa.get_fixed_points(Input=np.zeros(input_size))
fig_fp = dsa.plot_fixed_points(projection='2D')
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_fp, "fp_projection")
