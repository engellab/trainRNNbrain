import json
import os
import sys

sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.DataSaver import DataSaver
from src.DynamicSystemAnalyzer import DynamicSystemAnalyzerCDDM
from src.PerformanceAnalyzer import PerformanceAnalyzerCDDM
from src.RNN_numpy import RNN_numpy
from src.utils import get_project_root, numpify
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Task import *
from matplotlib import pyplot as plt
import torch
import time

# from src.datajoint_config import *

disp = True
activation = "relu"
taskname = "CDDM"
train_config_file = f"train_config_{taskname}_{activation}.json"
config_dict = json.load(
    open(os.path.join(get_project_root(), "data", "configs", train_config_file), mode="r", encoding='utf-8'))

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
# if run on the cluster
try:
    SLURM_JOB_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
    task_params["seed"] = np.random.randint(1000000)
    seed = task_params["seed"]
except:
    SLURM_JOB_ID = None

data_folder = os.path.join(config_dict["data_folder"], timestr + ('' if (SLURM_JOB_ID is None) else str(SLURM_JOB_ID)))

# # creating instances:
rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                      activation=activation, constrained=constrained,
                      sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                      connectivity_density_rec=connectivity_density_rec,
                      spectral_rad=spectral_rad,
                      random_generator=rng)
task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn_torch.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
trainer = Trainer(RNN=rnn_torch, Task=task,
                  max_iter=max_iter, tol=tol,
                  optimizer=optimizer, criterion=criterion,
                  lambda_orth=lambda_orth, lambda_r=lambda_r)

tic = time.perf_counter()
rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask, same_batch=same_batch)
toc = time.perf_counter()
print(f"Executed training in {toc - tic:0.4f} seconds")

# net_params = pickle.load(open(os.path.join(get_project_root(), "data", "trained_RNNs", "CDDM", "20230117-175732", "params_CDDM_0.03556.pkl"), "rb+"))
# validate
RNN_valid = RNN_numpy(N=net_params["N"],
                      dt=net_params["dt"],
                      tau=net_params["tau"],
                      activation=numpify(activation),
                      W_inp=net_params["W_inp"],
                      W_rec=net_params["W_rec"],
                      W_out=net_params["W_out"],
                      bias_rec=net_params["bias_rec"],
                      y_init=net_params["y_init"])

analyzer = PerformanceAnalyzerCDDM(RNN_valid)
score_function = lambda x, y: np.mean((x - y) ** 2)
input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid,
                                      mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
score = np.round(score, 7)
datasaver = DataSaver(data_folder)
# datasaver = None

print(f"MSE validation: {score}")
if not (datasaver is None): datasaver.save_data(config_dict, f"{score}_config.json")
if not (datasaver is None): datasaver.save_data(net_params, f"params_{taskname}_{score}.pkl")

fig_trainloss = plt.figure(figsize=(10, 3))
plt.plot(train_losses, color='r', label='train loss (log scale)')
plt.plot(val_losses, color='b', label='valid loss (log scale)')
plt.yscale("log")
plt.grid(True)
plt.legend(fontsize=16)
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_trainloss, f"{score}_train&valid_loss.png")

print(f"Plotting random trials")
inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
inputs = input_batch_valid[..., inds]
targets = target_batch_valid[..., inds]

fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_trials, f"{score}_random_trials.png")

print(f"Plotting psychometric data")
num_levels = len(config_dict["task_params"]["coherences"])
analyzer.calc_psychometric_data(task, mask, num_levels=num_levels, num_repeats=31, sigma_rec=0.03, sigma_inp=0.03)
fig_psycho = analyzer.plot_psychometric_data()
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_psycho, f"{score}_psychometric_data.png")
if not (datasaver is None): datasaver.save_data(analyzer.psychometric_data, f"{score}_psycho_data.pkl")

print(f"Analyzing fixed points")
dsa = DynamicSystemAnalyzerCDDM(RNN_valid)
params = {"fun_tol": 0.05,
          "diff_cutoff": 1e-4,
          "sigma_init_guess": 15,
          "patience": 100,
          "stop_length": 100,
          "mode": "approx"}
dsa.get_fixed_points(Input=np.array([1, 0, 0.5, 0.5, 0.5, 0.5]), **params)
dsa.get_fixed_points(Input=np.array([0, 1, 0.5, 0.5, 0.5, 0.5]), **params)
print(f"Calculating Line Attractor analytics")
dsa.calc_LineAttractor_analytics()

fig_LA3D = dsa.plot_LineAttractor_3D()
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_LA3D, f"{score}_LA_3D.png")
if not (datasaver is None): datasaver.save_data(dsa.fp_data, f"{score}_fp_data.pkl")
if not (datasaver is None): datasaver.save_data(dsa.LA_data, f"{score}_LA_data.pkl")

fig_RHS = dsa.plot_RHS_over_LA()
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_RHS, f"{score}_LA_RHS.png")

# rnn_dj = RNNDJ()
# task_dj = TaskDJ()
# trainer_dj = TrainerDJ()
# cddm_analysis_dj = CDDMRNNAnalysisDJ()
#
# task_id = 0
# trainer_id = 0
# rnn_timestamp = time.strftime("%Y%m%d%H%M%S")
#
# task_dj_dict = {"task_name": taskname + "_" + str(task_id),
#                 "n_steps": config_dict["n_steps"],
#                 "n_inputs": config_dict["num_inputs"],
#                 "n_outputs": config_dict["num_outputs"],
#                 "task_params": config_dict["task_params"],
#                 "mask": mask}
# trainer_dj_dict = {"task_name": taskname + "_" + str(task_id),
#                    "trainer_id": trainer_id,
#                    "max_iter": config_dict["max_iter"],
#                    "tol": config_dict["tol"],
#                    "lr": config_dict["lr"],
#                    "lambda_orth": config_dict["lambda_orth"],
#                    "lambda_r": config_dict["lambda_r"],
#                    "same_batch" : config_dict["same_batch"],
#                    "shuffle" : False}
# rnn_dj_dict = {"task_name": taskname + "_" + str(task_id),
#                "rnn_timestamp" : rnn_timestamp,
#                "trainer_id" : trainer_id,
#                "n": config_dict["N"],
#                "activation_name": config_dict["activation"],
#                "constrained": config_dict["constrained"],
#                "dt": config_dict["dt"],
#                "tau": config_dict["tau"],
#                "sr": config_dict["sr"],
#                "connectivity_density_rec": config_dict["connectivity_density_rec"],
#                "sigma_rec" : config_dict["sigma_rec"],
#                "sigma_inp": config_dict["sigma_inp"],
#                "w_inp" : net_params["W_inp"],
#                "w_rec" : net_params["W_rec"],
#                "w_out" : net_params["W_out"],
#                "b_rec" : 0 if net_params["bias_rec"] is None else net_params["bias_rec"]}
# cddm_analysis_dj_dict = {"task_name": taskname + "_" + str(task_id),
#                          "rnn_timestamp" : rnn_timestamp,
#                          "trainer_id": trainer_id,
#                          "mse_score": score,
#                          "psycho_data": deepcopy(analyzer.psychometric_data),
#                          "fp_data": deepcopy(dsa.fp_data),
#                          "la_data": deepcopy(dsa.LA_data)}
#
# task_dj.insert1(task_dj_dict, skip_duplicates=True)
# rnn_dj.insert1(rnn_dj_dict, skip_duplicates=True)
# trainer_dj.insert1(trainer_dj_dict, skip_duplicates=True)
# trainer_dj.insert1(trainer_dj_dict, skip_duplicates=True)
