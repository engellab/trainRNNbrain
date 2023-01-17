import json
import os
import numpy as np
from datetime import date
from src.utils import get_project_root
date = ''.join((list(str(date.today()).split("-"))[::-1]))

# RNN specific
N = 50
activation_name = 'tanh'
constrained = False
seed = None
sigma_inp = 0.03
sigma_rec = 0.03
dt = 1
tau = 10
sr = 1.2
connectivity_density_rec = 1.0

# task specific
task_name = 'MemoryAnti'
n_inputs = 3
n_outputs = 2
T = 300
n_steps = int(T / dt)
task_params = dict()
task_params["stim_on"] = 0
task_params["stim_off"] = n_steps // 4
task_params["recall_on"] = n_steps // 2
task_params["recall_off"] = n_steps
# mask = np.concatenate([np.arange(n_steps)]).tolist() # using the whole trial
mask = np.concatenate([np.arange(task_params["recall_on"]), 5 * n_steps // 6 + np.arange(1 * n_steps // 6) ]).tolist() # using the whole trial

# training specific
max_iter = 300
tol = 1e-10
lr = 0.005
weight_decay = 5e-6
lambda_orth = 0.3
lambda_r = 0.1
same_batch = True # generate new batch in each train loop
shuffle = False

data_folder = os.path.abspath(os.path.join(get_project_root(), "data", "trained_RNNs", f"{task_name}"))
tag = f'{task_name}_{activation_name}'

config_dict = {}
config_dict["N"] = N
config_dict["seed"] = seed
config_dict["activation"] = activation_name
config_dict["sigma_inp"] = sigma_inp
config_dict["sigma_rec"] = sigma_rec
config_dict["num_inputs"] = n_inputs
config_dict["num_outputs"] = n_outputs
config_dict["constrained"] = constrained
config_dict["dt"] = dt
config_dict["tau"] = tau
config_dict["sr"] = sr
config_dict["connectivity_density_rec"] = connectivity_density_rec
config_dict["max_iter"] = max_iter
config_dict["n_steps"] = n_steps
config_dict["task_params"] = task_params
config_dict["mask"] = mask
config_dict["tol"] = tol
config_dict["lr"] = lr
config_dict["same_batch"] = same_batch
config_dict["weight_decay"] = weight_decay
config_dict["lambda_orth"] = lambda_orth
config_dict["lambda_r"] = lambda_r
config_dict["data_folder"] = data_folder
config_dict["tag"] = tag

json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(get_project_root(), "data", "configs", f"train_config_{tag}.json"), mode="w")
outfile.write(json_obj)

