import json
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.utils import get_project_root

date = ''.join((list(str(date.today()).split("-"))[::-1]))

# RNN specific
N = 100
activation_name = 'relu'
constrained = True
seed = 0
sigma_inp = 0.05
sigma_rec = 0.05
dt = 1
tau = 10
sr = 1.3
connectivity_density_rec = 1.0

task_name = 'DMTS'
n_inputs = 3
n_outputs = 2
T = 120
n_steps = int(T / dt)
task_params = dict()
task_params["n_steps"] = n_steps
task_params["n_inputs"] = n_inputs
task_params["n_outputs"] = n_outputs
task_params["stim_on_sample"] = n_steps // 10
task_params["stim_off_sample"] = 2 * n_steps // 10
task_params["stim_on_match"] = 3 * n_steps // 10
task_params["stim_off_match"] = 4 * n_steps // 10
task_params["dec_on"] = 5 * n_steps // 10
task_params["dec_off"] = n_steps
task_params["random_window"] = n_steps // 10
task_params["seed"] = seed
mask = np.concatenate(
    [np.arange(int(5 * n_steps // 10)), int(6 * n_steps // 10) + np.arange(int(4 * n_steps // 10))]).tolist()

# training specific
max_iter = 2500
tol = 1e-10
lr = 0.01
weight_decay = 5e-6
lambda_orth = 0.3
orth_input_only = True
lambda_r = 0.00
same_batch = False
shuffle = False

data_folder = os.path.abspath(os.path.join(get_project_root(), "data", "trained_RNNs", f"{task_name}"))
config_tag = f'{task_name}_{activation_name}'

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
config_dict["orth_input_only"] = orth_input_only
config_dict["lambda_r"] = lambda_r
config_dict["data_folder"] = data_folder
config_dict["folder_tag"] = ''

json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(get_project_root(), "data", "configs", f"train_config_{config_tag}.json"), mode="w")
outfile.write(json_obj)