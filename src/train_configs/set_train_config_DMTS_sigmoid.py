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
activation_name = 'sigmoid'
constrained = True
exc_to_inh_ratio = 4
seed = 0
sigma_inp = 0.03
sigma_rec = 0.03
dt = 1
tau = 10
sr = 1.3
connectivity_density_rec = 1.0

task_name = 'DMTS'
n_inputs = 3
n_outputs = 1
T = 140
n_steps = int(T / dt)
task_params = dict()
task_params["n_steps"] = n_steps
task_params["n_inputs"] = n_inputs
task_params["n_outputs"] = n_outputs
task_params["stim_on_sample"] = 10
task_params["stim_off_sample"] = 20
task_params["stim_on_match"] = 80
task_params["stim_off_match"] = 90
task_params["dec_on"] = 100
task_params["dec_off"] = n_steps
task_params["random_window"] = 10
task_params["seed"] = seed
n_steps_out = (task_params["dec_on"] + 10)
mask = np.concatenate(
    [np.arange(task_params["dec_on"]), n_steps_out + np.arange(n_steps - n_steps_out)]).tolist()

# training specific
max_iter_1 = 10000
max_iter_2 = 10000
tol = 1e-10
lr = 0.001
weight_decay = 5e-6
lambda_orth = 0.3
orth_input_only = True
lambda_r_1 = 0.0
lambda_r_2 = 0.3
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
config_dict["exc_to_inh_ratio"] = exc_to_inh_ratio
config_dict["dt"] = dt
config_dict["tau"] = tau
config_dict["sr"] = sr
config_dict["connectivity_density_rec"] = connectivity_density_rec
config_dict["max_iter_1"] = max_iter_1
config_dict["max_iter_2"] = max_iter_2
config_dict["n_steps"] = n_steps
config_dict["task_params"] = task_params
config_dict["mask"] = mask
config_dict["tol"] = tol
config_dict["lr"] = lr
config_dict["same_batch"] = same_batch
config_dict["weight_decay"] = weight_decay
config_dict["lambda_orth"] = lambda_orth
config_dict["orth_input_only"] = orth_input_only
config_dict["lambda_r_1"] = lambda_r_1
config_dict["lambda_r_2"] = lambda_r_2
config_dict["data_folder"] = data_folder
config_dict["folder_tag"] = ''

json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(get_project_root(), "data", "configs", f"train_config_{config_tag}.json"), mode="w")
outfile.write(json_obj)
