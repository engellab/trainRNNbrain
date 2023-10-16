import json
import os
import sys

import numpy as np

sys.path.insert(0, '../')
sys.path.insert(0, '../../')
print(sys.path)
from src.utils import get_project_root
from datetime import date

date = ''.join((list(str(date.today()).split("-"))[::-1]))

# RNN sepcfication
N = 300
activation_name = "relu"
seed = None
sigma_inp = 0.05 # ?
sigma_rec = 0.05 # ?
constrained = True # enforce Dale's law
dt = 1
tau = 10 
sr = 1.2
connectivity_density_rec = 1.0

# task specific
task_name = "ALM"
n_inputs = 3
n_outputs = 2

T = 157 # from dataset
n_steps = int(T/dt)

n_rights = 100
n_lefts = 100
n_catches = 100
n_trials = n_rights + n_lefts + n_catches
directions = np.zeros(n_trials)

directions[:n_rights] = 1 # right: channel 1
directions[-n_catches:] = -1 # catch: -1
assert(np.sum(directions) == n_rights - n_catches)

task_params = {"cue_on": 7, "cue_off": 36,
               "go_on": 110,
               "n_steps": n_steps,
               "n_inputs": n_inputs, "n_outputs": n_outputs,
               "directions": directions}

# training specific
max_iter = 3000
tol = 1e-10
lr = 0.002
weight_decay = 1e-3
lambda_orth = 0.3
orth_input_only = True
lambda_r = 0.5
same_batch = True

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
config_dict["dt"] = dt
config_dict["tau"] = tau
config_dict["sr"] = sr
config_dict["connectivity_density_rec"] = connectivity_density_rec
config_dict["max_iter"] = max_iter
config_dict["n_steps"] = n_steps
config_dict["task_params"] = task_params
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