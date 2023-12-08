import json
import os
import sys
from datetime import date
import numpy as np
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.utils import get_project_root
import datetime
task_name = 'SquareNumber'

from pathlib import Path
home = str(Path.home())
if home == '/home/pt1290':
    projects_folder = home
    data_save_path = home + f'/rnn_coach/data/trained_RNNs/{task_name}'
    RNN_configs_path = home + '/rnn_coach/data/configs'
elif home == '/Users/tolmach':
    projects_folder = home + '/Documents/GitHub/'
    data_save_path = projects_folder + f'/rnn_coach/data/trained_RNNs/{task_name}'
    RNN_configs_path = projects_folder + '/rnn_coach/data/configs'
else:
    pass

# date = ''.join((list(str(date.today()).split("-"))[::-1]))

# RNN specific
N = 100
activation_name = 'relu'
constrained = True
seed = None
sigma_inp = 0.05
sigma_rec = 0.05
dt = 1
tau = 10
sr = 1.2
connectivity_density_rec = 1.0

# task specific
n_inputs = 2
n_outputs = 1
T = 150
n_steps = int(T / dt)

mask = (int(2 * n_steps // 10) + np.arange(int(8 * n_steps // 10))).tolist()
task_params = {"stim_on": 0, "stim_off": n_steps,
               "dec_on": int(n_steps//10), "dec_off": n_steps,
               "n_steps": n_steps, "n_inputs": n_inputs, "n_outputs": n_outputs}

task_params["seed"] = seed

# training specific
max_iter = 2000
tol = 1e-10
lr = 0.005
weight_decay = 5e-06
lambda_orth = 0.3
orth_input_only = True
lambda_r = 0.3
same_batch = True

data_folder = os.path.abspath(os.path.join(get_project_root(), "data", "trained_RNNs", f"{task_name}"))
config_tag = f'{task_name}_{activation_name}'

now = datetime.datetime.now()
year = now.year
month = now.month
day = now.day
timestr = f"{year}/{month}/{day}"

config_dict = {}
config_dict["time"] = timestr
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
config_dict["folder_tag"] = ''

json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(get_project_root(), "data", "configs", f"train_config_{config_tag}.json"), mode="w")
outfile.write(json_obj)
