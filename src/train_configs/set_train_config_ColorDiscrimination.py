import json
import os
import sys
import numpy as np
from datetime import date


taskname = 'ColorDiscrimination'
from pathlib import Path
home = str(Path.home())
if home == '/home/pt1290':
    projects_folder = home
    data_save_path = home + f'/rnn_coach/data/trained_RNNs/{taskname}'
    RNN_configs_path = home + '/rnn_coach/data/configs'
elif home == '/Users/tolmach':
    projects_folder = home + '/Documents/GitHub/'
    data_save_path = projects_folder + f'/rnn_coach/data/trained_RNNs/{taskname}'
    RNN_configs_path = projects_folder + '/rnn_coach/data/configs'
else:
    pass

date = ''.join((list(str(date.today()).split("-"))[::-1]))

# RNN specific
N = 150
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
n_inputs = 3
n_outputs = 12
T = 120
n_steps = int(T / dt)
mask = np.arange(int(n_steps // 3), n_steps).tolist()

task_params = {"color_on": 0,
               "color_off": n_steps,
               "n_steps": n_steps}
task_params["seed"] = seed

# training specific
max_iter = 1000
tol = 1e-10
lr = 0.01
weight_decay = 1e-3
lambda_orth = 0.3
orth_input_only = True
lambda_r = 0.5
lambda_smooth = 0.0
same_batch = True
shuffle = False
extra_info = f'{activation_name};N={N};lmbdr={lambda_r};lmbdo={lambda_orth};lmbds={lambda_smooth}'
name_tag = f'{taskname}_{extra_info}'

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
config_dict["shuffle"] = shuffle
config_dict["weight_decay"] = weight_decay
config_dict["lambda_orth"] = lambda_orth
config_dict["orth_input_only"] = orth_input_only
config_dict["lambda_r"] = lambda_r
config_dict["lambda_smooth"] = lambda_smooth
config_dict["folder_tag"] = ''

json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(RNN_configs_path, f"train_config_{name_tag}.json"), mode="w")
outfile.write(json_obj)