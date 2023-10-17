import json
import os
import sys
import numpy as np
import datetime

taskname = 'DelayDM'

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

# date = ''.join((list(str(date.today()).split("-"))[::-1]))

# RNN specific
N = 100
activation_name = 'relu'
constrained = False
seed = None
sigma_inp = 0.05
sigma_rec = 0.05
dt = 1
tau = 10
sr = 1.2
connectivity_density_rec = 1.0

# task specific
n_inputs = 3
n_outputs = 2
T = 157
n_steps = int(T / dt)
cue_on, cue_off = 7, 36
go_on = 110
mask = np.arange(n_steps).tolist()

task_params = {"cue_on": cue_on,
               "cue_off": cue_off,
               "go_on": go_on,
               "n_steps": n_steps,
               "n_inputs": n_inputs,
               "n_outputs": n_outputs}

n_rights = n_lefts = 10
n_catches = 1
directions = np.zeros(n_rights + n_lefts + n_catches)
directions[:n_rights] = 1 # right: channel 1
directions[-n_catches:] = -1 # catch: -1
assert(np.sum(directions) == n_rights - n_catches)
print(directions)
directions = directions.astype(int).tolist()

task_params["directions"] = directions
task_params["seed"] = seed

# training specific
max_iter = 1000
tol = 1e-10
lr = 0.005
weight_decay = 5e-6
lambda_orth = 0.3
orth_input_only = True
lambda_r = 0.02
same_batch = True
extra_info = f'{activation_name};N={N};lmbdr={lambda_r};lmbdo={lambda_orth};orth_inp_only={orth_input_only}'
name_tag = f'{taskname}_{extra_info}'

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
outfile = open(os.path.join(RNN_configs_path, f"train_config_{name_tag}.json"), mode="w")
outfile.write(json_obj)
