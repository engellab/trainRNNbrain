import json
import os
import sys

import numpy as np

sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.utils import get_project_root
from datetime import date

date = ''.join((list(str(date.today()).split("-"))[::-1]))

# RNN specific
N = 100
activation_name = 'relu'
constrained = True
seed = None
sigma_inp = 0.03
sigma_rec = 0.03
dt = 1
tau = 10
sr = 1.2
connectivity_density_rec = 1.0

# task specific
task_name = 'CDDM'
n_inputs = 6
n_outputs = 2
T = 750
n_steps = int(T / dt)
max_coherence = 0.8
coherence_lvls = 5

mask = np.concatenate([np.arange(int(n_steps // 3)), int(2 * n_steps // 3) + np.arange(int(n_steps // 3))]).tolist()
task_params = {"cue_on": 0, "cue_off": n_steps,
               "stim_on": int(n_steps // 3), "stim_off": n_steps,
               "dec_on": int(2 * n_steps // 3), "dec_off": n_steps,
               "n_steps": n_steps, "n_inputs": n_inputs, "n_outputs": n_outputs}
tmp = max_coherence * np.logspace(-(coherence_lvls - 1), 0, coherence_lvls, base=2)
coherences = np.concatenate([-np.array(tmp[::-1]), np.array([0]), np.array(tmp)]).tolist()
task_params["coherences"] = coherences
task_params["seed"] = seed

# training specific
max_iter = 1000
tol = 1e-10
lr = 0.02
weight_decay = 5e-6
lambda_orth = 0.3
lambda_r = 0.3
same_batch = True
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
config_dict["shuffle"] = shuffle

json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(get_project_root(), "data", "configs", f"train_config_{tag}.json"), mode="w")
outfile.write(json_obj)
