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
constrained = False
seed = None
sigma_inp = 0.05
sigma_rec = 0.05
dt = 1
tau = 10
sr = 1.3
connectivity_density_rec = 1.0
#Trial 22 finished with value: 0.0037636 and parameters: {'lr': 0.0022975299091267066, 'lmbd_orth': 0.1000504771878649, 'lmbd_r': 0.002122537293968812, 'spectral_rad': 1.3136631625741721, 'weight_decay': 1.1496845347878425e-05}. Best is trial 22 with value: 0.0037636.[0m
# task specific
task_name = 'MemoryAntiAngle'
n_outputs = 3
n_inputs = n_outputs + 2
T = 120
n_steps = int(T / dt)
task_params = dict()
task_params["stim_on"] = 1 * n_steps // 12
task_params["stim_off"] = 2 * n_steps // 12
task_params["random_window"] = 1 * n_steps // 12
task_params["recall_on"] = 9 * n_steps // 12
task_params["recall_off"] = n_steps
task_params["seed"] = seed
# mask = np.concatenate([np.arange(n_steps)]).tolist() # using the whole trial
mask = np.concatenate([np.arange(task_params["recall_on"]),
                       10 * n_steps // 12 + np.arange(2 * n_steps // 12)]).tolist()

# training specific
max_iter = 5000
tol = 1e-10
lr = 0.01
weight_decay = 4e-6
lambda_orth = 0.25
orth_input_only = True
lambda_r = 0.01 #0.0003
same_batch = False  # generate new batch in each train loop
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