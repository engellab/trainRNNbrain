import pickle
import os
import numpy as np
from datetime import date

seed = None
save_figs_locally = True
save_data = True
max_iter = 600
disp_figs = False
cue_on_throughout = True
num_outputs = 2
date = ''.join((list(str(date.today()).split("-"))[::-1]))
T = 450
dt = 1
tau = 10
n_steps = int(T / dt)
N = 50
sr = 1.0
tol = 1e-10
lr = 2e-3
max_coherence_train = 0.8
max_coherence_valid = 1.0
coherence_lvls = 5
weight_decay = 5e-6
w_noise = False
sigma_inp = 0.04
sigma_rec = 0.04
constrained = True
connectivity_density_rec = 1.0
lambda_o = 0.3
shuffle = False
data_folder = os.path.abspath(os.path.join("../data", "trained_RNNs"))
# data_folder = "/grid/engel/home/tolmach/selection_in_neural_circuits/data/trained_large_RNNs"

train_mask = np.concatenate([np.arange(int(n_steps // 3)), int(2 * n_steps // 3) + np.arange(int(n_steps // 3))])

protocol_dict = {"cue_on": 0, "cue_off": n_steps if cue_on_throughout else int(n_steps // 3),
                 "stim_on": int(n_steps // 3), "stim_off": n_steps,
                 "dec_on": int(2 * n_steps // 3), "dec_off": n_steps}
coherences_train = np.concatenate([-max_coherence_train * np.logspace(-6, 0, coherence_lvls, base=2)[::-1], [0], max_coherence_train * np.logspace(-6, 0, coherence_lvls, base=2)])
coherences_valid = np.linspace(-1, 1, 11)
tag = f'{date}_num_outputs={num_outputs}_N={N}_Nsteps={n_steps}_srec={sigma_rec}_sinp={sigma_inp}'

config_dict = {}
config_dict["num_outputs"] = num_outputs
config_dict["seed"] = seed
config_dict["max_iter"] = max_iter
config_dict["disp_figs"] = disp_figs
config_dict["save_data"] = save_data
config_dict["save_figs_locally"] = save_figs_locally
config_dict["N"] = N
config_dict["dt"] = dt
config_dict["n_steps"] = n_steps
config_dict["tau"] = tau
config_dict["sr"] = sr
config_dict["tol"] = tol
config_dict["lr"] = lr
config_dict["weight_decay"] = weight_decay
config_dict["connectivity_density_rec"] = connectivity_density_rec
config_dict["lambda_o"] = lambda_o
config_dict["w_noise"] = w_noise
config_dict["sigma_inp"] = sigma_inp
config_dict["sigma_rec"] = sigma_rec
config_dict["tag"] = tag
config_dict["constrained"] = constrained
config_dict["train_mask"] = train_mask
config_dict["protocol_dict"] = protocol_dict
config_dict["coherences_train"] = coherences_train
config_dict["coherences_valid"] = coherences_valid
config_dict["data_folder"] = data_folder
pickle.dump(config_dict, open(os.path.join("../data", "configs", f"train_config_{tag}.pkl"), "wb+"))

