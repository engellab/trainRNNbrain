import numpy as np
from matplotlib import pyplot as plt
import os
import json
from omegaconf import OmegaConf
from trainRNNbrain.utils import jsonify, unjsonify
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
import pickle
import hydra
from tqdm.auto import tqdm
import mplcursors
OmegaConf.register_new_resolver("eval", eval)

# User should provide:
# data_folder = '../../data/CDDM_relu_constrained=True_LambdaRSweep'
# hyperparam_name = 'trainer.lambda_r'

# data_folder = '../../data/CDDM_relu_constrained=True_WeightDecaySweep'
# hyperparam_name = 'trainer.weight_decay'

# data_folder = '../../data/CDDM_relu_constrained=True_LambdaZSweep'
# hyperparam_name = 'trainer.lambda_z'

# data_folder = '../../data/CDDM_relu_constrained=True_w&woStepwiseDO'
# hyperparam_name = 'trainer.dropout'

data_folder = '../../data/trained_RNNs/CDDM_relu_constrained=True_Categorical'
hyperparam_name = 'trainer.lambda_m'

save_name = f"{hyperparam_name.split(".")[1]}_Sweep.pkl"
aux_save_folder = '../../data/auxiliary_data/'
aux_save_path = os.path.join(aux_save_folder, save_name)

# =========== Utility ===========

def gini(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x = x - np.amin(x)
    mean = np.mean(x)
    if mean == 0:
        return 0.0
    diff_sum = np.abs(np.subtract.outer(x, x)).sum()
    return diff_sum / (2 * len(x)**2 * mean)

# ========== Try to load data first ==========
if os.path.exists(aux_save_path):
    print(f"Loading precomputed data from {aux_save_path}")
    with open(aux_save_path, "rb") as f:
        data = pickle.load(f)
else:
    print("No precomputed data found. Scanning folders and computing metrics...")
    folders = [f for f in os.listdir(data_folder)
               if not f.startswith('.') and os.path.isdir(os.path.join(data_folder, f))]
    data = {}
    # Dynamically find all unique values of hyperparam
    all_vals = set()
    for folder in tqdm(folders):
        try:
            R2 = float(folder.split("_")[0])
            cfg_path = os.path.join(data_folder, folder, f"{R2}_config.yaml")
            cfg = OmegaConf.load(cfg_path)
            v = OmegaConf.select(cfg, hyperparam_name)
            all_vals.add(v)
        except Exception:
            continue
    all_vals = sorted(list(all_vals))
    # Init data dict
    for v in all_vals:
        key = f"{hyperparam_name}={v}"
        data[key] = {"R2": [], "N": [], "gini": []}

    for folder in tqdm(folders):
        try:
            R2 = float(folder.split("_")[0])
            cfg_path = os.path.join(data_folder, folder, f"{R2}_config.yaml")
            cfg = OmegaConf.load(cfg_path)
            v = OmegaConf.select(cfg, hyperparam_name)
            key = f"{hyperparam_name}={v}"

            param_file = f'{R2}_params_CDDM.json'
            with open(os.path.join(data_folder, folder, param_file), 'r') as f:
                net_params = unjsonify(json.load(f))
            # defining the task
            task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
            task = hydra.utils.instantiate(task_conf)
            RNN = RNN_numpy(**net_params)
            analyzer = PerformanceAnalyzer(RNN)
            input_batch, target_batch, conditions = task.get_batch()
            trajectories, outputs = analyzer.get_firing_rate_trajectories(input_batch)
            vect = np.std(trajectories, axis=(1, 2)) + np.mean(trajectories, axis=(1, 2))
            participation = np.pad(vect, (0, cfg.model.N - len(vect)), mode='constant')
            g = gini(participation)
            N = np.sum((vect > 1e-3).astype(int))
            data[key]["R2"].append(R2)
            data[key]["N"].append(N)
            data[key]["gini"].append(g)
        except Exception as e:
            print(f"Skipping folder {folder}: {e}")
    # Save for future use
    os.makedirs(aux_save_folder, exist_ok=True)
    with open(aux_save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved computed data to {aux_save_path}")

# =========== Plotting ===========

markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan']

fig, ax = plt.subplots(figsize=(8, 6))
for i, key in enumerate(sorted(data.keys())):
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    r2 = data[key]['R2']
    n = data[key]['N']
    ax.scatter(
        r2, n,
        label=key,
        marker=marker,
        color=color,
        alpha=0.75,
        edgecolor='k',
        s=80
    )

ax.set_xlabel(r'$R^2$', fontsize=14)
ax.set_ylabel('Number of Active Units (N)', fontsize=14)
# ax.set_ylabel('Gini Index', fontsize=14)
ax.set_title(r'Number of Units vs $R^2$ (Hyperparameter comparison)', fontsize=16)
ax.legend(fontsize=12, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(labelsize=12)
fig.tight_layout()
mplcursors.cursor(hover=True) # Enable hover functionality
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for i, key in enumerate(sorted(data.keys())):
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    r2 = data[key]['R2']
    n = data[key]['gini']
    ax.scatter(
        r2, n,
        label=key,
        marker=marker,
        color=color,
        alpha=0.75,
        edgecolor='k',
        s=80
    )

ax.set_xlabel(r'$R^2$', fontsize=14)
ax.set_ylabel('Number of Active Units (N)', fontsize=14)
ax.set_ylabel('Gini Index', fontsize=14)
# ax.set_title(r'Number of Units vs $R^2$ (Hyperparameter comparison)', fontsize=16)
ax.legend(fontsize=12, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(labelsize=12)
fig.tight_layout()
mplcursors.cursor(hover=True) # Enable hover functionality
plt.show()
