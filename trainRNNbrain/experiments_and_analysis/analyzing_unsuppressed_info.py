import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
import os
import sys
import pickle
import json
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../")
from matplotlib import image as mpimg
import torch
from scipy.stats import zscore
from trainRNNbrain.rnn_coach.Task import TaskCDDM, TaskCDDMplus
from trainRNNbrain.rnn_coach.RNN_torch import RNN_torch
from scipy.sparse.linalg import lsqr
from copy import deepcopy
from sklearn.decomposition import PCA
import pandas as pd
import gc

# task_name = "CDDM"
# RNNs_path = os.path.join('../', '../', '../', "trainRNNbrain", "data", "trained_RNNs", task_name)
# rnns = []
# for folder in os.listdir(os.path.join('../', '../', "data", "trained_RNNs", task_name)):
#     if (folder == '.DS_Store'):
#         pass
#     else:
#         if "relu" in folder:
#             rnns.append(folder)
#
# scores = []
# Ns = []
# lmbdos = []
# lmbdrs = []
# lrs = []
# activations = []
# tags = []
# maxiters = []
# for folder in rnns:
#         files = os.listdir(os.path.join(RNNs_path, folder))
#         config_file = None
#         for file in files:
#             if "config" in file:
#                 config_file = file
#         config_data = json.load(open(os.path.join(RNNs_path, folder, config_file), "rb+"))
#         score = np.round(float(config_file.split("_")[0]), 7)
#         activation = config_data["activation"]
#         N = config_data["N"]
#         lmbdo = config_data["lambda_orth"]
#         lmbdr = config_data["lambda_r"]
#         lr=config_data["lr"]
#         maxiter=config_data["max_iter"]
#         extra_info = f"{task_name};{activation};N={N};lmbdo={lmbdo};lmbdr={lmbdr};lr={lr};maxiter={maxiter}"
#         name = f"{score}_{extra_info}"
#         tag = config_data["tag"]
#         scores.append(score)
#         Ns.append(N)
#         lmbdos.append(lmbdo)
#         lmbdrs.append(lmbdr)
#         lrs.append(lr)
#         tags.append(tag)
#         activations.append(activation)
#         maxiters.append(maxiter)
# df = pd.DataFrame({"name" : rnns, "scores" : scores, "N" : Ns, "activation": activations, "lmbdo" : lmbdos, "lmbdr": lmbdrs, "lr" : lrs, "maxiter" : maxiters})
# # additional filtering
# df = df[df['lr'] == 0.002]
# df = df[df['maxiter'] == 3000]
# df.sort_values("scores")
# top_RNNs = df.sort_values("scores")["name"].tolist()[:50]
#
# #loading rnns and analyzing them
# num_rnn = 0
# for num_rnn in range(len(top_RNNs)):
task_name = "CDDM"
RNNs_path = os.path.join('../', '../', '../', "trainRNNbrain", "data", "trained_RNNs", task_name)
# RNN_folder = sys.argv[1]
RNN_folder = '0.0113247_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000'
RNN_score = float(RNN_folder.split("_")[0])
RNN_path = os.path.join('../', '../', '../', "trainRNNbrain", "data", "trained_RNNs", task_name, RNN_folder)
RNN_data = json.load(open(os.path.join(RNN_path, f"{RNN_score}_params_{task_name}.json"), "rb+"))
RNN_config_file = json.load(open(os.path.join(RNN_path, f"{RNN_score}_config.json"), "rb+"))
W_out = np.array(RNN_data["W_out"])
W_rec = np.array(RNN_data["W_rec"])
W_inp = np.array(RNN_data["W_inp"])
bias_rec = np.array(RNN_data["bias_rec"])
y_init = np.array(RNN_data["y_init"])
activation = RNN_config_file["activation"]
mask = np.array(RNN_config_file["mask"])
input_size = RNN_config_file["num_inputs"]
output_size = RNN_config_file["num_outputs"]
task_params = RNN_config_file["task_params"]
n_steps = task_params["n_steps"]
sigma_inp = RNN_config_file["sigma_inp"]
sigma_rec = RNN_config_file["sigma_rec"]
dt = RNN_config_file["dt"]
tau = RNN_config_file["tau"]
task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
seed = np.random.randint(1000000)
print(f"seed: {seed}")
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
rng = torch.Generator(device=torch.device(device))
if not seed is None:
    rng.manual_seed(seed)

match activation:
    case 'relu':activation_RNN = lambda x: torch.maximum(x, torch.tensor(0))
    case "tanh": activation_RNN = torch.tanh
    case "sigmoid" : activation_RNN = lambda x: 1 / (1 + torch.exp(-x))
    case "softplus" : activation = activation_RNN = lambda x: torch.log(1 + torch.exp(5 * x))
RNN = RNN_torch(N=W_rec.shape[0], dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                activation=activation_RNN, random_generator=rng, device=device,
                sigma_rec=sigma_rec, sigma_inp=sigma_inp)
RNN_params = {"W_inp": W_inp,
              "W_rec": W_rec,
              "W_out": W_out,
              "b_rec": None,
              "y_init": np.zeros(W_rec.shape[0])}
RNN.set_params(RNN_params)
input_batch, target_batch, conditions_batch = task.get_batch()
n_trials = len(conditions_batch)
RNN.sigma_rec = RNN.sigma_inp = torch.tensor(0, device=RNN.device)
y, predicted_output_rnn = RNN(torch.from_numpy(input_batch.astype("float32")))
Y = np.hstack([y.detach().numpy()[:, :, i] for i in range(y.shape[-1])])

#TDR
Z = zscore(Y, axis = 1)
z = Z.reshape(-1, n_trials, n_steps)
z = np.swapaxes(z, 1, 2)
# PCA on Z
pca = PCA(n_components=20)
pca.fit(Z.T)
PCs = pca.components_
D = PCs.T @ PCs
Z_pca = D @ Z
z_pca = Z_pca.reshape(-1, n_trials, n_steps)
z_pca = np.swapaxes(z_pca, 1, 2)
choice = np.array([conditions_batch[i]['correct_choice'] for i in range(len(conditions_batch))])
motion_coh = np.array([conditions_batch[i]['motion_coh'] for i in range(len(conditions_batch))])
color_coh = np.array([conditions_batch[i]['color_coh'] for i in range(len(conditions_batch))])
context = np.array([(1 if conditions_batch[i]['context']=='motion' else -1) for i in range(len(conditions_batch))])
F = np.hstack([choice.reshape(-1, 1),
               motion_coh.reshape(-1, 1),
               color_coh.reshape(-1, 1),
               context.reshape(-1, 1),
               np.ones((n_trials, 1))])
B = np.zeros((Z.shape[0], n_steps, F.shape[1]))

for i in range(Z.shape[0]):
    for t in range(n_steps):
        betas_i_t = lsqr(F, z_pca[i, t, :], damp=100)[0]
        B[i, t, :] = deepcopy(betas_i_t)
ind_cont = np.argmax(np.linalg.norm(B[:, :, 0], axis=0))
ind_motion = np.argmax(np.linalg.norm(B[:, :, 1], axis=0))
ind_color = np.argmax(np.linalg.norm(B[:, :, 2], axis=0))
ind_choice = np.argmax(np.linalg.norm(B[:, :, 3], axis=0))
context_direction = B[:, ind_cont, 0]
motion_direction = B[:, ind_motion, 1]
color_direction = B[:, ind_color, 2]
choice_direction = B[:, ind_choice, 3]
B_max = np.hstack([context_direction.reshape(-1, 1),
                   motion_direction.reshape(-1, 1),
                   color_direction.reshape(-1, 1),
                   choice_direction.reshape(-1,1)])
U, s, V = np.linalg.svd(B_max)
B_orth = U[:, :V.shape[0]] @ V

TDR_decoded_vars = (B_orth.T @ Z_pca).reshape(-1, n_trials, n_steps)
TDR_decoded_vars = np.swapaxes(TDR_decoded_vars, 1, 2)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(TDR_decoded_vars[1, :, :225], color='r', alpha=0.1, label = 'relevant')
ax[0].plot(TDR_decoded_vars[1, :, 225:], color='b', alpha=0.1, label = 'irrelevant')
ax[0].set_ylim([-15, 15])
ax[1].plot(TDR_decoded_vars[2, :, :225], color='b', alpha=0.1)
ax[1].plot(TDR_decoded_vars[2, :, 225:], color='r', alpha=0.1)
ax[1].set_ylim([-15, 15])

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
ax[0].title.set_text('TDR motion, motion context')
ax[1].title.set_text('TDR color, color context')
ax[0].grid(True)
ax[1].grid(True)
# plt.show()
plt.savefig(f"../../img/dimensionality_reduction/{RNN_folder}_TDR.png")
plt.close(fig)

# Decoding
T = np.hstack([np.vstack([input_batch[:, :, i], target_batch[:, :, i]]) for i in range(target_batch.shape[-1])])
A = np.zeros((RNN.N, 8))
for i in range(T.shape[0]):
    a = lsqr(Y.T, T[i, :], damp=100)[0]
    A[:, i] = deepcopy(a)
U, s, V = np.linalg.svd(A)
A_orth = U[:, :V.shape[0]] @ V
Decoded_vars = (A_orth.T[:, :] @ Y[:, :]).reshape(-1, n_trials, n_steps)
Decoded_vars = np.swapaxes(Decoded_vars, 1, 2)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(Decoded_vars[2, :, :225] - Decoded_vars[3, :, :225], color='r', alpha=0.1, label = 'relevant')
ax[0].plot(Decoded_vars[2, :, 225:] - Decoded_vars[3, :, 225:], color='b', alpha=0.1, label = 'irrelevant')
# ax[0].set_ylim([-15, 15])
ax[1].plot(Decoded_vars[4, :, :225] - Decoded_vars[5, :, :225], color='b', alpha=0.1)
ax[1].plot(Decoded_vars[4, :, 225:] - Decoded_vars[5, :, 225:], color='r', alpha=0.1)
# ax[1].set_ylim([-15, 15])
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
ax[0].title.set_text('Decoded motion, motion context')
ax[1].title.set_text('Decoded color, color context')
ax[0].grid(True)
ax[1].grid(True)
# plt.show()
plt.savefig(f"../../img/dimensionality_reduction/{RNN_folder}_decoded.png")
plt.close(fig)

C = np.zeros((RNN.N, 8))
for i in range(T.shape[0]):
    c = lsqr(T.T, Y[i, :], damp=100)[0]
    C[i, :] = deepcopy(c)
U, s, V = np.linalg.svd(C)
C_orth = U[:, :V.shape[0]] @ V
Encoding_vars = (C_orth.T[:, :] @ Y[:, :]).reshape(-1, n_trials, n_steps)
Encoding_vars = np.swapaxes(Encoding_vars, 1, 2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(Encoding_vars[2, :, :225] - Encoding_vars[3, :, :225], color='r', alpha=0.1, label = 'relevant')
ax[0].plot(Encoding_vars[2, :, 225:] - Encoding_vars[3, :, 225:], color='b', alpha=0.1, label = 'irrelevant')
# ax[0].set_ylim([-15, 15])
ax[1].plot(Encoding_vars[4, :, :225] - Encoding_vars[5, :, :225], color='b', alpha=0.1)
ax[1].plot(Encoding_vars[4, :, 225:] - Encoding_vars[5, :, 225:], color='r', alpha=0.1)
# ax[1].set_ylim([-15, 15])

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
ax[0].title.set_text('Encoding direction motion, motion context')
ax[1].title.set_text('Encoding direction color, color context')
ax[0].grid(True)
ax[1].grid(True)
# plt.show()
plt.savefig(f"../../img/dimensionality_reduction/{RNN_folder}_encoding.png")
plt.close(fig)

gc.collect()
