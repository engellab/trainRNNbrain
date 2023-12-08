import json
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.DataSaver import DataSaver
from src.DynamicSystemAnalyzer import DynamicSystemAnalyzer
from src.PerformanceAnalyzer import PerformanceAnalyzerCDDM
from src.RNN_numpy import RNN_numpy
from src.utils import numpify, jsonify
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Tasks.TaskColorDiscriminationRGB import *
from matplotlib import pyplot as plt
import torch
import time
from sklearn.decomposition import PCA
# from src.datajoint_config import *

taskname = 'ColorDiscrimination'
activation = "relu"
train_config_file = f"train_config_{taskname}_{activation};N=70;lmbdr=0.5;lmbdo=0.3;lmbds=0.00;orth_inp_only=True.json"
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

disp = True
config_dict = json.load(
    open(os.path.join(RNN_configs_path, train_config_file), mode="r", encoding='utf-8'))

seed = np.random.randint(1000000)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
rng = torch.Generator(device=torch.device(device))
if not seed is None:
    rng.manual_seed(seed)

# defining RNN:
N = config_dict["N"]
activation_name = config_dict["activation"]
match activation_name:
    case 'relu': activation = lambda x: torch.maximum(torch.tensor(0.0), x)
    case 'tanh': activation = lambda x: torch.tanh(x)
    case 'sigmoid': activation = lambda x: 1 / (1 + torch.exp(-x))
    case 'softplus': activation = lambda x: torch.log(1 + torch.exp(5 * x))

dt = config_dict["dt"]
tau = config_dict["tau"]
constrained = config_dict["constrained"]
connectivity_density_rec = config_dict["connectivity_density_rec"]
spectral_rad = config_dict["sr"]
sigma_inp = config_dict["sigma_inp"]
sigma_rec = config_dict["sigma_rec"]

input_size = config_dict["num_inputs"]
output_size = config_dict["num_outputs"]

# Task:
n_steps = config_dict["n_steps"]
task_params = config_dict["task_params"]
task_params["seed"] = seed

# Trainer:
lambda_orth = config_dict["lambda_orth"]
orth_input_only = config_dict["orth_input_only"]
lambda_r = config_dict["lambda_r"]
mask = np.array(config_dict["mask"])
max_iter = config_dict["max_iter"]
tol = config_dict["tol"]
lr = config_dict["lr"]
weight_decay = config_dict["weight_decay"]
same_batch = config_dict["same_batch"]
lambda_smooth = config_dict["lambda_smooth"]

# General:
folder_tag = config_dict["folder_tag"]
timestr = time.strftime("%Y%m%d-%H%M%S")

# # creating instances:
rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                      activation=activation, constrained=constrained,
                      sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                      connectivity_density_rec=connectivity_density_rec,
                      spectral_rad=spectral_rad,
                      random_generator=rng)
task = eval("Task"+taskname)(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn_torch.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
trainer = Trainer(RNN=rnn_torch, Task=task,
                  max_iter=max_iter, tol=tol,
                  optimizer=optimizer, criterion=criterion,
                  lambda_orth=lambda_orth, orth_input_only=orth_input_only,
                  lambda_r=lambda_r)

tic = time.perf_counter()

def smoothness_penalty(states):
    # states has the shape (N, T, batch_size)
    batch_size = states.shape[-1]
    penalty = 0
    # the neighboring colors should not have drastically different responses!
    for i in range(batch_size):
        penalty += torch.mean(torch.pow(states[:, :, i] - states[:, :, (i + 1) % batch_size], 2))
    return penalty

penalty_dict = {"penalty_function" : smoothness_penalty, "lambda_smooth" : lambda_smooth}
rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask,
                                                                         same_batch=same_batch,
                                                                         penalty_dict=penalty_dict)
toc = time.perf_counter()
print(f"Executed training in {toc - tic:0.4f} seconds")

# throw out all the silent neurons!
######### clean the RNN from silent neurons!
input_batch, target_batch, conditions = task.get_batch()
rnn_torch.sigma_rec = rnn_torch.sigma_inp = torch.tensor(0, device=rnn_torch.device)
y, predicted_output_rnn = rnn_torch(torch.from_numpy(input_batch.astype("float32")).to(rnn_torch.device))
Y = torch.hstack([y.detach()[:, :, i] for i in range(y.shape[-1])]).T
Y_mean = torch.mean(torch.abs(Y), axis=0)
inds_fr = (torch.where(Y_mean > 0)[0]).tolist()
N_reduced = len(inds_fr)
config_dict["N"] = N_reduced
N = N_reduced
W_rec = net_params["W_rec"][inds_fr, :]
W_rec = W_rec[:, inds_fr]
net_params["W_rec"] = deepcopy(W_rec)
W_out = net_params["W_out"][:, inds_fr]
net_params["W_out"] = deepcopy(W_out)
W_inp = net_params["W_inp"][inds_fr, :]
net_params["W_inp"] = deepcopy(W_inp)
net_params["bias_rec"] = None
net_params["y_init"] = np.zeros(N_reduced)
RNN_params = {"W_inp": np.array(net_params["W_inp"]),
              "W_rec": np.array(net_params["W_rec"]),
              "W_out": np.array(net_params["W_out"]),
              "b_rec": np.array(net_params["bias_rec"]),
              "y_init": np.zeros(N)}
net_params["N"] = N_reduced
rnn_trained.set_params(RNN_params)
########

# validate
coherences_valid = np.linspace(-1, 1, 11)
task_params_valid = deepcopy(task_params)
task_params_valid["coherences"] = coherences_valid
task = eval("Task"+taskname)(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params_valid)



RNN_valid = RNN_numpy(N=net_params["N"],
                      dt=net_params["dt"],
                      tau=net_params["tau"],
                      activation=numpify(activation),
                      W_inp=net_params["W_inp"],
                      W_rec=net_params["W_rec"],
                      W_out=net_params["W_out"],
                      bias_rec=net_params["bias_rec"],
                      y_init=net_params["y_init"])

analyzer = PerformanceAnalyzerCDDM(RNN_valid)
score_function = lambda x, y: np.mean((x - y) ** 2)
input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask, sigma_rec=0, sigma_inp=0)
score = np.round(score, 7)
data_folder = f'{score}_{taskname};{activation_name};N={N_reduced};lmbdo={lambda_orth};orth_inp_only={orth_input_only};lmbdr={lambda_r};lr={lr};maxiter={max_iter}'
if folder_tag != '':
    data_folder+=f";tag={folder_tag}"
full_data_folder = os.path.join(data_save_path, data_folder)
datasaver = DataSaver(full_data_folder)

print(f"MSE validation: {score}")
if not (datasaver is None): datasaver.save_data(jsonify(config_dict), f"{score}_config.json")
if not (datasaver is None): datasaver.save_data(jsonify(net_params), f"{score}_params_{taskname}.json")

fig_trainloss = plt.figure(figsize=(10, 3))
plt.plot(train_losses, color='r', label='train loss (log scale)')
plt.plot(val_losses, color='b', label='valid loss (log scale)')
plt.yscale("log")
plt.grid(True)
plt.legend(fontsize=16)
if disp:
    plt.show()
if not (datasaver is None): datasaver.save_figure(fig_trainloss, f"{score}_train&valid_loss.png")

# colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]

colors = ["red", "#E34234", "orange", "#FFBF00",
          "yellow", "#7fff00", "green", "cyan",
          "blue", "#6D5ACF", "#7F00FF", "magenta"]

RNN_valid.clear_history()
RNN_valid.run(input_timeseries=input_batch_valid)
output = RNN_valid.get_output()
fig_output = plt.figure(figsize = (10, 4))
for i in range(len(colors)):
    plt.plot(output[i, -1, :].T, color=colors[i])
datasaver.save_figure(fig_output, f"{score}_outputs.png")

from tqdm.auto import tqdm
dsa = DynamicSystemAnalyzer(RNN_valid)
n = 90
points_colors = []
n_outputs = RNN_valid.W_out.shape[0]
for i in tqdm(range(n)):
    hue = i * (1./n)
    hsv = (hue, 1, 1)
    rgb = task.hsv_to_rgb(*hsv)
    input_stream, target_stream, condition = task.generate_input_target_stream(rgb)
    dsa.get_fixed_points(Input=input_stream[:, 0], patience = 10, stop_length=1, mode='exact')
    color_ind = int(((hue + 1.0 / (2 * n_outputs)) % 1) // (1.0 / n_outputs))
    points_colors.append(colors[color_ind])

points = []
for key in dsa.fp_data.keys():
    try:
        print(len(dsa.fp_data[key]['stable_fps']))
        points.append(dsa.fp_data[key]['stable_fps'])
    except:
        pass
points = np.vstack(points)

P = PCA(n_components=3)
P.fit(points)
points_pr = points @ P.components_.T

fig_fp, ax = plt.subplots(1, 1, figsize = (10, 4))
for i in range(points_pr.shape[0]):
    ax.scatter(points_pr[i, 0], points_pr[i, 1], color = points_colors[i], edgecolor = 'k')
datasaver.save_figure(fig_fp, f"{score}_fixedpoints.png")
datasaver.save_data(dsa.fp_data, f"{score}_fpdata.pkl")





