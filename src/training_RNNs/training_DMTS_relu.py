import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import json
from src.training_RNNs.training_utils import remove_silent_nodes, set_paths
from src.DataSaver import DataSaver
from src.PerformanceAnalyzer import PerformanceAnalyzer
from src.RNN_numpy import RNN_numpy
from src.utils import numpify, jsonify
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Tasks.TaskDMTS import *
from matplotlib import pyplot as plt
import torch
import time


disp = False
activation = "relu"
constrained = True
taskname = "DMTS"
info_tag = f'{taskname}_{activation}_constrained={constrained}'
train_config_file = f"train_config_{info_tag}.json"
save_folder_name = info_tag
home, data_save_path, RNN_configs_path = set_paths(save_folder_name)

config_dict = json.load(
    open(os.path.join(RNN_configs_path, train_config_file), mode="r", encoding='utf-8'))

# defining RNN:
N = config_dict["N"]
activation_name = config_dict["activation"]
match activation_name:
    case 'relu':
        activation = lambda x: torch.maximum(torch.tensor(0.0), x)
    case 'tanh':
        activation = lambda x: torch.tanh(x)
    case 'sigmoid':
        activation = lambda x: 1 / (1 + torch.exp(-x))
    case 'softplus':
        activation = lambda x: torch.log(1 + torch.exp(5 * x))

dt = config_dict["dt"]
tau = config_dict["tau"]
constrained = config_dict["constrained"]
exc_to_inh_ratio = config_dict["exc_to_inh_ratio"]
connectivity_density_rec = config_dict["connectivity_density_rec"]
spectral_rad = config_dict["sr"]
sigma_inp = config_dict["sigma_inp"]
sigma_rec = config_dict["sigma_rec"]
# seed = config_dict["seed"]
seed = None

input_size = config_dict["num_inputs"]
output_size = config_dict["num_outputs"]

# Task:
n_steps = config_dict["n_steps"]
task_params = config_dict["task_params"]

# Trainer:
lambda_orth = config_dict["lambda_orth"]
orth_input_only = config_dict["orth_input_only"]
lambda_r_1 = config_dict["lambda_r_1"]
lambda_r_2 = config_dict["lambda_r_2"]
mask = np.array(config_dict["mask"])
max_iter_1 = config_dict["max_iter_1"]
max_iter_2 = config_dict["max_iter_2"]
tol = config_dict["tol"]
lr = config_dict["lr"]
weight_decay = config_dict["weight_decay"]
same_batch = config_dict["same_batch"]

# General:
folder_tag = config_dict["folder_tag"]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

for tries in range(10):
    rng = torch.Generator(device=torch.device(device))
    if not seed is None:
        rng.manual_seed(seed)
    else:
        rng.manual_seed(np.random.randint(100000))

    # # creating instances:
    rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                          activation=activation, constrained=constrained,
                          exc_to_inh_ratio=exc_to_inh_ratio,
                          sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                          connectivity_density_rec=connectivity_density_rec,
                          spectral_rad=spectral_rad,
                          random_generator=rng)

    task = TaskDMTS(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rnn_torch.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    trainer = Trainer(RNN=rnn_torch, Task=task,
                      max_iter=max_iter_1, tol=tol,
                      optimizer=optimizer, criterion=criterion,
                      lambda_orth=lambda_orth, orth_input_only=orth_input_only,
                      lambda_r=lambda_r_1)

    tic = time.perf_counter()
    #multistage training

    rnn_trained_1, train_losses_1, val_losses_1, _ = trainer.run_training(train_mask=mask, same_batch=same_batch)
    trainer.lambda_r = lambda_r_2
    trainer.max_iter = max_iter_2
    trainer.RNN = rnn_trained_1

    rnn_trained_2, train_losses_2, val_losses_2, net_params = trainer.run_training(train_mask=mask, same_batch=same_batch)
    train_losses = train_losses_1 + train_losses_2
    val_losses = val_losses_1 + val_losses_2


    toc = time.perf_counter()
    print(f"Executed training in {toc - tic:0.4f} seconds")

    rnn_torch, net_params = remove_silent_nodes(rnn_torch=rnn_trained_2,
                                                task=task,
                                                net_params=net_params)

    # validate
    RNN_valid = RNN_numpy(N=net_params["N"],
                          dt=net_params["dt"],
                          tau=net_params["tau"],
                          activation=numpify(activation),
                          W_inp=net_params["W_inp"],
                          W_rec=net_params["W_rec"],
                          W_out=net_params["W_out"],
                          bias_rec=net_params["bias_rec"],
                          y_init=net_params["y_init"])

    analyzer = PerformanceAnalyzer(RNN_valid)
    score_function = lambda x, y: np.mean((x - y) ** 2)
    input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
    score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask, sigma_rec=0, sigma_inp=0)
    score = np.round(score, 7)
    data_folder = f'{score}_{taskname};{activation_name};N={net_params["N"]};lmbdo={lambda_orth};orth_inp_only={orth_input_only};lmbdr={lambda_r_2};lr={lr};maxiter={max_iter_2}'
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
    if disp: plt.show()
    if not (datasaver is None): datasaver.save_figure(fig_trainloss, f"{score}_train&valid_loss.png")

    batch_size = input_batch_valid.shape[2]
    RNN_valid.clear_history()
    RNN_valid.run(input_timeseries=input_batch_valid, sigma_rec=0, sigma_inp=0)
    RNN_trajectories = RNN_valid.get_history()
    RNN_output = RNN_valid.get_output()
    trajecory_data = {}
    trajecory_data["inputs"] = input_batch_valid
    trajecory_data["trajectories"] = RNN_trajectories
    trajecory_data["outputs"] = RNN_output
    trajecory_data["targets"] = target_batch_valid
    trajecory_data["conditions"] = conditions_valid
    datasaver.save_data(trajecory_data, f"{score}_RNNtrajdata_{taskname}.pkl")

    print(f"Plotting random trials")
    inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
    inputs = input_batch_valid[..., inds]
    targets = target_batch_valid[..., inds]

    fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    if disp:
        plt.show()
    if not (datasaver is None): datasaver.save_figure(fig_trials, "random_trials.png")
