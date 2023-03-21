import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

import json
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.DataSaver import DataSaver
from src.DynamicSystemAnalyzer import DynamicSystemAnalyzerCDDM
from src.PerformanceAnalyzer import PerformanceAnalyzerCDDM
from src.RNN_numpy import RNN_numpy
from src.utils import get_project_root, numpify, jsonify
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Task import *
from matplotlib import pyplot as plt
import torch
import time


def objective(trial):
    lambda_r = trial.suggest_loguniform('lambda_r', 0.01, 10)
    # # creating instances:
    rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                          activation=activation, constrained=constrained,
                          sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                          connectivity_density_rec=connectivity_density_rec,
                          spectral_rad=spectral_rad,
                          random_generator=rng)
    task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rnn_torch.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    trainer = Trainer(RNN=rnn_torch, Task=task,
                      max_iter=max_iter, tol=tol,
                      optimizer=optimizer, criterion=criterion,
                      lambda_orth=lambda_orth, lambda_r=lambda_r)

    tic = time.perf_counter()
    rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask, same_batch=same_batch)
    toc = time.perf_counter()
    print(f"Executed training in {toc - tic:0.4f} seconds")

    # validate
    coherences_valid = np.linspace(-1, 1, 11)
    task_params_valid = deepcopy(task_params)
    task_params_valid["coherences"] = coherences_valid
    task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params_valid)

    if activation_name == 'relu':
        activation_np = lambda x: np.maximum(x, 0)
    elif activation_name == 'tanh':
        activation_np = np.tanh
    elif activation_name == 'sigmoid':
        activation_np = lambda x: 1 / (1 + np.exp(-x))
    elif activation_name == 'softplus':
        activation_np = lambda x: np.log(1 + np.exp(5 * x))

    RNN_valid = RNN_numpy(N=net_params["N"],
                          dt=net_params["dt"],
                          tau=net_params["tau"],
                          activation=activation_np,
                          W_inp=net_params["W_inp"],
                          W_rec=net_params["W_rec"],
                          W_out=net_params["W_out"],
                          bias_rec=net_params["bias_rec"],
                          y_init=net_params["y_init"])

    analyzer = PerformanceAnalyzerCDDM(RNN_valid)
    score_function = lambda x, y: np.mean((x - y) ** 2)
    input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
    score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask, sigma_rec=0,
                                          sigma_inp=0)
    score = np.round(score, 7)

    return score


if __name__ == '__main__':
    disp = False
    activation = "relu"
    taskname = "CDDM"
    train_config_file = f"train_config_{taskname}_{activation}.json"
    config_dict = json.load(
        open(os.path.join(get_project_root(), "data", "configs", train_config_file), mode="r", encoding='utf-8'))

    seed = 0
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
    if activation_name == 'relu':
        activation = lambda x: torch.maximum(x, torch.tensor(0))
    elif activation_name == 'tanh':
        activation = torch.tanh
    elif activation_name == 'sigmoid':
        activation = lambda x: 1 / (1 + torch.exp(-x))
    elif activation_name == 'softplus':
        activation = lambda x: torch.log(1 + torch.exp(5 * x))

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
    lambda_r = config_dict["lambda_r"]
    mask = np.array(config_dict["mask"])
    max_iter = config_dict["max_iter"]
    tol = config_dict["tol"]
    lr = config_dict["lr"]
    weight_decay = config_dict["weight_decay"]
    same_batch = config_dict["same_batch"]



    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print(f"best parameters : {study.best_params}")
    root_folder = get_project_root()
    os.chdir(root_folder)
    os.chdir(os.path.join(os.getcwd(), "figures"))
    fig = plot_optimization_history(study).get_figure()
    fig.savefig("optimization_history_lambda_r.pdf")
    fig = plot_param_importances(study).get_figure()
    fig.savefig("param_importances_lambda_r.pdf")