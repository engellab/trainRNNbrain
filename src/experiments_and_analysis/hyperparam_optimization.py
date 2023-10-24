import sys
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from functools import partial
from src.PerformanceAnalyzer import PerformanceAnalyzer
from src.RNN_numpy import RNN_numpy
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from src.DataSaver import DataSaver
import numpy as np
np.set_printoptions(suppress=True)
from src.utils import get_project_root
import warnings
from src.utils import numpify, jsonify, orthonormalize
warnings.simplefilter("ignore", UserWarning)
import json
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Task import *
import torch
from pathlib import Path

def objective(trial, taskname, activation, num_repeats=7):

    train_config_file = f"train_config_{taskname}_{activation}.json"

    home = str(Path.home())
    RNN_configs_path = home + '/Documents/GitHub/rnn_coach/data/configs'
    seeds = np.arange(num_repeats)

    # define params to be varied
    lr = trial.suggest_loguniform('lr', 0.001, 0.05)
    lmbd_orth = trial.suggest_uniform('lmbd_orth', 0.0, 0.5)
    lmbd_r = trial.suggest_uniform('lmbd_r', 0.0, 0.5)
    spectral_rad = trial.suggest_uniform('spectral_rad', 0.8, 1.5)
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.2)

    config_dict = json.load(
        open(os.path.join(RNN_configs_path, train_config_file), mode="r", encoding='utf-8'))

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
    # spectral_rad = config_dict["sr"]
    sigma_inp = config_dict["sigma_inp"]
    sigma_rec = config_dict["sigma_rec"]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    rng = torch.Generator(device=torch.device(device))

    input_size = config_dict["num_inputs"]
    output_size = config_dict["num_outputs"]

    # Task:
    n_steps = config_dict["n_steps"]
    task_params = config_dict["task_params"]

    # Trainer:
    # lambda_orth = config_dict["lambda_orth"]
    orth_input_only = config_dict["orth_input_only"]
    # lambda_r = config_dict["lambda_r"]
    mask = np.array(config_dict["mask"])
    max_iter = config_dict["max_iter"]
    tol = config_dict["tol"]
    # lr = config_dict["lr"]
    # weight_decay = config_dict["weight_decay"]
    same_batch = config_dict["same_batch"]

    scores = []
    for seed in seeds:
        rng.manual_seed(int(seed))
        task_params["seed"] = seed

        # # creating instances:
        rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                              activation=activation, constrained=constrained,
                              sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                              connectivity_density_rec=connectivity_density_rec,
                              spectral_rad=spectral_rad,
                              random_generator=rng)

        task = eval("Task" + taskname)(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(rnn_torch.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        trainer = Trainer(RNN=rnn_torch, Task=task,
                          max_iter=max_iter, tol=tol,
                          optimizer=optimizer, criterion=criterion,
                          lambda_orth=lmbd_orth, orth_input_only=orth_input_only,
                          lambda_r=lmbd_r)

        rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask, same_batch=same_batch)

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
        RNN_valid = RNN_numpy(N=net_params["N"],
                              dt=net_params["dt"],
                              tau=net_params["tau"],
                              activation=numpify(activation),
                              W_inp=net_params["W_inp"],
                              W_rec=net_params["W_rec"],
                              W_out=net_params["W_out"],
                              bias_rec=net_params["bias_rec"],
                              y_init=net_params["y_init"],
                              seed=seed)

        analyzer = PerformanceAnalyzer(RNN_valid)
        score_function = lambda x, y: np.mean((x - y) ** 2)
        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask,
                                              sigma_rec=0,
                                              sigma_inp=0)
        score = np.round(score, 7)
        print(f"MSE validation: {np.round(score, 5)}")
        scores.append(deepcopy(score))

    return np.mean(score)


if __name__ == '__main__':
    taskname = 'MemoryAntiAngle'
    activation = 'tanh'
    num_repeats = 1
    n_trials = 5

    os.umask(0)
    img_path = os.path.abspath(os.path.join(get_project_root(), "img", f"{taskname}_hyperparam_optimization"))
    os.makedirs(img_path, exist_ok=True, mode=0o777)
    print(img_path)
    #set up img folder


    study = optuna.create_study(direction="maximize")
    objective = partial(objective, taskname=taskname, activation=activation, num_repeats=num_repeats)
    study.optimize(objective, n_trials=n_trials)
    print(f"best parameters : {study.best_params}")



    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image(os.path.join(img_path,f"{taskname}_optimization_history.pdf"), width=1920, height=1080)

    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image(os.path.join(img_path, f"{taskname}_importance.pdf"), width=1920, height=1080)