from datetime import date
import json
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.DataSaver import DataSaver
from src.PerformanceAnalyzer import PerformanceAnalyzer
from src.RNN_numpy import RNN_numpy
from src.utils import numpify, jsonify
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Tasks.Task import *
from matplotlib import pyplot as plt
import torch
import time
from pathlib import Path
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from src.utils import get_project_root

date = ''.join((list(str(date.today()).split("-"))[::-1]))
num_repeats = 5
# RNN specific
N = 50

taskname = 'MemoryAntiAngle'
for constrained in [True]:
    for activation_name in ['humpy_relu']:
        for i in range(num_repeats):

            seed = i
            sigma_inp = 0.05
            sigma_rec = 0.05
            dt = 1
            tau = 10
            sr = 1.3
            connectivity_density_rec = 1.0
            #Trial 22 finished with value: 0.0037636 and parameters: {'lr': 0.0022975299091267066, 'lmbd_orth': 0.1000504771878649, 'lmbd_r': 0.002122537293968812, 'spectral_rad': 1.3136631625741721, 'weight_decay': 1.1496845347878425e-05}. Best is trial 22 with value: 0.0037636.[0m
            # task specific
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
            max_iter = 10000
            tol = 1e-10
            lr = 0.01
            weight_decay = 4e-6
            lambda_orth = 0.25
            orth_input_only = True
            lambda_r = 0.01 #0.0003
            same_batch = False  # generate new batch in each train loop
            shuffle = False

            data_folder = os.path.abspath(os.path.join(get_project_root(), "data", "trained_RNNs", f"{taskname}"))
            config_tag = f'{taskname}_{activation_name}'

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

            train_config_file = f"train_config_{taskname}_{activation_name}.json"

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

            disp = False
            # defining RNN:
            N = config_dict["N"]
            activation_name = config_dict["activation"]
            match activation_name:
                case 'relu': activation = lambda x: torch.maximum(torch.tensor(0.0), x)
                case 'tanh': activation = lambda x: torch.tanh(x)
                case 'sigmoid': activation = lambda x: 1 / (1 + torch.exp(-x))
                case 'softplus': activation = lambda x: torch.log(1 + torch.exp(5 * x))
                case 'humpy_relu': activation = lambda x: torch.maximum(torch.tensor(0), x + 2 * torch.tanh(x)* torch.exp(-(torch.abs(x))))
                case 'humpy_line': activation = lambda x: x + 2 * torch.tanh(x) * torch.exp(-(torch.abs(x)))
                case 'leaky_relu': activation = lambda x: torch.maximum(torch.tensor(0.0), x) + torch.sign(x) * torch.maximum(torch.tensor(0.0), -0.1 * x)

            dt = config_dict["dt"]
            tau = config_dict["tau"]
            constrained = config_dict["constrained"]
            connectivity_density_rec = config_dict["connectivity_density_rec"]
            spectral_rad = config_dict["sr"]
            sigma_inp = config_dict["sigma_inp"]
            sigma_rec = config_dict["sigma_rec"]
            # seed = config_dict["seed"]
            seed = None

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            rng = torch.Generator(device=torch.device(device))

            if not seed is None:
                rng.manual_seed(seed)
            else:
                rng.manual_seed(np.random.randint(100000))

            input_size = config_dict["num_inputs"]
            output_size = config_dict["num_outputs"]

            # Task:
            n_steps = config_dict["n_steps"]
            task_params = config_dict["task_params"]

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

            # General:
            folder_tag = config_dict["folder_tag"]
            timestr = time.strftime("%Y%m%d-%H%M%S")
            data_folder = os.path.join(config_dict["data_folder"], timestr)

            # # creating instances:
            rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                                  activation=activation, constrained=constrained,
                                  sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                                  connectivity_density_rec=connectivity_density_rec,
                                  spectral_rad=spectral_rad,
                                  random_generator=rng)
            task = TaskMemoryAntiAngle(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size,
                                       task_params=task_params)
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
            rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask,
                                                                                     same_batch=same_batch)
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
            score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask,
                                                  sigma_rec=0, sigma_inp=0)
            score = np.round(score, 7)
            data_folder = f'{score}_{taskname};{activation_name};N={N_reduced};lmbdo={lambda_orth};orth_inp_only={orth_input_only};lmbdr={lambda_r};lr={lr};maxiter={max_iter}'
            if folder_tag != '':
                data_folder += f";tag={folder_tag}"
            full_data_folder = os.path.join(data_save_path, data_folder)
            datasaver = DataSaver(full_data_folder)

            fig_trainloss = plt.figure(figsize=(10, 3))
            plt.plot(train_losses, color='r', label='train loss (log scale)')
            plt.plot(val_losses, color='b', label='valid loss (log scale)')
            plt.yscale("log")
            plt.grid(True)
            plt.legend(fontsize=16)
            if disp:
                plt.show()
            if not (datasaver is None): datasaver.save_figure(fig_trainloss, f"{activation_name}_train&valid_loss")

            print(f"MSE validation: {score}")
            if not (datasaver is None): datasaver.save_data(jsonify(config_dict), f"{score}_{activation_name}_config.json")
            if not (datasaver is None): datasaver.save_data(jsonify(net_params), f"{score}_{activation_name}_params_{taskname}.json")

            print(f"Plotting random trials")
            inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
            inputs = input_batch_valid[..., inds]
            targets = target_batch_valid[..., inds]

            fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
            if disp:
                plt.show()
            if not (datasaver is None): datasaver.save_figure(fig_trials, f"{activation_name}_random_trials.png")


