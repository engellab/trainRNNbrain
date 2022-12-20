import os
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from src.utils import get_project_root
import numpy.random
from src.RNN_numpy import RNN_numpy
from src.RNN_torch import RNN_torch
from src.CDDM_training_utils import *
import pickle
import torch
from datetime import date
from src.CDDM_training_utils import trained_net_validation
from src.generate_CDDM_trials import *
from src.plots.local_analysis_1D import plot_projected_fixed_points_1D
from src.plots.local_analysis_3D import plot_LA_3D
from src.plots.plot_psychometric_function import get_psychometric_data, plot_psychometric_planes
from src.plots.plot_random_trials import plot_random_trials

'''
Train a bunch of large RNN on a CDDM task.
'''

def run_training(max_iter,
                 tag,
                 seed,
                 N,
                 num_outputs,
                 n_steps,
                 dt,
                 tau,
                 sr,
                 w_noise,
                 sigma_rec,
                 sigma_inp,
                 connectivity_density_rec,
                 lambda_o,
                 lr,
                 tol,
                 weight_decay,
                 protocol_dict,
                 train_mask,
                 coherences_train,
                 coherences_valid,
                 constrained,
                 disp_figs,
                 save_figs_locally,
                 save_data,
                 data_folder):
    '''
    :param max_iter: int, maximum number of iteration
    :param tag: string, put the tag on the resulting data which will be shown in the name of the file
    :param seed: int, seed, for reproducibility
    :param N: int, number of neurons in the RNN
    :param num_outputs: int (1 or 2) number of the output streams
    :param n_steps: int, number of steps in the trial
    :param dt: float, time resolution, < tau
    :param tau: float, internal time constant of individual neural node in the RNN
    :param sr: float, spectral radius of the initial W_rec matrix reflecting the neural connectivity
    :param w_noise: bool, training with or without noise
    :param sigma_rec: float, noise parameter in the recurrent dynamics
    :param sigma_inp: float, noise parameter in the input
    :param connectivity_density_rec: float, <=1 parameter controlling the sparcity of the recurrent connectivity
    :param lambda_o: regularization parameter making the columns of W_inp, and rows of W_out pair-wise orthogonal
    :param lr: float, learning rate
    :param tol: float, tolerance (before the training stops)
    :param weight_decay: floatm weight decay (similar to l2 orthogonalization of weights)
    :param protocol_dict: dictionary supplying the structure of the trial (see generate_CDDM_trials)
    :param train_mask: array, supplies the indices of the output time series which will be compared to the target timeseries
    :param coherences_train: list of coherences for motion and color used while training: e.g.[-1, -0.5, 0, 0.5, 0.5, 1]
    :param coherences_valid: list of coherences for motion and color used during validation
    :param constrained: bool, impose a bunch of hard constraints on weights of the RNN
    (Dale's law, positivity of input and output weights), see train_CDDM in CDDM_training_utils.py
    :param disp_figs: bool, suggesting showing the resulting figure at the end of the training
    :param save_figs_locally: bool, suggesting saving the figures
    :param save_data: bool, suggesting saving the resulting parameters and data
    :param data_folder: string, data folder to save the data and parameters to
    :return: rnn_trained - an instance of the trained RNN, data - trained parameters and a config for training
    '''
    config_data = deepcopy(locals())
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if seed is None:
        seed = np.random.randint(100000)
    generator_numpy = np.random.default_rng(seed)
    generator_torch = torch.Generator(device=device)
    generator_torch.manual_seed(seed)

    alpha = dt/tau
    rnn = RNN_torch(N=N, dt=dt, tau=tau,
                    spectral_rad=sr,
                    sigma_rec=sigma_rec,
                    sigma_inp=sigma_inp,
                    connectivity_density_rec=connectivity_density_rec,
                    lambda_o=lambda_o,
                    output_size=num_outputs,
                    random_generator=generator_torch).to(device)
    # dictionary of parameters needed to generate the trials to be trained on
    data_gen_params = {"n_steps": n_steps,
                       "coherences_train": coherences_train,
                       "coherences_valid": coherences_valid,
                       "protocol_dict": protocol_dict,
                       "num_outputs": num_outputs}
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=weight_decay)
    rnn_trained, train_losses, val_losses, best_net_params = train_CDDM(rnn,
                                                                        criterion=criterion,
                                                                        optimizer=optimizer,
                                                                        max_iter=max_iter,
                                                                        tol=tol,
                                                                        data_gen_params=deepcopy(data_gen_params),
                                                                        train_mask=train_mask,
                                                                        constrained=constrained,
                                                                        generator_numpy=generator_numpy)
    data = deepcopy(best_net_params)
    data["sr"] = sr
    data["lr"] = lr
    data["lambda_o"] = lambda_o
    data["coherences_train"] = coherences_train
    data["coherences_valid"] = coherences_valid
    data["n_steps"] = n_steps
    data["protocol_dict"] = protocol_dict
    data["dt"] = dt
    data["tau"] = tau
    data["w_noise"] = w_noise
    data["sigma_rec"] = sigma_rec
    data["sigma_inp"] = sigma_inp
    data["train_mask"] = train_mask

    W_out = data["W_out"]
    bias_out = data["bias_out"]
    W_rec = data["W_rec"]
    bias_rec = data["bias_rec"]
    W_inp = data["W_inp"]
    y_init = data["y_init"]

    # validation
    print("Validation after training...")
    generator_numpy = numpy.random.default_rng(seed+1) # use different seed to generate the noise in trials
    RNN_valid = RNN_numpy(N=N, dt=rnn.dt, tau=rnn.tau,
                          W_inp=W_inp,
                          W_rec=W_rec,
                          W_out=W_out,
                          bias_rec=bias_rec,
                          bias_out=bias_out,
                          y_init=y_init)
    score = trained_net_validation(RNN_valid,
                                   data_gen_params=deepcopy(data_gen_params),
                                   mask=train_mask,
                                   sigma_rec=sigma_rec,
                                   sigma_inp=sigma_inp,
                                   generator_numpy=generator_numpy)
    print(f"MSE validation: {score}")
    print(f"Plotting random trials")
    random_trials_figs = plot_random_trials(RNN_valid,
                                            num_trials_plot=15,
                                            mask=train_mask,
                                            sigma_rec=sigma_rec,
                                            sigma_inp=sigma_inp,
                                            data_gen_params=deepcopy(data_gen_params),
                                            disp=disp_figs)
    # plotting a bunch of figures: fixed point structure of the RNN projected on the choice axis,
    # psychometric plots, and the example of the trials
    print(f"Plotting fixed points projection on a choice vector")
    figs_1D = plot_projected_fixed_points_1D(RNN_valid, disp=disp_figs)
    print(f"Plotting slow point analysis in 3D")
    slow_points_data_dict, figs_RHS, fig_3D = plot_LA_3D(RNN_valid, disp=disp_figs)
    print(f"Plotting psychometric data")
    psycho_data = get_psychometric_data(RNN_valid, n_steps, protocol_dict, mask=train_mask,
                                        num_lvls=16, num_repetitions=11, sigma_inp=sigma_inp, sigma_rec=sigma_rec)
    psycho_figs = plot_psychometric_planes(psycho_data, disp=disp_figs)

    # Saving the data
    print(f"Saving the data")
    date_tag = ''.join((list(str(date.today()).split("-"))[::-1]))
    try:
        #if run on the cluster
        SGE_TASK_ID = int(os.environ["SGE_TASK_ID"])
    except:
        SGE_TASK_ID = None

    curdir = os.getcwd()
    os.chdir(get_project_root())
    os.chdir(data_folder)

    root_folder = f"RNN_{date_tag}"
    os.makedirs(root_folder, exist_ok=True)
    os.chdir(root_folder)

    name = f"{tag}_date={date.today()}_SGE_TASK_ID={SGE_TASK_ID}_MSEout={score}"
    folder = f"{tag}_date={date.today()}_SGE_TASK_ID={SGE_TASK_ID}_MSEout={score}"
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    data = {**data, **config_data}
    pickle.dump(data, open(f"data_{name}.pkl", "wb+"))

    if save_figs_locally:
        for i, f in enumerate(random_trials_figs):
            f.savefig(f"random_trial_{i}_{name}.png")
        for i, f in enumerate(figs_1D):
            ctx = "motion" if i == 0 else "color"
            f.savefig(f"fixed_points_1D_{ctx}_{name}.png")
        figs_RHS.savefig(f"RHS_norm_{name}.png")
        fig_3D.savefig(f"slow_points_{name}.png")
        for i, f in enumerate(psycho_figs):
            f.savefig(f"psychometric_function_{name}_{i}.png")
    os.chdir(curdir)
    return rnn_trained, data

if __name__ == '__main__':
    config_file = "train_config_20122022_num_outputs=2_N=50_Nsteps=450_srec=0.04_sinp=0.04.pkl"
    config_dict = pickle.load(open(os.path.join("../", "data", "configs", config_file), "rb+"))
    run_training(**config_dict)