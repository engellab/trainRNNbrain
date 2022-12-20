import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../')
from src.find_fp import get_LA_analytics, run
from src.RNN_numpy import RNN_numpy
from src.plots.color_scheme import get_colormaps
from matplotlib import pyplot as plt
from src.connectivity import get_small_connectivity_np
import pickle

def plot_LA_3D(RNN, N_points=200, nudge=0.075, steps_stim_on=500, steps_context_only_on=250, disp=True):
    nDim = 3
    W_rec = RNN.W_rec
    W_inp = RNN.W_inp
    W_out = RNN.W_out
    bias_rec = RNN.bias_rec
    bias_out = RNN.bias_out
    dt = RNN.dt
    tau = RNN.tau
    N = W_rec.shape[0]
    P_matrix = np.zeros((RNN.N, 3))
    P_matrix[:, 0] = W_out[0, :] - W_out[1, :]
    P_matrix[:, 1] = W_inp[:, 0] - W_inp[:, 1]
    P_matrix[:, 2] = W_inp[:, 2] + W_inp[:, 3] + W_inp[:, 4] + W_inp[:, 5]
    data_dict = get_LA_analytics(RNN, N_points=N_points, nudge = 0.01)
    RNN = RNN_numpy(N=N, dt=dt, tau=tau, W_inp=W_inp, W_rec=W_rec, W_out=W_out, bias_rec=bias_rec, bias_out=bias_out)

    trajectories = dict()
    trajectories["motion"] = {}
    trajectories["color"] = {}
    for ctxt in ["motion", "color"]:
        trajectories[ctxt] = {}
        for stim_status in ["relevant", "irrelevant"]:
            trajectories[ctxt][stim_status] = {}
            for period in ["context_only_on", "stim_on", "stim_off"]:
                trajectories[ctxt][stim_status][period] = {}

    colors, cmp = get_colormaps()
    red, blue, bluish, green, orange, lblue, violet = colors
    colors_trajectories = dict()
    colors_trajectories["motion"] = dict()
    colors_trajectories["color"] = dict()
    colors_trajectories["motion"]["relevant"] = colors[5]
    colors_trajectories["motion"]["irrelevant"] = colors[3]
    colors_trajectories["color"]["relevant"] = colors[1]
    colors_trajectories["color"]["irrelevant"] = colors[3]

    # pick a specific point and plot selection vectors:
    for ctxt in ["motion", "color"]:
        val = 1 if ctxt == 'motion' else 0
        for stim_status in ["relevant", "irrelevant"]:
            RNN.clear_history()
            rel_inds = [2, 3] if ctxt == 'motion' else [4, 5]
            irrel_inds = [4, 5] if ctxt == 'motion' else [2, 3]
            nudge_inds = rel_inds if stim_status == 'relevant' else irrel_inds

            input = np.array([val, 1 - val, 0.0, 0.0, 0.0, 0.0])
            x0 = 0.00 * np.random.randn(RNN.N)
            params = {"W_rec": W_rec, "W_inp": W_inp, "bias_rec": bias_rec,
                      "Input": input.reshape(-1, 1)}
            x_trajectory_context_only_on = run(steps_context_only_on, RNN.dt, RNN.tau, x0, **params)
            trajectories[ctxt][stim_status]["context_only_on"] = np.array(x_trajectory_context_only_on)

            RNN.clear_history()
            input = np.array([val, 1 - val, 0.5, 0.5, 0.5, 0.5])
            input[nudge_inds] += np.array([nudge, -nudge])
            params = {"W_rec": W_rec, "W_inp": W_inp, "bias_rec": bias_rec,
                      "Input": input.reshape(-1, 1)}
            x0 = x_trajectory_context_only_on[-1, :]
            x_trajectory_stim_on_right = run(steps_stim_on, RNN.dt, RNN.tau, x0, **params)
            trajectories[ctxt][stim_status]["stim_on"]["right"] = np.array(x_trajectory_stim_on_right)

            input = np.array([val, 1 - val, 0.5, 0.5, 0.5, 0.5])
            input[nudge_inds] += np.array([-nudge, +nudge])
            params = {"W_rec": W_rec, "W_inp": W_inp, "bias_rec": bias_rec,
                      "Input": input.reshape(-1, 1)}
            x0 = x_trajectory_context_only_on[-1, :]
            x_trajectory_stim_on_left = run(steps_stim_on, RNN.dt, RNN.tau, x0, **params)
            trajectories[ctxt][stim_status]["stim_on"]["left"] = np.array(x_trajectory_stim_on_left)

            input = np.array([val, 1 - val, 0.5, 0.5, 0.5, 0.5])
            params = {"W_rec": W_rec, "W_inp": W_inp, "bias_rec": bias_rec,
                      "Input": input.reshape(-1, 1)}
            x0 = x_trajectory_context_only_on[-1, :]
            x_trajectory_stim_on_center = run(steps_stim_on, RNN.dt, RNN.tau, x0, **params)
            trajectories[ctxt][stim_status]["stim_on"]["center"] = np.array(x_trajectory_stim_on_center)

    colors_trajectories = dict()
    colors_trajectories["motion"] = dict()
    colors_trajectories["color"] = dict()
    colors_trajectories["motion"]["relevant"] = orange
    colors_trajectories["motion"]["irrelevant"] = orange
    colors_trajectories["color"]["relevant"] = lblue
    colors_trajectories["color"]["irrelevant"] = lblue
    colors_LA = dict()
    colors_LA["motion"] = bluish
    colors_LA["color"] = green


    fig_3D = plt.figure()
    ax = fig_3D.add_subplot(projection='3d')
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel("Choice", fontsize=20)
    ax.set_ylabel("Context", fontsize=20)
    ax.set_zlabel("Sensory", fontsize=20)

    # initial point trajectory
    ax.scatter([0, 0], [0, 0], [0, 0], color='r', marker='o', s=10, alpha=0.9)

    for ctxt in ["motion", "color"]:
        slow_points_projected = data_dict[ctxt]["slow_points"] @ P_matrix
        ax.scatter(*(slow_points_projected[:, k] for k in range(nDim)), color=colors_LA[ctxt], marker='o', s=6,
                   alpha=0.2)
        ax.plot(*(slow_points_projected[:, k] for k in range(nDim)), color=colors_LA[ctxt])
        for stim_status in ["relevant"]:
            clr = colors_trajectories[ctxt][stim_status]

            period = "context_only_on"; alpha = 0.8; linewidth = 1.5;  linestyle = '-'
            trajectory_projected = trajectories[ctxt][stim_status][period] @ P_matrix
            ax.plot(*(trajectory_projected[:, t] for t in range(nDim)),
                    linestyle=linestyle, linewidth=linewidth, color=clr, alpha=alpha)

            period = "stim_on";  alpha = 0.9; linewidth = 3; linestyle = '-'
            trajectory_projected_right = trajectories[ctxt][stim_status][period]["right"] @ P_matrix
            ax.plot(*(trajectory_projected_right[:, t] for t in range(nDim)),
                    linestyle=linestyle, linewidth=linewidth, color=clr, alpha=alpha)

            trajectory_projected_left = trajectories[ctxt][stim_status][period]["left"] @ P_matrix
            ax.plot(*(trajectory_projected_left[:, t] for t in range(nDim)),
                    linestyle=':', linewidth=linewidth, color=clr, alpha=0.7)

            trajectory_projected_left = trajectories[ctxt][stim_status][period]["center"] @ P_matrix
            ax.plot(*(trajectory_projected_left[:, t] for t in range(nDim)),
                    linestyle='-', linewidth=linewidth, color='m', alpha=0.7)

    fig_3D.subplots_adjust()
    ax.view_init(12, 228)
    # ax.set_axis_off()
    # ax.view_init(30, 130)
    fig_3D.subplots_adjust()
    plt.tight_layout()
    if disp:
        plt.show()

    fig_RHS = plt.figure(figsize=(8, 4.5))
    # plt.suptitle(r"$\||RHS(x)\||^2$", fontsize = 20, y = 1.02)
    plt.axhline(0, color="gray", linewidth=2, alpha=0.2)
    plt.annotate(xy=(0, 0), xytext=(0.15, 0.0007), text=r"$\||RHS(x)\||^2$", fontsize=16)
    x = np.linspace(0, 1, N_points)
    plt.plot(x, np.array(data_dict["motion"]["fun_val"]), color=bluish, linewidth=3, linestyle='-', label="motion")
    plt.plot(x, np.array(data_dict["color"]["fun_val"]), color=green, linewidth=3, linestyle='-', label="color")
    plt.legend(fontsize=14)
    plt.xlabel("distance along the LA", fontsize = 16)
    plt.ylabel(r"$\||RHS(x)\||$", fontsize = 16)
    # plt.yticks([0.001], [r"$10^{-3}$"], fontsize=12, rotation=0)
    # plt.xticks([0, 0.1, 0.5, 0.9], fontsize=12)
    plt.grid(True)
    if disp:
        plt.show()
    return data_dict, fig_RHS, fig_3D

if __name__ == '__main__':
    # dt = 1
    # tau = 10
    # W_inp, W_rec, W_out = get_small_connectivity_np(rnd_perturb=1e-12)
    # N = W_rec.shape[0]
    # bias_rec = np.zeros(N)
    # bias_out = 0
    #
    # RNN = RNN_numpy(N=N, dt=dt, tau=tau,
    #                 W_inp=W_inp,
    #                 W_rec=W_rec,
    #                 W_out=W_out,
    #                 bias_rec=bias_rec,
    #                 bias_out=bias_out)
    data = pickle.load(open("../../data/trained_RNNs/RNN_20122022/20122022_num_outputs=2_N=50_Nsteps=750_srec=0.03_sinp=0.03_date=2022-12-20_SGE_TASK_ID=None_MSEout=0.0173588/data_20122022_num_outputs=2_N=50_Nsteps=750_srec=0.03_sinp=0.03_date=2022-12-20_SGE_TASK_ID=None_MSEout=0.0173588.pkl", "rb+"))
    RNN = RNN_numpy(N=data["N"], dt=data["dt"], tau=data["tau"],
                    W_inp=data["W_inp"],
                    W_rec=data["W_rec"],
                    W_out=data["W_out"],
                    bias_rec=data["bias_rec"],
                    bias_out=data["bias_out"])
    plot_LA_3D(RNN)
