from src.utils import get_colormaps
colors, cmp = get_colormaps()
red, blue, bluish, green, orange, lblue, violet = colors
mm = 1/25.4  # inches in mm


import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
import os
import sys
import json
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
import pandas as pd
import datetime
import pickle

task_name = "CDDM"
RNNs_path = os.path.join('../', '../', "data", "trained_RNNs", task_name)
RNNs = []
for folder in os.listdir(RNNs_path):
    if (folder == '.DS_Store'):
        pass
    else:
        if "relu" in folder:
            RNNs.append(folder)

names = []
scores = []
Ns = []
lmbdos = []
lmbdrs = []
lrs = []
activations = []
tags = []
maxiters = []
for folder in RNNs:
    day = float(datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%d'))
    month = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%m')
    if (month == '05') and (day >=18):
        files = os.listdir(os.path.join(RNNs_path, folder))
        config_file = None
        for file in files:
            if "config" in file:
                config_file = file
        config_data = json.load(open(os.path.join(RNNs_path, folder, config_file), "rb+"))
        score = np.round(float(config_file.split("_")[0]), 7)
        activation = config_data["activation"]
        N = config_data["N"]
        lmbdo = config_data["lambda_orth"]
        lmbdr = config_data["lambda_r"]
        lr=config_data["lr"]
        maxiter=config_data["max_iter"]
        extra_info = f"{task_name};{activation};N={N};lmbdo={lmbdo};lmbdr={lmbdr};lr={lr};maxiter={maxiter}"
        name = f"{score}_{extra_info}"
        tag = config_data["tag"]
        names.append(name)
        scores.append(score)
        Ns.append(N)
        lmbdos.append(lmbdo)
        lmbdrs.append(lmbdr)
        lrs.append(lr)
        tags.append(tag)
        activations.append(activation)
        maxiters.append(maxiter)

df = pd.DataFrame({"name" : names, "scores" : scores, "N" : Ns, "activation": activations, "lmbdo" : lmbdos, "lmbdr": lmbdrs, "lr" : lrs, "maxiter" : maxiters})
# additional filtering
df = df[df['lr'] == 0.002]
df = df[df['maxiter'] == 3000]
pd.set_option('display.max_rows', None)
df.sort_values("scores")
top_RNNs = df.sort_values("scores")["name"].tolist()[:50]

for num_rnn in range(len(top_RNNs)):
    RNN_subfolder = top_RNNs[num_rnn]
    RNN_score = float(top_RNNs[num_rnn].split("_")[0])
    RNN_path = os.path.join(RNNs_path, RNN_subfolder)
    os.listdir(RNN_path)
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
    RNN_subfolder = top_RNNs[num_rnn]
    RNN_score = float(top_RNNs[num_rnn].split("_")[0])
    LA_data = pickle.load(open(os.path.join(RNN_path, f"{RNN_score}_LA_data.pkl"), "rb+"))
    LC_folder = RNN_subfolder
    RNN_score = float(RNN_subfolder.split("_")[0])

    LC_folder_path = os.path.join('../', '../', '../', "latent_circuit_inference", "data", "inferred_LCs",
                                  LC_folder)
    subfolders = os.listdir(LC_folder_path)
    varianses = []
    variances_pr = []
    names = []
    for i, subfolder in enumerate(subfolders):
        if "8nodes" in subfolder or "8-nodes" in subfolder:
            score = float(subfolder.split("_")[0])
            score_pr = float(subfolder.split("_")[1])
            varianses.append(score)
            variances_pr.append(score_pr)
            names.append(subfolder)
    lc_df = pd.DataFrame({"name": names, "variance": varianses, "variance_pr": variances_pr})
    top_LCs = lc_df.sort_values("variance", ascending=False)["name"].tolist()
    LC_subfolder = top_LCs[0]
    print(LC_subfolder)
    score = float(LC_subfolder.split("_")[0])
    score_pr = float(LC_subfolder.split("_")[1])
    LC_path = os.path.join(LC_folder_path, LC_subfolder)
    LC_data = json.load(open(os.path.join(LC_path, f"{score}_{score_pr}_LC_params.json"), "rb+"))
    U = np.array(LC_data["U"])
    q = np.array(LC_data["q"])
    Q = U.T @ q
    w_out = np.array(LC_data["W_out"])
    w_rec = np.array(LC_data["W_rec"])
    w_inp = np.array(LC_data["W_inp"])
    N = LC_data["N"]
    dt = LC_data["dt"]
    tau = LC_data["tau"]
    names = ["ctx m", "ctx c", "mr", "ml", "cr", "cl", "OutR", "OutL"]
    n = 8

    print(num_rnn)

    w_rec_bar = Q.T @ W_rec @ Q
    fig_w_rec_comparison, ax_w_rec_comparison = plt.subplots(1, 2, figsize=(6, 3))
    # fig_w_rec_comparison.colorbar(im)

    ax_w_rec_comparison[0].set_xticks(np.arange(n))
    ax_w_rec_comparison[0].set_xticklabels(names)
    ax_w_rec_comparison[0].set_yticks(np.arange(n))
    ax_w_rec_comparison[0].set_yticklabels(names)

    ax_w_rec_comparison[1].set_xticks(np.arange(n))
    ax_w_rec_comparison[1].set_xticklabels(names)
    # ax_w_rec_comparison[1].set_yticks(np.arange(n))
    # ax_w_rec_comparison[1].set_yticklabels(names)

    # Set ticks on both sides of axes on
    # ax_w_rec_comparison[0].tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_w_rec_comparison[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax_w_rec_comparison[0].xaxis.get_major_ticks()], rotation=45,
             ha="right", va="center", rotation_mode="anchor")
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax_w_rec_comparison[0].xaxis.get_major_ticks()], rotation=45,
             ha="left", va="center", rotation_mode="anchor")
    # ax_w_rec_comparison[0].set_title(r"$Q W_{rec} Q^T$", fontsize = 16, pad=10)
    im_w_rec_bar = ax_w_rec_comparison[0].imshow(w_rec_bar, interpolation='none', vmin=-np.max(np.abs(w_rec)),
                                                 vmax=np.max(np.abs(w_rec)), cmap=cmp)

    # Set ticks on both sides of axes on
    # ax_w_rec_comparison[1].tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_w_rec_comparison[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax_w_rec_comparison[1].xaxis.get_major_ticks()], rotation=45,
             ha="right", va="center", rotation_mode="anchor")
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax_w_rec_comparison[1].xaxis.get_major_ticks()], rotation=45,
             ha="left", va="center", rotation_mode="anchor")
    # ax_w_rec_comparison[1].set_title(r"$w_{rec}$", fontsize = 16, pad=10)
    ax_w_rec_comparison[1].set_yticks([])
    im_w_rec = ax_w_rec_comparison[1].imshow(w_rec, interpolation='none', vmin=-np.max(np.abs(w_rec)),
                                             vmax=np.max(np.abs(w_rec)), cmap=cmp)

    fig_w_rec_comparison.subplots_adjust(right=0.8)
    cbar_ax = fig_w_rec_comparison.add_axes([0.83, 0.2, 0.02, 0.6])
    cbar = fig_w_rec_comparison.colorbar(im_w_rec, cax=cbar_ax)
    # cbar.ax.set_ylabel('value', rotation=270, labelpad=10, fontsize=16)
    plt.subplots_adjust(wspace=0.05)
    # plt.savefig("../figures/w_rec_vs_W_rec.pdf", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()