import os
import numpy as np
import matplotlib.pyplot as plt
import json
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments
import yaml
import hydra
from trainRNNbrain.utils import composite_lexicographic_sort, permute_matrices
OmegaConf.register_new_resolver("eval", eval)


def plot_matrices(W_inp, W_rec, W_out):
    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(W_inp.T, cmap='bwr', vmin=-1, vmax=1)
    ax[1].imshow(W_rec, cmap='bwr', vmin=-1, vmax=1)
    ax[2].imshow(W_out, cmap='bwr', vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()
    return None

def plot_trajectories(trajectories):
    F = trajectories.reshape(trajectories.shape[0], -1)
    pca = PCA(n_components=3)
    pca.fit_transform(F.T)
    P = pca.components_.T

    T = P.T @ F
    T = T.reshape(-1, trajectories.shape[1], trajectories.shape[2])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for k in range(T.shape[-1]):
        ax.plot(T[0, :, k], T[1, :, k], T[2, :, k], color='r', alpha = 0.1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    return None

def plot_selectivity(trajectories, PCs=(1,2,3), labels=None):
    F = trajectories.reshape(trajectories.shape[0], -1)
    pca = PCA(n_components=10)
    pca.fit_transform(F)
    P = pca.components_.T
    D = F @ P

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if labels is None:
        ax.scatter(*(D[:, j] for j in PCs), s=50, color='r', edgecolor='k')
    for i, l in enumerate(np.unique(labels)):
        inds = np.where(labels == l)[0]
        ax.scatter(*(D[inds, j] for j in PCs), s=50, color=colors[i % len(colors)], edgecolor='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    return None

def get_averaged_responses(trajectories, labels, dale_mask):
    averaged_responses = np.zeros((len(np.unique(labels)), trajectories.shape[1], trajectories.shape[2]))
    new_dale_mask = np.zeros(len(np.unique(labels)))
    for i, lbl in enumerate(np.unique(labels)):
        inds = np.where(labels == lbl)[0]
        averaged_responses[i, ...] = np.mean(trajectories[inds, ...], axis=0)
        new_dale_mask[i] = np.sign(np.mean(dale_mask[inds]))
    return averaged_responses, new_dale_mask

def plot_averaged_responses(averaged_responses, average_dale_mask=None):
    n = averaged_responses.shape[0]
    nr = int(np.floor(np.sqrt(n)))
    nc = int(np.ceil(n/nr))
    fig, ax = plt.subplots(nr, nc)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            k = i * ax.shape[1] + j
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['top'].set_visible(False)
            if k >= averaged_responses.shape[0]:
                ax[i,j].imshow(np.zeros_like(averaged_responses[-1, ...].T), cmap ='bwr', vmin=-1, vmax=1)
            else:
                ax[i,j].imshow(average_dale_mask[k] * averaged_responses[k, ...].T, cmap ='bwr', vmin=-1, vmax=1)
            if i != nr-1:
                ax[i, j].set_xticklabels([])
            if j != 0:
                ax[i, j].set_yticklabels([])
    plt.tight_layout()
    plt.show()
    return None

if __name__ == '__main__':
    subfolder = "0.9696697_CDDM_relu;N=85;lmbdO=10.0;OrthInpOnly=True;lmbdR=0.5;lmbdC=0.0;lmbdInac=0.5;LR=0.01;MaxIter=1000"
    common_path = '/Users/tolmach/Documents/GitHub/trainRNNbrain/data/trained_RNNs/CDDM_relu_constrained=True'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
    score = subfolder.split('_')[0]
    file_path = os.path.join(common_path, subfolder, f"{score}_params_CDDM.json")
    config_path = os.path.join(common_path, subfolder, f"{score}_config.yaml")
    with open(file_path) as f:
        data = json.load(f)
    W_inp = np.array(data["W_inp"])
    W_rec = np.array(data["W_rec"])
    W_out = np.array(data["W_out"])
    perm = composite_lexicographic_sort(W_inp, W_out.T, np.sign(np.sum(W_rec, axis = 0)))
    W_inp_, W_rec_, W_out_ = permute_matrices(W_inp, W_rec, W_out, perm)

    dale_mask = np.sign(np.sum(W_rec_, axis = 0))
    plot_matrices(W_inp_, W_rec_, W_out_)
    rnn_params = {"N": data["N"], "dt": data["dt"], "tau": data["tau"],
                  "activation_name": data["activation_name"], "activation_slope": data["activation_slope"],
                  "W_inp": W_inp_, "W_rec": W_rec_, "W_out": W_out_}
    rnn = RNN_numpy(**rnn_params)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(config_dict)
    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_conf)
    inputs, outputs, conditions = task.get_batch()
    rnn.run(inputs)
    trajectories = rnn.get_history()

    plot_trajectories(trajectories)
    labels = cluster_neurons(trajectories, dale_mask, n_clusters=(8, 8))
    plot_selectivity(trajectories, PCs=(0, 1, 2), labels=labels)

    averaged_responses, avg_dale_mask = get_averaged_responses(trajectories, labels, dale_mask)
    plot_averaged_responses(averaged_responses, avg_dale_mask)

    # # now based on the clustering I want to compute aver
    # w_inp, w_rec, w_out = compute_intercluster_weights(W_inp_, W_rec_, W_out_, labels)
    # perm = composite_lexicographic_sort(w_inp, w_out.T, np.sign(np.sum(w_rec, axis=0)))
    # w_inp_, w_rec_, w_out_ = permute_matrices(w_inp, w_rec, w_out, perm)
    # fig, ax = plt.subplots(3, 1)
    # ax[0].imshow(w_inp_.T, cmap = 'bwr', vmin=-1, vmax=1)
    # ax[1].imshow(w_rec_, cmap = 'bwr', vmin=-1, vmax=1)
    # ax[2].imshow(w_out_, cmap = 'bwr', vmin=-1, vmax=1)
    # plt.show()


