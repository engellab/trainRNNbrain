import os
import pickle
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../")
from trainRNNbrain.tasks.Task import *
from sklearn.decomposition import PCA

np.set_printoptions(suppress = True)
import numpy as np

np.set_printoptions(suppress=True)
import json
import torch
from trainRNNbrain.rnn_coach.Task import TaskCDDM
from trainRNNbrain.rnn_coach.RNN_torch import RNN_torch
from copy import deepcopy
import pandas as pd
import datetime
from tqdm.auto import tqdm


task_name = "CDDM"
RNNs_path = os.path.join('../', '../', '../', "trainRNNbrain", "data", "trained_RNNs", task_name)
RNNs = []
for folder in os.listdir(os.path.join('../', '../', "data", "trained_RNNs", task_name)):
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
dates = []
for folder in RNNs:
    day = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%d')
    month = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%m')
    year = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%Y')
    if (float(month) >= 5) and (float(day) >=19):
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
        dates.append(f"{day}/{month}/{year}")
df = pd.DataFrame({"name" : names, "scores" : scores, "N" : Ns, "date": dates, "activation": activations, "lmbdo" : lmbdos, "lmbdr": lmbdrs, "lr" : lrs, "maxiter" : maxiters})
# additional filtering
# df = df[df['lr'] == 0.005]
# df = df[df['maxiter'] == 3000]

pd.set_option('display.max_rows', None)
df.sort_values("scores")
top_RNNs = df.sort_values("scores")["name"].tolist()
feature_mats = []
RNN_labels = []
true_indices = []
for num_rnn in tqdm(range(len(top_RNNs))):
    RNN_subfolder = top_RNNs[num_rnn]
    RNN_score = float(top_RNNs[num_rnn].split("_")[0])
    RNN_path = os.path.join('../', '../', '../', "trainRNNbrain", "data", "trained_RNNs", task_name, RNN_subfolder)

    RNN_data = json.load(open(os.path.join(RNN_path, f"{RNN_score}_params_{task_name}.json"), "rb+"))
    RNN_config_file = json.load(open(os.path.join(RNN_path, f"{RNN_score}_config.json"), "rb+"))

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
        case 'relu': activation_RNN = lambda x: torch.maximum(x, torch.tensor(0))
        case 'tanh': activation_RNN = torch.tanh
        case 'sigmoid': activation_RNN = lambda x: 1 / (1 + torch.exp(-x))
        case 'softplus': lambda x: torch.log(1 + torch.exp(5 * x))

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

    task_params["coherences"] = [-0.8, -0.4, -0.2, 0, 0.2, 0.4, 0.8]
    task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
    input_batch, target_batch, conditions_batch = task.get_batch()
    n_trials = len(conditions_batch)

    RNN = RNN_torch(N=W_rec.shape[0], dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                    activation=activation_RNN, random_generator=rng, device=device,
                    sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    RNN_params = {"W_inp": W_inp,
                  "W_rec": W_rec,
                  "W_out": W_out,
                  "b_rec": None,
                  "y_init": np.zeros(W_rec.shape[0])}
    RNN.set_params(RNN_params)

    RNN.sigma_rec = RNN.sigma_inp = torch.tensor(0, device=RNN.device)
    y, predicted_output_rnn = RNN(torch.from_numpy(input_batch.astype("float32")))
    Y = np.hstack([y.detach().numpy()[:, :, i] for i in range(y.shape[-1])])
    traces = y.detach().numpy()
    Y_mean = np.mean(np.abs(Y), axis=1)
    th = np.percentile(Y_mean, 20)
    inds_fr = np.where(Y_mean > th)[0]

    inds_fr = np.where(np.mean(W_rec, axis=0) > 0)[0]
    traces = traces[inds_fr, ...]

    traces_z = deepcopy(traces)
    for i in range(traces.shape[0]):
        for j in range(traces.shape[2]):
            mean = np.mean(traces[i, :, j])
            if mean == 0.0:
                tr = 0 * np.ones(traces_z.shape[1])
            else:
                tr = (traces[i, :, j] - np.min(traces[i, :, j])) / (np.max(traces[i, :, j]) - np.min(traces[i, :, j]))
            traces_z[i, :, j] = deepcopy(tr)

    n_t = traces.shape[1]
    W_rec_reduced = W_rec[:, inds_fr]
    W_rec_reduced = W_rec_reduced[inds_fr, :]
    features1 = np.mean(traces_z[:, :n_t // 3, :], axis=1)
    features2 = np.mean(traces_z[:, 2 * (n_t // 3):, :], axis=1)
    features3 = np.mean(traces[:, :, :], axis=1)
    features4 = np.mean(W_rec_reduced, axis=0).reshape(-1, 1)
    features5 = np.mean(W_rec_reduced, axis=1).reshape(-1, 1)
    feature_mat = np.hstack([eval(f"features{i}") for i in range(1, 5)])
    feature_mats.append(feature_mat)
    RNN_labels.append(num_rnn * np.ones(feature_mat.shape[0]))
    true_indices.extend(inds_fr)

feature_mat = np.vstack(feature_mats)
RNN_labels = np.hstack(RNN_labels)

pca = PCA(n_components=10)
pca.fit(feature_mat)
print(pca.explained_variance_ratio_)
feature_mat_reduced = feature_mat @ pca.components_.T
data_dict = {"features" : feature_mat_reduced, "RNN_labels" : RNN_labels, "indices" : true_indices}
pickle.dump(data_dict, open(os.path.join('../', '../', '../', "trainRNNbrain", "data", "clustering_data", "many_neurons_>05182023_exc.pkl"), "wb+"))
#
# clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True, metric='manhattan')
# clusterer.fit(feature_mat_reduced)
# lbls = clusterer.labels_
#
# # %matplotlib notebook
# import pandas as pd
# colors = ['r', 'lightgreen', 'lightblue', 'yellow', 'orange', 'magenta', 'cyan', 'pink', 'k', 'gray', 'salmon', 'tomato', 'brown']
# fig = px.scatter()
# dct = {"xs": [], "ys" : [], "zs" : [], "cluster" : [], "color" : []}
# # dct = {"xs": [], "ys" : [], "zs" : [], "cluster" : []}
# for i, lbl in enumerate(lbls):
#     dct["xs"].append(feature_mat_reduced[i, 0])
#     dct["ys"].append(feature_mat_reduced[i, 1])
#     dct["zs"].append(feature_mat_reduced[i, 2])
#     dct["cluster"].append(lbl)
#     dct["color"].append('lightblue')
# df = pd.DataFrame(dct)
#
# fig = px.scatter(df, x="xs", y='ys', hover_data=["cluster"])
# fig.update_layout(
#     hoverlabel={
#         "bgcolor":"white",
#         "font_size": 16,
#         "font_family":"Rockwell"}
# )
# fig.show()