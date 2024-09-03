'''
plotting neurons from multiple rnns as points in the PC space
'''

import os
import pickle
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../")
from trainRNNbrain.tasks.Task import *
np.set_printoptions(suppress = True)
import numpy as np
import hdbscan
import plotly.graph_objects as go

data_dict = pickle.load(open(os.path.join('../', '../', '../', "trainRNNbrain", "data", "many_neurons_>05182023_exc.pkl"), "rb+"))
feature_mat_reduced = data_dict["features"]
RNN_labels = data_dict["RNN_labels"]
true_indices = data_dict["indices"]

clusterer = hdbscan.HDBSCAN(min_cluster_size=35, gen_min_span_tree=True, metric='l2')
clusterer.fit(feature_mat_reduced)
numerical_labels = clusterer.labels_

# # compute centroids of 12 clusters:
label_sorted_features = dict()
label_numbers = np.unique(np.array(numerical_labels))
label_numbers = label_numbers[1:] #skip the -1
for lbl_number in label_numbers:
    label_sorted_features[lbl_number] = []
for i, lbl_number in enumerate(numerical_labels):
    if lbl_number == -1:
        pass
    else:
        label_sorted_features[lbl_number].append(feature_mat_reduced[i, :])
centorids = dict()
for lbl_number in label_numbers:
    centorids[lbl_number] = np.mean(np.array(label_sorted_features[lbl_number]), axis = 0)

mR_coords = [-2.6, 2.4, -2.5, -2.4, 2.3, -1.3, 1.3, 0.15, 0.4, 0.3]
mL_coords = [2.9, 2.0, -2.4, 2.4, 2.3, -1.3, 1.3, 0.15, -0.5, -0.4]
cR_coords = [-2.6, -2.4, -2, 2, 2, 1.5, 0.9, -0.1, 0.15, 0]
cL_coords = [2.9, -2.4, -2.2, -2, 2.1, 1.5, 0.9, -0.15, -0.23, 0.24]
mRaux_coords = [-2.3, 4.2, -0.1, -2.6, 4, 1.2, 1.6, -0.2, -0.3, -0.4]
mLaux_coords = [2.6, 4.0, -0.1, 2.5, 3.7, 0.7, 1.4, -0.4, 0.4, 0.6]
cRaux_coords = [-2.4, -4.2, 0.3, 2.7, 4.2, -1.1, 1.7, 0.0, -0.4, 0.7]
cLaux_coords = [2.5, -4, -0.2, -2.6, 3.6, -0.4, 1.5, 0.0, 0.4, -0.6]
context_init_coords = [0, 0, 6.6, -0.3, 6.2, -0, 1.7, -0.3, 0, 0]
unsuppressed_coords = [0.0, -0.3, -6.5, 0.4, 7, 0.1, -0.15, -0.1, -0.1, 0]
OutR_coords = [-5, -0.15, -4, 0, 4, 0.2, 2.3, -0.1, -0.3, 0.4]
OutL_coords = [5, -0.15, -4, 0.2, 4, 0.2, 3.0, -0.2, 0, -0.2]
new_labels = ["mR", "mL", "cR", "cL", "mRaux", "mLaux", "cRaux", "cLaux", "context_init", "unsuppressed", "OutR", "OutL"]
approx_centroids = np.vstack([np.array(eval(f'{new_label}_coords')) for new_label in new_labels])

new_labels_sorted = []
for lbl_name in label_numbers:
    if lbl_name == -1:
        pass
    else:
        centroid = centorids[lbl_name]
        distances = np.mean((approx_centroids - np.vstack([centroid] * 12))**2, axis =1)
        new_labels_sorted.append(new_labels[np.argmin(distances)])

cluster_dict = {}
for i, label in enumerate(new_labels_sorted):
# for i, label in enumerate(numerical_labels):
    cluster_dict[label] = dict()
    cluster_dict[label]["cluster_num"] = label_numbers[i]
    cluster_dict[label]["centroid"] = centorids[label_numbers[i]]
    # for each RNN, put the indices such that these neurons belong to a cluster
    for j in np.unique(RNN_labels):
        cluster_dict[label][f"RNN{int(j)}"] = dict()
        nrn_inds_of_RNN = np.take(true_indices, np.where(RNN_labels == j)[0])
        num_lbls_of_nrns_of_RNN = np.take(numerical_labels, np.where(RNN_labels == j)[0])
        cluster_indices = np.take(nrn_inds_of_RNN, np.where(num_lbls_of_nrns_of_RNN == label_numbers[i])[0])
        cluster_dict[label][f"RNN{int(j)}"] = deepcopy(cluster_indices)
pickle.dump(cluster_dict, open(os.path.join('../', '../', '../', "trainRNNbrain", "data", "neurons_clustered.pkl"), "wb+"))


    # %matplotlib notebook
import pandas as pd
colors = ["plum", "powderblue", "purple", "red", "rosybrown",
            "royalblue", "rebeccapurple", "saddlebrown", "salmon",
            "sandybrown", "seagreen", "seashell", "sienna", "silver",
            "skyblue", "slateblue", "slategrey", "snow",
            "springgreen", "steelblue", "tan", "teal", "thistle", "tomato",
            "turquoise", "violet", "wheat", "white", "whitesmoke",
            "yellow", "yellowgreen"]

# fig.show()
# fig = px.scatter()
dct = {"xs": [], "ys" : [], "zs" : [], "cs" : [], 'features' : [], "cluster" : [], "color" : []}

for i, lbl in enumerate(numerical_labels):
    dct["xs"].append(feature_mat_reduced[i, 0])
    dct["ys"].append(feature_mat_reduced[i, 1])
    dct["zs"].append(feature_mat_reduced[i, 2])
    dct["cs"].append(feature_mat_reduced[i, 3])
    dct["features"].append(np.round(feature_mat_reduced[i, :], 2))
    dct["cluster"].append(lbl)
    dct["color"].append(colors[lbl])
dct["true_indices"] = true_indices
dct["RNN_label"] = RNN_labels
df = pd.DataFrame(dct)

for proj in ['xy', 'xz', 'yz']:
    match proj:
        case 'xy': x=dct["xs"]; y = dct["ys"]
        case 'xz': x = dct["xs"]; y = dct["zs"]
        case 'yz': x = dct["ys"]; y = dct["zs"]
    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        text=[f"Cluster: {new_labels_sorted[cluster]}<br>RNN: {RNN_label}<br>Features: {features}" for cluster, RNN_label, features in zip(dct['cluster'], dct['RNN_label'], dct['features'])],
        marker=dict(
            color=dct["color"],
            size=7,
            line=dict(width=0.25, color='black')
        )
    )

    inds = np.where(dct["RNN_label"] == 2)[0]
    trace_RNN0 = go.Scatter(
        x=np.take(x, inds),
        y=np.take(y, inds),
        mode='markers',
        text=[f"index: {true_indices[i]}<br>Cluster: {cluster}<br>RNN: {RNN_label}" for i, cluster, RNN_label in zip(inds, dct['cluster'], dct['RNN_label'])],
        marker=dict(
            color='white',
            size=15,
            line=dict(width=0.25, color='black')
        )
    )

    trace_particular_neuron = go.Scatter(
        x=np.take(x, [16]),
        y=np.take(y, [16]),
        mode='markers',
        marker=dict(
            color='black',
            size=15,
            line=dict(width=0.25, color='black')
        )
    )

    layout = go.Layout(
        title='Scatter Plot with Hover Information',
        xaxis=dict(title='X Axis'),
        yaxis=dict(title='Y Axis'),
        hovermode='closest'
    )

    # Create figure and add trace and layout
    # fig = go.Figure(data=[trace, trace_RNN0, trace_particular_neuron], layout=layout)
    fig = go.Figure(data=[trace], layout=layout)
    # Show plot
    fig.show()


        # colors = ['r', 'lightgreen', 'lightblue', 'yellow', 'orange', 'magenta', 'cyan', 'pink', 'k']
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for i, lbl in enumerate(lbls):
        #     xs = feature_mat_reduced[i, 0]
        #     ys = feature_mat_reduced[i, 1]
        #     zs = feature_mat_reduced[i, 2]
        #     cs = feature_mat_reduced[i, 3]
        #     ax.scatter(xs, ys, zs, color='r', s=10, edgecolor='k')
        # plt.show()
