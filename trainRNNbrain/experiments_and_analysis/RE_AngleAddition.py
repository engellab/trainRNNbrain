import colorsys
import itertools

from matplotlib.animation import FuncAnimation
from matplotlib import cm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.tasks.TaskReferenceFrame import TaskReferenceFrame
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from matplotlib import pyplot as plt
import numpy as np
import pickle
import json
import os
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask


def cluster_and_order_rows_with_filter(W_inp, low_norm_percentile=60, n_runs=500):
    # Step 1: Calculate the norm of each row
    row_norms = np.linalg.norm(W_inp, axis=1)

    # Step 2: Determine the threshold for low-norm rows (bottom 20%)
    low_norm_threshold = np.percentile(row_norms, low_norm_percentile)

    # Step 3: Separate rows into low-norm and high-norm groups
    low_norm_indices = np.where(row_norms <= low_norm_threshold)[0]
    high_norm_indices = np.where(row_norms > low_norm_threshold)[0]

    # Step 4: Cluster only the high-norm rows
    W_high_norm = W_inp[high_norm_indices]
    k = W_high_norm.shape[1]  # Number of clusters = number of columns

    if len(W_high_norm) > 0:  # Only cluster if there are high-norm rows
        best_clustering = None
        best_score = -1  # Initialize with the worst possible score

        # Try multiple runs of clustering and select the best one
        for _ in range(n_runs):
            # Use Agglomerative Clustering (or another algorithm)
            clustering = AgglomerativeClustering(n_clusters=k).fit(W_high_norm)
            labels = clustering.labels_

            # Calculate the silhouette score to evaluate the clustering
            if len(np.unique(labels)) > 1:  # Silhouette score requires at least 2 clusters
                score = silhouette_score(W_high_norm, labels)
                if score > best_score:
                    best_score = score
                    best_clustering = labels

        # If no valid clustering was found, fall back to K-means
        if best_clustering is None:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(W_high_norm)
            best_clustering = kmeans.labels_

        # Step 5: Group high-norm row indices by their cluster labels
        clusters = {}
        for idx, label in enumerate(best_clustering):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(high_norm_indices[idx])  # Map back to original indices

        # Step 6: Order the clusters based on proximity to canonical basis vectors
        canonical_basis = np.eye(k)
        cluster_means = {}
        for label, indices in clusters.items():
            cluster_means[label] = np.mean(W_inp[indices], axis=0)

        distances = {}
        for label, mean in cluster_means.items():
            distances[label] = [np.linalg.norm(mean - basis) for basis in canonical_basis]

        cluster_order = []
        for i in range(k):
            closest_label = min(distances.keys(), key=lambda x: distances[x][i])
            cluster_order.append(closest_label)
            del distances[closest_label]  # Remove the assigned cluster from further consideration

        # Step 7: Sort the clusters based on the order determined above
        sorted_clusters = []
        for label in cluster_order:
            sorted_clusters.append(clusters[label])
    else:
        sorted_clusters = []  # No high-norm rows to cluster

    # Step 8: Add the low-norm group as the last group
    if len(low_norm_indices) > 0:
        sorted_clusters.append(low_norm_indices.tolist())

    return sorted_clusters

network_name = '0.9953713_ReferenceFrame_relu;N=286;lmbdo=0.25;orth_inp_only=True;lmbdr=0.25;lr=0.005;maxiter=20000'
img_folder = "../../img/AngleAddition"
os.makedirs(img_folder, exist_ok=True)
networks_path = os.path.join('/Users/tolmach/Documents/GitHub/trainRNNbrain/data/trained_RNNs/ReferenceFrame_relu_constrained=True/ReferenceFrame_relu_constrained=True/')
coloring = "ego"


path = os.path.join(networks_path, network_name)
score = float(network_name.split('_')[0])
net_params = json.load(open(os.path.join(path, f"{score}_params_ReferenceFrame.json"), 'r'))
for key in ["W_inp", "W_rec", "W_out", "y_init"]:
    net_params[key] = np.array(net_params[key])
cfg = OmegaConf.load(open(os.path.join(path, f"{score}_config.yaml"), 'r'))
lambda_var = cfg.trainer.lambda_var
taskname = cfg.task.taskname

disp = cfg.display_figures

# defining the task
task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
# task_conf.batch_size = 12
task = hydra.utils.instantiate(task_conf)
mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)

W_inp = net_params["W_inp"]
W_rec = net_params["W_rec"]
W_out = net_params["W_out"]
groupings = cluster_and_order_rows_with_filter(W_inp)
permutation = list(itertools.chain.from_iterable(groupings))
W_inp = W_inp[permutation, :]
W_out = W_out[:, permutation]
W_rec = W_rec[:, permutation]
W_rec = W_rec[permutation, :]

# # W_inp = W_inp/np.linalg.norm(W_inp, axis = 0, keepdims=True)
# fig = plt.figure(figsize = (5, 5))
# plt.imshow(W_inp.T, vmin = -1, vmax = 1, cmap = 'bwr')
# plt.show()
#
# fig = plt.figure(figsize = (5, 5))
# plt.imshow(W_rec, vmin = -1, vmax = 1, cmap = 'bwr')
# plt.show()
#
# fig = plt.figure(figsize = (5, 5))
# plt.imshow(W_out, vmin = -1, vmax = 1, cmap = 'bwr')
# plt.show()


# validate
rnn = RNN_numpy(**net_params)
analyzer = PerformanceAnalyzer(rnn)
score_function = lambda x, y: np.mean((x - y) ** 2)
input_batch, target_batch, conditions_list = task.get_batch()

for i in range(12):
    inp, target, condition = task.generate_input_target_stream(theta_retina, theta_head=0)

rnn.reset_state()
rnn.clear_history()
rnn.run(input_batch)
trajectories = rnn.get_history()
trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)
inputs_flat = input_batch.reshape(input_batch.shape[0], -1)

print(network_name, f"lambda_var = {lambda_var}")

pca_tuning = PCA(n_components=5)
pca_tuning.fit(trajectories_flat)
P_tuning = pca_tuning.components_.T
tuning = trajectories_flat @ P_tuning
for projection in [(0, 1, 2)]:#, (0, 1, 3), (0, 2, 3), (1, 2, 3), (2, 3, 4)]:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*(tuning[:, p] for p in projection), color='r', s=30, edgecolor='k')
    plt.show()








