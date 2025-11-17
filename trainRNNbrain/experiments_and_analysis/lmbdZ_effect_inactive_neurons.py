import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.tasks.TaskCDDM import TaskCDDM
import os
import yaml
import json
import hydra
from trainRNNbrain.training.training_utils import prepare_task_arguments

OmegaConf.register_new_resolver("eval", eval)

path = '/Users/tolmach/Documents/GitHub/trainRNNbrain/data/VaryingLambdaZ/v3'
folders = [f for f in os.listdir(path) if not f.startswith('.')]
for folder in folders:
    if "lmbdZ=1.0" in folder:
        score = folder.split("_")[0]
        config_file = f'{score}_config.yaml'
        params_file = f'{score}_params_CDDM.json'
        config = OmegaConf.load(os.path.join(path, folder, config_file))
        print(config.trainer)
        N = int(folder.split(";")[1].split("=")[1])
        with open(os.path.join(path, folder, params_file), 'r') as f:
            params = json.load(f)

        for key in params.keys():
            if key in ['W_out', 'W_inp', 'W_rec', 'y_init', 'bias_rec']:
                params[key] = np.array(params[key])
        params["bias_rec"] = np.zeros(params["N"])


        # defining the task
        task_conf = prepare_task_arguments(cfg_task=config.task, dt=config.model.dt)
        task = hydra.utils.instantiate(task_conf)
        inputs, targets, conditions = task.get_batch()
        inputs = np.array(inputs)
        rnn = RNN_numpy(**params)
        rnn.run(inputs)
        trajectories = rnn.get_history()
        mean_activity = np.mean(np.abs(trajectories), axis=(1, 2))

        print(min(mean_activity))

        # # Create the histogram
        # fig, ax = plt.subplots()  # Get the figure and axes objects
        # ax.hist(mean_activity, bins=100, color='skyblue', edgecolor='black')
        #
        # # Turn off the top and right spines
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # # Add labels and title
        # ax.set_xlabel('Value')
        # ax.set_ylabel('Frequency')
        # ax.set_title(f'Histogram of mean neural activity, score={np.round(float(score), 2)}; N = {N}; lambda_z = {config.trainer.lambda_z}')
        # # Show the plot
        # plt.show()


        # fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        # ax.imshow(rnn.W_inp.T, vmin=-1, vmax=1, cmap='bwr')
        # plt.show()
        #
        #
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(rnn.W_rec, vmin=-1, vmax=1, cmap='bwr')
        # plt.show()
        #
        # fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        # ax.imshow(rnn.W_out, vmin=-1, vmax=1, cmap='bwr')
        # plt.show()



        #plot trajectories:
        #
        # # Flatten and apply PCA
        # F = trajectories.reshape(trajectories.shape[0], -1)
        # pca = PCA(n_components=3)
        # pca.fit_transform(F.T)  # Note: fitting on F.T
        # P = pca.components_.T
        # T = P.T @ F
        # T = T.reshape(-1, trajectories.shape[1], trajectories.shape[2])  # Shape: (3, T, batch_size)
        #
        # # Prepare figure and axis
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # # Hide default axes and ticks
        # # Plot all trajectories
        # for k in range(T.shape[-1]):
        #     ax.plot(T[0, :, k], T[1, :, k], T[2, :, k], color='r', alpha=0.1)
        #
        # # Keep only the last tick on each axis
        # ax.set_xticks([ax.get_xlim()[0], ax.get_xlim()[-1]])
        # ax.set_yticks([ax.get_ylim()[0], ax.get_ylim()[-1]])
        # ax.set_zticks([ax.get_zlim()[0], ax.get_zlim()[-1]])
        # # Remove the tick labels
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        #
        # # Set up animation update function
        # def update(frame):
        #     azim = frame
        #     elev = frame / 10 + frame / 60
        #     ax.view_init(elev=elev, azim=azim)
        #     fig.canvas.draw()
        #     return []
        #
        # # Create animation
        # num_frames = 360 * 2
        # ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames, 2), interval=1, blit=False)
        # plt.show()



        # Flatten and apply PCA
        F = trajectories.reshape(trajectories.shape[0], -1)
        pca = PCA(n_components=10)
        pca.fit_transform(F)
        P = pca.components_
        S = F @ P.T

        # Prepare the figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


        labels = None
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
        # Choose colors
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = [colors[i % len(colors)] for i in range(len(unique_labels))]
            point_colors = [colors[np.where(unique_labels == l)[0][0]] for l in labels]
        else:
            point_colors = ['r'] * S.shape[0]

        # Initial scatter plot
        scatter = ax.scatter(*(S[:, j] for j in (0, 1, 2)), c=point_colors, marker='o', s=40, edgecolor='k')

        # Keep only the last tick on each axis
        ax.set_xticks([ax.get_xlim()[0], ax.get_xlim()[-1]])
        ax.set_yticks([ax.get_ylim()[0], ax.get_ylim()[-1]])
        ax.set_zticks([ax.get_zlim()[0], ax.get_zlim()[-1]])
        # Remove the tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Set up animation update function
        def update(frame):
            azim = frame
            elev = frame / 10 + frame / 60
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw()
            return scatter,

        # Animation setup
        num_frames = 360 * 2
        ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames, 2), interval=50, blit=False)
        plt.show()
        # break





