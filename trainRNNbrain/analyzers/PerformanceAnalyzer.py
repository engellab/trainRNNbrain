from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class PerformanceAnalyzer():
    '''
    Generic class for analysis of the RNN performance on the given task
    '''

    def __init__(self, rnn_numpy, task=None):
        self.RNN = rnn_numpy
        self.Task = task
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                       '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                       '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']

    def get_validation_score(self, scoring_function,
                             input_batch, target_batch, mask,
                             sigma_rec=0, sigma_inp=0):
        batch_size = input_batch.shape[2]
        self.RNN.clear_history()
        self.RNN.run(input_timeseries=input_batch,
                     sigma_rec=sigma_rec,
                     sigma_inp=sigma_inp)
        output_prediction = self.RNN.get_output()

        if mask is None:
            mask = np.arange(output_prediction.shape[1])
        scores = [scoring_function(output_prediction[:, mask, i], target_batch[:, mask, i]) for i in range(batch_size)]
        scores = [v for v in scores if not math.isnan(v) and not math.isinf(v)]
        return np.mean(scores)

    def plot_trials(self, input_batch, target_batch, mask, sigma_rec=0.03, sigma_inp=0.03, labels=None, conditions=None):
        n_inputs = input_batch.shape[0]
        n_steps = input_batch.shape[1]
        batch_size = input_batch.shape[2]

        fig_output, axes = plt.subplots(batch_size, 1, figsize=(5, batch_size * 1))

        self.RNN.clear_history()
        self.RNN.y = deepcopy(self.RNN.y_init)
        self.RNN.run(input_timeseries=input_batch,
                    sigma_rec=sigma_rec,
                    sigma_inp=sigma_inp)
        predicted_output = self.RNN.get_output()
        colors = ["r", "blue", "g", "c", "m", "y", 'k']
        n_outputs = self.RNN.W_out.shape[0]
        for k in range(batch_size):
            if not (conditions is None):
                condition_str = ''.join([f"{key}: {conditions[k][key] if type(conditions[k][key]) == str else np.round(conditions[k][key], 3)}\n" for key in conditions[k].keys()])
                axes[k].text(s=condition_str, x=n_steps // 10, y=0.05, color='darkviolet')

            for i in range(n_outputs):
                tag = labels[i] if not (labels is None) else ''
                axes[k].plot(predicted_output[i, :, k], color=colors[i], linewidth=2, label=f'predicted {tag}')
                axes[k].plot(mask, target_batch[i, mask, k], color=colors[i], linewidth=2, linestyle='--', label=f'target {tag}')
            axes[k].set_ylim([-0.1, 1.2])
            axes[k].spines.right.set_visible(False)
            axes[k].spines.top.set_visible(False)
            if k != batch_size - 1:
                axes[k].set_xticks([])
        axes[0].legend(frameon=False, loc=(0.05, 1.1), ncol=2)
        axes[batch_size // 2].set_ylabel("Output")
        axes[-1].set_xlabel("time step, ms")
        fig_output.tight_layout()
        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        return fig_output

    def animate_trajectories(self, trajectories):
        # Flatten and apply PCA
        F = trajectories.reshape(trajectories.shape[0], -1)
        pca = PCA(n_components=3)
        pca.fit_transform(F.T)  # Note: fitting on F.T
        P = pca.components_.T
        T = P.T @ F
        T = T.reshape(-1, trajectories.shape[1], trajectories.shape[2])  # Shape: (3, T, batch_size)

        # Prepare figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # Hide default axes and ticks

        # Plot all trajectories
        for k in range(T.shape[-1]):
            ax.plot(T[0, :, k], T[1, :, k], T[2, :, k], color='r', alpha=0.1)

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
            return []

        # Create animation
        num_frames = 360 * 2
        ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames, 2), interval=1, blit=False)
        return ani

    def animate_selectivity(self, trajectories, axes=(0,1,2), labels=None):
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

        # Choose colors
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = [self.colors[i % len(self.colors)] for i in range(len(unique_labels))]
            point_colors = [colors[np.where(unique_labels == l)[0][0]] for l in labels]
        else:
            point_colors = ['r'] * S.shape[0]

        # Initial scatter plot
        scatter = ax.scatter(*(S[:, j] for j in axes), c=point_colors, marker='o', s=40, edgecolor='k')

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
        ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames, 2), interval=1, blit=False)
        return ani

    def plot_matrices(self):
        import matplotlib.pyplot as plt

        fig_matrices, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot input weights
        im0 = axes[0].imshow(self.RNN.W_inp.T, vmin=-1, vmax=1, cmap='bwr')
        axes[0].set_title("W_inp.T")

        # Plot recurrent weights
        im1 = axes[1].imshow(self.RNN.W_rec, vmin=-1, vmax=1, cmap='bwr')
        axes[1].set_title("W_rec")

        # Plot output weights
        im2 = axes[2].imshow(self.RNN.W_out, vmin=-1, vmax=1, cmap='bwr')
        axes[2].set_title("W_out")

        # Remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # Adjust layout first
        fig_matrices.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for colorbar

        # Add horizontal colorbar below all plots
        cbar_ax = fig_matrices.add_axes([0.25, 0.04, 0.5, 0.03])  # [left, bottom, width, height]
        fig_matrices.colorbar(im2, cax=cbar_ax, orientation='horizontal')

        return fig_matrices

    def composite_lexicographic_sort(self, matrix1, matrix2, dale_mask):
        """
        Sorts rows by:
        1. Lexicographic order of matrix1
        2. Then matrix2
        3. Then dale_mask (-1 before +1)
        Args:
            matrix1 (np.ndarray): shape (N, D1), one-hot rows
            matrix2 (np.ndarray): shape (N, D2), one-hot rows
            dale_mask (np.ndarray): shape (N,), values in {-1, +1}
        Returns:
            perm (np.ndarray): permutation of row indices
        """
        assert matrix1.shape[0] == matrix2.shape[0] == dale_mask.shape[0], "Mismatch in row count"
        idx1 = np.argmax(matrix1, axis=1)  # primary key
        idx2 = np.argmax(matrix2, axis=1)  # secondary key
        perm = np.lexsort((dale_mask, idx2, idx1))
        return perm

    def compute_intercluster_weights(self, W_inp, W_rec, W_out, labels):
        """
        Compute average inter-cluster connectivity matrices.

        Args:
            W_inp: (N, I)
            W_rec: (N, N)
            W_out: (O, N)
            labels: (N,) array of integers in [0, C-1]

        Returns:
            w_inp: (C, I)
            w_rec: (C, C)
            w_out: (O, C)
        """
        N = labels.shape[0]
        C = np.max(labels) + 1  # number of clusters

        w_inp = np.zeros((C, W_inp.shape[1]))
        w_rec = np.zeros((C, C))
        w_out = np.zeros((W_out.shape[0], C))

        for i in range(C):
            idx_i = np.where(labels == i)[0]
            if len(idx_i) == 0:
                continue
            # Average input into cluster i
            w_inp[i] = W_inp[idx_i].mean(axis=0)

            for j in range(C):
                idx_j = np.where(labels == j)[0]
                if len(idx_j) == 0:
                    continue
                # Average recurrent weight from cluster j to i
                submatrix = W_rec[np.ix_(idx_i, idx_j)]
                w_rec[i, j] = submatrix.mean()

        # Average output from each cluster
        for j in range(C):
            idx_j = np.where(labels == j)[0]
            if len(idx_j) == 0:
                continue
            w_out[:, j] = W_out[:, idx_j].mean(axis=1)

        return w_inp, w_rec, w_out

    def permute_matrices(self, W_inp, W_rec, W_out, dale_mask, perm):
        W_inp_ = W_inp[perm, :]
        W_rec_ = W_rec[perm, :]
        W_rec_ = W_rec_[:, perm]
        W_out_ = W_out[:, perm]
        dale_mask_ = dale_mask[perm]
        return W_inp_, W_rec_, W_out_, dale_mask_

    def cluster_neurons(self, trajectories, dale_mask=None, n_clusters=(8, 4)):
        if dale_mask is None:
            F = trajectories.reshape(trajectories.shape[0], -1)
            pca = PCA(n_components=10)
            pca.fit_transform(F)
            P = pca.components_.T
            D = F @ P
            cl = KMeans(n_clusters=n_clusters)
            cl.fit(D)
            labels = cl.labels_
        else:
            idx_pos = np.where(dale_mask==True)[0]
            idx_neg = np.where(dale_mask==False)[0]
            trajectories_pos = trajectories[idx_pos, ...]
            trajectories_neg = trajectories[idx_neg, ...]
            labels_pos = self.cluster_neurons(trajectories_pos, dale_mask=None, n_clusters=n_clusters[0])
            labels_neg = self.cluster_neurons(trajectories_neg, dale_mask=None, n_clusters=n_clusters[1])

            labels = np.zeros(dale_mask.shape)
            labels_neg = len(np.unique(labels_pos)) + np.array(labels_neg)
            labels[idx_pos] = labels_pos
            labels[idx_neg] = labels_neg
        return labels


    def get_averaged_responses(self, trajectories, dale_mask, labels):
        averaged_responses = np.zeros((len(np.unique(labels)), trajectories.shape[1], trajectories.shape[2]))
        new_dale_mask = np.zeros(len(np.unique(labels)))

        for lbl in np.unique(labels):
            inds = np.where(labels == lbl)[0]
            i = int(lbl)
            averaged_responses[i, ...] = np.mean(trajectories[inds, ...], axis=0)
            new_dale_mask[i] = (np.mean(dale_mask[inds]) >= 0.5)
        return averaged_responses, new_dale_mask.astype(bool)

    def plot_averaged_responses(self, averaged_responses, average_dale_mask=None, show=True, save=False, name=None):
        n = averaged_responses.shape[0]
        nr = int(np.floor(np.sqrt(n)))
        nc = int(np.ceil(n / nr))
        fig, ax = plt.subplots(nr, nc, figsize=(nc * 2, nr * 2))
        ax = np.atleast_2d(ax)

        for i in range(nr):
            for j in range(nc):
                k = i * nc + j
                # print(f"Plotting responses of cluster {k}")
                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['top'].set_visible(False)

                if k >= n:
                    ax[i, j].imshow(np.zeros_like(averaged_responses[-1].T), cmap='bwr', vmin=-1, vmax=1)
                else:
                    if average_dale_mask is not None:
                        s = 2 * (average_dale_mask[k].astype(float) - 0.5)
                    else:
                        s = 1.0
                    ax[i, j].imshow(s * averaged_responses[k].T, cmap='bwr', vmin=-1, vmax=1)

                if i != nr - 1:
                    ax[i, j].set_xticklabels([])
                if j != 0:
                    ax[i, j].set_yticklabels([])

        plt.tight_layout()
        return fig


    def get_trajectories(self, inputs):
        self.RNN.clear_history()
        self.RNN.y = self.RNN.y_init
        self.RNN.run(input_timeseries=inputs, sigma_rec=0, sigma_inp=0)
        trajectories = self.RNN.get_history()
        outputs = self.RNN.get_output()
        return trajectories, outputs


class PerformanceAnalyzerCDDM(PerformanceAnalyzer):
    def __init__(self, rnn_numpy):
        PerformanceAnalyzer.__init__(self, rnn_numpy)

    def calc_psychometric_data(self,
                               task,
                               mask,
                               num_levels=7,
                               num_repeats=7,
                               sigma_rec=0.03,
                               sigma_inp=0.03,
                               coh_bounds=(-1, 1)):
        coherence_lvls = np.linspace(coh_bounds[0], coh_bounds[1], num_levels)
        psychometric_data = {}
        psychometric_data["coherence_lvls"] = coherence_lvls
        psychometric_data["motion"] = {}
        psychometric_data["color"] = {}
        psychometric_data["motion"]["right_choice_percentage"] = np.empty((num_levels, num_levels))
        psychometric_data["color"]["right_choice_percentage"] = np.empty((num_levels, num_levels))
        psychometric_data["motion"]["MSE"] = np.empty((num_levels, num_levels))
        psychometric_data["color"]["MSE"] = np.empty((num_levels, num_levels))

        task.coherences = coherence_lvls
        input_batch, target_batch, conditions = task.get_batch()
        batch_size = input_batch.shape[-1]
        input_batch = np.repeat(input_batch, axis=-1, repeats=num_repeats)
        target_batch = np.repeat(target_batch, axis=-1, repeats=num_repeats)
        self.RNN.clear_history()
        self.RNN.y = deepcopy(self.RNN.y_init)
        self.RNN.run(input_timeseries=input_batch,
                     sigma_rec=sigma_rec,
                     sigma_inp=sigma_inp,
                     save_history=True)
        output = self.RNN.get_output()
        out_dim = output.shape[0]
        output = output.reshape((*output.shape[:-1], 2, num_levels, num_levels, num_repeats))
        target_batch = target_batch.reshape((*target_batch.shape[:-1], 2, num_levels, num_levels, num_repeats))
        if out_dim == 1:
            choices = np.sign(output[-1, :])
        else:
            choices = np.sign(output[0, -1, ...] - output[1, -1, ...])

        errors = np.sum(np.sum((target_batch[:, mask, ...] - output[:, mask, ...]) ** 2, axis=0), axis=0) / mask.shape[0]

        choices_to_right = (choices + 1) / 2
        # This reshaping pattern relies on the batch-structure from the CDDM task.
        # If you mess up with a batch generation function it may affect the psychometric function
        mean_choices_to_right = np.mean(choices_to_right, axis=-1)
        mean_error = np.mean(errors, axis=-1)
        # the color coh is the first dim initianlly, hence needs to transpose
        psychometric_data["motion"]["right_choice_percentage"] = mean_choices_to_right[0, ...].T
        psychometric_data["motion"]["MSE"] = mean_error[0, ...].T
        # the color coh is the second dim initianlly, hence No needs to transpose
        psychometric_data["color"]["right_choice_percentage"] = mean_choices_to_right[1, ...]
        psychometric_data["color"]["MSE"] = mean_error[1, ...]
        self.psychometric_data = deepcopy(psychometric_data)
        return psychometric_data

    def plot_psychometric_data(self,
                               show_MSE_surface=True,
                               show_colorbar=False,
                               show_axeslabels=True, cmap='bwr'):
        coherence_lvls = self.psychometric_data["coherence_lvls"]

        # invert cause of the axes running from the bottom to the top
        Motion_rght_prcntg = self.psychometric_data["motion"]["right_choice_percentage"][::-1, :]
        Motion_MSE = self.psychometric_data["motion"]["MSE"]
        Color_rght_prcntg = self.psychometric_data["color"]["right_choice_percentage"][::-1, :]
        Color_MSE = self.psychometric_data["color"]["MSE"][::-1, :]
        num_lvls = Color_rght_prcntg.shape[0]

        n_rows = 2 if show_MSE_surface else 1

        fig, axes = plt.subplots(n_rows, 2, figsize=(80 * mm, n_rows * 80 * mm))

        if n_rows == 1:
            axes = axes[np.newaxis, :]

        # the plots themselves:
        for i, ctxt in enumerate(["Motion", "Color"]):
            im1 = axes[0, i].imshow(eval(f"{ctxt}_rght_prcntg"), cmap=cmap, interpolation="bicubic")
            if show_MSE_surface:
                im2 = axes[1, i].imshow(eval(f"{ctxt}_MSE"), cmap=cmap, interpolation="bicubic")

        # axes labels:
        if show_axeslabels == False:
            for i, ctxt in enumerate(["Motion", "Color"]):
                for j in range(axes.shape[0]):
                    if len(coherence_lvls) % 2 == 0:
                       tick_positions = [0, len(coherence_lvls)//2, len(coherence_lvls)//2 +1, len(coherence_lvls)-1]
                    else:
                        tick_positions = [0, len(coherence_lvls) // 2, len(coherence_lvls)-1]
                    axes[j, i].set_xticks(tick_positions)
                    axes[j, i].set_xticklabels([])
                    axes[j, i].set_yticks(tick_positions)
                    axes[j, i].set_yticklabels([])

        if show_axeslabels:
            fig.suptitle("Psychometric data")
            for i, ctxt in enumerate(["Motion", "Color"]):
                axes[0, i].title.set_text(f"{ctxt}, % right")

                if show_colorbar:
                    plt.colorbar(im1, ax=axes[0, i], orientation='vertical')
                if show_MSE_surface:
                    axes[1, i].title.set_text(f"{ctxt}, MSE surface")
                    if show_colorbar:
                        plt.colorbar(im2, ax=axes[1, i], orientation='vertical')

                for i, ctxt in enumerate(["Motion", "Color"]):
                    for j in range(axes.shape[0]):
                        if j == axes.shape[0] - 1:
                            axes[j, i].set_xticks(np.arange(num_lvls))
                            axes[j, i].set_xticklabels(np.round(coherence_lvls, 2))
                            axes[j, i].set_xlabel("Motion coherence")
                        else:
                            axes[j, i].set_xticks(np.arange(num_lvls))
                            axes[j, i].set_xticklabels([])
                        if i == 0:
                            axes[j, i].set_yticks(np.arange(num_lvls))
                            axes[j, i].set_yticklabels(labels=np.round(coherence_lvls, 2)[::-1])
                            axes[j, i].set_ylabel("Color coherence")
                        else:
                            axes[j, i].set_yticks(np.arange(num_lvls))
                            axes[j, i].set_yticklabels([])

        # fig.tight_layout()
        # plt.subplots_adjust(wspace=0.125, hspace=0.15)
        return fig
