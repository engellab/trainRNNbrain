from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt


class PerformanceAnalyzer():
    '''
    Generic class for analysis of the RNN performance on the given task
    '''

    def __init__(self, rnn_numpy, task=None):
        self.RNN = rnn_numpy
        self.Task = task

    def get_validation_score(self, scoring_function,
                             input_batch, target_batch, mask,
                             sigma_rec=0.03, sigma_inp=0.03,
                             rng_numpy=None):
        n_inputs = input_batch.shape[0]
        n_steps = input_batch.shape[1]
        batch_size = input_batch.shape[2]
        self.RNN.clear_history()
        # self.RNN.y = np.repeat(deepcopy(self.RNN.y)[:, np.newaxis], axis=-1, repeats=batch_size)
        trajectories, output_prediction = self.RNN.run_multiple_trajectories(input_timeseries=input_batch,
                                                                             sigma_rec=sigma_rec,
                                                                             sigma_inp=sigma_inp,
                                                                             generator_numpy=rng_numpy)
        avg_score = np.mean(
            [scoring_function(output_prediction[:, mask, i], target_batch[:, mask, i]) for i in range(batch_size)])
        return avg_score

    def plot_trials(self, input_batch, target_batch, mask, sigma_rec=0.03, sigma_inp=0.03, labels=None, rng_numpy=None):
        n_inputs = input_batch.shape[0]
        n_steps = input_batch.shape[1]
        batch_size = input_batch.shape[2]
        fig_output, axes = plt.subplots(batch_size, 1, figsize=(7, 8))
        self.RNN.clear_history()
        self.RNN.y = deepcopy(self.RNN.y_init)
        self.RNN.run_multiple_trajectories(input_timeseries=input_batch,
                                           sigma_rec=sigma_rec,
                                           sigma_inp=sigma_inp,
                                           generator_numpy=rng_numpy)
        predicted_output = self.RNN.get_output()
        colors = ["r", "b", "g", "c", "m", "y", 'k']
        n_outputs = self.RNN.W_out.shape[0]
        for k in range(batch_size):
            for i in range(n_outputs):
                tag = labels[i] if not (labels is None) else ''
                axes[k].plot(predicted_output[i, :, k], color=colors[i], label=f'predicted {tag}')
                axes[k].plot(mask, target_batch[i, mask, k], color=colors[i], linestyle='--', label=f'target {tag}')
            axes[k].spines.right.set_visible(False)
            axes[k].spines.top.set_visible(False)

        axes[0].legend(fontsize=12, frameon=False, bbox_to_anchor=(1.0, 1.0))
        axes[batch_size // 2].set_ylabel("Output", fontsize=12)
        axes[-1].set_xlabel("time step, ms", fontsize=12)
        fig_output.tight_layout()
        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        return fig_output


class PerformanceAnalyzerCDDM(PerformanceAnalyzer):
    def __init__(self, rnn_numpy):
        PerformanceAnalyzer.__init__(self, rnn_numpy)

    def calc_psychometric_data(self, task, mask, num_levels=7, num_repeats=7, sigma_rec=0.03, sigma_inp=0.03):
        coherence_lvls = np.linspace(-1, 1, num_levels)
        psychometric_data = {}
        psychometric_data["coherence_lvls"] = coherence_lvls
        psychometric_data["motion"] = {}
        psychometric_data["color"] = {}
        psychometric_data["motion"]["right_choice_percentage"] = np.empty((num_levels, num_levels))
        psychometric_data["color"]["right_choice_percentage"] = np.empty((num_levels, num_levels))
        psychometric_data["motion"]["MSE"] = np.empty((num_levels, num_levels))
        psychometric_data["color"]["MSE"] = np.empty((num_levels, num_levels))

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
        if output.shape[0] == 2:
            choices = np.sign(output[0, -1, :] - output[1, -1, :])
        elif output.shape[0] == 1:
            choices = np.sign(output[0, -1, :])
        errors = np.sum(np.sum((target_batch[:, mask, :] - output[:, mask, :]) ** 2, axis=0), axis=0) / mask.shape[0]

        choices_to_right = (choices + 1) / 2
        # This reshaping pattern relies on the batch-structure from the CDDM task.
        # If you mess up with a batch generation function it may affect the psychometric function
        mean_choices_to_right = np.mean(choices_to_right.reshape(2, num_levels, num_levels, num_repeats), axis=-1)
        mean_error = np.mean(errors.reshape(2, num_levels, num_levels, num_repeats), axis=-1)
        psychometric_data["motion"]["right_choice_percentage"] = mean_choices_to_right[0, ...].T
        psychometric_data["color"]["right_choice_percentage"] = mean_choices_to_right[1, ...]
        psychometric_data["motion"]["MSE"] = mean_error[0, ...].T
        psychometric_data["color"]["MSE"] = mean_error[1, ...]
        self.psychometric_data = deepcopy(psychometric_data)
        return psychometric_data

    def plot_psychometric_data(self):
        coherence_lvls = self.psychometric_data["coherence_lvls"]
        Motion_rght_prcntg = self.psychometric_data["motion"]["right_choice_percentage"]
        Color_rght_prcntg = self.psychometric_data["color"]["right_choice_percentage"][::-1, :]
        Motion_MSE = self.psychometric_data["motion"]["MSE"]
        Color_MSE = self.psychometric_data["color"]["MSE"][::-1, :]
        num_lvls = Color_rght_prcntg.shape[0]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Psychometric data", fontsize=14)
        tag = ["Motion", "Color"]
        for i in range(2):
            axes[0, i].title.set_text(f"{tag[i]}, % right")
            im1 = axes[0, i].imshow(eval(f"{tag[i]}_rght_prcntg"), cmap="coolwarm", interpolation="bicubic")
            plt.colorbar(im1, ax=axes[0, i], orientation='vertical')

            axes[1, i].title.set_text(f"{tag[i]}, MSE surface")
            im2 = axes[1, i].imshow(eval(f"{tag[i]}_MSE"), cmap="coolwarm", interpolation="bicubic")
            plt.colorbar(im2, ax=axes[1, i], orientation='vertical')

            for j in range(2):
                axes[j, i].set_xticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2), rotation=50)
                axes[j, i].set_yticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2)[::-1])
        axes[0, 0].set_xticklabels([])
        axes[0, 1].set_xticklabels([])
        axes[0, 0].set_ylabel("Coherence of color", fontsize=16)
        axes[1, 0].set_ylabel("Coherence of color", fontsize=16)
        axes[1, 0].set_xlabel("Coherence of motion", fontsize=16)
        axes[1, 1].set_xlabel("Coherence of motion", fontsize=16)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.125, hspace=0.15)
        return fig
