from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import os
os.system('python ../../style/style_setup.py')
mm = 1.0/25.4

class PerformanceAnalyzer():
    '''
    Generic class for analysis of the RNN performance on the given task
    '''

    def __init__(self, rnn_numpy, task=None):
        self.RNN = rnn_numpy
        self.Task = task

    def get_validation_score(self, scoring_function,
                             input_batch, target_batch, mask,
                             sigma_rec=0, sigma_inp=0):
        n_inputs = input_batch.shape[0]
        n_steps = input_batch.shape[1]
        batch_size = input_batch.shape[2]
        self.RNN.clear_history()
        # self.RNN.y = np.repeat(deepcopy(self.RNN.y)[:, np.newaxis], axis=-1, repeats=batch_size)
        self.RNN.run(input_timeseries=input_batch,
                     sigma_rec=sigma_rec,
                     sigma_inp=sigma_inp)
        output_prediction = self.RNN.get_output()
        if mask is None:
            avg_score = np.mean(
                [scoring_function(output_prediction[:, :, i], target_batch[:, :, i]) for i in range(batch_size)])
        else:
            avg_score = np.mean(
                [scoring_function(output_prediction[:, mask, i], target_batch[:, mask, i]) for i in range(batch_size)])
        return avg_score

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
                               coh_bouds=(-1, 1)):
        coherence_lvls = np.linspace(coh_bouds[0], coh_bouds[1], num_levels)
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

        errors = np.sum(np.sum((target_batch[:, mask, ...] - output[:, mask, ...]) ** 2, axis=0), axis=0) / mask.shape[
            0]

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
                               show_axeslabels=True):
        coherence_lvls = self.psychometric_data["coherence_lvls"]

        # invert cause of the axes running from the bottom to the top
        Motion_rght_prcntg = self.psychometric_data["motion"]["right_choice_percentage"][::-1, :]
        Motion_MSE = self.psychometric_data["motion"]["MSE"]
        Color_rght_prcntg = self.psychometric_data["color"]["right_choice_percentage"][::-1, :]
        Color_MSE = self.psychometric_data["color"]["MSE"][::-1, :]
        num_lvls = Color_rght_prcntg.shape[0]

        n_rows = 2 if show_MSE_surface else 1

        fig, axes = plt.subplots(n_rows, 2, figsize=(n_rows * 80 * mm, 160 * mm))

        if n_rows == 1:
            axes = axes[np.newaxis, :]

        # the plots themselves:
        for i, ctxt in enumerate(["Motion", "Color"]):
            im1 = axes[0, i].imshow(eval(f"{ctxt}_rght_prcntg"), cmap="bwr", interpolation="bicubic")
            if show_MSE_surface:
                im2 = axes[1, i].imshow(eval(f"{ctxt}_MSE"), cmap="bwr", interpolation="bicubic")

        # axes labels:
        if show_axeslabels == False:
            for i, ctxt in enumerate(["Motion", "Color"]):
                for j in range(axes.shape[0]):
                    axes[j, i].set_xticks([], labels=[])
                    axes[j, i].set_yticks([], labels=[])

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
                            axes[j, i].set_xticks(np.arange(num_lvls),
                                                  labels=np.round(coherence_lvls, 2), rotation=50)
                            axes[j, i].set_xlabel("Motion coherence")
                        else:
                            axes[j, i].set_xticks(np.arange(num_lvls), labels=[])
                        if i == 0:
                            axes[j, i].set_yticks(np.arange(num_lvls),
                                                  labels=np.round(coherence_lvls, 2)[::-1])
                            axes[j, i].set_ylabel("Color coherence")
                        else:
                            axes[j, i].set_yticks(np.arange(num_lvls), labels=[])

        # fig.tight_layout()
        # plt.subplots_adjust(wspace=0.125, hspace=0.15)
        return fig
