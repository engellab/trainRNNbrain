import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../')
from src.connectivity import get_small_connectivity_np
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy
from src.generate_CDDM_trials import generate_input_target_stream


def get_psychometric_data(RNN, n_steps, protocol_dict, mask, num_lvls=16,
                          num_repetitions=11, sigma_inp=0.03, sigma_rec=0.03, baseline_stimulus=False):
    coherence_lvls = np.linspace(-1, 1, num_lvls)
    res_dict = {}
    res_dict["coherence_lvls"] = coherence_lvls
    res_dict["ctx_motion"] = {}
    res_dict["ctx_color"] = {}
    res_dict["ctx_motion"]["right_choice_percentage"] = np.empty((num_lvls, num_lvls))
    res_dict["ctx_color"]["right_choice_percentage"] = np.empty((num_lvls, num_lvls))
    res_dict["ctx_motion"]["MSE"] = np.empty((num_lvls, num_lvls))
    res_dict["ctx_color"]["MSE"] = np.empty((num_lvls, num_lvls))

    for i, cm in tqdm(enumerate(coherence_lvls)):
        for j, cc in enumerate(coherence_lvls):
            for ctx in [1, 0]:
                ctx_text = "motion" if ctx == 1 else "color"
                choices = []
                errors = []
                input_stream, target_stream = generate_input_target_stream(context=ctx_text,
                                                                           motion_coh=cm, color_coh=cc,
                                                                           num_outputs=RNN.W_out.shape[0],
                                                                           n_steps=n_steps,
                                                                           protocol_dict=protocol_dict)
                for k in range(num_repetitions):
                    RNN.clear_history()
                    RNN.y = deepcopy(RNN.y_init)
                    input_coeffs = input_stream #np.vstack([np.array([ctx, 1 - ctx, mr, ml, cr, cl]).reshape(1, -1)] * n_steps)
                    RNN.run(num_steps=n_steps, Inputs=input_coeffs, sigma_rec=sigma_rec, sigma_inp=sigma_inp, save_history=True)
                    output = RNN.get_output()
                    if output.shape[-1] == 2:
                        choice = np.sign(output[-1, 0]-output[-1, 1]) if (not ((output[-1, 0] == 0.0) and (output[-1, 1] == 0.0))) else 0.5
                    else:
                        choice = np.sign(output[-1]) if (output[-1] != 0.0) else 0.5
                    error = np.sum((target_stream[mask, :] - output[mask, :]) ** 2) / mask.shape[0]
                    errors.append(deepcopy(error))
                    choices.append(deepcopy(choice))
                choices_to_right = np.sum((np.array(choices) + 1) / 2) / len(choices)
                mean_error = np.mean(errors)
                key = "ctx_motion" if ctx == 1 else "ctx_color"
                res_dict[key]["right_choice_percentage"][j, i] = choices_to_right
                res_dict[key]["MSE"][j, i] = mean_error
    return res_dict

def plot_psychometric_planes(data_dict, disp=True):
    coherence_lvls = data_dict["coherence_lvls"]
    Motion_ctx_rght_prcntg = data_dict["ctx_motion"]["right_choice_percentage"][::-1, :]
    Color_ctx_rght_prcntg = data_dict["ctx_color"]["right_choice_percentage"][::-1, :]
    Motion_ctx_MSE = data_dict["ctx_motion"]["MSE"][::-1, :]
    Color_ctx_MSE = data_dict["ctx_color"]["MSE"][::-1, :]
    num_lvls = Color_ctx_rght_prcntg.shape[0]

    fig1 = plt.figure()#figsize=(7,7))
    plt.suptitle("Motion context, percentage to the right", fontsize=16)
    im = plt.imshow(Motion_ctx_rght_prcntg, cmap="coolwarm", interpolation="bicubic")
    plt.xticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2), rotation=52)
    plt.yticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2)[::-1])
    plt.ylabel("Coherence of color", fontsize=16)
    plt.xlabel("Coherence of motion", fontsize=16)
    plt.colorbar(im)
    if disp:
        plt.tight_layout()
        plt.show()

    fig2 = plt.figure()#figsize=(7,7))
    plt.suptitle("Color context, percentage to the right", fontsize=16)
    im = plt.imshow(Color_ctx_rght_prcntg, cmap="coolwarm", interpolation="bicubic")
    plt.xticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2), rotation=52)
    plt.yticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2)[::-1])
    plt.ylabel("Coherence of color", fontsize=16)
    plt.xlabel("Coherence of motion", fontsize=16)
    plt.colorbar(im)
    if disp:
        plt.tight_layout()
        plt.show()

    fig3 = plt.figure()#figsize=(7,7))
    plt.suptitle("MSE surface, Motion context", fontsize=16)
    im = plt.imshow(Motion_ctx_MSE, cmap="coolwarm", interpolation="bicubic" )
    plt.xticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2), rotation=50)
    plt.yticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2)[::-1])
    plt.ylabel("Coherence of color", fontsize=16)
    plt.xlabel("Coherence of motion", fontsize=16)
    plt.colorbar(im)
    if disp:
        plt.tight_layout()
        plt.show()

    fig4 = plt.figure()#figsize=(7,7))
    plt.suptitle("MSE surface, Color context", fontsize=16)
    im = plt.imshow(Color_ctx_MSE, cmap="coolwarm", interpolation="bicubic")
    plt.xticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2), rotation=50)
    plt.yticks(np.arange(num_lvls), labels=np.round(coherence_lvls, 2)[::-1])
    plt.ylabel("Coherence of color", fontsize=16)
    plt.xlabel("Coherence of motion", fontsize=16)
    plt.colorbar(im)
    if disp:
        plt.tight_layout()
        plt.show()

    return [fig1, fig2, fig3, fig4]


if __name__ == '__main__':
    import os
    from src.RNN_numpy import RNN_numpy

    n_steps = 300
    max_coherence = 0.4
    dt = 1
    tau = 10
    W_inp, W_rec, W_out = get_small_connectivity_np(rnd_perturb=1e-12)
    N = W_rec.shape[0]
    bias_rec = np.zeros(N)
    bias_out = 0
    mask = np.concatenate([np.arange(100), np.arange(100)+200])
    protocol_dict = {"cue_on": 0, "cue_off": n_steps,
                     "stim_on": int(n_steps//3), "stim_off": n_steps,
                     "dec_on": int(2 * n_steps//3), "dec_off": n_steps}
    RNN = RNN_numpy(N=N, dt=dt, tau=tau,
                    W_inp=W_inp,
                    W_rec=W_rec,
                    W_out=W_out,
                    bias_rec=bias_rec,
                    bias_out=bias_out)
    psycho_data = get_psychometric_data(RNN, n_steps, mask=mask, num_lvls=16, num_repetitions=11,
                                        sigma_inp=0.03, sigma_rec=0.03,
                                        protocol_dict=protocol_dict)
    psycho_figs = plot_psychometric_planes(psycho_data, disp=True)
