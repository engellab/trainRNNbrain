import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../')
from src.connectivity import get_small_connectivity_np
from src.find_fp import *
from matplotlib import pyplot as plt

'''
Find fixed points of the system in different contexts (given the contextual inputs)
and project these fixed points onto w_out vector
'''

def plot_projected_fixed_points_1D(RNN, fun_tol=1e-6, patience=100, stop_length=100, eig_cutoff=1e-9, sigma_init_guess=10, disp=True):
    W_rec = RNN.W_rec
    W_inp = RNN.W_inp
    W_out = RNN.W_out
    bias_rec = RNN.bias_rec
    bias_out = RNN.bias_out

    x = 0.5
    y = 0.5
    figs = []
    for I in [(np.array([[1.0, 0.0, x, 1 - x, y, 1 - y]])).T, (np.array([[0.0, 1.0, x, 1 - x, y, 1 - y]])).T]:
        ctxt = 'motion' if I[0, 0] == 1.0 else 'color'
        params = {"W_rec" : W_rec, "W_inp" : W_inp, "bias_rec" : bias_rec, "Input" : I}
        if W_out.shape[0] == 2:
            choice_axis = W_out[0, :] - W_out[1, :]
        else:
            choice_axis = W_out.flatten()
        ufps, sfps, msfps = find_fixed_points(**params, fun_tol=fun_tol, patience=patience,
                                              stop_length=stop_length, sigma_init_guess=sigma_init_guess, eig_cutoff=eig_cutoff)
        if sfps.shape[0] != 0:
            projected_sfps = np.round((sfps @ choice_axis + bias_out).flatten(), 4)
        if ufps.shape[0] != 0:
            projected_ufps = np.round((ufps @ choice_axis + bias_out).flatten(), 4)
        if msfps.shape[0] != 0:
            projected_msfps = np.round((msfps @ choice_axis + bias_out).flatten(), 4)

        fig = plt.figure(figsize=(12, 3))
        plt.suptitle(r"Fixed points projected on $W_{out}$ " + f"{ctxt} context", fontsize=16)
        plt.plot(np.linspace(-1.1, 1.1, 100), np.zeros(100), alpha=0.5, color='k')
        if sfps.shape[0] != 0:
            for i in range(projected_sfps.shape[0]):
                plt.scatter(projected_sfps[i], 0, marker="o", s=100, color="blue")
        if ufps.shape[0] != 0:
            for i in range(projected_ufps.shape[0]):
                plt.scatter(projected_ufps[i], 0, marker="x", s=100, color="red")
        if msfps.shape[0] != 0:
            for i in range(projected_msfps.shape[0]):
                plt.scatter(projected_msfps[i], 0, marker="o", s=100, color="k")
        plt.grid(True)
        if disp:
            plt.show()
        figs.append(fig)
    return figs

if __name__ == '__main__':
    dt = 1
    tau = 10
    W_inp, W_rec, W_out = get_small_connectivity_np(rnd_perturb=1e-12)
    N = W_rec.shape[0]
    bias_rec = np.zeros(N)
    bias_out = 0

    RNN = RNN_numpy(N=N, dt=dt, tau=tau,
                    W_inp=W_inp,
                    W_rec=W_rec,
                    W_out=W_out,
                    bias_rec=bias_rec,
                    bias_out=bias_out)

    plot_projected_fixed_points_1D(RNN,
                                   fun_tol=1e-12,
                                   patience=300,
                                   stop_length=100,
                                   eig_cutoff=1e-9,
                                   sigma_init_guess=10,
                                   disp=True)
