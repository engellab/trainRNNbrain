import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../')
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import random
from scipy.stats import uniform
from numpy.linalg import eig
from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent

def in_the_list(x, x_list, cutoff_diff=1e-6):
    for i in range(len(x_list)):
        diff = np.linalg.norm(x-x_list[i],2)
        if diff < cutoff_diff:
            return True
    return False

def orthonormalize(W):
    for i in range(W.shape[-1]):
        for j in range(i):
            W[:, i] = W[:, i] - W[:, j] * np.dot(W[:, i], W[:, j])
        W[:, i] = W[:, i]/np.linalg.norm(W[:, i])
    return W

def ReLU(x):
    return np.maximum(x, 0)

def generate_recurrent_weights(N, density, sr):
    A = (1.0 / (density * np.sqrt(N))) * np.array(random(N, N, density, data_rvs=uniform(-1, 2).rvs).todense())
    # get eigenvalues
    w, v = eig(A)
    A = A * (sr / np.max(np.abs(w)))
    return A

def sort_eigs(E, R):
    # sort eigenvectors
    data = np.hstack([E.reshape(-1, 1), R.T])
    data = np.array(sorted(data, key=lambda l: np.real(l[0])))[::-1, :]
    E = data[:, 0]
    R = data[:, 1:].T
    return E, R

def cosine_sim(A, B):
    v1 = A.flatten()/np.linalg.norm(A.flatten())
    v2 = B.flatten()/np.linalg.norm(B.flatten())
    return np.round(np.dot(v1, v2),3)

def plot_trial(inputs, output, condition):
    # input vector:
    # 0, 1 - motion context, color context
    # 2, 3 - motion right, motion left
    # 3, 4 - color right color left
    # 5, 6 - output right, output left

    fig, ax = plt.figure(figsize=(12, 3))
    ctxt = "motion" if (inputs[0, 0] >= 1.0) else "color"
    relevant_coherence = condition["motion_coh"] if ctxt == 'motion' else condition["color_coh"]
    irrelevant_coherence = condition["color_coh"] if ctxt == 'motion' else condition["motion_coh"]
    correct_choice = condition['correct_choice']
    relevant_signals = inputs[:, 2:4] if ctxt == 'motion' else inputs[:, 4:6]
    irrelevant_signals = inputs[:, 4:6] if ctxt == 'motion' else inputs[:, 2:4]
    decision_output = (output[:, 0] - output[:, 1])
    correctness = True if (np.sign(decision_output[-1]) == correct_choice) else False
    plt.plot(decision_output, color="k", label='decision')
    plt.suptitle(
        f"Context = {ctxt}, correct choice = {correct_choice}, coherence = {np.round(relevant_coherence, 2)}",
        color='green' if correctness else "red", fontsize=16)
    plt.plot(relevant_signals[:, 0] - relevant_signals[:, 1], color='r', linewidth=2, label="relevant signal")
    plt.plot(irrelevant_signals[:, 0] - irrelevant_signals[:, 1], color='b', label="irrelevant signal",
             alpha=0.4)
    plt.legend(fontsize=16, loc=3)
    plt.grid(True)
    plt.ylim([-1.1, 1.1])
    return ax
