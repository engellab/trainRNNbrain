import os
import sys
sys.path.insert(0, "../")
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import torch
from torch import nn
from src.utils import orthonormalize

'''
lightweight numpy implementation of RNN for validation and quick testing and plotting
'''
class RNN_numpy():
    
    def __init__(self, N, dt, tau, W_inp, W_rec, W_out, bias_rec=None, bias_out=None, activation="ReLU", y_init=None):
        self.N = N
        self.W_inp = W_inp
        self.W_rec = W_rec
        self.W_out = W_out
        if bias_rec is None:
            self.bias_rec = np.zeros(self.N)
        else:
            self.bias_rec = bias_rec
        if bias_out is None:
            self.bias_out = 0
        else:
            self.bias_out = bias_out
        self.dt = dt
        self.tau = tau
        self.alpha = self.dt/self.tau
        if not (y_init is None):
            self.y_init = y_init
        else:
            self.y_init = np.zeros(self.N)
        self.y = deepcopy(self.y_init)
        self.y_history = []
        self.activation_name = activation

        if activation == "ReLU":
            self.activation = lambda x: np.maximum(x, 0)
        elif activation == "Tanh":
            self.activation = lambda x: np.tanh(x)

    def rhs(self, I, sigma_rec=None, sigma_inp=None, generator_numpy=None):
        if (generator_numpy is None):
            generator_numpy = np.random.default_rng(np.random.randint(10000))
        rec_noise_term = np.sqrt((2 / self.alpha) * sigma_rec**2) * generator_numpy.standard_normal(self.N) if (not (sigma_rec is None)) else np.zeros(self.N)
        inp_noise_term = np.sqrt((2 / self.alpha) * sigma_inp ** 2) * generator_numpy.standard_normal(6) if (not (sigma_inp is None)) else np.zeros(6)
        return (-self.y +
                self.activation(self.W_rec @ (self.y).reshape(-1, 1)
                                + self.W_inp @ (I + inp_noise_term).reshape(-1, 1)
                                + self.bias_rec.reshape(-1, 1) + rec_noise_term.reshape(-1, 1)).flatten()
               )

    def rhs_jac(self, I):
        if self.activation_name == "ReLU":
            N = self.W_rec.shape[0]
            # y = np.round(self.y.reshape(-1, 1), 3)
            arg = ((self.W_rec @ self.y).flatten() + (self.W_inp @ I.reshape(-1, 1)).flatten() + self.bias_rec.flatten())
            m = 0.5
            D = np.diag(np.heaviside(arg, m))
            return -np.eye(N) + self.W_rec @ D
        else:
            return ValueError("Not Implemented!")

    def step(self, I, sigma_rec=None, sigma_inp=None, generator_numpy=None):
        self.y += (self.dt / self.tau) * self.rhs(I, sigma_rec, sigma_inp, generator_numpy)

    def run(self, num_steps, Inputs, save_history=False, sigma_rec=None, sigma_inp=None, generator_numpy=None):
        for i in range(num_steps):
            if save_history == True:
                self.y_history.append(deepcopy(self.y))
            self.step(Inputs[i, :], sigma_rec=sigma_rec, sigma_inp=sigma_inp, generator_numpy=generator_numpy)
        return None

    def get_history(self):
        return np.array(self.y_history)

    def clear_history(self):
        self.y_history = []

    def reset_state(self):
        self.y = np.zeros(self.N)

    def get_output(self, collapse=False):
        output = (np.array(self.y_history) @ self.W_out.T + self.bias_out)
        if self.W_out.shape[0] == 2 and collapse == True:
            output = (output[:, 0] - output[:, 1])
        return output

    def run_multiple_trajectories(self, inputs, sigma_rec):
        trajectories = []
        outputs = []
        for i in range(inputs.shape[0]):
            self.clear_history()
            self.y = deepcopy(self.y_init)
            self.run(num_steps=inputs.shape[1], Inputs=inputs[i, :, :], save_history=True, sigma_rec=sigma_rec)
            trajectories.append(deepcopy(self.get_history()))
            outputs.append(deepcopy(self.get_output()))
        trajectories = np.stack(trajectories)
        outputs = np.stack(outputs)
        return trajectories, outputs

if __name__ == '__main__':
    dt = 1
    tau = 10
    n = 2
    N = 150
    T_steps = 400

    w_rec = np.array([[0, 0], [1, 0]])
    w_inp = np.array([1, 0]).reshape(-1, 1)
    w_out = np.array([0, 1]).reshape(1, -1)

    c = RNN_numpy(n, dt, tau, w_inp, w_rec, w_out)
    c.y = np.zeros(n)
    Input = np.array([1])
    x_dot = []
    for t in range(T_steps):
        c.step(I=Input)
        c.y_history.append(deepcopy(c.y))
        x_dot.append(c.rhs(I=Input))

    x_dot = np.array(x_dot)
    x = c.get_history()

    fig = plt.figure(figsize=(12, 3))
    plt.plot(x[:, 0], color='b')
    plt.plot(x[:, 1], color='r')
    plt.grid(True)
    plt.ylim([0, 1.1])
    plt.show()