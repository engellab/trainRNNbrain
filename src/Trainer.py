'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''

import os
import pickle
import sys
import numpy as np
import torch
from copy import deepcopy
from src.RNN_numpy import RNN_numpy


def L2_ortho(rnn, X = None, y = None):
    # regularization of the input and ouput matrices
    b = torch.cat((rnn.input_layer.weight, rnn.output_layer.weight.t()), dim=1)
    b = b / torch.norm(b, dim=0)
    return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)

def print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss):
    gr_prfx = '\033[92m'
    gr_sfx = '\033[0m'

    train_prfx = gr_prfx if (train_loss <= min_train_loss) else ''
    train_sfx = gr_sfx if (train_loss <= min_train_loss) else ''
    if not (val_loss is None):
        val_prfx = gr_prfx if (val_loss <= min_val_loss) else ''
        val_sfx = gr_sfx if (val_loss <= min_val_loss) else ''
        print(f"iteration {iter},"
              f" train loss: {train_prfx}{np.round(train_loss, 6)}{train_sfx},"
              f" validation loss: {val_prfx}{np.round(val_loss, 6)}{val_sfx}")
    else:
        print(f"iteration {iter},"
              f" train loss: {train_prfx}{np.round(train_loss, 6)}{train_sfx}")

class Trainer():
    def __init__(self, RNN, Task, max_iter, tol, criterion, optimizer, lambda_orth, lambda_r):
        '''
        :param RNN: pytorch RNN (specific template class)
        :param Task: task (specific template class)
        :param max_iter: maximum number of iterations
        :param tol: float, such that if the cost function reaches tol the optimization terminates
        :param criterion: function to evaluate loss
        :param optimizer: pytorch optimizer (Adam, SGD, etc.)
        :param lambda_ort: float, regularization softly imposing a pair-wise orthogonality
         on columns of W_inp and rows of W_out
        :param lambda_r: float, regularization of the mean firing rates during the trial
        '''
        self.RNN = RNN
        self.Task = Task
        self.max_iter = max_iter
        self.tol = tol
        self.criterion = criterion
        self.optimizer = optimizer
        self.lambda_orth = lambda_orth
        self.lambda_r = lambda_r

    def train_step(self, input, target_output, mask):
        states, predicted_output = self.RNN(input)
        loss = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :]) + \
               self.lambda_orth * L2_ortho(self.RNN) + \
               self.lambda_r * torch.mean(states ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        error_vect = torch.sum(((target_output[:, mask, :] - predicted_output[:, mask, :]) ** 2).squeeze(), dim=1) / len(mask)
        return loss.item(), error_vect

    def eval_step(self, input, target_output, mask):
        with torch.no_grad():
            self.RNN.eval()
            states, predicted_output_val = self.RNN(input, w_noise=False)
            val_loss = self.criterion(target_output[:, mask, :], predicted_output_val[:, mask, :]) + \
                       self.lambda_orth * L2_ortho(self.RNN) +\
                       self.lambda_r * torch.mean(states ** 2)
            return float(val_loss.numpy())


    def run_training(self, train_mask, same_batch=False):
        train_losses = []
        val_losses = []
        self.RNN.train() #puts the RNN into training mode (sets update_grad = True)
        min_train_loss = np.inf
        min_val_loss = np.inf
        best_net_params = deepcopy(self.RNN.get_params())
        if same_batch:
            input_batch, target_batch, conditions_batch = self.Task.get_batch()
            input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
            target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)
            input_val = deepcopy(input_batch)
            target_output_val = deepcopy(target_batch)
            # input_val, target_output_val, conditions_val = self.Task.get_batch()
            # input_val = torch.from_numpy(input_val.astype("float32")).to(self.RNN.device)
            # target_output_val = torch.from_numpy(target_output_val.astype("float32")).to(self.RNN.device)

        for iter in range(self.max_iter):
            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch()
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)
                input_val, target_output_val, conditions_val = self.Task.get_batch()
                input_val = torch.from_numpy(input_val.astype("float32")).to(self.RNN.device)
                target_output_val = torch.from_numpy(target_output_val.astype("float32")).to(self.RNN.device)

            train_loss, error_vect = self.train_step(input=input_batch, target_output=target_batch, mask=train_mask)
            if self.RNN.constrained:
                # positivity of entries of W_inp and W_out
                self.RNN.output_layer.weight.data = torch.maximum(self.RNN.output_layer.weight.data, torch.tensor(0))
                self.RNN.input_layer.weight.data = torch.maximum(self.RNN.input_layer.weight.data, torch.tensor(0))
                # Dale's law
                self.RNN.recurrent_layer.weight.data = torch.maximum(self.RNN.recurrent_layer.weight.data * self.RNN.dale_mask, torch.tensor(0)) * self.RNN.dale_mask

            # validation
            val_loss = self.eval_step(input_val, target_output_val, train_mask)
            # keeping track of train and valid losses and printing
            print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                best_net_params = deepcopy(self.RNN.get_params())
            if train_loss <= min_train_loss:
                min_train_loss = train_loss

            if val_loss <= self.tol:
                self.RNN.set_params(best_net_params)
                return self.RNN, train_losses, val_losses, best_net_params

        self.RNN.set_params(best_net_params)
        return self.RNN, train_losses, val_losses, best_net_params

if __name__ == '__main__':
    from src.RNN_torch import RNN_torch
    from src.Task import TaskCDDM
    from src.PerformanceAnalyzer import AnalyzerCDDM
    from src.DynamicSystemAnalyzer import DynamicSystemAnalyzerCDDM
    from matplotlib import pyplot as plt
    from src.utils import numpify

    # RNN:
    N = 50
    activation = lambda x: torch.maximum(x, torch.tensor(0))
    # activation = lambda x: 1/(1 + torch.exp(-6 * (x-0.4)))
    # activation = lambda x: 0.1 * torch.log(1 + torch.exp(10 * x))
    # activation = lambda x: torch.tanh(x)
    rng = torch.Generator()
    rng.manual_seed(2)
    sigma_inp = 0.03
    sigma_rec = 0.03
    n_inputs = 6
    n_outputs = 2
    constrained = False
    rnn_torch = RNN_torch(N=N, input_size=n_inputs, output_size=n_outputs,
                          activation=activation, constrained=constrained,
                          sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                          random_generator=rng)
    # Task:
    n_steps = 300
    task_params = dict()
    max_coherence_train = 0.8
    coherence_lvls = 5
    tmp = max_coherence_train * np.logspace(-4, 0, coherence_lvls, base=2)
    protocol_dict = dict()
    protocol_dict["cue_on"] = 0
    protocol_dict["cue_off"] = n_steps
    protocol_dict["stim_on"] = n_steps//3
    protocol_dict["stim_off"] = n_steps
    protocol_dict["dec_on"] = 2 * (n_steps//3)
    protocol_dict["dec_off"] = n_steps
    task_params["protocol_dict"] = protocol_dict
    task_params["protocol_dict"]["coherences_train"] = np.concatenate([-np.array(tmp[::-1]), np.array([0]), np.array(tmp)])
    task_params["protocol_dict"]["coherences_valid"] = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    task = TaskCDDM(n_steps, n_inputs, n_outputs, task_params)

    # Trainer:
    max_iter = 300
    tol = 1e-6
    lr = 0.02
    weight_decay = 1e-5
    lambda_orth = 0.3
    lambda_r = 0.1
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rnn_torch.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask = np.concatenate([np.arange(n_steps//3), np.arange(n_steps//3) + 2 * (n_steps//3)])
    trainer = Trainer(RNN=rnn_torch, Task=task, max_iter=max_iter, tol=tol,
                      optimizer=optimizer, criterion=criterion,
                      lambda_orth=lambda_orth, lambda_r=lambda_r)


    rnn_trained, train_losses, val_losses, best_net_params = trainer.run_training(train_mask=train_mask, same_batch=True)
    # pickle.dump(best_net_params, open(os.path.join("../", "data", "trained_RNNs", "data_tanh.pkl"), "wb+"))

    # plt.figure(figsize = (10, 3))
    # plt.plot(np.log(train_losses), color='r', label='train loss (log scale)')
    # plt.plot(np.log(val_losses), color='b', label='valid loss (log scale)')
    # plt.grid(True)
    # plt.legend(fontsize=16)
    # plt.show()

    # best_net_params = pickle.load(open(os.path.join("../", "data", "trained_RNNs", "data_tanh.pkl"), "rb+"))
    # validate
    # show a bunch of trajectories
    RNN_valid = RNN_numpy(N=best_net_params["N"], dt=best_net_params["dt"], tau=best_net_params["tau"],
                          activation = numpify(activation), #lambda x: np.maximum(x, 0),
                          W_inp=best_net_params["W_inp"],
                          W_rec=best_net_params["W_rec"],
                          W_out=best_net_params["W_out"],
                          bias_rec=best_net_params["bias_rec"],
                          y_init=best_net_params["y_init"])

    analyzer = AnalyzerCDDM(RNN_valid)
    score_function = lambda x, y: np.mean((x - y) ** 2)
    input_batch_valid, target_batch_valid, conditions_valid = task.get_batch(mode='valid')
    score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid,
                                          train_mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    print(f"MSE validation: {np.round(score, 5)}")

    print(f"Plotting random trials")
    inds = np.random.choice(np.arange(input_batch_valid.shape[0]), 10)
    inputs = input_batch_valid[inds, :, :]
    targets = target_batch_valid[inds, :, :]

    fig = analyzer.plot_trials(inputs, targets, train_mask, labels = ["OutR", "OutL"], sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    plt.show()

    analyzer.calc_psychometric_data(task, train_mask, num_levels=11, num_repeats=7, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    fig = analyzer.plot_psychometric_data(disp=True)
    plt.show()

    dsa = DynamicSystemAnalyzerCDDM(RNN_valid)
    dsa.calc_LineAttractor_analytics(N_points=11, stop_length=20, patience=50)

    fig = dsa.plot_fp_2D()
    plt.show()

    fig_RHS, fig_3D = dsa.plot_LineAttractor_3D(N_points=11)
    plt.show()






