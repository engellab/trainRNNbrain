'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''

from copy import deepcopy

import numpy as np
import torch


def L2_ortho(rnn, X=None, y=None):
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
        error_vect = torch.sum(((target_output[:, mask, :] - predicted_output[:, mask, :]) ** 2).squeeze(),
                               dim=1) / len(mask)
        return loss.item(), error_vect

    def eval_step(self, input, target_output, mask):
        with torch.no_grad():
            self.RNN.eval()
            states, predicted_output_val = self.RNN(input, w_noise=False)
            val_loss = self.criterion(target_output[:, mask, :], predicted_output_val[:, mask, :]) + \
                       self.lambda_orth * L2_ortho(self.RNN) + \
                       self.lambda_r * torch.mean(states ** 2)
            return float(val_loss.cpu().numpy())

    def run_training(self, train_mask, same_batch=False):
        train_losses = []
        val_losses = []
        self.RNN.train()  # puts the RNN into training mode (sets update_grad = True)
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
                self.RNN.recurrent_layer.weight.data = (
                            torch.maximum(self.RNN.recurrent_layer.weight.data.cpu() * self.RNN.dale_mask.cpu(),
                                          torch.tensor(0)) * self.RNN.dale_mask).to(self.RNN.device)

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
