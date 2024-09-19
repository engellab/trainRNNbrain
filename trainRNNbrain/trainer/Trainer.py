'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''

from copy import deepcopy
import numpy as np
import torch


def L2_ortho(rnn, X=None, y=None, orth_input_only = True):
    # regularization of the input and ouput matrices
    # Pair-wise orthogonalization of both the input columns and output rows
    if not orth_input_only:
        b = torch.cat((rnn.W_inp, rnn.W_out.t()), dim=1)
        b = b / torch.norm(b, dim=0)
        return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)
    else:
        # Pair-wise orthogonalization input columns only
        b = rnn.W_inp
        b = b / torch.norm(b, dim=0)
    return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)


def print_iteration_info(iter, train_loss, min_train_loss, val_loss=None, min_val_loss=None):
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
    def __init__(self, RNN, Task, criterion, optimizer,
                 max_iter=1000,
                 tol=1e-12,
                 lambda_orth=0.3,
                 orth_input_only=True,
                 lambda_r=0.5,
                 p = 2):
        '''
        :param RNN: pytorch RNN (specific template class)
        :param Task: task (specific template class)
        :param max_iter: maximum number of iterations
        :param tol: float, such that if the cost function reaches tol the optimization terminates
        :param criterion: function to evaluate loss
        :param optimizer: pytorch optimizer (Adam, SGD, etc.)
        :param lambda_ort: float, regularization softly imposing a pair-wise orthogonality
         on columns of W_inp and rows of W_out
        :param orth_input_only: bool, if impose penalties only on the input columns,
         or extend the penalty onto the output rows as well
        :param lambda_r: float, regularization of the mean firing rates during the trial
        '''
        self.RNN = RNN
        self.Task = Task
        self.max_iter = max_iter
        self.tol = tol
        self.criterion = criterion
        self.optimizer = optimizer
        self.lambda_orth = lambda_orth
        self.orth_input_only = orth_input_only
        self.lambda_r = lambda_r
        self.p = p

    def train_step(self, input, target_output, mask):
        states, predicted_output = self.RNN(input)
        loss = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :]) + \
               self.lambda_orth * L2_ortho(self.RNN, orth_input_only=self.orth_input_only) + \
               self.lambda_r * torch.mean(torch.abs(states) ** self.p)

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
                       self.lambda_orth * L2_ortho(self.RNN, orth_input_only=self.orth_input_only) + \
                       self.lambda_r * torch.mean(torch.abs(states) ** self.p)
            return float(val_loss.cpu().numpy())

    def run_training(self, train_mask, same_batch=False, shuffle=False):
        train_losses = []
        val_losses = []
        self.RNN.train()  # puts the RNN into training mode (sets update_grad = True)
        min_train_loss = np.inf
        min_val_loss = np.inf
        best_net_params = deepcopy(self.RNN.get_params())
        if same_batch:
            input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
            input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
            target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)
            input_val = deepcopy(input_batch)
            target_output_val = deepcopy(target_batch)

        for iter in range(self.max_iter):
            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)
                input_val = deepcopy(input_batch)
                target_output_val = deepcopy(target_batch)

            train_loss, error_vect = self.train_step(input=input_batch,
                                                     target_output=target_batch,
                                                     mask=train_mask)

            # positivity of entries of W_inp and W_out
            self.RNN.W_inp.data = torch.maximum(self.RNN.W_inp.data, torch.tensor(0.0))
            self.RNN.W_out.data = torch.maximum(self.RNN.W_out.data, torch.tensor(0.0))

            if self.RNN.constrained:
                # Dale's law
                self.RNN.W_out.data *= self.RNN.output_mask.to(self.RNN.device)
                self.RNN.W_inp.data *= self.RNN.input_mask.to(self.RNN.device)

                incorrect_rec_vals_mask = (self.RNN.W_rec.data.to(self.RNN.device) * self.RNN.dale_mask.to(self.RNN.device) < 0).to(self.RNN.device)
                self.RNN.W_rec.data[incorrect_rec_vals_mask] = torch.tensor(0.0).to(self.RNN.device)

                incorrect_out_vals_mask = (self.RNN.W_out.data.to(self.RNN.device) * self.RNN.dale_mask.to(self.RNN.device) < 0).to(self.RNN.device)
                self.RNN.W_out.data[incorrect_out_vals_mask] = torch.tensor(0.0).to(self.RNN.device)

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
