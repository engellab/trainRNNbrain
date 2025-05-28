'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''

from copy import deepcopy
import numpy as np
import torch
from tqdm.auto import tqdm

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
                 lambda_z=0.1,
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
        self.lambda_z = lambda_z
        self.loss_monitor = {"behavior": [],
                            "channel overlap": [],
                            "metabolic": [],
                            "inactivity": []}
        self.p = p

    def categorical_penalty(self, states, dale_mask):
        X = states.view(states.shape[0], -1)
        loss = torch.tensor(0.0, device=states.device, dtype=states.dtype)
        if dale_mask is None:
            dale_mask = torch.ones(X.shape[0], device=states.device, dtype=states.dtype)

        for nrn_sign in [1, -1]:
            X_subpop = X[dale_mask == nrn_sign, :]
            X_norm_sq = (X_subpop ** 2).sum(dim=1, keepdim=True)
            D = (X_norm_sq + X_norm_sq.T - 2 * X_subpop @ X_subpop.T)
            D = D.clamp(min=1e-9).sqrt()
            d = D.view(-1)
            m = torch.mean(d).detach() + 1e-8
            s = torch.std(d).detach() + 1e-8
            loss += (torch.mean(d[d < m] / m) +
                     torch.mean(1.0 - torch.clip((d[d >= m] - m) / (2 * s), 0.0, 1.0)))
        return loss

    def inactivity_penalty(self, states):
        X = states.view(states.shape[0], -1)
        participation = torch.mean(torch.abs(X), dim=1)
        threshold = torch.quantile(participation, 0.3).detach() + 1e-8
        diff = threshold - participation
        return torch.mean((diff[diff > 0] / threshold) ** 2)

    def channel_overlap_penalty(self):
        b = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        b = b / (torch.linalg.vector_norm(b, dim=0) + 1e-12)
        G = torch.tril(b.T @ b, diagonal=-1)
        lower_tri_mask = torch.tril(torch.ones_like(G), diagonal=-1)
        return torch.sqrt(torch.mean(G[lower_tri_mask == 1.0] ** 2))

    def train_step(self, input, target_output, mask):
        states, predicted_output = self.RNN(input)
        behavior_mismatch_penalty = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :])
        channel_overlap_penalty = self.lambda_orth * self.channel_overlap_penalty()
        metabolic_penalty = self.lambda_r * torch.mean(torch.abs(states) ** self.p)
        inactivity_penalty = self.lambda_z * self.inactivity_penalty(states)

        loss = (behavior_mismatch_penalty +
                channel_overlap_penalty +
                metabolic_penalty +
                inactivity_penalty)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_monitor["behavior"].append(behavior_mismatch_penalty.detach())
        self.loss_monitor["channel overlap"].append(channel_overlap_penalty.detach())
        self.loss_monitor["metabolic"].append(metabolic_penalty.detach())
        self.loss_monitor["inactivity"].append(inactivity_penalty.detach())
        return loss.item()

    def eval_step(self, input, target_output, mask):
        with torch.no_grad():
            self.RNN.eval()
            states, predicted_output_val = self.RNN(input, w_noise=False)
            val_loss = self.criterion(target_output[:, mask, :], predicted_output_val[:, mask, :]) + \
                       self.lambda_orth * self.channel_overlap_penalty() + \
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

        for iter in tqdm(range(self.max_iter)):
            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)
                input_val = deepcopy(input_batch)
                target_output_val = deepcopy(target_batch)

            train_loss = self.train_step(input=input_batch,
                                         target_output=target_batch,
                                         mask=train_mask)
            eps = 1e-12
            # positivity of entries of W_inp and W_out
            self.RNN.W_inp.data = torch.maximum(self.RNN.W_inp.data, torch.tensor(eps))
            self.RNN.W_out.data = torch.maximum(self.RNN.W_out.data, torch.tensor(eps))

            if self.RNN.constrained:
                # Dale's law
                self.RNN.W_out.data *= self.RNN.output_mask.to(self.RNN.device)
                self.RNN.W_inp.data *= self.RNN.input_mask.to(self.RNN.device)

                W_rec = self.RNN.W_rec.data
                W_out = self.RNN.W_out.data
                dale_mask = self.RNN.dale_mask  # assumed already on the same device

                # Apply Dale's law by clamping incorrect signs
                W_rec[(W_rec * dale_mask < 0)] = eps
                W_out[(W_out * dale_mask < 0)] = eps

            # validation
            # keeping track of train and valid losses and printing
            # if iter % 100 == 0:
            #     print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss)
            # val_loss = self.eval_step(input_val, target_output_val, train_mask)
            # train_losses.append(train_loss)
            # val_losses.append(val_loss)
            # if val_loss <= min_val_loss:
            #     min_val_loss = val_loss
            #     best_net_params = deepcopy(self.RNN.get_params())
            # if train_loss <= min_train_loss:
            #     min_train_loss = train_loss
            #
            # if val_loss <= self.tol:
            #     self.RNN.set_params(best_net_params)
            #     return self.RNN, train_losses, val_losses, best_net_params

            print_iteration_info(iter, train_loss, min_train_loss)
            train_losses.append(train_loss)
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                best_net_params = deepcopy(self.RNN.get_params())
            if train_loss <= self.tol:
                self.RNN.set_params(best_net_params)
                return self.RNN, train_losses, val_losses, best_net_params

        self.RNN.set_params(best_net_params)
        return self.RNN, train_losses, val_losses, best_net_params
