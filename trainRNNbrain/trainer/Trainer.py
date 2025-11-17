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
                 lambda_z=0.0,
                 dropout=False,
                 drop_rate=0.3,
                 p=2):
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
                            "activity": [],
                            "isolation": []}
        self.p = p
        self.dropout = dropout
        self.drop_rate = drop_rate

    # def categorical_penalty(self, states, dale_mask):
    #     X = states.view(states.shape[0], -1)
    #     loss = torch.tensor(0.0, device=states.device, dtype=states.dtype)
    #     if dale_mask is None:
    #         dale_mask = torch.ones(X.shape[0], device=states.device, dtype=states.dtype)
    #
    #     for nrn_sign in [1, -1]:
    #         X_subpop = X[dale_mask == nrn_sign, :]
    #         X_norm_sq = (X_subpop ** 2).sum(dim=1, keepdim=True)
    #         D = (X_norm_sq + X_norm_sq.T - 2 * X_subpop @ X_subpop.T)
    #         D = D.clamp(min=1e-9).sqrt()
    #         d = D.view(-1)
    #         m = torch.mean(d).detach() + 1e-8
    #         s = torch.std(d).detach() + 1e-8
    #         loss += (torch.mean(d[d < m] / m) +
    #                  torch.mean(1.0 - torch.clip((d[d >= m] - m) / (2 * s), 0.0, 1.0)))
    #     return loss
    #
    # def isolation_penalty(self, states, target_frac=0.3, beta=20):
    #     X = states.view(states.shape[0], -1)
    #     participation = torch.mean(torch.abs(X), dim=1) + torch.std(torch.abs(X), dim = 1)
    #     threshold = torch.quantile(participation, target_frac).detach() + 1e-8
    #     penalty = torch.nn.functional.softplus((threshold - participation) * beta) / (threshold + 1e-8)
    #     return penalty.mean()

    def channel_overlap_penalty(self):
        b = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        b = b / (torch.linalg.vector_norm(b, dim=0) + 1e-8)
        G = torch.tril(b.T @ b, diagonal=-1)
        lower_tri_mask = torch.tril(torch.ones_like(G), diagonal=-1)
        return torch.sqrt(torch.mean(G[lower_tri_mask == 1.0] ** 2))

    def activity_penalty(self, states):
        mu = torch.mean(torch.abs(states))
        target_activity = 1 / self.RNN.N
        term_above = (torch.relu(mu - target_activity) ** 2)
        term_below = (torch.relu(target_activity - mu) ** 2) * (self.RNN.N ** 2)
        return term_below + term_above

    # def activity_penalty(self, states):
    #     return torch.mean(torch.abs(states) ** self.p) * (self.RNN.N ** (self.p - 1))

    def isolation_penalty(self, percent=0.3, beta=20):
        # Compute connectedness per neuron
        incoming = self.RNN.W_rec.abs().sum(dim=1) + self.RNN.W_inp.abs().sum(dim=1)  # [N]
        outgoing = self.RNN.W_rec.abs().sum(dim=0) + self.RNN.W_out.abs().sum(dim=0)  # [N]
        incoming_threshold = torch.quantile(incoming, percent).detach()
        outgoing_threshold = torch.quantile(outgoing, percent).detach()
        incoming_penalty = torch.sigmoid((incoming_threshold - incoming) * beta)
        outgoing_penalty = torch.sigmoid((outgoing_threshold - outgoing) * beta)
        penalty = 0.5 * (incoming_penalty.mean() + outgoing_penalty.mean())
        return penalty

    def train_step(self, input, target_output, mask):
        states, predicted_output = self.RNN(input, w_noise=True, dropout=self.dropout, drop_rate=self.drop_rate)
        behavior_mismatch_penalty = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :])
        channel_overlap_penalty = self.lambda_orth * self.channel_overlap_penalty()
        activity_penalty = self.lambda_r * self.activity_penalty(states)
        isolation_penalty = self.lambda_z * self.isolation_penalty()
        loss = (behavior_mismatch_penalty +
                channel_overlap_penalty +
                activity_penalty +
                isolation_penalty)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_monitor["behavior"].append(behavior_mismatch_penalty.detach())
        self.loss_monitor["channel overlap"].append(channel_overlap_penalty.detach())
        self.loss_monitor["activity"].append(activity_penalty.detach())
        self.loss_monitor["isolation"].append(isolation_penalty.detach())
        return loss.item()

    def eval_step(self, input, target_output, mask):
        with torch.no_grad():
            self.RNN.eval()
            states, predicted_output_val = self.RNN(input, w_noise=False, dropout=False)
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

        for iter in tqdm(range(self.max_iter)):
            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)

            train_loss = self.train_step(input=input_batch,
                                         target_output=target_batch,
                                         mask=train_mask)
            eps = 1e-8
            # positivity of entries of W_inp and W_out
            self.RNN.W_inp.data = torch.maximum(self.RNN.W_inp.data, torch.tensor(eps))
            self.RNN.W_out.data = torch.maximum(self.RNN.W_out.data, torch.tensor(eps))
            if self.RNN.constrained:
                self.enforce_dale(eps)
            self.RNN.W_inp.data *= self.RNN.input_mask
            self.RNN.W_rec.data *= self.RNN.recurrent_mask
            self.RNN.W_out.data *= self.RNN.output_mask

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

    def enforce_dale(self, eps=1e-8):
        with torch.no_grad():
            dale_mask = self.RNN.dale_mask  # shape (N,)

            # --- Recurrent weights (W_rec): shape (N, N) ---
            W_rec = self.RNN.W_rec
            dale_mask_rec = dale_mask.view(1, -1).expand(W_rec.shape[0], -1)  # each column = presynaptic
            sign_violation_rec = (W_rec * dale_mask_rec < 0)
            W_rec[sign_violation_rec] = eps * dale_mask_rec[sign_violation_rec]  # set to small correct-sign value

            # --- Output weights (W_out): shape (output_dim, N) ---
            W_out = self.RNN.W_out
            dale_mask_out = dale_mask.view(1, -1).expand(W_out.shape[0], -1)  # same shape as W_out
            sign_violation_out = (W_out * dale_mask_out < 0)
            W_out[sign_violation_out] = eps * dale_mask_out[sign_violation_out]

        return None
