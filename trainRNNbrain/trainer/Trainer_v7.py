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
                 lambda_r=0.05,
                 lambda_z=0.05,
                 lambda_p=0.05,
                 lambda_m=0.05,
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
        self.lambda_m = lambda_m
        self.lambda_p = lambda_p
        self.loss_monitor = {"behavior": [],
                             "channel overlap": [],
                             "activity": [],
                             "isolation": []}
        self.p = p
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.counter = 0

    def categorical_penalty(self, states, dale_mask):
        X = states.view(states.shape[0], -1)
        loss = torch.tensor(0.0, device=states.device, dtype=states.dtype)
        if dale_mask is None:
            dale_mask = torch.ones(X.shape[0], device=states.device, dtype=states.dtype)

        for nrn_sign in [1, -1]:
            X_subpop = X[dale_mask == nrn_sign, :]
            X_norm_sq = (X_subpop ** 2).sum(dim=1, keepdim=True)
            D = (X_norm_sq + X_norm_sq.T - 2 * X_subpop @ X_subpop.T)
            D = D.clamp(min=1e-8).sqrt()
            d = D.view(-1)
            m = torch.mean(d).detach() + 1e-8
            s = torch.std(d).detach() + 1e-8
            loss += (torch.mean(d[d < m] / m) +
                     torch.mean(1.0 - torch.clip((d[d >= m] - m) / (2 * s), 0.0, 1.0)))
        return loss

    def channel_overlap_penalty(self):
        b = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        b = b / (torch.linalg.vector_norm(b, dim=0) + 1e-8)
        G = torch.tril(b.T @ b, diagonal=-1)
        lower_tri_mask = torch.tril(torch.ones_like(G), diagonal=-1)
        return torch.sqrt(torch.mean(G[lower_tri_mask == 1.0] ** 2))

    def isolation_penalty(self, percent=0.3, beta=20, min_threshold=0.01):
        # Compute connectedness per neuron
        incoming = self.RNN.W_rec.abs().sum(dim=1) + self.RNN.W_inp.abs().sum(dim=1)  # [N]
        outgoing = self.RNN.W_rec.abs().sum(dim=0) + self.RNN.W_out.abs().sum(dim=0)  # [N]
        incoming_threshold = torch.maximum(torch.quantile(incoming, percent).detach(), torch.tensor(min_threshold))
        outgoing_threshold = torch.maximum(torch.quantile(outgoing, percent).detach(), torch.tensor(min_threshold))
        incoming_penalty = torch.sigmoid((incoming_threshold - incoming) * beta)
        outgoing_penalty = torch.sigmoid((outgoing_threshold - outgoing) * beta)
        penalty = 0.5 * (incoming_penalty.mean() + outgoing_penalty.mean())
        return penalty

    def activity_penalty(self, states):
        mu = torch.mean(torch.abs(states))
        target_activity = 1 / self.RNN.N
        term_above = (torch.relu(mu - target_activity) ** 2) * (self.RNN.N)
        # harshly penalize neurons for very low activity, so that it scales with N
        term_below = (torch.relu(target_activity - mu) ** 2) * (self.RNN.N ** 2)
        return term_below + term_above

    def train_step(self, input, target_output, mask):
        states, predicted_output = self.RNN(input, w_noise=True, dropout=self.dropout, drop_rate=self.drop_rate)
        behavior_mismatch_penalty = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :])
        channel_overlap_penalty = self.lambda_orth * self.channel_overlap_penalty() if self.lambda_orth != 0 else 0
        activity_penalty = self.lambda_r * self.activity_penalty(states) if self.lambda_r != 0 else 0
        isolation_penalty = self.lambda_z * self.isolation_penalty() if self.lambda_z != 0 else 0

        # # Only keep grad for states, don't update RNN weights here
        # grads = torch.autograd.grad(
        #     outputs=behavior_mismatch_penalty,
        #     inputs=states,
        #     retain_graph=True,
        #     create_graph=True,  # Optional: set True if you’ll backprop through the penalty
        # )[0]  # grads.shape = (N, T, K)
        # # Compute mean |grad| across time and batch for each neuron
        # g = grads.abs().mean(dim=(1, 2))  # shape: (N,)
        # p = g / (g.sum() + 1e-12)
        # entropy = -torch.sum(p * torch.log(p + 1e-12))
        # max_entropy = torch.log(torch.tensor(self.RNN.N, dtype=torch.float32, device=states.device))
        # participation_penalty = self.lambda_p * (1.0 - entropy / max_entropy)  # bounded [0,1])

        loss = (behavior_mismatch_penalty +
                channel_overlap_penalty +
                activity_penalty +
                isolation_penalty)
        self.counter += 1
        if self.counter == 100:
            pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        def to_item(x):
            return x.detach() if torch.is_tensor(x) else torch.tensor(x)

        self.loss_monitor["behavior"].append(to_item(behavior_mismatch_penalty))
        self.loss_monitor["channel overlap"].append(to_item(channel_overlap_penalty))
        self.loss_monitor["activity"].append(to_item(activity_penalty))
        self.loss_monitor["isolation"].append(to_item(isolation_penalty))
        # self.loss_monitor["participation"].append(participation_penalty.detach())
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

            print_iteration_info(iter, train_loss, min_train_loss)
            train_losses.append(train_loss)
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                best_net_params = deepcopy(self.RNN.get_params())
            if train_loss <= self.tol:
                print("Reached tolerance!")
                self.RNN.set_params(best_net_params)
                return self.RNN, train_losses, val_losses, best_net_params

        self.RNN.set_params(best_net_params)
        return self.RNN, train_losses, val_losses, best_net_params

    def enforce_dale(self, eps=1e-8):
        with torch.no_grad():
            # W_rec
            W_rec = self.RNN.W_rec
            dale_mask_expanded_rec = self.RNN.dale_mask.unsqueeze(0).repeat(W_rec.shape[0], 1)
            abberant_mask_rec = (W_rec * dale_mask_expanded_rec < 0)
            corrected_rec = W_rec.clone()
            corrected_rec[abberant_mask_rec] = eps * dale_mask_expanded_rec[abberant_mask_rec]
            self.RNN.W_rec.copy_(corrected_rec)

            # W_out
            W_out = self.RNN.W_out
            dale_mask_expanded_out = self.RNN.dale_mask.unsqueeze(0).repeat(W_out.shape[0], 1)
            abberant_mask_out = (W_out * dale_mask_expanded_out < 0)
            corrected_out = W_out.clone()
            corrected_out[abberant_mask_out] = eps * dale_mask_expanded_out[abberant_mask_out]
            self.RNN.W_out.copy_(corrected_out)
            return None
