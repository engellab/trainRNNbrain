'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
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
                 lambda_cv = 0.05,
                 lambda_h = 0.1,
                 lambda_hv = 0.05,
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
        self.lambda_h = lambda_h
        self.lambda_hv = lambda_hv
        self.lambda_cv = lambda_cv
        self.loss_monitor = {"behavior": [],
                             "channel overlap": [], #lambda_orth
                             "activity": [], #lambda_r
                             "isolation": [], # lambda_p
                             "net inputs variability": [], #lambda_hv
                             "net inputs mean" : [] #lambda_h
                             # "afferents variability": []
                             # "E2I ratios": [],
                             }
        self.p = p
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.counter = 0

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
    #         D = D.clamp(min=1e-8).sqrt()
    #         d = D.view(-1)
    #         m = torch.mean(d).detach() + 1e-8
    #         s = torch.std(d).detach() + 1e-8
    #         loss += (torch.mean(d[d < m] / m) +
    #                  torch.mean(1.0 - torch.clip((d[d >= m] - m) / (2 * s), 0.0, 1.0)))
    #     return loss

    def channel_overlap_penalty(self):
        '''
        Encourages input channels to be non overlapping: so that a neuron receives at most one input channel
        '''
        b = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        b = b / (torch.linalg.vector_norm(b, dim=0) + 1e-8)
        G = torch.tril(b.T @ b, diagonal=-1)
        lower_tri_mask = torch.tril(torch.ones_like(G), diagonal=-1)
        return torch.sqrt(torch.mean(G[lower_tri_mask == 1.0] ** 2))

    # def E2I_penalty(self, threshold=0.5):
    #     '''
    #     Penalizes units for having excitation/inhibition below defined threshold
    #     '''
    #     # Excitatory and inhibitory indices
    #     exc_idx = torch.where(self.RNN.dale_mask == 1)[0]
    #     inh_idx = torch.where(self.RNN.dale_mask == -1)[0]
    #     # Sum of excitatory/inhibitory inputs **to each neuron**
    #     E = torch.sum(self.RNN.W_rec[:, exc_idx], dim=1)
    #     I = torch.sum(-self.RNN.W_rec[:, inh_idx], dim=1) + 1e-8
    #     E2I_ratios = E / I  # Per neuron
    #     # Penalize only if E/I < threshold
    #     penalty = F.softplus(threshold - E2I_ratios, beta=20)
    #     return torch.mean((penalty / threshold) ** 2)

    def isolation_penalty(self, percent=0.5, min_threshold=0.01):
        '''
        If average (excitatory) incoming connections are below the threshold (either hard or dynamic)
        the unit is penalized depending on how far below threshold its connectedness lies
        '''
        if not self.RNN.dale_mask is None:
            exc_inds = torch.where(self.RNN.dale_mask == 1.0)[0]
            incoming_weights = self.RNN.W_rec[:, exc_inds].abs().sum(dim=1)
        else:
            incoming_weights = self.RNN.W_rec.abs().sum(dim=1)
        min_threshold = torch.tensor(min_threshold, device=self.RNN.device)
        threshold = torch.maximum(torch.quantile(incoming_weights, percent).detach(), min_threshold)
        incoming_penalty = torch.mean((F.softplus(threshold - incoming_weights, beta=20) / threshold) ** 2)
        return incoming_penalty

    # def afferents_variability_penalty(self, percent=0.5, min_threshold=0.02):
    #     '''
    #     Encourages incoming weight variability (seen in the units with the highest participation)
    #     If the incoming (excitatory) weights variability is below a certain threshold, it is penalized for it.
    #     '''
    #     if not self.RNN.dale_mask is None:
    #         exc_inds = torch.where(self.RNN.dale_mask == 1.0)[0]
    #         incoming_weights_varaibility = self.RNN.W_rec[:, exc_inds].abs().std(dim=1)
    #     else:
    #         incoming_weights_varaibility = self.RNN.W_rec.abs().std(dim=1)
    #     min_threshold = torch.tensor(min_threshold, device=self.RNN.device)
    #     threshold = torch.maximum(torch.quantile(incoming_weights_varaibility, percent).detach(), min_threshold)
    #     weights_variability_penalty = torch.mean((F.softplus(threshold - incoming_weights_varaibility, beta=20) / threshold) ** 2)
    #     return weights_variability_penalty

    def net_inputs_variability_penalty(self, states, input, percent=0.5, min_threshold=0.05):
        '''
        Encourages variability of net inputs across trials:
        Computes timewise standard deviation across trials, and then takes a mean across time
        Penalizes a unit if it's net input variability is below a certain threshold
        '''
        h = (torch.einsum('ij, jkl->ikl', self.RNN.W_rec, states)
             + torch.einsum('ij, jkl->ikl', self.RNN.W_inp, input))
        neural_trail_variability = torch.mean(torch.std(h, dim=-1), dim=-1)
        min_threshold = torch.tensor(min_threshold, device=self.RNN.device)
        threshold = torch.maximum(torch.quantile(neural_trail_variability, percent).detach(), min_threshold)
        trial_variability_penalty = torch.mean((F.softplus(threshold - neural_trail_variability, beta=20) / threshold) ** 2)
        return trial_variability_penalty

    def activity_penalty(self, states,
                         low_percent=0.4, high_percent=0.6,
                         hard_min=None, hard_max=1.0,
                         alpha=1.0, beta=1.0):
        '''
        Penalizes either too weak or excessive activity of units
        '''
        # Per-neuron mean absolute activity
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)

        # Hard thresholds
        if hard_min is None:
            hard_min = torch.tensor(1.0 / self.RNN.N, device=self.RNN.device)
        else:
            hard_min = torch.tensor(hard_min, device=self.RNN.device)
        if hard_max is None:
            hard_max = torch.tensor(1.0, device=self.RNN.device)
        else:
            hard_max = torch.tensor(hard_max, device=self.RNN.device)

        low_threshold = torch.max(torch.quantile(activity, q=low_percent).detach(), hard_min)
        high_threshold = torch.min(torch.quantile(activity, q=high_percent).detach(), hard_max)

        # Penalize neurons below low and high thresholds
        low_activity_penalty = torch.mean((F.softplus(low_threshold - activity, beta=20) / low_threshold) ** 2)

        high_activity_penalty = torch.mean((F.softplus(activity - high_threshold, beta=20) / high_threshold) ** 2)
        return alpha * low_activity_penalty + beta * high_activity_penalty

    def net_inputs_penalty(self, states, input,
                           low_percent=0.4, high_percent=0.6,
                           hard_min=-1, hard_max=1.0,
                           alpha=1.0, beta=1.0):
        # Per-neuron mean absolute activity
        h = (torch.einsum('ij, jkl->ikl', self.RNN.W_rec, states)
             + torch.einsum('ij, jkl->ikl', self.RNN.W_inp, input))
        mean_h = torch.mean(h, dim=(1, 2))
        # Hard thresholds
        if hard_min is None:
            hard_min = torch.tensor(1.0 / self.RNN.N, device=self.RNN.device)
        else:
            hard_min = torch.tensor(hard_min, device=self.RNN.device)
        if hard_max is None:
            hard_max = torch.tensor(1.0, device=self.RNN.device)
        else:
            hard_max = torch.tensor(hard_max, device=self.RNN.device)

        low_threshold = torch.max(torch.quantile(mean_h, q=low_percent).detach(), hard_min)
        high_threshold = torch.min(torch.quantile(mean_h, q=high_percent).detach(), hard_max)

        # Penalize neurons below low and high thresholds
        low_h_penalty = torch.mean((F.softplus(low_threshold - mean_h, beta=20) / low_threshold) ** 2)

        high_h_penalty = torch.mean((F.softplus(mean_h - high_threshold, beta=20) / high_threshold) ** 2)
        return alpha * low_h_penalty + beta * high_h_penalty

    def train_step(self, input, target_output, mask):
        states, predicted_output = self.RNN(input, w_noise=True, dropout=self.dropout, drop_rate=self.drop_rate)
        behavior_mismatch_penalty = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :])
        channel_overlap_penalty = self.lambda_orth * self.channel_overlap_penalty() if self.lambda_orth != 0 else 0
        activity_penalty = self.lambda_r * self.activity_penalty(states) if self.lambda_r != 0 else 0
        net_inputs_variability_penalty = self.lambda_hv * self.net_inputs_variability_penalty(states, input) if self.lambda_hv != 0 else 0
        isolation_penalty = self.lambda_p * self.isolation_penalty() if self.lambda_p != 0 else 0
        net_inputs_penalty = self.lambda_h * self.net_inputs_penalty(states, input) if self.lambda_h != 0 else 0
        # afferents_variability_penalty = self.lambda_cv * self.afferents_variability_penalty() if self.lambda_cv != 0 else 0
        # E2I_penalty = self.lambda_z * self.E2I_penalty() if self.lambda_z != 0 else 0
        loss = (behavior_mismatch_penalty
                + channel_overlap_penalty
                + activity_penalty
                + net_inputs_variability_penalty
                + isolation_penalty
                + net_inputs_penalty
                # + afferents_variability_penalty
                # + E2I_penalty
                )
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
        self.loss_monitor["net inputs variability"].append(to_item(net_inputs_variability_penalty))
        self.loss_monitor["isolation"].append(to_item(isolation_penalty))
        self.loss_monitor["net inputs mean"].append(to_item(net_inputs_penalty))
        # self.loss_monitor["afferents variability"].append(to_item(afferents_variability_penalty))
        # self.loss_monitor["E2I ratios"].append(to_item(E2I_penalty))
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
