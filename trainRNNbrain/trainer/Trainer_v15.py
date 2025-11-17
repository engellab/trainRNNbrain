'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score, mean_squared_error
from tqdm.auto import tqdm

def print_iteration_info(
    iter,
    train_loss,
    min_train_loss,
    val_loss=None,
    min_val_loss=None,
    train_direction='min',
    val_direction='min'
):
    """
    Print training and validation metrics with green highlight if improved.

    Args:
        iter: iteration number
        train_loss: current training loss or metric
        min_train_loss: best training value so far
        val_loss: current validation loss or metric (optional)
        min_val_loss: best validation value so far (optional)
        train_direction: 'min' or 'max' — defines if lower or higher is better for training
        val_direction: 'min' or 'max' — defines if lower or higher is better for validation
    """
    gr_prfx = '\033[92m'
    gr_sfx = '\033[0m'

    def is_improved(current, best, direction):
        if direction == 'min':
            return current <= best
        elif direction == 'max':
            return current >= best
        else:
            raise ValueError("Direction must be 'min' or 'max'")

    # Evaluate improvement
    train_improved = is_improved(train_loss, min_train_loss, train_direction)
    train_prfx = gr_prfx if train_improved else ''
    train_sfx = gr_sfx if train_improved else ''

    if val_loss is not None and min_val_loss is not None:
        val_improved = is_improved(val_loss, min_val_loss, val_direction)
        val_prfx = gr_prfx if val_improved else ''
        val_sfx = gr_sfx if val_improved else ''
        print(f"iteration {iter},"
              f" train: {train_prfx}{np.round(train_loss, 6)}{train_sfx},"
              f" val: {val_prfx}{np.round(val_loss, 6)}{val_sfx}")
    else:
        print(f"iteration {iter},"
              f" train: {train_prfx}{np.round(train_loss, 6)}{train_sfx}")



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
                 lambda_d=0.05,
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
        self.lambda_d = lambda_d
        self.lambda_hv = lambda_hv
        self.lambda_cv = lambda_cv
        self.loss_monitor = {"behavior": [],
                             "channel overlap": [], #lambda_orth
                             "activity": [], #lambda_r
                             "isolation": [], # lambda_p
                             "net inputs variability": [], #lambda_hv
                             "net inputs mean": [], #lambda_h
                             "categorical": [], #lambda_m
                             "dimensionality": [] #lambda_d
                             }
        self.p = p
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.counter = 0

    def dimensionality_penalty(self, states, rank_target=5, beta=20.):
        N, T, K = states.shape
        X = states.reshape(N, T * K)  # [N, TK]
        # Singular values (sorted in descending order by default)
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
        S = S[S > 1e-6 * S[0]]  # Only consider significant singular values
        P = S / (S.sum() + 1e-8)
        entropy = -torch.sum(P * torch.log(P + 1e-8))
        eff_rank = torch.exp(entropy)
        penalty = F.softplus(eff_rank - rank_target, beta=beta)
        return penalty


    def categorical_penalty(self, states, attract_margin=0.1, repell_margin=0.3, diameter_quantile=0.9, beta=20.0):
        X = states.view(states.shape[0], -1)
        loss = torch.tensor(0.0, device=states.device, dtype=states.dtype)
        if getattr(self.RNN, 'dale_mask', None) is None:
            dale_mask = torch.ones(X.shape[0], device=states.device, dtype=states.dtype)
        else:
            dale_mask = self.RNN.dale_mask

        for nrn_sign in [1, -1]:
            X_subpop = X[dale_mask == nrn_sign, :]
            if X_subpop.shape[0] <= 1:
                continue
            X_norm_sq = (X_subpop ** 2).sum(dim=1, keepdim=True)
            D2 = X_norm_sq + X_norm_sq.T - 2 * X_subpop @ X_subpop.T
            D = D2.clamp(min=1e-8).sqrt()
            # Remove diagonal (self-distances)
            i, j = torch.triu_indices(D.shape[0], D.shape[1], offset=1)
            dists = D[i, j]

            diameter = torch.quantile(dists, diameter_quantile).detach()
            attract_thresh = attract_margin * diameter
            repell_thresh = repell_margin * diameter

            # Triangle penalty (fully smooth)
            term1 = F.softplus(dists / (attract_thresh + 1e-8), beta=beta)
            term2 = F.softplus((dists - attract_thresh) / (attract_thresh + 1e-8), beta=beta)
            term3 = F.softplus((dists - attract_thresh) / ((repell_thresh - attract_thresh) + 1e-8), beta=beta)
            triangle = F.softplus(term1 - term2 - term3, beta=beta)
            loss += triangle.mean()
        return loss

    def channel_overlap_penalty(self):
        '''
        Encourages input channels to be non overlapping: so that a neuron receives at most one input channel
        '''
        b = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        b = b / (torch.linalg.vector_norm(b, dim=0) + 1e-8)
        G = torch.tril(b.T @ b, diagonal=-1)
        lower_tri_mask = torch.tril(torch.ones_like(G), diagonal=-1)
        return torch.sqrt(torch.mean(G[lower_tri_mask == 1.0] ** 2))

    def isolation_penalty(self, percent=0.5, min_threshold=0.01, beta=20.0):
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
        incoming_penalty = torch.mean((F.softplus(threshold - incoming_weights, beta=beta) / threshold) ** 2)
        return incoming_penalty

    # def afferents_variability_penalty(self, percent=0.5, min_threshold=0.02, beta=20.0):
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
    #     weights_variability_penalty = torch.mean((F.softplus(threshold - incoming_weights_varaibility, beta=beta) / threshold) ** 2)
    #     return weights_variability_penalty

    def net_inputs_variability_penalty(self, states, input, percent=0.5, min_threshold=0.05, beta=20):
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
        trial_variability_penalty = torch.mean((F.softplus(threshold - neural_trail_variability, beta=beta) / threshold) ** 2)
        return trial_variability_penalty

    def activity_penalty(self, states,
                         low_percent=0.4, high_percent=0.6,
                         hard_min=None, hard_max=1.0, beta=20.0):
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
        low_activity_penalty = torch.mean((F.softplus(low_threshold - activity, beta=beta) / low_threshold) ** 2)

        high_activity_penalty = torch.mean((F.softplus(activity - high_threshold, beta=beta) / high_threshold) ** 2)
        return low_activity_penalty + high_activity_penalty

    def net_inputs_penalty(self, states, input,
                           low_percent=0.4, high_percent=0.6,
                           hard_min=-1, hard_max=1.0, beta=20.0):
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
        low_h_penalty = torch.mean((F.softplus(low_threshold - mean_h, beta=beta) / low_threshold) ** 2)

        high_h_penalty = torch.mean((F.softplus(mean_h - high_threshold, beta=beta) / high_threshold) ** 2)
        return low_h_penalty + high_h_penalty

    def train_step(self, input, target_output, mask):
        states, predicted_output = self.RNN(input, w_noise=True, dropout=self.dropout, drop_rate=self.drop_rate)
        behavior_mismatch_penalty = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :])
        channel_overlap_penalty = self.lambda_orth * self.channel_overlap_penalty() if self.lambda_orth != 0 else 0
        activity_penalty = self.lambda_r * self.activity_penalty(states) if self.lambda_r != 0 else 0
        net_inputs_variability_penalty = self.lambda_hv * self.net_inputs_variability_penalty(states, input) if self.lambda_hv != 0 else 0
        isolation_penalty = self.lambda_p * self.isolation_penalty() if self.lambda_p != 0 else 0
        net_inputs_penalty = self.lambda_h * self.net_inputs_penalty(states, input) if self.lambda_h != 0 else 0
        categorical_penalty = self.lambda_m * self.categorical_penalty(states) if self.lambda_m != 0 else 0
        dimensionality_penalty = self.lambda_d * self.dimensionality_penalty(states) if self.lambda_d != 0 else 0
        loss = (behavior_mismatch_penalty
                + channel_overlap_penalty
                + activity_penalty
                + net_inputs_variability_penalty
                + isolation_penalty
                + net_inputs_penalty
                + categorical_penalty
                + dimensionality_penalty
                )
        self.counter += 1
        if self.counter == 300:
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
        self.loss_monitor["categorical"].append(to_item(categorical_penalty))
        self.loss_monitor["dimensionality"].append(to_item(dimensionality_penalty))
        return loss.item()

    def eval_step(self, input, target_output, mask, metric="R2"):
        with torch.no_grad():
            self.RNN.eval()
            states, predicted_output_val = self.RNN(input, w_noise=False, dropout=False)
            if metric=='R2':
                val_loss = r2_score(target_output[:, mask, :], predicted_output_val[:, mask, :])
            elif metric == 'MSE':
                val_loss = mean_squared_error(target_output[:, mask, :], predicted_output_val[:, mask, :])
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
