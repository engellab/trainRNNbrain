from copy import deepcopy
import torch
import numpy as np
from torch.nn.functional import softplus
'''
forward pass return 3 arguments instead of 2
'''

# Connectivity defining methods
def sparse(tnsr: torch.Tensor,
           sparsity: float,
           mean: float = 0.0,
           std:  float = 1.0,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Fill a 2D tensor with N(mean, std) and set the same number of rows to zero in each column,
    such that the column-wise zero fraction equals `sparsity`.

    Logic kept identical to original:
      - Uses `generator.device` if a generator is provided; otherwise picks CUDA if available.
      - Draws Gaussian noise, then for each column zeros out ceil(sparsity * rows) entries via randperm.
    """
    if tnsr.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    # Decide device (same rule as original)
    device = (generator.device if generator is not None
              else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

    rows, cols = tnsr.size()
    zeros_per_col = int(np.ceil(float(sparsity) * rows))

    # Local generator (optional) so we can consistently pass one to ops
    gen = generator if generator is not None else torch.Generator(device=device)

    with torch.no_grad():
        # 1) Gaussian fill on chosen device
        out = torch.normal(mean=float(mean), std=float(std),
                           size=(rows, cols), generator=gen, device=device, dtype=tnsr.dtype)

        # 2) Impose sparsity column-wise
        if zeros_per_col > 0:
            for c in range(cols):
                idx = torch.randperm(rows, generator=gen, device=device)[:zeros_per_col]
                out[idx, c] = 0
    return out


def get_connectivity(N: int,
                     num_inputs: int,
                     num_outputs: int,
                     radius: float = 1.2,
                     recurrent_density: float = 1.0,
                     input_density: float = 1.0,
                     output_density: float = 1.0,
                     generator: torch.Generator | None = None):
    """
    Unconstrained connectivity (no Dale):
      - Spectral radius of W_rec ≈ `radius`
      - Column-wise sparsity set by *_density
    Logic preserved; device-safe and compact.
    """
    device = (generator.device if generator is not None
              else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
    dtype = torch.float32

    # Gaussian params (match original scaling)
    mu_rec, std = 0.0, 1.0 / float(N)
    mu_pos = 1.0 / np.sqrt(N)

    # Recurrent weights (no self-connections)
    rec_spars = 1.0 - float(recurrent_density)
    W_rec = sparse(torch.empty(N, N, device=device, dtype=dtype),
                   rec_spars, mu_rec, std, generator)
    W_rec = W_rec - torch.diag(torch.diag(W_rec))  # remove self-connections

    # Spectral radius adjustment (robust to complex eigvals)
    eigvals = torch.linalg.eigvals(W_rec)
    spec_rad = eigvals.abs().max().real.clamp_min(1e-12)
    W_rec = (float(radius) / float(spec_rad)) * W_rec
    W_rec = W_rec.to(dtype)

    # Input / Output weights (nonnegative mean scaling like original)
    in_spars  = 1.0 - float(input_density)
    out_spars = 1.0 - float(output_density)

    W_inp = sparse(torch.empty(N, num_inputs, device=device, dtype=dtype),
                   in_spars, mu_pos, std, generator).to(dtype)
    W_out = sparse(torch.empty(num_outputs, N, device=device, dtype=dtype),
                   out_spars, mu_pos, std, generator).to(dtype)

    # Masks
    input_mask     = (W_inp != 0).to(dtype)
    output_mask    = (W_out != 0).to(dtype)
    recurrent_mask = torch.ones(N, N, device=device, dtype=dtype)  # (self-conn already zeroed)

    return W_rec, W_inp, W_out, recurrent_mask, output_mask, input_mask


def get_connectivity_Dale(N: int,
                          num_inputs: int,
                          num_outputs: int,
                          radius: float = 1.2,
                          recurrent_density: float = 1.0,
                          input_density: float = 1.0,
                          output_density: float = 1.0,
                          exc_to_inh_ratio: float = 4.0,
                          generator: torch.Generator | None = None):
    """
    Build Dale-constrained connectivity:
      - Ne excitatory (positive rows), Ni inhibitory (negative rows)
      - W_rec spectral radius ≈ `radius`
      - Column-wise sparsity for rec/in/out set by *_density
    Logic preserved; code simplified & device-safe.
    """
    device = (generator.device if generator is not None
              else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
    dtype = torch.float32

    # E/I split
    exc_frac = exc_to_inh_ratio / (exc_to_inh_ratio + 1.0)
    Ne = int(np.floor(N * exc_frac)); Ni = N - Ne

    # Gaussian params (match original scaling)
    mu_E, mu_I = 1.0 / np.sqrt(N), exc_to_inh_ratio / np.sqrt(N)
    std = 1.0 / N

    # Recurrent blocks (row-wise signs): [E | I] columns
    rec_spars = 1.0 - float(recurrent_density)
    E_E =  torch.abs(sparse(torch.empty(Ne, Ne, device=device, dtype=dtype), rec_spars, mu_E, std, generator))
    E_I = -torch.abs(sparse(torch.empty(Ne, Ni, device=device, dtype=dtype), rec_spars, mu_I, std, generator))
    I_E =  torch.abs(sparse(torch.empty(Ni, Ne, device=device, dtype=dtype), rec_spars, mu_E, std, generator))
    I_I = -torch.abs(sparse(torch.empty(Ni, Ni, device=device, dtype=dtype), rec_spars, mu_I, std, generator))
    W_rec = torch.cat([torch.cat([E_E, E_I], dim=1), torch.cat([I_E, I_I], dim=1)], dim=0)

    # No self-connections
    W_rec = W_rec - torch.diag(torch.diag(W_rec))

    # Spectral radius adjustment (handle complex eigs safely)
    eigvals = torch.linalg.eigvals(W_rec)                # (N,) complex
    spec_rad = eigvals.abs().max().real.clamp_min(1e-12)
    W_rec = (float(radius) / float(spec_rad)) * W_rec
    W_rec = W_rec.to(dtype)

    # Input weights (nonnegative)
    in_spars = 1.0 - float(input_density)
    W_inp = torch.abs(sparse(torch.empty(N, num_inputs, device=device, dtype=dtype),
                             in_spars, mu_E, std, generator)).to(dtype)

    # Output weights: only from excitatory neurons; zeros for inhibitory cols
    out_spars = 1.0 - float(output_density)
    W_out_E = torch.abs(sparse(torch.empty(num_outputs, Ne, device=device, dtype=dtype),
                               out_spars, mu_E, std, generator))
    W_out = torch.hstack([W_out_E, torch.zeros(num_outputs, Ni, device=device, dtype=dtype)]).to(dtype)

    # Masks
    dale_mask     = torch.cat([torch.ones(Ne, device=device, dtype=dtype),
                               -torch.ones(Ni, device=device, dtype=dtype)], dim=0)
    output_mask   = (W_out != 0).to(dtype)
    input_mask    = (W_inp != 0).to(dtype)
    recurrent_mask= torch.ones(N, N, device=device, dtype=dtype)  # (no self-inhibition already enforced)

    return W_rec, W_inp, W_out, recurrent_mask, dale_mask, output_mask, input_mask


'''
Continuous-time RNN class implemented in pytorch to train with BPTT
'''


class RNN_torch(torch.nn.Module):
    def __init__(self,
                 N,
                 activation_name,
                 activation_slope=1.0,
                 dt=1,
                 tau=10,
                 constrained=True,
                 exc_to_inh_ratio=1.0,
                 connectivity_density_rec=1.0,
                 spectral_rad=1.2,
                 sigma_rec=.03,
                 sigma_inp=.03,
                 gamma=0.1,
                 d=0.0,
                 y_init=None,
                 seed=None,
                 input_size=6,
                 output_size=2):
        '''
        :param N: int, number of neural nodes in the RNN
        :param activation_name: name of the activation function in the dynamics of the RNN
        :param constrained: whether the connectivity is constrained to comply with Dales law and elements of W_inp, W_out > 0
        :param connectivity_density_rec: float, defines the sparsity of the connectivity
        :param spectral_rad: float, spectral radius of the initial connectivity matrix W_rec
        :param dt: float, time resolution of RNN
        :param tau: float, internal time constant of the RNN-neural nodes
        :param sigma_rec: float, std of the gaussian noise in the recurrent dynamics
        :param sigma_inp: float, std of the gaussian noise in the input to the RNN
        :param y_init: array of N values, initial value of the RNN dynamics
        :param seed: seed for torch random generator, for reproducibility
        :param output_size: number of the output channels of the RNN
        :param device:
        '''
        super(RNN_torch, self).__init__()
        # self.device = torch.device('mps')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Using {self.device} for RNN!")
        self.N = N
        self.activation_slope = torch.tensor(activation_slope).to(self.device)
        self.activation_name = activation_name
        if activation_name == 'relu':
            self.activation = lambda x: torch.maximum(torch.tensor(0.), self.activation_slope * x)
        elif activation_name == 'tanh':
            self.activation = lambda x: torch.tanh(self.activation_slope * x)
        elif activation_name == 'sigmoid':
            self.activation = lambda x: torch.sigmoid(self.activation_slope * x)
        elif activation_name == 'softplus':
            self.activation = lambda x: torch.nn.Softplus(beta=self.activation_slope)(x)

        self.tau = tau
        self.dt = dt
        self.alpha = torch.tensor((dt / tau)).to(self.device)
        self.sigma_rec = torch.from_numpy(np.array(sigma_rec)).to(self.device)
        self.sigma_inp = torch.from_numpy(np.array(sigma_inp)).to(self.device)
        self.gamma = torch.tensor(gamma).to(self.device)
        self.d = torch.tensor(d).to(self.device)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.spectral_rad = torch.from_numpy(np.array(spectral_rad)).to(self.device)
        self.connectivity_density_rec = connectivity_density_rec
        self.constrained = constrained
        self.exc2inhR = exc_to_inh_ratio
        self.dale_mask = None
        self.output_mask = None
        self.last_dropout_mask = torch.ones(self.N).to(self.device)

        if not (y_init is None):
            self.y_init = y_init
        else:
            self.y_init = torch.zeros(self.N)

        self.random_generator = torch.Generator(device=self.device)
        if not (seed is None):
            self.random_generator.manual_seed(seed)
        else:
            seed = np.random.randint(1000000)
            self.random_generator.manual_seed(seed)


        if self.constrained:
            # imposing a bunch of constraint on the connectivity:
            # positivity of W_inp, W_out,
            # W_rec has to be subject to Dale's law
            W_rec, W_inp, W_out, self.recurrent_mask, self.dale_mask, self.output_mask, self.input_mask = \
                get_connectivity_Dale(N=self.N, num_inputs=self.input_size, num_outputs=self.output_size,
                                      radius=self.spectral_rad,
                                      exc_to_inh_ratio=self.exc2inhR,
                                      generator=self.random_generator,
                                      recurrent_density=self.connectivity_density_rec)
        else:
            W_rec, W_inp, W_out, self.recurrent_mask, self.output_mask, self.input_mask = \
                get_connectivity(N=self.N, num_inputs=self.input_size, num_outputs=self.output_size,
                                 radius=self.spectral_rad,
                                 generator=self.random_generator,
                                 recurrent_density=self.connectivity_density_rec)
        self.W_out = torch.nn.Parameter(W_out.to(self.device))
        self.W_rec = torch.nn.Parameter(W_rec.to(self.device))
        self.W_inp = torch.nn.Parameter(W_inp.to(self.device))

    def rhs(self, s, I, i_noise, r_noise, dropout_mask=None):
        s_do = s if dropout_mask is None else s * dropout_mask  # (N,B) · (N,1)
        h = self.W_rec @ s_do + self.W_inp @ (I + i_noise)
        return - (s - self.d) + self.activation(h) + r_noise - self.gamma * s ** 3

    def adversarial_dropout_mask(self, participation, drop_rate, adv=5.0,
                                 mass_preserve=True, eps=1e-8, cap=1.25):
        N = participation.numel()
        K = int(round(drop_rate * N))
        K = max(0, min(N - 1, K))
        if K == 0:
            return torch.ones(N, 1, device=participation.device, dtype=participation.dtype)
        with torch.no_grad():
            p = (participation / participation.max().clamp_min(eps)).detach()
            s = torch.softmax(adv * p, dim=0)
            idx_drop = torch.multinomial(s, num_samples=K, replacement=False)
            keep = torch.ones(N, device=p.device, dtype=p.dtype)
            keep[idx_drop] = 0.0
            if mass_preserve:
                Pk = (p * keep).sum()
                Pd = (p * (1.0 - keep)).sum()
                c = 1.0 + Pd / (Pk + eps)
                m = (keep * c).unsqueeze(-1)
            else:
                m = keep.unsqueeze(-1)
            if cap is not None:
                m = m.clamp_max(cap)
        return m

    def forward(self, u, w_noise=True, dropout=False, drop_rate=0.05, participation=None):
        T_steps, batch_size = u.shape[1], u.shape[-1]
        train_do = dropout and (drop_rate > 0)
        do_mask = self.adversarial_dropout_mask(participation, drop_rate) if train_do else None  # (N,1)

        states = torch.zeros(self.N, 1, batch_size, device=self.device)
        states[:, 0, :] = self.y_init.reshape(-1, 1).repeat(1, batch_size)

        rec_noise = torch.zeros(self.N, T_steps, batch_size, device=self.device)
        inp_noise = torch.zeros(self.input_size, T_steps, batch_size, device=self.device)
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) * torch.randn(
                *rec_noise.shape, generator=self.random_generator, device=self.device
            )
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) * torch.randn(
                *inp_noise.shape, generator=self.random_generator, device=self.device
            )

        states_list = [states[:, 0, :]]
        for t in range(1, T_steps):
            rhs_val = self.rhs(
                s=states_list[-1],
                I=u[:, t - 1, :],
                i_noise=inp_noise[:, t - 1, :],
                r_noise=rec_noise[:, t - 1, :],
                dropout_mask=do_mask
            )
            states_list.append(states_list[-1] + self.alpha * rhs_val)

        states_new = torch.stack(states_list, dim=1)  # (N, T, B)
        with torch.no_grad():
            do_mask = torch.ones(self.N, 1, 1, device=self.device) if do_mask is None else do_mask
            unscaled_do_mask = do_mask.float().clamp(0, 1).reshape(self.N, 1, 1)
        outputs = torch.einsum("oj,jtk->otk", self.W_out, states_new * unscaled_do_mask)
        return states_new, outputs

    def get_params(self):
        '''
        Save crucial parameters of the RNN as numpy arrays
        :return: parameter dictionary containing connectivity parameters, initial conditions,
         number of nodes, dt and tau
        '''
        param_dict = {}
        W_out = deepcopy(self.W_out.data.cpu().detach().numpy())
        W_rec = deepcopy(self.W_rec.data.cpu().detach().numpy())
        W_inp = deepcopy(self.W_inp.data.cpu().detach().numpy())
        y_init = deepcopy(self.y_init.detach().cpu().numpy())
        param_dict["activation_name"] = self.activation_name
        param_dict["activation_slope"] = self.activation_slope
        param_dict["W_out"] = W_out
        param_dict["W_inp"] = W_inp
        param_dict["W_rec"] = W_rec
        param_dict["y_init"] = y_init
        param_dict["N"] = self.N
        param_dict["dt"] = self.dt
        param_dict["tau"] = self.tau
        param_dict["gamma"] = self.gamma
        param_dict["d"] = self.d
        return param_dict

    def set_params(self, params):
        self.N = params["W_rec"].shape[0]
        self.W_out.data = torch.from_numpy(np.float32(params["W_out"])).to(self.device)
        self.W_inp.data = torch.from_numpy(np.float32(params["W_inp"])).to(self.device)
        self.W_rec.data = torch.from_numpy(np.float32(params["W_rec"])).to(self.device)
        self.gamma = torch.tensor(params["gamma"]).to(self.device)
        self.d = torch.tensor(params["d"]).to(self.device)
        self.y_init = torch.from_numpy(np.float32(params["y_init"])).to(self.device)
        self.activation_slope = torch.tensor(params["activation_slope"]).to(self.device)
        return None


if __name__ == '__main__':
    N = 100
    activation_name = 'tanh'
    rnn_torch = RNN_torch(N=N, activation_name=activation_name, constrained=True)
    param_dict = rnn_torch.get_params()
    print(param_dict)
