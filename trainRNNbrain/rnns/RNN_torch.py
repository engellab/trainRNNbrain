from copy import deepcopy
import torch
import numpy as np

'''
forward pass return 3 arguments instead of 2
'''

# Connectivity defining methods
def sparse(tnsr, sparsity, mean=0.0, std=1.0, generator=None):
    if tnsr.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    if not (generator is None):
        device = generator.device
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    rows, cols = tnsr.shape
    num_zeros = int(np.ceil(sparsity * rows))

    with torch.no_grad():
        device_cpu = torch.device('cpu')
        m = torch.from_numpy(np.array(mean)).to(device_cpu)
        std = torch.from_numpy(np.array(std)).to(device_cpu)
        (rows, cols) = (torch.tensor(rows).to(device_cpu), torch.tensor(cols).to(device_cpu))
        # it seems it can only use generator on CPU?
        generator_cpu = torch.Generator(device=torch.device(device_cpu))
        if not (generator is None):
            generator_cpu.manual_seed(int(generator.initial_seed()))
        else:
            generator_cpu.manual_seed(np.random.randint(100000))
        # first create tensor on CPU then move it to gpu
        tnsr = torch.normal(m, std, (rows, cols), generator=generator_cpu).to(device)

        for col_idx in range(cols):
            row_indices = torch.randperm(rows, generator=generator, device=device)
            zero_indices = row_indices[:num_zeros]
            tnsr[zero_indices, col_idx] = 0
    return tnsr


def get_connectivity_Dale(N, num_inputs, num_outputs, radius=1.5, recurrent_density=1.0, input_density=1.0,
                          output_density=1.0, exc2inhR=4, generator=None):
    device = generator.device if generator is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    exc_percentage = exc2inhR / (exc2inhR + 1)
    Ne = int(np.floor(N * exc_percentage))
    Ni = N - Ne

    mu_E = 1 / np.sqrt(N)
    mu_I = exc2inhR / np.sqrt(N)
    std = 1 / N
    recurrent_sparsity = 1 - recurrent_density

    rowE = torch.cat([
        torch.abs(sparse(torch.empty(Ne, Ne, device=device), recurrent_sparsity, mu_E, std, generator)),
        -torch.abs(sparse(torch.empty(Ne, Ni, device=device), recurrent_sparsity, mu_I, std, generator))
    ], dim=1)
    rowI = torch.cat([
        torch.abs(sparse(torch.empty(Ni, Ne, device=device), recurrent_sparsity, mu_E, std, generator)),
        -torch.abs(sparse(torch.empty(Ni, Ni, device=device), recurrent_sparsity, mu_I, std, generator))
    ], dim=1)

    W_rec = torch.cat([rowE, rowI], dim=0)
    W_rec = W_rec - torch.diag(torch.diag(W_rec))

    w = torch.linalg.eigvals(W_rec)
    spec_radius = torch.max(torch.abs(w))
    W_rec = (radius / (spec_radius + 1e-12)) * W_rec
    W_rec = W_rec.float()

    input_sparsity = 1 - input_density
    W_inp = torch.abs(sparse(torch.zeros(N, num_inputs, device=device), input_sparsity, mu_E, std, generator)).float()

    output_sparsity = 1 - output_density
    W_out = torch.abs(sparse(torch.zeros(num_outputs, Ne, device=device), output_sparsity, mu_E, std, generator))
    W_out = torch.hstack([W_out, torch.zeros(num_outputs, Ni, device=device)]).float()

    dale_mask = torch.cat([torch.ones(Ne, device=device), -torch.ones(Ni, device=device)])
    output_mask = (W_out != 0).float()
    input_mask = (W_inp != 0).float()
    recurrent_mask = (torch.ones(N, N, device=device) - torch.eye(N, device=device)).float()

    return W_rec, W_inp, W_out, recurrent_mask, dale_mask, output_mask, input_mask

'''
Continuous-time RNN class implemented in pytorch to train with BPTT
'''


class RNN_torch(torch.nn.Module):
    def __init__(self,
                 N,
                 activation_args={"name": "relu", "slope": 1.0},
                 dt=1,
                 tau=10,
                 exc2inhR=4.0,
                 connectivity_density_rec=1.0,
                 spectral_rad=1.2,
                 sigma_rec=.03,
                 sigma_inp=.03,
                 gamma=0.1,
                 bias_init_amp=0.0,
                 y_init=None,
                 seed=None,
                 input_size=6,
                 output_size=2):
        '''
        :param N: int, number of neural nodes in the RNN
        :param activation_args: dictionary containing the name and parameters of the activation function in the dynamics of the RNN
        :param connectivity_density_rec: float, defines the sparcity of the connectivity
        :param spectral_rad: float, spectral radius of the initial connectivity matrix W_rec
        :param dt: float, time resolution of RNN
        :param tau: float, internal time constant of the RNN-neural nodes
        :param exc2inhR: float, ratio of excitatory to inhibitory recurrent connections
        :param bias_init_amp: float, amplitude of the initial bias vector
        :param gamma: float, coefficient of the cubic nonlinearity in the RNN dynamics
        :param sigma_rec: float, std of the gaussian noise in the recurrent dynamics
        :param sigma_inp: float, std of the gaussian noise in the input to the RNN
        :param y_init: array of N values, initial value of the RNN dynamics
        :param seed: seed for torch random generator, for reproducibility
        :param input_size: number of the input channels of the RNN
        :param output_size: number of the output channels of the RNN
        '''
        super(RNN_torch, self).__init__()
        # self.device = torch.device('mps')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Using {self.device} for RNN!")
        self.N = N
        self.activation_args = activation_args

        self.activation = self.configure_activation_(activation_args)
        self.tau = tau
        self.dt = dt
        self.alpha = torch.tensor((dt / tau)).to(self.device)
        self.sigma_rec = torch.from_numpy(np.array(sigma_rec)).to(self.device)
        self.sigma_inp = torch.from_numpy(np.array(sigma_inp)).to(self.device)
        self.input_size = torch.from_numpy(np.array(input_size)).to(self.device)
        self.output_size = torch.from_numpy(np.array(output_size)).to(self.device)
        self.spectral_rad = torch.from_numpy(np.array(spectral_rad)).to(self.device)
        self.bias_init_amp = torch.tensor(bias_init_amp).to(self.device)
        self.connectivity_density_rec = connectivity_density_rec
        self.exc2inhR = exc2inhR
        self.gamma = gamma
        self.dale_mask = None
        self.output_mask = None

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


        # imposing a bunch of constraint on the connectivity:
        # positivity of W_inp, W_out,
        # W_rec has to be subject to Dale's law
        W_rec, W_inp, W_out, self.recurrent_mask, self.dale_mask, self.output_mask, self.input_mask = \
            get_connectivity_Dale(N=self.N, num_inputs=self.input_size, num_outputs=self.output_size,
                                    radius=self.spectral_rad,
                                    exc2inhR=self.exc2inhR,
                                    generator=self.random_generator,
                                    recurrent_density=self.connectivity_density_rec)
        if self.bias_init_amp not in (None, 0):
            b = self.bias_init_amp * torch.rand(self.N, device=self.device, generator=self.random_generator)
            self.bias = torch.nn.Parameter(b)
        self.W_out = torch.nn.Parameter(W_out.to(self.device))
        self.W_rec = torch.nn.Parameter(W_rec.to(self.device))
        self.W_inp = torch.nn.Parameter(W_inp.to(self.device))

    @staticmethod
    def configure_activation_(activation_args):
        activation_name = activation_args["name"]
        if activation_name == 'relu':
            return lambda x: torch.maximum(torch.tensor(0.), activation_args["slope"] * x)
        elif activation_name == 'leaky_relu':
            return lambda x: torch.maximum(torch.tensor(0.), activation_args["slope"] * x) + torch.minimum(torch.tensor(0.), activation_args["leak_slope"] * x)
        elif activation_name == 'tanh':
            return lambda x: torch.tanh(activation_args["slope"] * x)
        elif activation_name == 'sigmoid':
            return lambda x: torch.sigmoid(activation_args["slope"] * x)
        elif activation_name == 'softplus':
            return lambda x: torch.nn.Softplus(beta=activation_args["beta"])(activation_args["slope"] * x)
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def get_dropout_mask(self, dropout_args, participation=None):
        drop_rate = dropout_args.get("drop_rate", 0.0)
        beta = dropout_args.get("dropout_beta", 1.0)

        sm = dropout_args["sampling_method"]
        if sm == "uniform":
            v = torch.ones(self.N, device=self.device)
        elif sm == "participation":
            if participation is None:
                raise ValueError("sampling_method='participation' requires participation")
            v = participation.reshape(self.N).to(self.device).float()
        elif sm == "output_weights":
            v = torch.sum(torch.abs(self.W_out), dim=0)

        w = torch.softmax(beta * v, dim=0)
        p_drop = torch.clamp(drop_rate * self.N * w, 0.0, 0.999)
        keep_p = 1.0 - p_drop

        return torch.bernoulli(keep_p.reshape(self.N, 1), generator=self.random_generator)

    def rhs(self, s, I, i_noise, r_noise, dropout_mask=None, dropout_kind=None):
        b = 0.0 if (self.bias_init_amp is None) or (torch.abs(self.bias_init_amp) < 1e-8) else self.bias.unsqueeze(1).expand(-1, s.shape[1])
        if dropout_mask is not None and dropout_kind == "dead":
            m = (dropout_mask > 0).view(-1, 1).to(s)  # force 0/1 for dead
            h = self.W_rec @ (s * m) + self.W_inp @ (I + i_noise) + b
            a = self.activation(h) * m
            rn = r_noise * m
        else:
            h = self.W_rec @ s + self.W_inp @ (I + i_noise) + b
            a = self.activation(h)
            rn = r_noise
        return -s + a + rn - self.gamma * torch.pow(s, 3)

    def forward(self, u, w_noise=True, dropout=False, dropout_args=None, participation=None):
        if dropout_args is None:
            dropout_args = {"dropout_kind": None, "sampling_method": None, "drop_rate": 0.0, "dropout_beta": 1.0}
        T_steps = u.shape[1]
        batch_size = u.shape[-1]
        dk = dropout_args.get("dropout_kind", None)
        sm = dropout_args.get("sampling_method", None)
        part = participation if sm == "participation" else None

        dropout_mask = None
        if dropout and dk in ("dead", "mute"):
            dropout_mask = self.get_dropout_mask(dropout_args, part)

        states = torch.zeros(self.N, 1, batch_size, device=self.device)
        states[:, 0, :] = self.y_init.reshape(-1, 1).repeat(1, batch_size)

        rec_noise = torch.zeros(self.N, T_steps, batch_size, device=self.device)
        inp_noise = torch.zeros(self.input_size, T_steps, batch_size, device=self.device)
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) * \
                        torch.randn(*rec_noise.shape, generator=self.random_generator, device=self.device)
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) * \
                        torch.randn(*inp_noise.shape, generator=self.random_generator, device=self.device)

        states_list = [states[:, 0, :]]

        for t in range(1, T_steps):
            rhs_val = self.rhs(
                s=states_list[-1],
                I=u[:, t - 1, :],
                i_noise=inp_noise[:, t - 1, :],
                r_noise=rec_noise[:, t - 1, :],
                dropout_kind=dk,
                dropout_mask=dropout_mask
            )
            next_state = states_list[-1] + self.alpha * rhs_val
            states_list.append(next_state)

        states_new = torch.stack(states_list, dim=1)  # (N, T, B)
        if dropout_mask is None:
            W_out = self.W_out
        elif dk == "mute" or dk == "dead":
            W_out = self.W_out * (dropout_mask > 0).view(1, -1)    # binary mask
        else:
            W_out = self.W_out
        outputs = torch.einsum("oj,jtk->otk", W_out, states_new)
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
        if self.bias_init_amp != 0.0 and not (self.bias_init_amp is None):
            bias = deepcopy(self.bias.detach().cpu().numpy())
        else:
            bias = None
        param_dict["activation_args"] = self.activation_args
        param_dict["W_out"] = W_out
        param_dict["W_inp"] = W_inp
        param_dict["W_rec"] = W_rec
        param_dict["y_init"] = y_init
        param_dict["bias"] = bias
        param_dict["N"] = self.N
        param_dict["dt"] = self.dt
        param_dict["tau"] = self.tau
        param_dict["gamma"] = self.gamma
        param_dict["dale_mask"] = None if self.dale_mask is None else self.dale_mask.detach().cpu().numpy()
        param_dict["input_mask"] = self.input_mask.detach().cpu().numpy()
        param_dict["recurrent_mask"] = self.recurrent_mask.detach().cpu().numpy()
        param_dict["output_mask"] = self.output_mask.detach().cpu().numpy()
        return param_dict

    def set_params(self, params):
        self.N = params["W_rec"].shape[0]
        self.W_out.data = torch.from_numpy(np.float32(params["W_out"])).to(self.device)
        self.W_inp.data = torch.from_numpy(np.float32(params["W_inp"])).to(self.device)
        self.W_rec.data = torch.from_numpy(np.float32(params["W_rec"])).to(self.device)
        self.y_init = torch.from_numpy(np.float32(params["y_init"])).to(self.device)
        self.gamma = torch.tensor(params["gamma"]).to(self.device)
        self.dt = torch.tensor(params["dt"]).to(self.device)
        self.tau = torch.tensor(params["tau"]).to(self.device)
        self.alpha = torch.tensor((self.dt / self.tau)).to(self.device)
        if not (params["bias"] is None):
            self.bias = torch.from_numpy(np.float32(params["bias"])).to(self.device)
        self.activation_args = params["activation_args"]
        self.activation = self.configure_activation_(self.activation_args)
        self.dale_mask = None if params["dale_mask"] is None else torch.from_numpy(np.float32(params["dale_mask"])).to(self.device)
        self.input_mask = torch.from_numpy(np.float32(params["input_mask"])).to(self.device)
        self.recurrent_mask = torch.from_numpy(np.float32(params["recurrent_mask"])).to(self.device)
        self.output_mask = torch.from_numpy(np.float32(params["output_mask"])).to(self.device)
        return None


if __name__ == '__main__':
    N = 100
    activation_name = 'tanh'
    rnn_torch = RNN_torch(N=N, activation_args={"name": activation_name, "slope": 1.0}, bias_init_amp=0.1)
    param_dict = rnn_torch.get_params()
    print(param_dict)
