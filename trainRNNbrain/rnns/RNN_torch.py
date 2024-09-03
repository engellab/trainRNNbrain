from copy import deepcopy
import torch
import numpy as np


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


def get_connectivity(N, num_inputs, num_outputs, radius=1.2, recurrent_density=1.0, input_density=1.0,
                     output_density=1.0, generator=None):
    '''
    generates W_inp, W_rec and W_out matrices of RNN, with specified parameters
    :param N: number of neural nodes
    :param num_inputs: number of input channels, input dimension
    :param num_outputs: number of output channels, output dimension
    :param radius: spectral radius of the generated cnnectivity matrix: controls the maximal abs value of eigenvectors.
    the greater the parameter is the more sustained and chaotic activity the network exchibits, the lower - the quicker
    the network relaxes back to zero.
    :param recurrent_density: oppposite of sparcirty of the reccurrent matrix. 1.0 - fully connected recurrent matrix
    :param input_density: 1.0 - fully connected input matrix, 0 - maximally sparce matrix
    :param output_density: 1.0 - fully connected output matrix, 0 - maximally sparce matrix
    :param generator: torch random generator, for reproducibility
    :return:
    '''
    if not (generator is None):
        device = generator.device
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    # Balancing parameters
    mu = 0
    mu_pos = 1 / np.sqrt(N)
    var = 1 / N
    recurrent_sparsity = 1 - recurrent_density
    W_rec = sparse(torch.empty(N, N, device=device), recurrent_sparsity, mu, var, generator)

    # spectral radius adjustment
    W_rec = W_rec - torch.diag(torch.diag(W_rec))
    w, v = torch.linalg.eig(W_rec)
    spec_radius = torch.max(torch.absolute(w))
    # W_rec = torch.tensor(radius).to(device=device) * W_rec.to(device=device) / spec_radius.to(device=device)
    W_rec = torch.tensor(radius/spec_radius).to(device=device) * W_rec.to(device=device)
    W_inp = torch.zeros([N, num_inputs], device=device).float()
    input_sparsity = 1 - input_density
    W_inp = sparse(W_inp, input_sparsity, mu_pos, var, generator)

    output_sparsity = 1 - output_density
    W_out = sparse(torch.empty(num_outputs, N), output_sparsity, mu_pos, var, generator)

    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_inp != 0).to(device=device).float()
    # recurrent_mask = torch.ones(N, N) - torch.eye(N)
    recurrent_mask = torch.ones(N, N)# - torch.eye(N)

    return W_rec.to(device=device).float(), \
           W_inp.to(device=device).float(), \
           W_out.to(device=device).float(), \
           recurrent_mask.to(device=device).float(), \
           output_mask.to(device=device).float(), \
           input_mask.to(device=device).float()


def get_connectivity_Dale(N, num_inputs, num_outputs, radius=1.5, recurrent_density=1.0, input_density=1.0,
                          output_density=1.0, exc_to_inh_ratio=4, generator=None):
    '''
    generates W_inp, W_rec and W_out matrices of RNN, with specified parameters, subject to a Dales law,
    and about 20:80 ratio of inhibitory neurons to exchitatory ones.
    Following the paper "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive tasks:
    A Simple and Flexible Framework" - Song et al. (2016)

    :param N: number of neural nodes
    :param num_inputs: number of input channels, input dimension
    :param num_outputs: number of output channels, output dimension
    :param radius: spectral radius of the generated cnnectivity matrix: controls the maximal abs value of eigenvectors.
    the greater the parameter is the more sustained and chaotic activity the network exchibits, the lower - the quicker
    the network relaxes back to zero.
    :param recurrent_density: oppposite of sparcirty of the reccurrent matrix. 1.0 - fully connected recurrent matrix
    :param input_density: 1.0 - fully connected input matrix, 0 - maximally sparce matrix
    :param output_density: 1.0 - fully connected output matrix, 0 - maximally sparce matrix
    :param generator: torch random generator, for reproducibility
    :return:
    '''
    if not (generator is None):
        device = generator.device
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')


    exc_percentage = exc_to_inh_ratio / (exc_to_inh_ratio + 1)
    Ne = int(np.ceil(N * exc_percentage))
    Ni = N - Ne

    # Initialize W_rec
    W_rec = torch.empty([0, N], device=device)

    # Balancing parameters
    mu_E = 1 / np.sqrt(N)
    mu_I = exc_to_inh_ratio / np.sqrt(N)

    var = 1 / N
    # generating excitatory part of connectivity and an inhibitory part of connectivity:
    rowE = torch.empty([Ne, 0], device=device)
    rowI = torch.empty([Ni, 0], device=device)
    recurrent_sparsity = 1 - recurrent_density
    rowE = torch.cat((rowE, torch.abs(sparse(torch.empty(Ne, Ne, device=device), recurrent_sparsity, mu_E, var, generator))), 1)
    rowE = torch.cat((rowE, -torch.abs(sparse(torch.empty(Ne, Ni, device=device), recurrent_sparsity, mu_I, var, generator))), 1)
    rowI = torch.cat((rowI, torch.abs(sparse(torch.empty(Ni, Ne, device=device), recurrent_sparsity, mu_E, var, generator))), 1)
    rowI = torch.cat((rowI, -torch.abs(sparse(torch.empty(Ni, Ni, device=device), recurrent_sparsity, mu_I, var, generator))), 1)

    W_rec = torch.cat((W_rec, rowE), 0)
    W_rec = torch.cat((W_rec, rowI), 0)

    #  spectral radius adjustment
    W_rec = W_rec - torch.diag(torch.diag(W_rec))
    w, v = torch.linalg.eig(W_rec)
    spec_radius = torch.max(torch.absolute(w))
    # W_rec = torch.tensor(radius).to(device=device) * W_rec.to(device=device) / spec_radius.to(device=device)
    W_rec = torch.tensor(radius / spec_radius).to(device=device) * W_rec.to(device=device)
    W_rec = W_rec.float()

    W_inp = torch.zeros([N, num_inputs]).float()
    input_sparsity = 1 - input_density
    W_inp = torch.abs(sparse(W_inp, input_sparsity, mu_E, var, generator)).float()

    W_out = torch.zeros([num_outputs, Ne]).float()
    output_sparsity = 1 - output_density
    W_out = torch.abs(sparse(W_out, output_sparsity, mu_E, var, generator))
    W_out = torch.hstack([W_out, torch.zeros([num_outputs, Ni], device=device)]).float()

    dale_mask = torch.cat([torch.ones(Ne), -torch.ones(Ni)])
    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_inp != 0).to(device=device).float()
    # No self connectivity constraint
    # recurrent_mask = torch.ones(N, N) - torch.eye(N)
    recurrent_mask = torch.ones(N, N)# - torch.eye(N)
    return W_rec, W_inp, W_out, recurrent_mask.to(device=device).float(), dale_mask, output_mask, input_mask


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
                 y_init=None,
                 seed=None,
                 input_size=6,
                 output_size=2):
        '''
        :param N: int, number of neural nodes in the RNN
        :param activation_name: name of the activation function in the dynamics of the RNN
        :param constrained: whether the connectivity is constrained to comply with Dales law and elements of W_inp, W_out > 0
        :param connectivity_density_rec: float, defines the sparcity of the connectivity
        :param spectral_rad: float, spectral radius of the initial connectivity matrix W_rec
        :param dt: float, time resolution of RNN
        :param tau: float, internal time constant of the RNN-neural nodes
        :param sigma_rec: float, std of the gaussian noise in the recurrent dynamics
        :param sigma_inp: float, std of the gaussian noise in the input to the RNN
        :param bias_rec: array of N values, (inhibition/excitation of neural nodes from outside of the network)
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
        self.input_size = torch.from_numpy(np.array(input_size)).to(self.device)
        self.output_size = torch.from_numpy(np.array(output_size)).to(self.device)
        self.spectral_rad = torch.from_numpy(np.array(spectral_rad)).to(self.device)
        self.connectivity_density_rec = connectivity_density_rec
        self.constrained = constrained
        self.exc_to_inh_ratio = exc_to_inh_ratio
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


        if self.constrained:
            # imposing a bunch of constraint on the connectivity:
            # positivity of W_inp, W_out,
            # W_rec has to be subject to Dale's law
            W_rec, W_inp, W_out, self.recurrent_mask, self.dale_mask, self.output_mask, self.input_mask = \
                get_connectivity_Dale(N=self.N, num_inputs=self.input_size, num_outputs=self.output_size,
                                      radius=self.spectral_rad, exc_to_inh_ratio=self.exc_to_inh_ratio,
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

    def rhs(self, s, I, i_noise, r_noise):
        return -s + self.activation(self.W_rec @ s + self.W_inp @ (I + i_noise)) + r_noise

    def forward(self, u, w_noise=True):
        '''
        forward dynamics of the RNN (full trial)
        :param u: array of input vectors (self.input_size, T_steps, batch_size)
        :param w_noise: bool, pass forward with or without noise
        :return: the full history of the internal variables and the outputs
        '''
        T_steps = u.shape[1]
        batch_size = u.shape[-1]
        states = torch.zeros(self.N, 1, batch_size, device=self.device)
        states[:, 0, :] = deepcopy(self.y_init).reshape(-1, 1).repeat(1, batch_size)
        rec_noise = torch.zeros(self.N, T_steps, batch_size, device=self.device)
        inp_noise = torch.zeros(self.input_size, T_steps, batch_size, device=self.device)
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) \
                        * torch.randn(*rec_noise.shape, generator=self.random_generator, device=self.device)
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) \
                        * torch.randn(*inp_noise.shape, generator=self.random_generator, device=self.device)

        for i in range(T_steps - 1):
            state_new = states[:, i, :] + self.alpha * self.rhs(states[:, i, :],
                                                                u[:, i, :],
                                                                inp_noise[:, i, :],
                                                                rec_noise[:, i, :])
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        outputs = torch.einsum("oj, jtk -> otk", self.W_out, states)
        return states, outputs

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
        return param_dict

    def set_params(self, params):
        self.N = params["W_rec"].shape[0]
        self.W_out.data = torch.from_numpy(np.float32(params["W_out"])).to(self.device)
        self.W_inp.data = torch.from_numpy(np.float32(params["W_inp"])).to(self.device)
        self.W_rec.data = torch.from_numpy(np.float32(params["W_rec"])).to(self.device)
        self.y_init = torch.from_numpy(np.float32(params["y_init"])).to(self.device)
        self.activation_slope = torch.tensor(params["activation_slope"]).to(self.device)
        return None


if __name__ == '__main__':
    N = 100
    activation_name = 'tanh'
    rnn_torch = RNN_torch(N=N, activation_name=activation_name, constrained=True)
    param_dict = rnn_torch.get_params()
    print(param_dict)
