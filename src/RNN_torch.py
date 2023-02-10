import sys
sys.path.insert(0, "../")
from copy import deepcopy
import torch
import numpy as np
from numpy import linalg


# Connectivity defining methods

def sparse(tensor, sparsity, mean=0, std=1, generator=None):
    r"""Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(np.ceil(sparsity * rows))

    with torch.no_grad():
        tensor.normal_(mean, std, generator=generator)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows, generator=generator)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


def get_connectivity(device, N, num_inputs, num_outputs, radius=1.5, recurrent_density=1, input_density=1,
                     output_density=1, generator=None):
    '''
    generates W_inp, W_rec and W_out matrices of RNN, with specified parameters
    :param device: torch related: CPU or GPU
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

    # Balancing parameters
    mu = 0
    mu_pos = 1 / np.sqrt(N)
    var = 1 / N

    recurrent_sparsity = 1 - recurrent_density
    W_rec = sparse(torch.empty(N, N), recurrent_sparsity, mu, var, generator)

    # spectral radius adjustment
    W_rec = W_rec - torch.diag(torch.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius * W_rec / spec_radius

    W_inp = torch.zeros([N, num_inputs]).float()
    input_sparsity = 1 - input_density
    W_inp = sparse(W_inp, input_sparsity, mu_pos, var, generator)

    output_sparsity = 1 - output_density
    W_out = sparse(torch.empty(num_outputs, N), output_sparsity, mu_pos, var, generator)

    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_inp != 0).to(device=device).float()
    recurrent_mask = torch.ones(N, N) - torch.eye(N)
    return W_rec.to(device=device).float(), \
           W_inp.to(device=device).float(), \
           W_out.to(device=device).float(), \
           recurrent_mask.to(device=device).float(), \
           output_mask.to(device=device).float(), \
           input_mask.to(device=device).float()


def get_connectivity_Dale(device, N, num_inputs, num_outputs, radius=1.5, recurrent_density=1, input_density=1,
                          output_density=1, generator=None):
    '''
    generates W_inp, W_rec and W_out matrices of RNN, with specified parameters, subject to a Dales law,
    and about 20:80 ratio of inhibitory neurons to exchitatory ones.
    Following the paper "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks:
    A Simple and Flexible Framework" - Song et al. (2016)

    :param device: torch related: CPU or GPU
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
    Ne = int(N * 0.8)
    Ni = int(N * 0.2)

    # Initialize W_rec
    W_rec = torch.empty([0, N])

    # Balancing parameters
    mu_E = 1 / np.sqrt(N)
    mu_I = 4 / np.sqrt(N)

    var = 1 / N
    # generating excitatory part of connectivity and an inhibitory part of connectivity:
    rowE = torch.empty([Ne, 0])
    rowI = torch.empty([Ni, 0])
    recurrent_sparsity = 1 - recurrent_density
    rowE = torch.cat((rowE, torch.abs(sparse(torch.empty(Ne, Ne), recurrent_sparsity, mu_E, var, generator))), 1)
    rowE = torch.cat((rowE, -torch.abs(sparse(torch.empty(Ne, Ni), recurrent_sparsity, mu_I, var, generator))), 1)
    rowI = torch.cat((rowI, torch.abs(sparse(torch.empty(Ni, Ne), recurrent_sparsity, mu_E, var, generator))), 1)
    rowI = torch.cat((rowI, -torch.abs(sparse(torch.empty(Ni, Ni), recurrent_sparsity, mu_I, var, generator))), 1)

    W_rec = torch.cat((W_rec, rowE), 0)
    W_rec = torch.cat((W_rec, rowI), 0)

    #  spectral radius adjustment
    W_rec = W_rec - torch.diag(torch.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius * W_rec / spec_radius

    W_inp = torch.zeros([N, num_inputs]).float()
    input_sparsity = 1 - input_density
    W_inp = torch.abs(sparse(W_inp, input_sparsity, mu_E, var, generator))

    W_out = torch.zeros([num_outputs, N])
    output_sparsity = 1 - output_density
    W_out = torch.abs(sparse(W_out, output_sparsity, mu_E, var, generator))

    dale_mask = torch.sign(W_rec).to(device=device).float()
    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_inp != 0).to(device=device).float()
    recurrent_mask = torch.ones(N, N) - torch.eye(N)
    return W_rec.to(device=device).float(), W_inp.to(device=device).float(), W_out.to(
        device=device).float(), recurrent_mask.to(device=device).float(), dale_mask, output_mask, input_mask


'''
Continuous-time RNN class implemented in pytorch to train with BPTT
'''


class RNN_torch(torch.nn.Module):
    def __init__(self,
                 N,
                 activation,
                 dt=1,
                 tau=10,
                 constrained=True,
                 connectivity_density_rec=1.0,
                 spectral_rad=1.2,
                 sigma_rec=.03,
                 sigma_inp=.03,
                 bias_rec=None,
                 y_init=None,
                 random_generator=None,
                 input_size=6,
                 output_size=2,
                 device=None):
        '''
        :param N: int, number of neural nodes in the RNN
        :param activation: torch function, activation function in the dynamics of the RNN
        :param constrained: whether the connectivity is constrained to comply with Dales law and elements of W_inp, W_out > 0
        :param connectivity_density_rec: float, defines the sparcity of the connectivity
        :param spectral_rad: float, spectral radius of the initial connectivity matrix W_rec
        :param dt: float, time resolution of RNN
        :param tau: float, internal time constant of the RNN-neural nodes
        :param sigma_rec: float, std of the gaussian noise in the recurrent dynamics
        :param sigma_inp: float, std of the gaussian noise in the input to the RNN
        :param bias_rec: array of N values, (inhibition/excitation of neural nodes from outside of the network)
        :param y_init: array of N values, initial value of the RNN dynamics
        :param random_generator: torch random generator, for reproducibility
        :param output_size: number of the output channels of the RNN
        :param device:
        '''
        super(RNN_torch, self).__init__()
        self.N = N
        self.activation = activation
        self.tau = tau
        self.dt = dt
        self.alpha = (dt / tau)
        self.sigma_rec = torch.tensor(sigma_rec)
        self.sigma_inp = torch.tensor(sigma_inp)
        self.input_size = input_size
        self.output_size = output_size
        self.spectral_rad = spectral_rad
        self.connectivity_density_rec = connectivity_density_rec
        self.constrained = constrained
        self.dale_mask = None

        if not (y_init is None):
            self.y_init = y_init
        else:
            self.y_init = torch.zeros(self.N)
        # self.device = torch.device('cpu')
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        print(f"Using {self.device} for RNN!")

        self.random_generator = random_generator
        self.input_layer = (torch.nn.Linear(self.input_size, self.N, bias=False)).to(self.device)
        self.recurrent_layer = torch.nn.Linear(self.N, self.N, bias=(False if (bias_rec is None) else bias_rec)).to(
            self.device)
        self.output_layer = torch.nn.Linear(self.N, self.output_size, bias=False).to(self.device)

        if self.constrained:
            # imposing a bunch of constraint on the connectivity:
            # positivity of W_inp, W_out,
            # W_rec has to be subject to Dale's law
            W_rec, W_inp, W_out, self.recurrent_mask, self.dale_mask, self.output_mask, self.input_mask = \
                get_connectivity_Dale(device, self.N, num_inputs=self.input_size, num_outputs=self.output_size,
                                      radius=self.spectral_rad, generator=self.random_generator,
                                      recurrent_density=self.connectivity_density_rec)
        else:
            W_rec, W_inp, W_out, self.recurrent_mask, self.output_mask, self.input_mask = \
                get_connectivity(device, self.N, num_inputs=self.input_size, num_outputs=self.output_size,
                                 radius=self.spectral_rad,
                                 generator=self.random_generator,
                                 recurrent_density=self.connectivity_density_rec)
        self.output_layer.weight.data = W_out.to(self.device)
        self.input_layer.weight.data = W_inp.to(self.device)
        self.recurrent_layer.weight.data = W_rec.to(self.device)
        if bias_rec is None:
            self.recurrent_layer.bias = None

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
        inp_noise = torch.zeros(self.input_size, T_steps, batch_size)
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) \
                        * torch.randn(*rec_noise.shape, generator=self.random_generator)
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) \
                        * torch.randn(*inp_noise.shape, generator=self.random_generator)
        # passing through layers require batch-first shape!
        # that's why we need to reshape the inputs and states!
        states = torch.swapaxes(states, 0, -1)
        u = torch.swapaxes(u, 0, -1).to(self.device)
        rec_noise = torch.swapaxes(rec_noise, 0, -1).to(self.device)
        inp_noise = torch.swapaxes(inp_noise, 0, -1).to(self.device)
        for i in range(T_steps - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + \
                        self.alpha * (
                                self.activation(
                                    self.recurrent_layer(states[:, i, :]) +
                                    self.input_layer(u[:, i, :] + inp_noise[:, i, :])) +
                                rec_noise[:, i, :]
                        )
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        outputs = torch.swapaxes(self.output_layer(states), 0, -1)
        states = torch.swapaxes(states, 0, -1)
        return states, outputs

    def get_params(self):
        '''
        Save crucial parameters of the RNN as numpy arrays
        :return: parameter dictionary containing connectivity parameters, initial conditions,
         number of nodes, dt and tau
        '''
        param_dict = {}
        W_out = deepcopy(self.output_layer.weight.data.cpu().detach().numpy())
        W_rec = deepcopy(self.recurrent_layer.weight.data.cpu().detach().numpy())
        W_inp = deepcopy(self.input_layer.weight.data.cpu().detach().numpy())
        y_init = deepcopy(self.y_init.detach().cpu().numpy())
        if not (self.recurrent_layer.bias is None):
            bias_rec = deepcopy(self.recurrent_layer.bias.data.cpu().detach().numpy())
        else:
            bias_rec = None
        param_dict["W_out"] = W_out
        param_dict["W_inp"] = W_inp
        param_dict["W_rec"] = W_rec
        param_dict["bias_rec"] = bias_rec
        param_dict["y_init"] = y_init
        param_dict["N"] = self.N
        param_dict["dt"] = self.dt
        param_dict["tau"] = self.tau
        return param_dict

    def set_params(self, params):
        self.output_layer.weight.data = torch.from_numpy(params["W_out"]).to(self.device)
        self.input_layer.weight.data = torch.from_numpy(params["W_inp"]).to(self.device)
        self.recurrent_layer.weight.data = torch.from_numpy(params["W_rec"]).to(self.device)
        if not (self.recurrent_layer.bias is None):
            self.recurrent_layer.bias.data = torch.from_numpy(params["bias_rec"]).to(self.device)
        self.y_init = torch.from_numpy(params["y_init"]).to(self.device)
        return None


if __name__ == '__main__':
    N = 100
    activation = lambda x: torch.maximum(x, torch.tensor(0))
    rnn_torch = RNN_torch(N=N, activation=activation, constrained=True)
    param_dict = rnn_torch.get_params()
    print(param_dict)
