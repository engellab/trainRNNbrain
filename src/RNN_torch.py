import sys
sys.path.insert(0, "../")
from src.connectivity import get_connectivity, get_connectivity_Dale
from copy import deepcopy
import torch

'''
Continuous-time RNN class implemented in pytorch to train with BPTT
'''

class RNN_torch(torch.nn.Module):
    def __init__(self,
                 N,
                 activation,
                 dt=1,
                 tau=10,
                 constrained = True,
                 connectivity_density_rec=1.0,
                 spectral_rad=1.2,
                 sigma_rec=.03,
                 sigma_inp=.03,
                 bias_rec=None,
                 y_init= None,
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
        self.alpha = (dt/tau)
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
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.random_generator = random_generator
        self.recurrent_layer = torch.nn.Linear(self.N, self.N, bias=(False if (bias_rec is None) else bias_rec))
        self.input_layer = (torch.nn.Linear(self.input_size, self.N, bias=False))
        self.output_layer = torch.nn.Linear(self.N, self.output_size, bias=False)

        if self.constrained:
            # imposing a bunch of constraint on the connectivity:
            # positivity of W_inp, W_out,
            # W_rec has to be subject to Dale's law
            W_rec, W_inp, W_out, self.recurrent_mask, self.dale_mask, self.output_mask, self.input_mask =\
                get_connectivity_Dale(device, self.N, num_inputs=self.input_size, num_outputs=self.output_size,
                                      radius=self.spectral_rad, generator=self.random_generator,
                                      recurrent_density=self.connectivity_density_rec)
        else:
            W_rec, W_inp, W_out, self.recurrent_mask, self.output_mask, self.input_mask = \
                get_connectivity(device, self.N, num_inputs=self.input_size, num_outputs=self.output_size, radius=self.spectral_rad,
                                      generator=self.random_generator,
                                      recurrent_density=self.connectivity_density_rec)
        self.output_layer.weight.data = W_out
        self.input_layer.weight.data = W_inp
        self.recurrent_layer.weight.data = W_rec

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
        rec_noise = torch.zeros(self.N, T_steps, batch_size).to(device=self.device)
        inp_noise = torch.zeros(self.input_size, T_steps, batch_size).to(device=self.device)
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) \
                    * torch.randn(*rec_noise.shape, generator=self.random_generator).to(device=self.device)
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) \
                        * torch.randn(*inp_noise.shape, generator=self.random_generator).to(device=self.device)
        # passing thorugh layers require batch-first shape!
        # that's why we need to reshape the inputs and states!
        states = torch.swapaxes(states, 0, -1)
        u = torch.swapaxes(u, 0, -1)
        rec_noise = torch.swapaxes(rec_noise, 0, -1)
        inp_noise = torch.swapaxes(inp_noise, 0, -1)
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
        W_out = deepcopy(self.output_layer.weight.data.detach().numpy())
        W_rec = deepcopy(self.recurrent_layer.weight.data.detach().numpy())
        W_inp = deepcopy(self.input_layer.weight.data.detach().numpy())
        y_init = deepcopy(self.y_init.detach().numpy())
        if not (self.recurrent_layer.bias is None):
            bias_rec = deepcopy(self.recurrent_layer.bias.data.detach().numpy())
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
        self.output_layer.weight.data = torch.from_numpy(params["W_out"])
        self.input_layer.weight.data = torch.from_numpy(params["W_inp"])
        self.recurrent_layer.weight.data = torch.from_numpy(params["W_rec"])
        if not (self.recurrent_layer.bias is None):
            self.recurrent_layer.bias.data = torch.from_numpy(params["bias_rec"])
        self.y_init = torch.from_numpy(params["y_init"])
        return None

if __name__ == '__main__':
    N = 100
    activation = lambda x: torch.maximum(x, torch.tensor(0))
    rnn_torch = RNN_torch(N=N, activation=activation, constrained=True)
    param_dict = rnn_torch.get_params()
    print(param_dict)
