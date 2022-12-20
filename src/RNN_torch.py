import sys
sys.path.insert(0, "../")
from src.connectivity import get_connectivity
from copy import deepcopy
import torch

'''
Continuous-time RNN class implemented in pytorch to train with BPTT
'''

class RNN_torch(torch.nn.Module):
    def __init__(self,
                 N,
                 connectivity_density_rec=0.2,
                 spectral_rad=1.2,
                 lambda_o=0.0,
                 dt=1,
                 tau=10,
                 sigma_rec=.05,
                 sigma_inp=.01,
                 bias_rec=None,
                 bias_out=None,
                 y_init= None,
                 random_generator=None,
                 output_size=1,
                 device=None):
        '''
        :param N: int, number of neural nodes in the RNN
        :param connectivity_density_rec: float, defines the sparcity of the connectivity
        :param spectral_rad: float, spectral radius of the initial connectivity matrix W_rec
        :param lambda_o: float, regularization softly imposing a pair-wise orthogonality
         on columns of W_inp and rows of W_out
        :param dt: float, time resolution of RNN
        :param tau: float, internal time constant of the RNN-neural nodes
        :param sigma_rec: float, std of the gaussian noise in the recurrent dynamics
        :param sigma_inp: float, std of the gaussian noise in the input to the RNN
        :param bias_rec: array of N values, (inhibition/excitation of neural nodes from outside of the network)
        :param bias_out: float, bias for the output
        :param y_init: array of N values, initial value of the RNN dynamics
        :param random_generator: torch random generator, for reproducibility
        :param output_size: number of the output channels of the RNN
        :param device:
        '''
        super(RNN_torch, self).__init__()
        self.N = N
        self.tau = tau
        self.dt = dt
        self.alpha = (dt/tau)
        self.sigma_rec = torch.tensor(sigma_rec)
        self.sigma_inp = torch.tensor(sigma_inp)
        self.input_size = 6 # [context motion, context color, motion right, motion left, color right, color left]
        self.output_size = output_size
        self.spectral_rad = spectral_rad
        self.connectivity_density_rec = connectivity_density_rec
        self.lambda_o = lambda_o
        self.activation = torch.nn.ReLU()
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
        self.output_layer = torch.nn.Linear(self.N, self.output_size,
                                            bias=(False if (bias_out is None) else bias_out))

        try:
            # imposing a bunch of constraint on the connectivity:
            # positivity of W_inp, W_out,
            # W_rec has to be subject to Dale's law
            W_rec, W_inp, W_out, self.recurrent_mask, self.dale_mask, self.output_mask, self.input_mask =\
                get_connectivity(device, self.N, num_outputs=self.output_size, radius=self.spectral_rad, generator=self.random_generator,
                                 recurrent_density=self.connectivity_density_rec)
            self.output_layer.weight.data = W_out
            self.input_layer.weight.data = W_inp
            self.recurrent_layer.weight.data = W_rec

            if bias_rec is None:
                self.recurrent_layer.bias = None
            if bias_out is None:
                self.output_layer.bias = None
        except RuntimeError:
            pass


    def forward(self, u, w_noise=True):
        '''
        forward dynamics of the RNN (full trial)
        :param u: array of input vectors (batch_size, T_steps, 6)
        :param w_noise: bool, pass forward with or without noise
        :return: the full history of the internal variables and the outputs
        '''
        batch_size = u.shape[0]
        T_steps = u.shape[1]
        states = torch.zeros(batch_size, 1, self.N, device=self.device)
        states[:, 0, :] = self.y_init
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) \
                    * torch.randn(batch_size, T_steps, self.N, generator=self.random_generator).to(device=self.device)
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) \
                        * torch.randn(batch_size, T_steps, 6, generator=self.random_generator).to(device=self.device)
        else:
            rec_noise = torch.zeros(batch_size, T_steps, self.N).to(device=self.device)
            inp_noise = torch.zeros(batch_size, T_steps, 6).to(device=self.device)

        for i in range(T_steps - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + \
                        self.alpha * (
                            self.activation(
                                self.recurrent_layer(states[:, i, :]) +
                                self.input_layer(u[:, i, :] + inp_noise[:, i, :])) +
                                rec_noise[:, i, :]
                        )
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states, self.output_layer(states)

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
        if not (self.output_layer.bias is None):
            bias_out = deepcopy(self.output_layer.bias.data.detach().numpy())
        else:
            bias_out = None
        param_dict["W_out"] = W_out
        param_dict["W_inp"] = W_inp
        param_dict["W_rec"] = W_rec
        param_dict["bias_out"] = bias_out
        param_dict["bias_rec"] = bias_rec
        param_dict["y_init"] = y_init
        param_dict["N"] = self.N
        param_dict["dt"] = self.dt
        param_dict["tau"] = self.tau
        return param_dict

