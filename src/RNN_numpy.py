import sys

sys.path.insert(0, "../")
import numpy as np
from copy import deepcopy
import numdifftools as nd

'''
lightweight numpy implementation of RNN for validation and quick testing and plotting
'''


def ReLU(x):
    return np.maximum(x, 0)


class RNN_numpy():
    def __init__(self, N, dt, tau, W_inp, W_rec, W_out, bias_rec=None, activation=ReLU, y_init=None):
        self.N = N
        self.W_inp = W_inp
        self.W_rec = W_rec
        self.W_out = W_out
        if bias_rec is None:
            self.bias_rec = np.zeros(self.N)
        else:
            self.bias_rec = bias_rec
        self.dt = dt
        self.tau = tau
        self.alpha = self.dt / self.tau
        if not (y_init is None):
            self.y_init = y_init
        else:
            self.y_init = np.zeros(self.N)
        self.y = deepcopy(self.y_init)
        self.y_history = []
        self.activation = activation

    def rhs(self, y, input, sigma_rec=None, sigma_inp=None, generator_numpy=None):
        if (generator_numpy is None): generator_numpy = np.random.default_rng(np.random.randint(10000))
        if len(y.shape) == 2:
            # Check that the batch_size (last dimension) is the same as the Input's last dimension
            if y.shape[-1] != input.shape[-1]:
                raise ValueError(
                    f"The last dimension of the RNN state and the Input (representing batch size) should be equal!" +
                    f" {x.shape[-1]} != {input.shape[-1]}")
            batch_size = y.shape[-1]
            bias_rec = np.repeat(self.bias_rec[:, np.newaxis], repeats=batch_size, axis=-1)
        else:
            bias_rec = self.bias_rec

        if ((sigma_rec is None) and (sigma_inp is None)) or ((sigma_rec == 0) and (sigma_inp == 0)):
            return -y + self.activation(self.W_rec @ y + self.W_inp @ input + self.bias_rec)
        else:
            rec_noise_term = np.sqrt((2 / self.alpha) * sigma_rec ** 2) * generator_numpy.standard_normal(y.shape) \
                if (not (sigma_rec is None)) else np.zeros(x.shape)
            inp_noise_term = np.sqrt((2 / self.alpha) * sigma_inp ** 2) * generator_numpy.standard_normal(input.shape) \
                if (not (sigma_inp is None)) else np.zeros(input.shape)
            return -y + self.activation(
                self.W_rec @ y + self.W_inp @ (input + inp_noise_term) + bias_rec + rec_noise_term)

    def rhs_noisless(self, y, input):
        '''
        Bare version of RHS for efficient fixed point analysis
        supposed to work only with one point at the state-space at the time (no batches!)
        '''
        return -y + self.activation(self.W_rec @ y + self.W_inp @ input + self.bias_rec)

    def rhs_jac(self, y, input):
        # efficient calculation of Jacobian using a finite difference (lesser hustle than with autograd!)
        return nd.Jacobian(self.rhs_noisless)(y, input)

    def step(self, input, sigma_rec=None, sigma_inp=None, generator_numpy=None):
        self.y += (self.dt / self.tau) * self.rhs(self.y, input, sigma_rec, sigma_inp, generator_numpy)

    def run(self, input_timeseries, save_history=False, sigma_rec=None, sigma_inp=None, generator_numpy=None):
        '''
        :param Inputs: an array, has to be iether (n_inputs x n_steps) dimensions or (n_inputs x n_steps x batch_batch_size)
        :param save_history: bool, whether to save the resulting trajectory
        :param sigma_rec: noise parameter in the recurrent dynamics
        :param sigma_inp: noise parameter in the input channel
        :param generator_numpy: numpy random number generator, for reproducibility
        :return: None
        '''
        num_steps = input_timeseries.shape[1]  # second dimension
        if len(input_timeseries.shape) == 3:
            batch_size = input_timeseries.shape[-1]  # last dimension
            # if the state is a 1D vector, repeat it batch_size number of times to match with Input dimension
            if len(self.y.shape) == 1:
                self.y = np.repeat(deepcopy(self.y)[:, np.newaxis], axis=1, repeats=batch_size)
        for i in range(num_steps):
            if save_history == True:
                self.y_history.append(deepcopy(self.y))
            self.step(input_timeseries[:, i, ...], sigma_rec=sigma_rec, sigma_inp=sigma_inp,
                      generator_numpy=generator_numpy)
        return None

    def get_history(self):
        return np.array(self.y_history)

    def clear_history(self):
        self.y_history = []

    def reset_state(self):
        self.y = np.zeros(self.N)

    def get_output(self):
        y_history = np.stack(self.y_history, axis=0)
        output = np.swapaxes((self.W_out @ y_history), 0, 1)
        return output

    def run_multiple_trajectories(self, input_timeseries, sigma_rec, sigma_inp, generator_numpy=None):
        if type(input_timeseries) == list:
            input_timeseries = np.array(input_timeseries)
        # Check that inputs have 3 dimensions: n_inputs x n_steps x batch_size
        if len(input_timeseries.shape) != 3:
            raise ValueError("Inputs dimension have to be n_inputs x n_steps x batch_size!")
        self.clear_history()
        batch_size = input_timeseries.shape[-1]
        self.y = np.repeat(deepcopy(self.y_init)[:, np.newaxis], axis=-1, repeats=batch_size)
        self.run(input_timeseries=input_timeseries,
                 sigma_rec=sigma_rec,
                 sigma_inp=sigma_inp,
                 save_history=True,
                 generator_numpy=generator_numpy)
        trajectories = np.stack(self.y_history, axis=1)  # N x n_steps x batch_size
        outputs = self.get_output()  # n_outputs x n_steps x batch_size
        return trajectories, outputs


if __name__ == '__main__':
    N = 100
    x = np.random.randn(N)
    W_rec = np.random.randn(N, N)
    W_inp = np.random.randn(N, 6)
    W_out = np.random.randn(2, N)
    bias_rec = np.random.randn(N)

    # Input = np.ones(6)
    dt = 0.1
    tau = 10
    batch_size = 1
    input = np.ones((6, batch_size))
    activation_fun = np.tanh
    rnn = RNN_numpy(N=N, W_rec=W_rec, W_inp=W_inp, W_out=W_out, dt=dt, tau=tau, activation=activation_fun)
    rnn.y = np.random.randn(N)
    print(rnn.rhs_noisless(rnn.y, input=input).shape)
    print((rnn.rhs_jac(rnn.y, input=input)))
    # generate multidimensional inputs
    # input_timeseries = 0.1 * np.ones((6, 301, 32))
    # rnn.run(input_timeseries=input_timeseries, sigma_rec=0.03, sigma_inp=0.03)
    # trajectories, outputs = rnn.run_multiple_trajectories(input_timeseries=input_timeseries,
    #                                                       sigma_rec=0.03,
    #                                                       sigma_inp=0.03)
    # print(trajectories.shape)
    # print(outputs.shape)
