import numpy as np
from copy import deepcopy
import numdifftools as nd

'''
lightweight numpy implementation of RNN for validation and quick testing and plotting
'''

class RNN_numpy():
    def __init__(self,
                 N, dt, tau,
                 W_inp, W_rec, W_out,
                 activation_name, activation_slope,
                 bias_rec=None, y_init=None, seed=None, beta=None):
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
        self.activation_name = activation_name
        self.activation_slope = activation_slope
        if activation_name == 'relu':
            self.activation = lambda x: np.maximum(0, self.activation_slope * x)
        elif activation_name == 'sigmoid':
            self.activation = lambda x: 1.0 / (1 + np.exp(-self.activation_slope * x))
        elif activation_name == 'tanh':
            self.activation = lambda x: np.tanh(self.activation_slope * x)
        elif activation_name == 'softplus':
            self.beta = beta
            self.activation = np.log(1 + np.exp(self.beta * x))/self.beta

        if seed is None:
            self.rng = np.random.default_rng(np.random.randint(10000))
        else:
            self.rng = np.random.default_rng(seed)



    def rhs(self, y, input, sigma_rec=None, sigma_inp=None):
        if len(y.shape) == 2:
            # Check that the batch_size (last dimension) is the same as the Input's last dimension
            if y.shape[-1] != input.shape[-1]:
                raise ValueError(
                    f"The last dimension of the RNN state and the Input (representing batch size) should be equal!" +
                    f" {y.shape[-1]} != {input.shape[-1]}")
            batch_size = y.shape[-1]
            bias_rec = np.repeat(self.bias_rec[:, np.newaxis], repeats=batch_size, axis=-1)
        else:
            bias_rec = self.bias_rec

        if ((sigma_rec is None) and (sigma_inp is None)) or ((sigma_rec == 0) and (sigma_inp == 0)):
            return -y + self.activation(self.W_rec @ y + self.W_inp @ input + bias_rec)
        else:
            rec_noise_term = np.sqrt((2 / self.alpha) * sigma_rec ** 2) * self.rng.standard_normal(y.shape) \
                if (not (sigma_rec is None)) else np.zeros(x.shape)
            inp_noise_term = np.sqrt((2 / self.alpha) * sigma_inp ** 2) * self.rng.standard_normal(input.shape) \
                if (not (sigma_inp is None)) else np.zeros(input.shape)
            return -y + self.activation(
                self.W_rec @ y + self.W_inp @ (input + inp_noise_term) + bias_rec + rec_noise_term)

    def rhs_noisless(self, y, input):
        '''
        Bare version of RHS for efficient fixed point analysis
        supposed to work only with one point at the state-space at the time (no batches!)
        '''
        return -y + self.activation(self.W_rec @ y + self.W_inp @ input + self.bias_rec)

    # def rhs_jac(self, y, input):
    #     if len(input.shape) > 1:
    #         raise ValueError("Jacobian computations work only for single point and a single input-vector. It doesn't yet work in the batch mode")
    #     # efficient calculation of Jacobian using a finite difference
    #     return nd.Jacobian(self.rhs_noisless)(y, input)

    def rhs_jac(self, y, input):
        if len(input.shape) > 1:
            raise ValueError("Jacobian computations work only for single point and a single input-vector. It doesn't yet work in the batch mode")
        arg = self.W_rec @ y + self.W_inp @ input + self.bias_rec
        if self.activation_name == 'relu':
            f_prime = self.activation_slope * np.heaviside(self.activation_slope * arg, 0.5)
        elif self.activation_name == 'tanh':
            f_prime = self.activation_slope * (1 - np.tanh(self.activation_slope * arg) ** 2)
        elif self.activation_name == 'sigmoid':
            sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
            f_prime = self.activation_slope * (sigmoid(self.activation_slope * arg)) * (1.0 - sigmoid(self.activation_slope * arg))
        elif self.activation_name == 'softplus':
            f_prime = lambda x: np.exp(self.beta * x)/(np.exp(self.beta * x) + 1)
        return -np.eye(self.N) + np.diag(f_prime) @ self.W_rec


    # def jax_rhs_noisless(self, y, input):
    #     '''
    #     Bare version of RHS for efficient fixed point analysis
    #     supposed to work only with one point at the state-space at the time (no batches!)
    #     '''
    #     return -y + self.activation_jax(jnp.array(self.W_rec) @ y + jnp.array(self.W_inp) @ input + jnp.array(self.bias_rec))
    #
    # def jax_rhs_jac(self, y, input):
    #     if len(input.shape) > 1:
    #         raise ValueError("Jacobian computations work only for single point and a single input-vector. It doesn't yet work in the batch mode")
    #     jac_fun = jax.jacfwd(self.jax_rhs_noisless, argnums=0)
    #     return jac_fun(jnp.array(y), jnp.array(input))

    def rhs_noisless_h(self, h, input):
        '''
        h = W_rec y + W_inp u + b_rec
        '''
        return -h + self.W_rec @ self.activation(h) + self.W_inp @ input + self.bias_rec

    def rhs_jac_h(self, h, input):
        if len(input.shape) > 1:
            raise ValueError(
                "Jacobian computations work only for single point and a single input-vector. It doesn't yet work in the batch mode")
        return nd.Jacobian(self.rhs_noisless_h)(h, input)

    # def rhs_jac_explicit(self, y, input): #explicit Jacobian computation for RELU ONLY
    #     arg = ((self.W_rec @ y).flatten() + (self.W_inp @ input.reshape(-1, 1)).flatten())
    #     m = 0.5
    #     D = np.diag(np.heaviside(arg, m))
    #     J = -np.eye(self.N) + D @ self.W_rec
    #     return J

    def step(self, input, sigma_rec=None, sigma_inp=None):
        self.y += (self.dt / self.tau) * self.rhs(self.y, input, sigma_rec, sigma_inp)

    def run(self, input_timeseries, save_history=True, sigma_rec=None, sigma_inp=None):
        '''
        :param Inputs: an array, has to be iether (n_inputs x n_steps) dimensions or (n_inputs x n_steps x batch_batch_size)
        :param save_history: bool, whether to save the resulting trajectory
        :param sigma_rec: noise parameter in the recurrent dynamics
        :param sigma_inp: noise parameter in the input channel
        :return: None
        '''
        num_steps = input_timeseries.shape[1]  # second dimension
        if len(input_timeseries.shape) == 3:
            batch_size = input_timeseries.shape[-1]  # last dimension
            # if the state is a 1D vector, repeat it batch_size number of times to match with Input dimension
            if len(self.y.shape) == 1:
                self.y = np.repeat(deepcopy(self.y)[:, np.newaxis], axis=1, repeats=batch_size)
            if len(self.y.shape) == 2:
                pass

        for i in range(num_steps):
            if save_history == True:
                self.y_history.append(deepcopy(self.y))
            self.step(input_timeseries[:, i, ...], sigma_rec=sigma_rec, sigma_inp=sigma_inp)
        return None

    def get_history(self):
        # N x T x Batch_size or N x T if Batch_size = 1
        return np.swapaxes(np.array(self.y_history), 0, 1)

    def reset_state(self):
        self.y = deepcopy(self.y_init)

    def clear_history(self):
        self.y_history = []
        self.y = deepcopy(self.y_init)

    def get_output(self):
        y_history = np.stack(self.y_history, axis=0)
        if len(y_history.shape) == 3:
            output = np.swapaxes((self.W_out @ y_history), 0, 1)
        elif len(y_history.shape) == 2:
            output = self.W_out @ y_history.T
        else:
            raise ValueError("y_history variable should have either 2 or 3 dimensions!")
        return output

if __name__ == '__main__':
    N = 100
    activation_name = 'relu'
    x = np.random.randn(N)
    W_rec = np.random.randn(N, N)
    W_inp = np.random.randn(N, 6)
    W_out = np.random.randn(2, N)
    bias_rec = np.random.randn(N)

    # Input = np.ones(6)
    dt = 0.1
    tau = 10
    batch_size = 11
    input = np.ones((6))

    rnn = RNN_numpy(N=N, W_rec=W_rec, W_inp=W_inp, W_out=W_out, dt=dt, tau=tau, activation_name=activation_name)

    rnn.y = np.random.randn(N)
    input_timeseries = 0.1 * np.ones((6, 301))
    rnn.run(input_timeseries=input_timeseries)
    output = rnn.get_output()