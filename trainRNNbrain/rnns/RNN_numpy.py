import numpy as np
from copy import deepcopy
import jax
import jax.numpy as jnp

'''
lightweight numpy implementation of RNN for validation and quick testing and plotting
'''

class RNN_numpy():
    def __init__(self,
                 N, dt, tau,
                 W_inp, W_rec, W_out,
                 activation_args={"name": "relu", "slope": 1.0},
                 gamma=0.1,
                 bias=None,
                 y_init=None,
                 seed=None
                 ):
        self.N = N
        self.W_inp = W_inp
        self.W_rec = W_rec
        self.W_out = W_out
        if bias is None:
            self.bias = np.zeros(self.N)
        else:
            self.bias = bias
        self.dt = dt
        self.tau = tau
        self.alpha = self.dt / self.tau
        if not (y_init is None):
            self.y_init = y_init
        else:
            self.y_init = np.zeros(self.N)
        self.y = deepcopy(self.y_init)
        self.y_history = []
        self.activation_args = activation_args
        self.activation = self.configure_activation_(activation_args)
        self.activation_name = activation_args["name"]
        self.gamma = gamma

        if seed is None:
            self.rng = np.random.default_rng(np.random.randint(10000))
        else:
            self.rng = np.random.default_rng(seed)

    @staticmethod
    def configure_activation_(activation_args):
        activation_name = activation_args["name"]
        if activation_name == 'relu':
            slope = activation_args.get("slope", 1.0)
            return lambda x: np.maximum(0., slope * x)
        elif activation_name == 'tanh':
            slope = activation_args.get("slope", 1.0)
            return lambda x: np.tanh(slope * x)
        elif activation_name == 'sigmoid':
            slope = activation_args.get("slope", 1.0)
            sigmoid = lambda x: 1.0 / (1.0 + np.exp(-slope * x))
            return lambda x: sigmoid(slope * x)
        elif activation_name == 'softplus':
            beta = activation_args.get("beta", 1.0)
            slope = activation_args.get("slope", 1.0)
            return lambda x: np.log(1 + np.exp(beta * (slope * x))) / beta
        elif activation_name == 'leaky_relu':
            slope = activation_args.get("slope", 1.0)
            leak_slope = activation_args.get("leak_slope", 0.01)
            return lambda x: np.where(x > 0, slope * x, leak_slope * x)
        else:
            raise ValueError(f"Activation function {activation_name} is not recognized!")

    def rhs(self, y, input, sigma_rec=None, sigma_inp=None):
        sr = 0.0 if sigma_rec is None else float(sigma_rec)
        si = 0.0 if sigma_inp is None else float(sigma_inp)
        if sr == 0.0 and si == 0.0:
            h = self.W_rec @ y + self.W_inp @ input
            return -y + self.activation(h) - self.gamma * y ** 3
        a = float(self.alpha)
        rec_noise = ((2.0 / a) ** 0.5 * sr) * self.rng.standard_normal(y.shape, dtype=y.dtype)
        inp_noise = ((2.0 / a) ** 0.5 * si) * self.rng.standard_normal(input.shape, dtype=input.dtype)
        h = self.W_rec @ y + self.W_inp @ (input + inp_noise)
        return -y + self.activation(h) + rec_noise - self.gamma * y ** 3

    def rhs_noiseless(self, y, input):
        '''
        Bare version of RHS for efficient fixed point analysis
        supposed to work only with one point at the state-space at the time (no batches!)
        '''
        return -y + self.activation(self.W_rec @ y + self.W_inp @ input + self.bias) - self.gamma * y ** 3


    def rhs_jac(self, y, input):
        if len(input.shape) > 1:
            raise ValueError("Jacobian computations work only for single point and a single input-vector. It doesn't yet work in the batch mode")
        arg = self.W_rec @ y + self.W_inp @ input + self.bias
        if self.activation_args["name"] == 'relu':
            f_prime = self.activation_args["slope"] * np.heaviside(self.activation_args["slope"] * arg, 0.5)
        elif self.activation_args["name"] == 'tanh':
            f_prime = self.activation_args["slope"] * (1 - np.tanh(self.activation_args["slope"] * arg) ** 2)
        elif self.activation_args["name"] == 'sigmoid':
            sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
            f_prime = self.activation_args["slope"] * (sigmoid(self.activation_args["slope"] * arg)) * (1.0 - sigmoid(self.activation_args["slope"] * arg))
        elif self.activation_args["name"] == 'softplus':
            beta = self.activation_args.get("beta", 1.0)
            slope = self.activation_args.get("slope", 1.0)
            f_prime = lambda x: slope / (1 + np.exp(-beta * slope * x))
        elif self.activation_args["name"] == 'leaky_relu':
            f_prime = self.activation_args["slope"] * np.heaviside(arg, 0.5) + self.activation_args["leak_slope"] * np.heaviside(-arg, 0.5)
        return -np.eye(self.N) + np.diag(f_prime) @ self.W_rec

    def rhs_jac_h(self, h, input):
        """
        Computes the Jacobian of rhs_noisless_h with respect to h at the given h and input.
        Returns a NumPy array.
        """

        # Wrap the function so JAX knows to differentiate with respect to h
        def fun(h_):
            return self.rhs_noiseless_h(h_, input)

        # JAX expects jnp arrays
        h_jax = jnp.asarray(h)
        jac = jax.jacfwd(fun)(h_jax)
        # Optionally convert to np if needed downstream
        return np.asarray(jac)

    def rhs_noiseless_h(self, h, input):
        '''
        h = W_rec y + W_inp u + b_rec
        '''
        return -h + self.W_rec @ self.activation(h) + self.W_inp @ input + self.bias

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
    bias = np.random.randn(N)

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