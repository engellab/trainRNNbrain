import numpy as np
import jax.numpy as jnp
import jax
import inspect
import re

def activation(x):
    return np.maximum(0, x)

function_code = inspect.getsource(activation)
jnp_activation_str = re.sub(r"np", "jnp", function_code)
jnp_activation_str = re.sub(r"activation", "activation_jax", jnp_activation_str)
exec(jnp_activation_str)  # cretes a function defined in the string

def rhs(x, W_rec, W_inp, input):
    return -x + activation_jax(W_rec @ x + W_inp @ input)

def rhs_jac(x, W_rec, W_inp, input):
    jac_fn =  jax.jacfwd(rhs, argnums=0)
    return jac_fn(x, W_rec, W_inp, input)

def rhs_jac_explicit(x, W_rec, W_inp, input):
    arg = ((W_rec @ x).flatten() + (W_inp @ input.reshape(-1, 1)).flatten())
    m = 0.5
    D = np.diag(np.heaviside(arg, m))
    J = -np.eye(x.shape[0]) + D @ W_rec
    return J

if __name__ == '__main__':


    N = 100
    x = np.random.randn(N)
    W_rec = np.random.randn(N, N)
    W_inp = np.random.randn(N, 6)
    W_out = np.random.randn(2, N)
    input = np.ones((6))


    J_jax = np.array(rhs_jac(x=jnp.array(x), W_rec = jnp.array(W_rec), W_inp = jnp.array(W_inp), input=jnp.array(input)))
    J_exp = rhs_jac_explicit(x=x, W_rec=W_rec, W_inp=W_inp, input=input)
    print(np.linalg.norm((J_jax - J_exp), 2))
