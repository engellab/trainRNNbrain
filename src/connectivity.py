import math
import numpy as np
import torch
from numpy import linalg

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
    num_zeros = int(math.ceil(sparsity * rows))

    with torch.no_grad():
        tensor.normal_(mean, std, generator=generator)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows, generator=generator)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor

def get_connectivity(device, N, num_outputs=1, radius=1.5, recurrent_density=1, input_density=1, output_density=1, generator=None):
    Ne = int(N * 0.8)
    Ni = int(N * 0.2)

    # Initialize W_rec
    W_rec = torch.empty([0, N])

    # Balancing parameters
    mu_E = 1 / np.sqrt(N)
    mu_I = 4 / np.sqrt(N)

    var = 1 / N

    rowE = torch.empty([Ne, 0])
    rowI = torch.empty([Ni, 0])
    recurrent_sparsity = 1-recurrent_density
    rowE = torch.cat((rowE, torch.abs(sparse(torch.empty(Ne, Ne), recurrent_sparsity, mu_E, var, generator))), 1)
    rowE = torch.cat((rowE, -torch.abs(sparse(torch.empty(Ne, Ni), recurrent_sparsity, mu_I, var, generator))), 1)
    rowI = torch.cat((rowI, torch.abs(sparse(torch.empty(Ni, Ne), recurrent_sparsity, mu_E, var, generator))), 1)
    rowI = torch.cat((rowI, -torch.abs(sparse(torch.empty(Ni, Ni), recurrent_sparsity, mu_I, var, generator))), 1)

    W_rec = torch.cat((W_rec, rowE), 0)
    W_rec = torch.cat((W_rec, rowI), 0)

    W_rec = W_rec - torch.diag(torch.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius * W_rec / spec_radius

    W_in = torch.zeros([N, 6]).float()
    input_sparsity = 1-input_density
    W_in = torch.abs(sparse(W_in, input_sparsity, mu_E, var, generator))

    W_out = torch.zeros([num_outputs, N])
    output_sparsity = 1 - output_density
    W_out[:, :Ne] = torch.abs(sparse(torch.empty(1, Ne), output_sparsity, mu_E, var, generator))

    dale_mask = torch.sign(W_rec).to(device=device).float()
    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_in != 0).to(device=device).float()
    recurrent_mask = torch.ones(N, N) - torch.eye(N)
    return W_rec.to(device=device).float(), W_in.to(device=device).float(), W_out.to(
        device=device).float(), recurrent_mask.to(device=device).float(), dale_mask, output_mask, input_mask

def get_small_connectivity_np(rnd_perturb=1e-12):
    '''
    An example connectivity of 8 nodes, solving the context dependet decision making task
    '''
    N = 8
    W_rec = np.zeros((N, N))
    # context mechanism
    W_rec[4, 0] = W_rec[5, 0] = W_rec[2, 1] = W_rec[3, 1] = -1.6 + rnd_perturb * np.random.randn()
    W_rec[2, 0] = W_rec[3, 0] = W_rec[4, 1] = W_rec[5, 1] = 0.1
    # Output
    W_rec[6, 2] = W_rec[6, 4] = W_rec[7, 3] = W_rec[7, 5] = 1.2 + rnd_perturb * np.random.randn()
    # Output competition
    W_rec[6, 7] = W_rec[7, 6] = -1.0
    # stimulus competition
    # W_rec[2, 3] = W_rec[3, 2] = W_rec[4, 5] = W_rec[5, 4] = -1.0 + rnd_perturb * np.random.randn()

    # signal channel competition
    W_rec[2, 4] = W_rec[2, 5] = W_rec[3, 4] = W_rec[3, 5] = -0.3 + rnd_perturb * np.random.randn()
    W_rec[4, 2] = W_rec[4, 3] = W_rec[5, 2] = W_rec[5, 3] = -0.3 + rnd_perturb * np.random.randn()

    W_inp = np.zeros((8, 6))
    W_inp[:6, :6] = np.eye(6)

    W_out = np.zeros((2, 8))
    W_out[0, 6] = 2
    W_out[1, 7] = 2

    return W_inp, W_rec, W_out
