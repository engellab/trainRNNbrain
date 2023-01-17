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

def get_connectivity(device, N, num_inputs, num_outputs, radius=1.5, recurrent_density=1, input_density=1, output_density=1, generator=None):
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

    recurrent_sparsity = 1-recurrent_density
    W_rec = sparse(torch.empty(N, N), recurrent_sparsity, mu, var, generator)

    # spectral radius adjustment
    W_rec = W_rec - torch.diag(torch.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius * W_rec / spec_radius

    W_inp = torch.zeros([N, num_inputs]).float()
    input_sparsity = 1-input_density
    W_inp = sparse(W_inp, input_sparsity, mu_pos, var, generator)

    output_sparsity = 1 - output_density
    W_out = sparse(torch.empty(num_outputs, N), output_sparsity, mu_pos, var, generator)

    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_inp != 0).to(device=device).float()
    recurrent_mask = torch.ones(N, N) - torch.eye(N)
    return W_rec.to(device=device).float(),\
           W_inp.to(device=device).float(),\
           W_out.to(device=device).float(),\
           recurrent_mask.to(device=device).float(),\
           output_mask.to(device=device).float(),\
           input_mask.to(device=device).float()

def get_connectivity_Dale(device, N, num_inputs, num_outputs, radius=1.5, recurrent_density=1, input_density=1, output_density=1, generator=None):
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
    recurrent_sparsity = 1-recurrent_density
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

    W_inp= torch.zeros([N, num_inputs]).float()
    input_sparsity = 1-input_density
    W_inp= torch.abs(sparse(W_inp, input_sparsity, mu_E, var, generator))

    W_out = torch.zeros([num_outputs, N])
    output_sparsity = 1 - output_density
    W_out = torch.abs(sparse(W_out, output_sparsity, mu_E, var, generator))

    dale_mask = torch.sign(W_rec).to(device=device).float()
    output_mask = (W_out != 0).to(device=device).float()
    input_mask = (W_inp!= 0).to(device=device).float()
    recurrent_mask = torch.ones(N, N) - torch.eye(N)
    return W_rec.to(device=device).float(), W_inp.to(device=device).float(), W_out.to(
        device=device).float(), recurrent_mask.to(device=device).float(), dale_mask, output_mask, input_mask
