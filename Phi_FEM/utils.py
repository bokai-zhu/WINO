import numpy as np
import random
import os
import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

seed = 2023
random.seed(seed)
np.random.seed(seed)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
# from prepare_data import (
#     call_phi,
#     call_G,
# )
import torch

def smooth_padding(W, pad_size):
    if isinstance(pad_size, tuple) or isinstance(pad_size, list):
        pad_left = pad_size[0]
        pad_right = pad_size[1]
    else:
        pad_left = pad_size
        pad_right = pad_size

    assert (
        pad_left < W.shape[-1] and pad_right < W.shape[-1]
    ), f"we must have pad_left < W.shape[-1] and pad_right < W.shape[-1]. Here:pad_left={pad_left},pad_right={pad_right} and W.shape[-1]={W.shape[-1]}"

    left_value = W[..., 0:1]
    right_value = W[..., -1:]

    left = W[..., 1 : pad_left + 1] - left_value
    right = W[..., -1 - pad_right : -1] - right_value
    left = -left[..., :]
    right = -right[..., :]
    left += left_value
    right += right_value
    return torch.cat((left, W, right), dim=-1)

def pad_1d(W, pad_size, axis):
    dim = len(W.shape)
    if not (axis == -1 or axis == dim - 1):
        permutation = list(range(dim))
        permutation[-1] = axis % dim
        permutation[axis] = dim - 1

        W = W.permute(permutation)
        W = smooth_padding(W, pad_size)
        W = W.permute(permutation)
    else:
        W = smooth_padding(W, pad_size)

    return W

def slice_at_given_axis(W, begin, size, axis):
    beg = [0 for _ in W.shape]
    sizes = [i for i in W.shape]
    beg[axis] = begin % W.shape[axis]
    sizes[axis] = size
    res = W[tuple(slice(b, b + s) for b, s in zip(beg, sizes))]
    return res

class Derivator_fd:
    def __init__(
        self,
        derivative_symbols,
        interval_lengths,
        axes,
        centred=True,
        keep_size=True,
    ):
        self.axes = axes
        self.centred = centred

        self.derivative_symbols_initial = None
        if isinstance(derivative_symbols, dict):
            self.derivative_symbols_initial = derivative_symbols
            self.derivative_symbols = list(derivative_symbols.values())
        else:
            self.derivative_symbols = derivative_symbols

        self.interval_lengths = interval_lengths
        self.keep_size = keep_size

    def D_axis(self, U, i):
        axis = self.axes[i]
        interval_length = self.interval_lengths[i]
        axis_size = U.shape[axis]
        h = interval_length / (axis_size - 1)
        if self.centred:
            U1_ = slice_at_given_axis(U, 2, axis_size - 2, axis)
            U_1 = slice_at_given_axis(U, 0, axis_size - 2, axis)
            D = (U1_ - U_1) / (2 * h)
            if self.keep_size:
                D = pad_1d(D, 1, axis)
            return D
        else:
            U1_ = slice_at_given_axis(U, 1, axis_size - 1, axis)
            U_1 = slice_at_given_axis(U, 0, axis_size - 1, axis)
            D = (U1_ - U_1) / h
            if self.keep_size:
                D = pad_1d(D, (0, 1), axis)
                return D

    def one_symbol_rec(self, U, res, symbol):
        if len(symbol) == 0:
            return U

        previous_der = res.get(symbol[:-1])

        if previous_der is None:
            previous_der = self.one_symbol_rec(U, res, symbol[:-1])

        der = self.D_axis(previous_der, symbol[-1])
        res[symbol] = der
        return der

    def __call__(self, U):
        res = {}
        for symbol in self.derivative_symbols:
            self.one_symbol_rec(U, res, symbol)

        if self.derivative_symbols_initial is not None:
            res_2 = {}
            for k_init, v_init in self.derivative_symbols_initial.items():
                res_2[k_init] = res[v_init]
        else:
            res_2 = []
            for symbol in self.derivative_symbols:
                res_2.append(res[symbol])

        return res_2

def convert_numpy_matrix_to_fenicsx(X, nb_vert):
    """
    Convert a numpy matrix to a FEniCSx function (for vector functions of dimension 2).

    Parameters:
        X (array): Input array of size nb_dof_x x nb_dof_y.
        nb_vert (int): Number of vertices in each direction.

    Returns:
        dolfinx.Function: Output function that can be used with FEniCSx.
    """
    nb_cell_x = nb_vert - 1
    nb_cell_y = nb_vert - 1

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0, 0], [1, 1]]),
        np.array([nb_cell_x, nb_cell_y]),
    )
    # Define the function space
    V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))

    # Get the degrees of freedom (dof) coordinates and map them
    dof_coords = V.tabulate_dof_coordinates()
    dof_coords = dof_coords.reshape((-1, 3))[:, :2]
    # Sort the coordinates by y first, then by x
    sorted_indices = np.lexsort((dof_coords[:, 0], dof_coords[:, 1]))
    sorted_indices = sorted_indices.astype(np.int32)

    # Create the Function
    X_fenicsx = dolfinx.fem.Function(V)

    X_x = X[:, :, 0].flatten()
    X_y = X[:, :, 1].flatten()

    X_fenicsx_vector = np.zeros(X_fenicsx.x.array[:].shape)
    X_fenicsx_vector_x = np.zeros((dof_coords.shape[0],))
    X_fenicsx_vector_y = np.zeros((dof_coords.shape[0],))
    X_fenicsx_vector_x[sorted_indices] = X_x
    X_fenicsx_vector_y[sorted_indices] = X_y

    X_fenicsx_vector[::2] = X_fenicsx_vector_x
    X_fenicsx_vector[1::2] = X_fenicsx_vector_y

    X_fenicsx.x.array[:] = X_fenicsx_vector
    return X_fenicsx

# def generate_phi_numpy(param_holes, nb_vert):
#     """
#     Generate a level-set function using the given parameters.

#     Parameters:
#         x_0 (float): x position of the center.
#         y_0 (float): y position of the center.
#         lx (float): width of the domain.
#         ly (float): height of the domain.
#         theta (float): angle of rotation (radians, rotation of center (x_0, y_0)).
#         nb_vert (int): number of pixels.

#     Returns:
#         array: Values of the level-set function.
#     """
#     xy = np.linspace(0.0, 1.0, nb_vert)
#     XX, YY = np.meshgrid(xy, xy)
#     XX = XX.flatten()
#     YY = YY.flatten()
#     XXYY = np.stack([XX, YY])
#     phi = call_phi(XXYY, param_holes)
#     if len(param_holes.shape) == 3:
#         phi = np.reshape(phi, [param_holes.shape[0], nb_vert, nb_vert]).transpose(
#             0, 2, 1
#         )
#     else:
#         phi = np.reshape(phi, [1, nb_vert, nb_vert]).transpose(0, 2, 1)
#     return phi

# def generate_G_numpy(gamma_G, nb_vert):
#     """Generation of a boundary condition using the given parameters.

#     Args:
#         alpha (float)
#         beta (float)
#         nb_vert (int): number of pixels

#     Returns:
#         array: values of the function.
#     """
#     xy = np.linspace(0.0, 1.0, nb_vert)
#     XX, YY = np.meshgrid(xy, xy)
#     XX = XX.flatten()
#     YY = YY.flatten()
#     XXYY = np.stack([XX, YY])
#     G = call_G(XXYY, gamma_G)
#     if isinstance(gamma_G, np.ndarray):
#         nb_data = gamma_G.shape[0]
#         G = G.reshape((2, nb_data, nb_vert, nb_vert)).transpose((1, 0, 3, 2))
#     else:
#         G = G.reshape((2, 1, nb_vert, nb_vert)).transpose((1, 0, 3, 2))
#     return G

def generate_manual_new_data_numpy(phi, G, dtype=torch.float32):
    """
    Generate input vectors for a FNO model using the provided phi, F, and G.

    Parameters:
        phi (array): Values of the level-set functions.
        F (array): Values of the force fields.
        G (array): Values of the boundary conditions.
        dtype (type, optional): Type of TensorFlow float. Default is tf.float32.

    Returns:
        tensor: Tensor of shape (nb_data, nb_vert, nb_vert, 4).
    """

    phi = torch.tensor(phi, dtype=dtype)
    G = torch.tensor(G, dtype=dtype)

    X = torch.stack([phi, G[:, 1, :, :]], dim=1)
    return X
