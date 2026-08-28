import os
import numpy as np
import torch
from typing import Tuple
from scipy.fftpack import idct
from timeit import default_timer
from torch.utils.data import Dataset

# from .pde_solvers import darcy2D_solver, FGMBeam_solver, PlateHole_solver


def GRF(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.randn(s, s)

    # Define the (square root of) eigenvalues of the covariance operator
    k1, k2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = tau ** (alpha - 1) * (np.pi ** 2 * (k1 ** 2 + k2 ** 2) + tau ** 2) ** (-alpha / 2)

    # Construct the KL coefficients
    l = s * coef * xi
    l[0, 0] = 0

    # 2D inverse discrete cosine transform
    u = idct(idct(l, axis=0, norm='ortho'), axis=1, norm='ortho')
    return u


def generate_inputs(N, m, alpha, tau):
    """
    Generates all the N inputs

    Parameters
    ----------
    N : (int)
        Number of training functions u.

    m : (integer)
        number of sensor points.

    tau, alpha : (float)
        parameters of covariance

    Returns
    -------
    f_all : (3D arrays)
        the values of the input function at the sensor points, one input function per row and per column

    """
    f_all = np.zeros((N, m, m), dtype=np.float64)
    # sensor_pts = np.linspace(0, 1, m + 1, endpoint=False, dtype=np.float64)[1:]
    # x_new_grid, y_new_grid = np.meshgrid(sensor_pts, sensor_pts)
    for i in range(N):
        if (i+1) % 200 == 0:
            print(f"Generating the {i+1}th input")
        in_data = GRF(alpha, tau, m)
        in_data = np.where(in_data >= 0, 12., 4.)
        f_all[i] = in_data
    return f_all


def gaussian_normalize(x: np.ndarray, eps=0.00001) -> Tuple[np.ndarray, float, float]:
    """
    Attributes:
        x (np.ndarray): Input array
        eps (float): Small number to avoid division by zero

    Returns:
        Tuple[np.ndarray, float, float]: Normalized array, mean and standard deviation
    """
    mean = np.mean(x, 0, keepdims=True)
    std = np.std(x, 0, keepdims=True)
    x = (x - mean) / (std + eps)
    return x, mean, std


class GaussianNormalizer(torch.nn.Module): # inherit nn.Module
    def __init__(self, x=None, mean=None, std=None, eps=1e-5):
        super().__init__()
        if x is not None:
            # Compute mean and standard deviation
            mean = torch.mean(x, dim=(0, 1, 2), keepdims=True)
            std = torch.std(x, dim=(0, 1, 2), keepdims=True)
        # Register as buffers so they are saved in state_dict and move with model.to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return (x * (self.std + self.eps)) + self.mean


def ElementsToNodes(elementValues):
    temp = np.pad(elementValues, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='edge')
    return (temp[:, :-1, :-1, :] + temp[:, 1:, :-1, :] + temp[:, :-1, 1:, :] + temp[:, 1:, 1:, :]) / 4


class LinearelasticityDataset(Dataset):
    def __init__(self, model_data):
        if not os.path.exists(model_data["dir"]):
            os.makedirs(model_data["dir"])

        if os.path.exists(model_data["path"]):
            print("Found saved dataset at", model_data["path"])
            loaded_data = np.load(model_data["path"])
            inputs = loaded_data['inputs']
            targets = loaded_data['targets']
        else:
            print("NOT Found saved dataset at", model_data["path"])
            raise ValueError("Please run the python code for generating dataset or download the database.")

        # Extract data and put batch dimension in front
        self.x = torch.from_numpy(inputs).to(torch.float64).to(model_data["device"])
        self.y = torch.from_numpy(targets).to(torch.float64).to(model_data["device"])

        # Add channel dimension at the end
        # self.x = self.x[..., np.newaxis]
        # self.y = self.y[..., np.newaxis]

        # Normalize data
        if model_data["normalized"]:
            #self.x, _, _ = gaussian_normalize(self.x)
            #self.y, _, _ = gaussian_normalize(self.y)

            self.normalizer_x = GaussianNormalizer(self.x)
            self.normalizer_y = GaussianNormalizer(self.y)
            self.x = self.normalizer_x.encode(self.x)
            self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class HyperelasticityDataset(Dataset):
    def __init__(self, model_data, path, normalizers=None):
        if not os.path.exists(model_data["dir"]):
            os.makedirs(model_data["dir"])

        if os.path.exists(path):
            print("Found saved dataset at", path)
            loaded_data = np.load(path)
            inputs = loaded_data['inputs']
            targets = loaded_data['targets']
        else:
            print("NOT Found saved dataset at", path)
            raise ValueError("Please run the python code for generating dataset or download the database.")

        # Extract data and put batch dimension in front
        self.x = torch.from_numpy(inputs).to(torch.float64).to(model_data["device"])
        self.y = torch.from_numpy(targets).to(torch.float64).to(model_data["device"])
        
        # Add channel dimension at the end
        # self.x = self.x[..., np.newaxis]
        # self.y = self.y[..., np.newaxis]

        # Normalize data
        if model_data["normalized"]:
            #self.x, _, _ = gaussian_normalize(self.x)
            #self.y, _, _ = gaussian_normalize(self.y)

            if normalizers is not None:
                self.normalizer_x, self.normalizer_y = normalizers[0], normalizers[1]
            else:
                self.normalizer_x = GaussianNormalizer(self.x)
                self.normalizer_y = GaussianNormalizer(self.y)
            self.x = self.normalizer_x.encode(self.x)
            self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]





