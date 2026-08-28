#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model classes and functions for the Fourier Neural Operator in 2D.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer compatible with SOAP/Shampoo.
        Weights are stored as float (Real) with an extra dimension of size 2,
        and viewed as complex during forward pass.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.double))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.double))

    # Complex multiplication
    @staticmethod
    def compl_mul2d(input_ft, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. Compute Fourier coefficients
        # rfft2 returns complex tensors
        x_ft = torch.fft.rfft2(x)

        # View real weights as complex for the spectral multiply
        w1 = torch.view_as_complex(self.weights1)
        w2 = torch.view_as_complex(self.weights2)

        # 3. Multiply relevant Fourier modes
        # Output tensor needs to be complex double to match weights/input precision
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cdouble)

        # Processing modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)

        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)

        # 4. Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        # use convolution with kernel size 1 as fully connected layers
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, model_data, input_range=None):
        super(FNO2d, self).__init__()

        if input_range is None:
            input_range = [-100, 100]

        self.modes1 = model_data["fno"]["modes"]
        self.modes2 = model_data["fno"]["modes"]
        self.width = model_data["fno"]["width"]
        self.input = model_data["fno"]["input"]
        self.output = model_data["fno"]["output"]
        self.padding = 2
        self.x_min = input_range[0]
        self.x_max = input_range[1]

        self.fc0 = nn.Linear(self.input, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output)

        self.W = model_data["W"]
        self.H = model_data["H"]

        # Automatically call initialization
        # self.initialize_weights()

    def initialize_weights(self):
        # 1. Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 2. 1x1 Conv layers
        for m in [self.w0, self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # SpectralConv2d real-valued weights
        for m in [self.conv0, self.conv1, self.conv2, self.conv3]:
            if hasattr(m, 'weights1'):
                nn.init.normal_(m.weights1, mean=0.0, std=1e-2)
            if hasattr(m, 'weights2'):
                nn.init.normal_(m.weights2, mean=0.0, std=1e-2)

        # 4. Last layer
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Input x shape: (batch, sx, sy, channels)
        grid = self.get_grid(x.shape, x.device, self.W, self.H)

        # Concatenate grid
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def get_grid(shape, device, W, H):
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]

        gridx = torch.tensor(np.linspace(0, W, size_x), dtype=torch.float64)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batch_size, 1, size_y, 1])

        gridy = torch.tensor(np.linspace(0, H, size_y), dtype=torch.float64)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batch_size, size_x, 1, 1])

        # Result shape: (batch, x, y, 2)
        return torch.cat((gridx, gridy), dim=-1).to(device)