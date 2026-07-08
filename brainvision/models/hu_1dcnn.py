"""
1D Convolutional Neural Network for HSI classification.
Hu et al. (2015) — "Deep Convolutional Neural Networks for
Hyperspectral Image Classification"
Journal of Sensors, 2015. https://doi.org/10.1155/2015/258619
"""

import math

import torch
import torch.nn as nn
from torch.nn import init


class HuEtAl1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for HSI classification.
    Hu et al. (2015) — https://doi.org/10.1155/2015/258619

    Input:  (B, input_channels)  — one pixel spectrum per sample
    Output: (B, n_classes)       — raw logits
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(
        self,
        input_channels: int = 128,
        n_classes: int = 4,
        kernel_size: int = None,
        pool_size: int = None,
    ):
        super().__init__()
        self.input_channels = input_channels

        # kernel size = ceil(n_bands / 9) as per paper
        if kernel_size is None:
            kernel_size = math.ceil(input_channels / 9)
        # pool size = ceil(kernel_size / 5) as per paper
        if pool_size is None:
            pool_size = math.ceil(kernel_size / 5)

        # C1: 20 kernels of size kernel_size
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)

        self.features_size = self._get_final_flattened_size()

        # FC layers: features → 100 → n_classes
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        # x: (B, C) → unsqueeze to (B, 1, C) for Conv1d
        x = x.unsqueeze(1)
        x = torch.tanh(self.pool(self.conv(x)))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)
