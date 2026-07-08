"""
3D Deep Learning for HSI Classification.
Ben Hamida et al. (2018) — "3-D Deep Learning Approach for Remote
Sensing Image Classification"
IEEE TGRS, 2018. https://doi.org/10.1109/TGRS.2018.2818945
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class HamidaEtAl3DCNN(nn.Module):
    """
    3D Deep Learning for HSI Classification.
    Ben Hamida et al. (2018) — https://doi.org/10.1109/TGRS.2018.2818945

    Input:  (B, C, patch_size, patch_size)  — spatial-spectral patch
    Output: (B, n_classes)                  — raw logits
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels, self.patch_size, self.patch_size)
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
        return x.numel()

    def __init__(
        self, input_channels: int = 128, n_classes: int = 4, patch_size: int = 5
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # Conv1: 20 kernels (3,3,3)
        self.conv1 = nn.Conv3d(
            1, 20, (3, 3, 3), stride=(1, 1, 1), padding=0 if patch_size > 3 else 1
        )
        # Pool1: spectral stride=2
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        # Conv2: 35 kernels (3,3,3)
        self.conv2 = nn.Conv3d(20, 35, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0))
        # Pool2: spectral stride=2
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        # Conv3, Conv4: reduce spectral dimension further
        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(35, 35, (2, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        # x: (B, C, P, P) → (B, 1, C, P, P)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        return self.fc(x)
