# fabelo_cnn.py
"""
AlexNet-inspired 2D-CNN for HSI patch classification.
Fabelo et al. (2019) — "Deep Learning-Based Framework for In Vivo
Identification of Glioblastoma Tumor using Hyperspectral Images of Human Brain"
Sensors, 19(4), 920. https://doi.org/10.3390/s19040920

Architecture described as "based approximately on AlexNet" (Section 2.3):
  - 3 convolutional layers
  - 1 average pooling layer
  - 1 fully connected layer
  - Input: 11x11 pixel patches centred on each labelled pixel

Note: Exact filter counts and kernel sizes were not published in Table 2.
This implementation follows the described structural pattern scaled to
11x11 HSI patches. Exact replication is not possible from the paper alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Fabelo2DCNN(nn.Module):
    """
    2D-CNN for HSI patch classification.
    Fabelo et al. (2019) — Sensors, 19(4), 920.

    Input:  (B, input_channels, 11, 11)  — spatial patch
    Output: (B, n_classes)               — raw logits
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            init.zeros_(m.bias)

    def _get_flat_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels,
                            self.patch_size, self.patch_size)
            x = self.features(x)
        return x.numel()

    def __init__(self,
                 input_channels: int,
                 n_classes:      int,
                 patch_size:     int = 11):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size     = patch_size

        # AlexNet-inspired: 3 conv layers → avg pool → FC
        # Kernel sizes scaled to fit 11×11 patches
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,             64,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,             128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.flat_size  = self._get_flat_size()
        self.classifier = nn.Linear(self.flat_size, n_classes)

        self.apply(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, P, P)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)