"""
2D-CNN for HSI classification
"""

import torch
import torch.nn as nn
from torch.nn import init


class Simple2DCNN(nn.Module):
    """
    Minimal 2D-CNN for HSI patch classification.
    Designed to mirror FabeloDNN's simplicity (4,936 params)
    but with spatial context from a 5x5 patch.

    Architecture:
      Conv2d(128→16, 3x3) → ReLU
      Conv2d(16→8,  3x3)  → ReLU
      GlobalAvgPool        → (B, 8)
      Linear(8→4)

    Philosophy:
      - No padding → spatial dims shrink naturally (5→3→1)
      - Global average pooling instead of flatten — forces the
        network to learn spatially invariant features rather
        than memorising patch positions
      - Minimal filters — 16 and 8 rather than 32/64/128
      - No dropout, no BatchNorm — matches FabeloDNN's bare structure

    Input:  (B, input_channels, patch_size, patch_size)
    Output: (B, n_classes)

    Params with input_channels=128, patch_size=5, n_classes=4:
      Conv1 : 128x16x3x3 + 16  =  18,448
      Conv2 :  16x8x3x3  +  8  =   1,160
      FC    :   8x4       +  4  =      36
      Total :                      19,644
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            init.zeros_(m.bias)

    def __init__(self,
                 input_channels: int,
                 n_classes:      int,
                 patch_size:     int = 5):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size     = patch_size

        # Two conv layers — no padding so 5×5 → 3×3 → 1×1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

        # Global average pool — collapses spatial dims to (B, 8)
        # regardless of patch size, avoids fixed flatten size
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, n_classes)

        self.apply(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, P, P)
        x = self.conv1(x)   # (B, 16, P-2, P-2)
        x = self.conv2(x)   # (B,  8, P-4, P-4)
        x = self.gap(x)     # (B,  8,   1,   1)
        x = x.view(x.size(0), -1)   # (B, 8)
        return self.classifier(x)