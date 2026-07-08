"""
Contextual Deep CNN for HSI Classification.
Lee & Kwon (2016) — "Contextual Deep CNN Based Hyperspectral Classification"
IGARSS 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LeeEtAl2DCNN(nn.Module):
    """
    Contextual Deep CNN for HSI Classification.
    Lee & Kwon (2016) — IGARSS 2016.

    Input:  (B, C, patch_size, patch_size)  — spatial-spectral patch
    Output: (B, n_classes)                  — raw logits

    Note: expects patch_size >= 3. The model processes the full spectral depth
    in the first inception layer, then applies 2D residual convolutions on the
    resulting feature maps.
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(
        self, input_channels: int = 128, n_classes: int = 4, patch_size: int = 5
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # ── Inception module: two 3D convs collapsing the spectral dimension ──
        # Outputs (B, 256, 1, H, W) — spectral dim collapsed to 1
        self.conv_3x3 = nn.Conv3d(
            1, 128, (input_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        self.conv_1x1 = nn.Conv3d(
            1, 128, (input_channels, 1, 1), stride=(1, 1, 1), padding=0
        )

        # ── 2D residual blocks operating on (B, 256, H, W) ───────────────────
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))  # residual block 1
        self.conv3 = nn.Conv2d(128, 128, (1, 1))
        self.conv4 = nn.Conv2d(128, 128, (1, 1))  # residual block 2
        self.conv5 = nn.Conv2d(128, 128, (1, 1))
        self.conv6 = nn.Conv2d(128, 128, (1, 1))  # final convolutions
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # x: (B, C, P, P) → add channel dim → (B, 1, C, P, P)
        x = x.unsqueeze(1)

        # Inception — collapse spectral dimension
        x_3x3 = self.conv_3x3(x)  # (B, 128, 1, P, P)
        x_1x1 = self.conv_1x1(x)  # (B, 128, 1, P, P)
        x = torch.cat([x_3x3, x_1x1], dim=1)  # (B, 256, 1, P, P)
        x = x.squeeze(2)  # (B, 256, P, P)

        x = F.relu(self.lrn1(x))
        x = self.conv1(x)  # (B, 128, P, P)
        x = F.relu(self.lrn2(x))

        # Residual block 1
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Residual block 2
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)  # (B, n_classes, P, P)

        # Extract centre pixel prediction
        centre = self.patch_size // 2
        return x[:, :, centre, centre]  # (B, n_classes)
