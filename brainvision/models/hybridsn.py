"""
HybridSN: Hybrid Spectral-Spatial 3D-CNN + 2D-CNN for HSI classification.
Roy et al. (2020) — https://doi.org/10.1109/LGRS.2019.2918719
"""

import torch
import torch.nn as nn


class HybridSN(nn.Module):
    """
    HybridSN: Hybrid Spectral-Spatial 3D-CNN + 2D-CNN for HSI classification.
    Roy et al. (2020) — https://doi.org/10.1109/LGRS.2019.2918719

    Input:  (B, C, patch_size, patch_size)
    Output: (B, n_classes)
    """

    def __init__(
        self, input_channels: int = 128, patch_size: int = 5, n_classes: int = 4
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # ── 3D conv blocks ────────────────────────────────────────────────────
        # Spatial dims preserved via padding=(0,1,1)
        # Spectral dim reduces naturally (no spectral padding)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self._shape_3d = self._get_shape_after_3dconv()

        # ── 2D conv block ─────────────────────────────────────────────────────
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                self._shape_3d[1] * self._shape_3d[2],
                64,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),  # preserve spatial
            nn.ReLU(inplace=True),
        )

        self._flat_size = self._get_flat_size()

        # ── FC blocks ─────────────────────────────────────────────────────────
        self.dense1 = nn.Sequential(
            nn.Linear(self._flat_size, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.4)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.4)
        )
        self.dense3 = nn.Linear(128, n_classes)

    def _get_shape_after_3dconv(self) -> torch.Size:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels, self.patch_size, self.patch_size)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape  # (1, 32, D, patch_size, patch_size)

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(
                1,
                self._shape_3d[1] * self._shape_3d[2],
                self._shape_3d[3],
                self._shape_3d[4],
            )
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, C, P, P)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(
            x.shape[0],
            x.shape[1] * x.shape[2],  # merge spectral+filter
            x.shape[3],
            x.shape[4],
        )
        x = self.conv4(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
