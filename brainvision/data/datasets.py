"""
brainvision.data.datasets — PyTorch Dataset classes for HSI classification.

Provides pixel-level (1D) and patch-level (2D/3D) datasets for training
neural networks on preprocessed hyperspectral patient cubes.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class HSIPixelDataset(Dataset):
    """
    Flattens preprocessed patient cubes into (spectrum, label) pixel pairs.
    Excludes unlabelled pixels (label == 0).
    Labels are shifted to 0-indexed: 1→0, 2→1, 3→2, 4→3.
    """

    def __init__(self, patients: list[dict]):
        spectra, labels = [], []
        for p in patients:
            cube = p["processed"]
            flat_pixels = cube.reshape(-1, cube.shape[-1])
            flat_labels = p["labels"].reshape(-1)
            mask = flat_labels > 0
            spectra.append(flat_pixels[mask])
            labels.append(flat_labels[mask])

        self.X = torch.tensor(np.concatenate(spectra), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(labels), dtype=torch.long) - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HSIPatchDataset(Dataset):
    """
    Extracts fixed-size spatial patches centred on each labelled pixel.
    Used for 2D-CNN and 3D-CNN models that need spatial context.

    Each sample:
      patch : (C, patch_size, patch_size)  float32  — spectral-spatial patch
      label : int64                                  — 0-indexed class label
    """

    def __init__(self, patients: list[dict], patch_size: int = 5):
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.patch_size = patch_size
        self.half = patch_size // 2

        self.patches = []
        self.labels = []

        for p in patients:
            cube = p["processed"]  # (H, W, C)
            labels = p["labels"]  # (H, W)
            H, W, C = cube.shape

            # Pad cube spatially so border pixels get full patches
            padded = np.pad(
                cube,
                ((self.half, self.half), (self.half, self.half), (0, 0)),
                mode="reflect",
            )  # (H+pad, W+pad, C)

            ys, xs = np.where(labels > 0)  # labelled pixel coords
            for y, x in zip(ys, xs):
                patch = padded[y : y + patch_size, x : x + patch_size, :]  # (P, P, C)
                self.patches.append(patch.transpose(2, 0, 1))  # (C, P, P)
                self.labels.append(labels[y, x] - 1)  # 0-indexed

        self.patches = torch.tensor(np.stack(self.patches), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]
