"""
brainvision.data.datasets — PyTorch Dataset classes for HSI classification.

Provides pixel-level (1D) and patch-level (2D/3D) datasets for training
neural networks on preprocessed hyperspectral patient cubes.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import MiniBatchKMeans

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
        half = patch_size // 2

        patches, labels = [], []   # ← local variables, not self

        for p in patients:
            cube      = p["processed"]
            label_map = p["labels"]
            H, W, C   = cube.shape

            padded = np.pad(
                cube,
                ((half, half), (half, half), (0, 0)),
                mode="reflect",
            )

            ys, xs = np.where(label_map > 0)
            for y, x in zip(ys, xs):
                patch = padded[y:y+patch_size, x:x+patch_size, :]
                patches.append(patch.transpose(2, 0, 1))
                labels.append(label_map[y, x] - 1)

        self.patches = torch.tensor(np.stack(patches), dtype=torch.float32)
        self.y       = torch.tensor(labels,            dtype=torch.long)

    def __len__(self):          return len(self.patches)
    def __getitem__(self, idx): return self.patches[idx], self.y[idx]


def reduce_training_pixels(dataset: HSIPixelDataset,
                            n_per_class: int = 1000, n_classes: int = 4, seed: int = 42) -> HSIPixelDataset:
    """
    Reduce training pixels to n_per_class per class using K-Means centroids.
    Follows Fabelo et al. (2023) — 100 clusters per class, n most similar pixels.
    Balances classes and reduces redundancy.
    """
    n_clusters  = 100
    n_similar   = n_per_class // n_clusters   # pixels per centroid
    X           = dataset.X.numpy()
    y           = dataset.y.numpy()
    keep_idx    = []

    for c in range(n_classes):
        mask     = y == c
        X_c      = X[mask]
        idx_c    = np.where(mask)[0]

        if len(X_c) <= n_per_class:
            # Not enough pixels — keep all
            keep_idx.append(idx_c)
            continue

        # K-Means clustering
        kmeans   = MiniBatchKMeans(n_clusters=n_clusters,
                                   random_state=seed,
                                   n_init=3)
        kmeans.fit(X_c)
        centroids = kmeans.cluster_centers_   # (100, 128)

        # For each centroid find n_similar most similar pixels using SAM
        selected = []
        for centroid in centroids:
            # Spectral Angle Mapper — smaller angle = more similar
            norm_px  = X_c / (np.linalg.norm(X_c, axis=1, keepdims=True) + 1e-6)
            norm_c   = centroid / (np.linalg.norm(centroid) + 1e-6)
            angles   = np.arccos(np.clip(norm_px @ norm_c, -1, 1))
            nearest  = np.argsort(angles)[:n_similar]
            selected.extend(idx_c[nearest])

        keep_idx.append(np.array(selected[:n_per_class]))

    all_idx     = np.concatenate(keep_idx)
    dataset.X   = torch.tensor(X[all_idx], dtype=torch.float32)
    dataset.y   = torch.tensor(y[all_idx], dtype=torch.long)

    print(f"Reduced training set: {len(dataset.X)} pixels "
          f"({n_per_class} per class x {n_classes} classes)")
    return dataset