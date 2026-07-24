"""
brainvision.data.datasets — PyTorch Dataset classes for HSI classification.

Provides pixel-level (1D) and patch-level (2D/3D) datasets for training
neural networks on preprocessed hyperspectral patient cubes.
"""

import numpy as np
import torch
import random
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
    Memory-efficient patch dataset — extracts patches on-the-fly
    rather than pre-storing all patches in memory.
    """

    def __init__(self, patients:   list[dict],
                 patch_size:    int  = 7,
                 balance:       bool = False,
                 augment:       bool = False,
                 seed:          int  = 42,
                 n_classes:     int = 4):
        assert patch_size % 2 == 1
        half             = patch_size // 2
        self.patch_size  = patch_size
        self.half        = half
        self.augment     = augment

        # Store padded cubes in memory (much smaller than patches)
        # (H+pad, W+pad, C) float32 per patient
        # e.g. (393, 349, 128) × 4 bytes = ~70 MB per patient
        # 35 patients × 70 MB = ~2.4 GB — manageable
        self.padded_cubes = []
        candidates        = {c: [] for c in range(n_classes)}

        for p in patients:
            cube      = p['processed']       # (H, W, C)
            label_map = p['labels']          # (H, W)

            padded    = np.pad(
                cube,
                ((half, half), (half, half), (0, 0)),
                mode='reflect'
            ).astype(np.float32)

            pidx = len(self.padded_cubes)
            self.padded_cubes.append(padded)

            ys, xs = np.where(label_map > 0)
            for y, x in zip(ys, xs):
                c = int(label_map[y, x]) - 1
                candidates[c].append((pidx, y, x))

        # Balance centre pixels if requested
        if balance:
            rng       = random.Random(seed)
            min_count = min(len(v) for v in candidates.values() if v)
            print(f"  Balancing → {min_count} centres per class")
            for c in candidates:
                if len(candidates[c]) > min_count:
                    candidates[c] = rng.sample(candidates[c], min_count)

        # Store only centre pixel coordinates — NOT the patches themselves
        self.centres = []   # list of (pidx, y, x, label)
        for c, centres in candidates.items():
            for (pidx, y, x) in centres:
                self.centres.append((pidx, y, x, c))

        # Shuffle
        rng2 = random.Random(seed + 1)
        rng2.shuffle(self.centres)

        # Extract labels into self.y for compute_class_weights compatibility
        self.y = torch.tensor([c for _, _, _, c in self.centres], dtype=torch.long)

        print(f"  Lazy dataset: {len(self.centres):,} patches "
              f"(extracted on-the-fly)")

    def __len__(self):
        return len(self.centres)

    def __getitem__(self, idx):
        pidx, y, x, label = self.centres[idx]
        padded = self.padded_cubes[pidx]
        p      = self.patch_size

        # Extract patch on-the-fly — (P, P, C) → (C, P, P)
        patch = padded[y:y+p, x:x+p, :].transpose(2, 0, 1).copy()
        patch = torch.tensor(patch, dtype=torch.float32)

        if self.augment:
            if random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])
            if random.random() > 0.5:
                patch = torch.flip(patch, dims=[1])
            k = random.randint(0, 3)
            if k > 0:
                patch = torch.rot90(patch, k=k, dims=[1, 2])

        return patch, torch.tensor(label, dtype=torch.long)


class _HSIPatchDataset(Dataset):
    """
    Extracts fixed-size spatial patches centred on each labelled pixel.
    Used for 2D-CNN, 3D-CNN, and HybridSN models.

    Each sample:
      patch : (C, patch_size, patch_size)  float32
      label : int64  — 0-indexed (0=NT, 1=TT, 2=BV, 3=BG)

    Args:
        patients  : list of preprocessed patient dicts
        patch_size: spatial size of each patch (must be odd)
        balance   : if True, undersample all classes to minority class count
                    following Fabelo et al. (2019) random balancing approach.
                    Applied before patch extraction — only minority-class-sized
                    subsets of centre pixels are used per class.
        augment   : if True, apply random flips and rotations on-the-fly
                    following Fabelo et al. (2019) 800% augmentation.
                    Should be True for training, False for val/test.
        seed      : random seed for reproducible balancing
    """

    def __init__(self, patients:   list[dict],
                 patch_size:    int  = 7,
                 balance:       bool = False,
                 augment:       bool = False,
                 seed:          int  = 42,
                 n_classes:     int = 4):
        assert patch_size % 2 == 1, "patch_size must be odd"
        half = patch_size // 2

        # Collect all (cube, y, x, label) centre candidates
        # We collect candidate centre pixels first — before extracting patches
        # This lets us balance at the centre-pixel level cheaply
        candidates = {c: [] for c in range(n_classes)}   # class → list of (cube_padded, y, x)

        padded_cubes = []
        for p in patients:
            cube      = p['processed']                    # (H, W, C)
            label_map = p['labels']                       # (H, W)
            H, W, C   = cube.shape

            padded = np.pad(cube,
                            ((half, half), (half, half), (0, 0)),
                            mode='reflect')               # (H+2*half, W+2*half, C)
            padded_idx = len(padded_cubes)
            padded_cubes.append(padded)

            ys, xs = np.where(label_map > 0)
            for y, x in zip(ys, xs):
                c = int(label_map[y, x]) - 1             # 0-indexed class
                candidates[c].append((padded_idx, y, x))

        # Balance — undersample to minority class count
        if balance:
            rng       = random.Random(seed)
            min_count = min(len(v) for v in candidates.values() if len(v) > 0)
            print(f"  Balancing patches → {min_count} centre pixels per class")
            for c in candidates:
                if len(candidates[c]) > min_count:
                    candidates[c] = rng.sample(candidates[c], min_count)

        # Extract patches from selected centres
        patches, labels = [], []
        for c, centres in candidates.items():
            for (pidx, y, x) in centres:
                padded = padded_cubes[pidx]
                patch  = padded[y:y+patch_size,
                                x:x+patch_size, :]        # (P, P, C)
                patches.append(patch.transpose(2, 0, 1))  # (C, P, P)
                labels.append(c)

        self.patches   = torch.tensor(np.stack(patches), dtype=torch.float32)
        self.y         = torch.tensor(labels,            dtype=torch.long)
        self.augment   = augment
        self.patch_size = patch_size

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]                          # (C, P, P)
        label = self.y[idx]

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])
            # Random vertical flip
            if random.random() > 0.5:
                patch = torch.flip(patch, dims=[1])
            # Random 90° rotation (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                patch = torch.rot90(patch, k=k, dims=[1, 2])

        return patch, label


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