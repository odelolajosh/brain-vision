"""
brainvision.validation — Patient-level data splitting and validation procedures.

Implements three validation strategies from the project proposal:
  VP1 : Campaign-based holdout (C1+C2 train/val, C3 test)
  VP2 : Stratified 60/20/20 across all campaigns
  VP3 : K-Fold cross-validation on C1+C2, C3 held out
"""

import random

import numpy as np
from sklearn.model_selection import KFold

from brainvision.data.io import load_processed_patients  # noqa: F401 — re-exported


def _patient_level_split(
    patients: list[dict], val_ratio: float = 0.20, seed: int = 42
) -> tuple[list, list]:
    """
    Split a patient list into train/val at patient level.
    Returns (train_patients, val_patients).
    """
    patient_ids = sorted(set(p["id"].split("-")[0] for p in patients))
    rng = random.Random(seed)
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_ratio))
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])

    train = [p for p in patients if p["id"].split("-")[0] in train_ids]
    val = [p for p in patients if p["id"].split("-")[0] in val_ids]
    return train, val


def _print_splits(splits: dict, label: str):
    """Print a summary of a train/val/test split."""
    print(f"\n{label} splits:")
    for s, patients in splits.items():
        pids = sorted(set(p["id"].split("-")[0] for p in patients))
        print(f"  {s:5s}: {len(patients):3d} images  {len(pids):2d} patients → {pids}")


def verify_no_leakage(splits: dict):
    """Assert no patient appears in more than one split."""
    train_p = set(p["id"].split("-")[0] for p in splits["train"])
    val_p = set(p["id"].split("-")[0] for p in splits["val"])
    test_p = set(p["id"].split("-")[0] for p in splits["test"])

    assert train_p.isdisjoint(val_p), "Leak: patient in both train and val!"
    assert train_p.isdisjoint(test_p), "Leak: patient in both train and test!"
    assert val_p.isdisjoint(test_p), "Leak: patient in both val and test!"
    print("✅ No patient-level leakage detected")


def build_splits_vp1(campaigns: dict[int, list[dict]], seed: int = 42) -> dict:
    """
    Validation Procedure 1:
      Train + Val : Campaign 1 + Campaign 2  (80/20 patient-level split)
      Test        : Campaign 3 (all patients)
    """
    train_val = campaigns[1] + campaigns[2]
    test = campaigns[3]

    train, val = _patient_level_split(train_val, val_ratio=0.20, seed=seed)

    splits = {"train": train, "val": val, "test": test}
    _print_splits(splits, "VP1")
    return splits


def build_splits_vp2(campaigns: dict[int, list[dict]], seed: int = 42) -> dict:
    """
    Validation Procedure 2:
      Stratified 60/20/20 split across all three campaigns.
      Each campaign contributes equally to every split.
    """
    splits = {"train": [], "val": [], "test": []}

    for campaign_id, patients in campaigns.items():
        patient_ids = sorted(set(p["id"].split("-")[0] for p in patients))
        rng = random.Random(seed + campaign_id)
        shuffled = patient_ids.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = max(1, round(n * 0.20))
        n_val = max(1, round(n * 0.20))
        n_train = n - n_test - n_val

        train_ids = set(shuffled[:n_train])
        val_ids = set(shuffled[n_train : n_train + n_val])
        test_ids = set(shuffled[n_train + n_val :])

        for p in patients:
            pid = p["id"].split("-")[0]
            if pid in train_ids:
                splits["train"].append(p)
            elif pid in val_ids:
                splits["val"].append(p)
            else:
                splits["test"].append(p)

        print(
            f"  Campaign {campaign_id}: "
            f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients"
        )

    _print_splits(splits, "VP2")
    return splits


def build_splits_vp3(
    campaigns: dict[int, list[dict]], n_folds: int = 5, seed: int = 42
) -> list[dict]:
    """
    Validation Procedure 3:
      K-Fold cross-validation on Campaign 1 + Campaign 2 (patient level).
      Test: Campaign 3 (held out across all folds).

    Returns a list of fold dicts:
      [
        { 'fold': 1, 'train': [...], 'val': [...], 'test': [...] },
        ...
      ]
    """
    train_val_patients = campaigns[1] + campaigns[2]
    test_patients = campaigns[3]

    # Unique patient IDs for fold assignment
    patient_ids = sorted(set(p["id"].split("-")[0] for p in train_val_patients))
    patient_ids = np.array(patient_ids)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        train_ids = set(patient_ids[train_idx])
        val_ids = set(patient_ids[val_idx])

        fold_train = [
            p for p in train_val_patients if p["id"].split("-")[0] in train_ids
        ]
        fold_val = [p for p in train_val_patients if p["id"].split("-")[0] in val_ids]

        folds.append(
            {
                "fold": fold_idx + 1,
                "train": fold_train,
                "val": fold_val,
                "test": test_patients,
            }
        )

        print(
            f"  Fold {fold_idx + 1}: "
            f"{len(fold_train)} train / {len(fold_val)} val images  "
            f"| val patients: {sorted(val_ids)}"
        )

    print(f"  Test (Campaign 3): {len(test_patients)} images — fixed across all folds")
    return folds


# def build_splits_fabelo(campaigns: dict[int, list[dict]],
#                         n_folds:   int = 5,
#                         seed:      int = 42) -> list[dict]:
#     """
#     Replicates Fabelo et al. (2023) three-way data partition with 5-fold CV.
#     Leon et al. npj Precision Oncology, 2023.

#     - All three campaigns pooled at patient level
#     - Each fold: random 60/20/20 patient-level split
#     - 5 independent folds with different random seeds
#     - Results should be reported as median ± std across folds

#     Returns list of 5 fold dicts:
#       [
#         { 'fold': 1, 'train': [...], 'val': [...], 'test': [...] },
#         ...
#       ]
#     """
#     # Pool all patients from all campaigns
#     all_patients = [p for patients in campaigns.values() for p in patients]

#     # Build patient → images mapping
#     patient_map = {}
#     for p in all_patients:
#         pid = p['id'].split('-')[0]
#         patient_map.setdefault(pid, []).append(p)

#     patient_ids = sorted(patient_map.keys())
#     n           = len(patient_ids)

#     # Test set - 20% of n
#     n_test      = max(1, round(n * 0.20))
#     rng_test    = random.Random(seed)
#     shuffled    = patient_ids.copy()
#     rng_test.shuffle(shuffled)

#     test_ids    = set(shuffled[:n_test])
#     pool_ids    = shuffled[n_test:]   # remaining 80% — used for train/val

#     test_patients = [p for pid in test_ids for p in patient_map[pid]]

#     n_pool  = len(pool_ids)
#     n_val   = max(1, round(n_pool * 0.25))   # 25% of 80% = 20% of n
#     n_train = n_pool - n_val                  # 75% of 80% = 60% of n

#     print(f"Total patients        : {n}")
#     print(f"Test set (fixed)      : {n_test} patients  "
#           f"({n_test/n*100:.0f}% of X)  — shared across all folds")
#     print(f"Pool for train/val    : {n_pool} patients  "
#           f"({n_pool/n*100:.0f}% of X)")
#     print(f"Per fold → train      : ~{n_train} patients  "
#           f"({n_train/n*100:.0f}% of X)")
#     print(f"Per fold → val        : ~{n_val} patients  "
#           f"({n_val/n*100:.0f}% of X)")
#     print(f"Folds                 : {n_folds}\n")

#     # Build folds — each reshuffles the 80% pool independently
#     folds = []

#     for fold_idx in range(n_folds):
#         # Independent seed per fold — different train/val partition each time
#         rng      = random.Random(seed + fold_idx + 1)
#         shuffled_pool = pool_ids.copy()
#         rng.shuffle(shuffled_pool)

#         val_ids   = set(shuffled_pool[:n_val])
#         train_ids = set(shuffled_pool[n_val:])

#         fold_train = [p for pid in train_ids for p in patient_map[pid]]
#         fold_val   = [p for pid in val_ids   for p in patient_map[pid]]

#         folds.append({
#             'fold'  : fold_idx + 1,
#             'train' : fold_train,
#             'val'   : fold_val,
#             'test'  : test_patients,   # ← same for every fold
#         })

#         print(f"  Fold {fold_idx+1}: "
#               f"train={len(fold_train):2d} images "
#               f"({len(train_ids)} patients)  "
#               f"val={len(fold_val):2d} images "
#               f"({len(val_ids)} patients)  "
#               f"test={len(test_patients):2d} images "
#               f"({n_test} patients, fixed)")

#     print(f"\n  Test patients: {sorted(test_ids)}")
#     return folds


def build_splits_fabelo(campaigns: dict[int, list[dict]],
                        n_folds:   int = 5,
                        seed:      int = 42) -> list[dict]:
    """
    Replicates Fabelo et al. (2023) three-way data partition with 5-fold CV.

    Structure:
      - All three campaigns pooled at patient level
      - 20% of patients held out as a FIXED test set across all folds
      - Remaining 80% split into n_folds using K-Fold CV
      - K-Fold guarantees every pool patient appears in val exactly once
      - This gives approximately 60/20/20 of total X per fold:
          Train : (n_folds-1)/n_folds × 80% ≈ 60% of X
          Val   :          1/n_folds  × 80% ≈ 20% of X  (with n_folds=5)
          Test  : 20% of X  (fixed, identical across all folds)

    Args:
        campaigns : dict mapping campaign_id → list of patient dicts
        n_folds   : number of cross-validation folds (default 5)
        seed      : random seed for test set selection and K-Fold shuffle

    Returns list of fold dicts, each with:
      {
        'fold'  : int,
        'train' : list[dict],
        'val'   : list[dict],
        'test'  : list[dict],   ← identical across all folds
      }
    """
    from sklearn.model_selection import KFold

    # ── Pool all patients ─────────────────────────────────────────────────────
    all_patients = [p for patients in campaigns.values() for p in patients]

    patient_map = {}
    for p in all_patients:
        pid = p['id'].split('-')[0]
        patient_map.setdefault(pid, []).append(p)

    patient_ids = sorted(patient_map.keys())
    n           = len(patient_ids)

    # ── Carve out fixed test set — 20% of X ──────────────────────────────────
    n_test   = max(1, round(n * 0.20))
    rng      = random.Random(seed)
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)

    test_ids      = set(shuffled[:n_test])
    pool_ids      = np.array(sorted(set(shuffled[n_test:])))
    test_patients = [p for pid in test_ids for p in patient_map[pid]]

    n_pool = len(pool_ids)

    print(f"Total patients        : {n}")
    print(f"Test set (fixed)      : {n_test} patients  "
          f"({n_test/n*100:.0f}% of X)  — shared across all folds")
    print(f"Pool for K-Fold CV    : {n_pool} patients  "
          f"({n_pool/n*100:.0f}% of X)")
    print(f"K-Fold                : {n_folds} folds  "
          f"(every pool patient appears in val exactly once)")
    print(f"Per fold → train      : ~{n_pool*(n_folds-1)//n_folds} patients  "
          f"({n_pool*(n_folds-1)/n_folds/n*100:.0f}% of X)")
    print(f"Per fold → val        : ~{n_pool//n_folds} patients  "
          f"({n_pool//n_folds/n*100:.0f}% of X)")
    print(f"Test patients         : {sorted(test_ids)}\n")

    # ── K-Fold on pool — guarantees full coverage ─────────────────────────────
    kf    = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(pool_ids)):
        train_ids = set(pool_ids[train_idx])
        val_ids   = set(pool_ids[val_idx])

        fold_train = [p for pid in train_ids for p in patient_map[pid]]
        fold_val   = [p for pid in val_ids   for p in patient_map[pid]]

        folds.append({
            'fold'  : fold_idx + 1,
            'train' : fold_train,
            'val'   : fold_val,
            'test'  : test_patients,    # ← fixed across all folds
        })

        print(f"  Fold {fold_idx+1}: "
              f"train={len(fold_train):2d} images "
              f"({len(train_ids):2d} pts)  "
              f"val={len(fold_val):2d} images "
              f"({len(val_ids):2d} pts)  "
              f"test={len(test_patients):2d} images "
              f"({n_test} pts, fixed)")

    return folds


def build_splits_lopo(campaigns: dict[int, list[dict]],
                      seed:      int = 42) -> list[dict]:
    """
    Leave-One-Patient-Out (LOPO) cross-validation.

    Follows Fabelo et al. (2019) — Sensors, 19(4), 920.
    https://doi.org/10.3390/s19040920

    - Train+Val pool: Campaign 1 + Campaign 2 (benchmark subset)
    - Each fold: one patient held out as validation, all others train
    - Test: Campaign 3 (fixed, held out across all folds)
    - Number of folds = number of unique patients in C1+C2

    Used specifically for FabeloDNN replication experiments.
    Results reported as mean ± std across all folds.

    Returns list of fold dicts:
      [
        {
          'fold'        : int,
          'val_patient' : str,   # patient ID held out e.g. '012'
          'train'       : [...],
          'val'         : [...],
          'test'        : [...],
        },
        ...
      ]
    """
    train_val_patients = campaigns[1] + campaigns[2]
    test_patients      = campaigns[3]

    # Group images by patient
    patient_map = {}
    for p in train_val_patients:
        pid = p['id'].split('-')[0]
        patient_map.setdefault(pid, []).append(p)

    patient_ids = sorted(patient_map.keys())
    n_patients  = len(patient_ids)

    print(f"LOPO cross-validation")
    print(f"  Train+Val pool : Campaign 1 + Campaign 2")
    print(f"  Unique patients: {n_patients}")
    print(f"  Folds          : {n_patients}  (one per patient)")
    print(f"  Test           : Campaign 3 ({len(test_patients)} images, fixed)\n")

    folds = []

    for i, val_pid in enumerate(patient_ids):
        val_images   = patient_map[val_pid]
        train_images = [
            p
            for pid, images in patient_map.items()
            if pid != val_pid
            for p in images
        ]

        folds.append({
            'fold'        : i + 1,
            'val_patient' : val_pid,
            'train'       : train_images,
            'val'         : val_images,
            'test'        : test_patients,
        })

        print(f"  Fold {i+1:>2} — val patient: {val_pid}  "
              f"({len(val_images)} images)  "
              f"train: {len(train_images)} images")

    print(f"\n  Total folds: {len(folds)}")
    return folds


_split_cache = {}

def get_splits(campaigns: dict[int, list[dict]], strategy: str) -> list[dict]:
    """Build splits for a strategy — cached in memory."""
    if strategy in _split_cache:
        return _split_cache[strategy]

    if strategy == 'vp1':
        s = build_splits_vp1(campaigns)
        verify_no_leakage(s)
        splits = [{'fold': None, **s}]
    elif strategy == 'vp2':
        s = build_splits_vp2(campaigns)
        verify_no_leakage(s)
        splits = [{'fold': None, **s}]
    elif strategy == 'vp3':
        splits = build_splits_vp3(campaigns, n_folds=5)
        for f in splits: verify_no_leakage(f)
    elif strategy == 'lopo':
        splits = build_splits_lopo(campaigns)
        for f in splits: verify_no_leakage(f)
    elif strategy == 'vp_fabelo':
        splits = build_splits_fabelo(campaigns, n_folds=5)
        for f in splits: verify_no_leakage(f)
    else:
        raise ValueError(
                f"Unknown STRATEGY '{strategy}'. "
                f"Choose 'vp1', 'vp2', 'vp3', 'lopo', or 'vp_fabelo'."
            )

    _split_cache[strategy] = splits
    return splits
