"""
brainvision.data — Data loading and dataset utilities.

Submodules
----------
io        : ENVI file loading, zip handling, campaign summaries.
datasets  : PyTorch Dataset classes for pixel-level and patch-level HSI data.
"""

from brainvision.data.datasets import HSIPatchDataset, HSIPixelDataset
from brainvision.data.io import (
    all_campaigns_summary,
    campaign_summary,
    find_envi_pair,
    list_patient_zips,
    load_envi_cube,
    load_labels,
    load_patient,
    load_processed_patients,
    print_campaign_summary,
    unzip_patient,
)

__all__ = [
    # Datasets
    "HSIPixelDataset",
    "HSIPatchDataset",
    # I/O
    "unzip_patient",
    "find_envi_pair",
    "load_envi_cube",
    "load_labels",
    "load_patient",
    "list_patient_zips",
    "campaign_summary",
    "print_campaign_summary",
    "all_campaigns_summary",
    "load_processed_patients",
]
