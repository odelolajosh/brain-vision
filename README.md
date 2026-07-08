# Brain Vision

Hyperspectral image (HSI) classification for intraoperative brain tumour detection using deep learning.

Built on the [HELICoiD](https://doi.org/10.3390/s19102459) VNIR dataset, this project compares six neural architectures and four loss functions for pixel-level tissue classification into Normal Tissue, Tumour, Blood Vessel, and Background.

## Project Structure

```
brainvision/
├── brainvision/                    # Python package
│   ├── constants.py              # All hyperparameters and paths
│   ├── losses.py                 # CE, Focal, Dice, Unified Focal Loss
│   ├── validation.py             # Patient-level splits (VP1, VP2, VP3)
│   ├── data/
│   │   ├── io.py                 # ENVI loading, zip handling, campaign utils
│   │   └── datasets.py           # HSIPixelDataset, HSIPatchDataset
│   └── models/
│       ├── baseline_dnn.py       # Baseline1DDNN  (Fabelo et al. 2023)
│       ├── hu_1dcnn.py           # HuEtAl1DCNN    (Hu et al. 2015)
│       ├── lee_2dcnn.py          # LeeEtAl2DCNN   (Lee & Kwon 2016)
│       ├── hamida_3dcnn.py       # HamidaEtAl3DCNN (Ben Hamida et al. 2018)
│       ├── hybridsn.py           # HybridSN       (Roy et al. 2020)
│       └── spectralformer.py     # SpectralFormer  (Hong et al. 2021)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Calibration, smoothing, decimation
│   ├── 03_training.ipynb         # Model training loop
│   └── 04_evaluation.ipynb       # Metrics, confusion matrices, plots
├── checkpoints/                  # Saved model weights (.pt)
├── datasets/                     # Raw HELICoiD campaign ZIPs (gitignored)
├── processed/                    # Preprocessed .npz cubes (gitignored)
├── results/                      # Training histories and metrics
├── docs/                         # Reference docs and figures
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
# Clone the repository
git clone https://github.com/odelolajosh/brainvision.git
cd brainvision

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode (so notebook imports work)
pip install -e .
```

## Models

| Model | Type | Reference | Parameters |
|-------|------|-----------|------------|
| Baseline1DDNN | Fully connected DNN | Fabelo et al. (2023) | 17.1M |
| HuEtAl1DCNN | 1D-CNN | Hu et al. (2015) | 77K |
| LeeEtAl2DCNN | 2D-CNN + Inception | Lee & Kwon (2016) | 297K |
| HamidaEtAl3DCNN | 3D-CNN | Ben Hamida et al. (2018) | 33K |
| HybridSN | 3D+2D CNN | Roy et al. (2020) | 2.6M |
| SpectralFormer | Vision Transformer | Hong et al. (2021) | 198K |

Detailed architecture summaries are in [`docs/model_summaries.md`](docs/model_summaries.md).

## Loss Functions

| Loss | Description | Reference |
|------|-------------|-----------|
| Cross-Entropy | Standard CE | — |
| Focal Loss | Down-weights easy examples (γ=2.0) | Lin et al. (2018) |
| Dice Loss | Optimises set overlap directly | — |
| Unified Focal Loss | λ·Focal + (1-λ)·Dice (λ=0.5) | Yeung et al. (2022) |

## Usage

```python
from brainvision.models import SpectralFormer, HybridSN
from brainvision.losses import FocalLoss, UnifiedFocalLoss
from brainvision.data import HSIPixelDataset, HSIPatchDataset, load_processed_patients
from brainvision.constants import *

# Load preprocessed data
patients = load_processed_patients("processed/first_campaign")

# Create datasets
pixel_ds = HSIPixelDataset(patients)           # for 1D models
patch_ds = HSIPatchDataset(patients, patch_size=5)  # for 2D/3D models

# Initialise a model
model = SpectralFormer(input_channels=N_DECIMATED_BANDS, n_classes=N_CLASSES)
```

## Validation Procedures

- **VP1**: Campaign 1+2 (80/20 patient split) → train/val; Campaign 3 → test
- **VP2**: Stratified 60/20/20 across all campaigns
- **VP3**: 5-fold cross-validation on C1+C2; C3 held out