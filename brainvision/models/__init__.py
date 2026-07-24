"""
brainvision.models — Neural network architectures for HSI classification.

Available models:
  Baseline1DDNN    : Fully connected DNN (Fabelo et al., 2023)
  HuEtAl1DCNN      : 1D-CNN (Hu et al., 2015)
  LeeEtAl2DCNN     : Contextual 2D-CNN (Lee & Kwon, 2016)
  HamidaEtAl3DCNN  : 3D-CNN (Ben Hamida et al., 2018)
  HybridSN         : Hybrid 3D+2D CNN (Roy et al., 2020)
  SpectralFormer   : Transformer-based (Hong et al., 2021)
"""

from brainvision.models.baseline_dnn import Baseline1DDNN
from brainvision.models.fabelo_dnn import FabeloDNN
from brainvision.models.fabelo_2dcnn import Fabelo2DCNN
from brainvision.models.simple_2dcnn import Simple2DCNN
from brainvision.models.hamida_3dcnn import HamidaEtAl3DCNN
from brainvision.models.hu_1dcnn import HuEtAl1DCNN
from brainvision.models.hybridsn import HybridSN
from brainvision.models.lee_2dcnn import LeeEtAl2DCNN
from brainvision.models.spectralformer import SpectralFormer

__all__ = [
    "Baseline1DDNN",
    "FabeloDNN",
    "Fabelo2DCNN",
    "HuEtAl1DCNN",
    "Simple2DCNN",
    "LeeEtAl2DCNN",
    "HamidaEtAl3DCNN",
    "HybridSN",
    "SpectralFormer",
]
