"""
1D Deep Neural Network architectures for HSI classification.

Faithful replication of Fabelo et al. (2019) Sensors
28→40 nodes, ReLU, no BatchNorm, no dropout
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class FabeloDNN(nn.Module):
    """
    Faithful replication of the 1D-DNN from:

    Fabelo, H., Halicek, M., Ortega, S., et al. (2019).
    Deep Learning-Based Framework for In Vivo Identification of
    Glioblastoma Tumor using Hyperspectral Images of Human Brain.
    Sensors, 19(4), 920. https://doi.org/10.3390/s19040920

    Architecture (Section 2.3):
      - Two hidden layers: 28 nodes → 40 nodes
      - Activation: ReLU
      - No BatchNorm, no Dropout (not mentioned in paper)
      - Training: LR=0.1, 45 epochs (multiclass), LOPO cross-validation
      - Balancing: random balance to minority class (TT)
      - Framework: TensorFlow (replicated here in PyTorch)

    Input:  (B, input_channels)  — one pixel spectrum per sample
    Output: (B, n_classes)       — raw logits
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity="relu")
            init.zeros_(m.bias)

    def __init__(self, input_channels: int = 128, n_classes: int = 4):
        super().__init__()

        self.fc1 = nn.Linear(input_channels, 28)
        self.fc2 = nn.Linear(28, 40)
        self.fc3 = nn.Linear(40, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
