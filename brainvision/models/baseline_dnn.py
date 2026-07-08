"""
Baseline 1D Deep Neural Network for HSI classification.
Fabelo et al. (2023).
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Baseline1DDNN(nn.Module):
    """1D-NN: Fully connected DNN. Fabelo et al. (2023)."""

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity="relu")
            init.zeros_(m.bias)

    def __init__(self, input_channels=128, n_classes=4, dropout=True, dropout_rate=0.5):
        super().__init__()
        self.use_dropout = dropout
        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        return self.fc4(x)
