"""
brainvision.losses — Loss functions for HSI classification.

Includes Cross-Entropy, Focal Loss, Dice Loss, and Unified Focal Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CrossEntropyLoss = nn.CrossEntropyLoss


# ── FOCAL LOSS ────────────────────────────────────────────────────────────────
# Lin et al. (2018) — "Focal Loss for Dense Object Detection"
# Down-weights easy examples so training focuses on hard minority classes
# FL(p) = -alpha * (1 - p)^gamma * log(p)
# gamma=2.0 is the value used in the original paper and your proposal


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    Lin et al. (2018) — https://doi.org/10.1109/TPAMI.2018.2858826

    Args:
        alpha  : class weight tensor (n_classes,) — handles class imbalance
        gamma  : focusing parameter — higher = more focus on hard examples
                 0.0 reduces to standard cross entropy
    Input:
        logits  : (B, C) raw unnormalised scores
        targets : (B,)   ground truth class indices
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample CE loss without reduction
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ── DICE LOSS ─────────────────────────────────────────────────────────────────
# Directly optimises the overlap between predicted and ground truth regions
# More robust to class imbalance than CE — large classes don't dominate
# Dice = 2 * |P ∩ G| / (|P| + |G|)
# Loss = 1 - Dice (minimise)


class DiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss.

    Args:
        eps : smoothing term to avoid division by zero
    Input:
        logits  : (B, C) raw unnormalised scores
        targets : (B,)   ground truth class indices
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C)
        # One-hot encode targets
        one_hot = F.one_hot(targets, n_classes).float()  # (B, C)

        # Compute per-class Dice score
        intersection = (probs * one_hot).sum(dim=0)  # (C,)
        union = (probs + one_hot).sum(dim=0)  # (C,)
        dice_per_class = (2 * intersection + self.eps) / (union + self.eps)  # (C,)

        # Average across classes and convert to loss
        return 1 - dice_per_class.mean()


# ── UNIFIED FOCAL LOSS ───────────────────────────────────────────────────────
# Yeung et al. (2022) — "Unified Focal Loss: Generalising Dice and cross entropy
#                         based losses to handle class imbalanced medical image
#                         segmentation"
# Combines Focal Loss and Dice Loss into a single objective
# UFL = lambda * FocalLoss + (1 - lambda) * DiceLoss
# lambda=0.5, delta=0.6 as per your proposal


class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss.
    Yeung et al. (2022) — https://doi.org/10.1016/j.neunet.2021.11.015

    Args:
        alpha  : class weight tensor (n_classes,) passed to FocalLoss
        gamma  : focusing parameter for FocalLoss          (default 2.0)
        lmbda  : weight balancing FocalLoss vs DiceLoss    (default 0.5)
        delta  : foreground/background weighting in Dice   (default 0.6)
        eps    : smoothing term for DiceLoss               (default 1e-6)
    Input:
        logits  : (B, C) raw unnormalised scores
        targets : (B,)   ground truth class indices
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        lmbda: float = 0.5,
        delta: float = 0.6,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lmbda = lmbda
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.lmbda * focal_loss + (1 - self.lmbda) * dice_loss
