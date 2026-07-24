"""
brainvision.metrics — Evaluation metrics for HSI classification.

All notebooks import from here to guarantee consistency.
Metrics follow Fabelo et al. (2023) conventions:
  - macro F1 and Dice excluding Background (no_bg variants)
  - per-class sensitivity, specificity, Dice
  - Overall Accuracy (OA)
"""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from brainvision.constants import N_CLASSES, CLASS_NAMES


def compute_metrics(targets:  np.ndarray,
                    preds:    np.ndarray,
                    run_name: str = "") -> dict:
    """
    Compute full evaluation metrics from ground truth and predictions.

    Follows Fabelo et al. (2023) — macro F1 and Dice excluding BG
    are the primary reported metrics.

    Args:
        targets  : (N,) int array of ground truth labels (0-indexed)
        preds    : (N,) int array of predicted labels   (0-indexed)
        run_name : optional identifier stored in the returned dict

    Returns dict with:
        run_name          : str
        oa                : float  — overall accuracy
        macro_f1          : float  — macro F1, all classes
        macro_f1_no_bg    : float  — macro F1, NT+TT+BV only  ← Fabelo metric
        macro_dice        : float  — macro Dice, all classes
        macro_dice_no_bg  : float  — macro Dice, NT+TT+BV only
        sensitivity       : (N_CLASSES,) float — per-class sensitivity
        specificity       : (N_CLASSES,) float — per-class specificity
        dice              : (N_CLASSES,) float — per-class Dice
        confusion_matrix  : (N_CLASSES, N_CLASSES) int
    """
    n   = N_CLASSES
    cm  = confusion_matrix(targets, preds, labels=list(range(n)))

    sensitivity = np.zeros(n)
    specificity = np.zeros(n)
    dice        = np.zeros(n)

    for i in range(n):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        sensitivity[i] = TP / (TP + FN + 1e-6)
        specificity[i] = TN / (TN + FP + 1e-6)
        dice[i]        = (2 * TP) / (2 * TP + FP + FN + 1e-6)

    macro_f1         = f1_score(targets, preds, average='macro',
                                 zero_division=0)
    macro_f1_no_bg   = f1_score(targets, preds, labels=[0, 1, 2],
                                 average='macro', zero_division=0)
    macro_dice       = dice.mean()
    macro_dice_no_bg = dice[:3].mean()    # NT=0, TT=1, BV=2 — exclude BG=3
    oa               = (preds == targets).mean()

    return {
        'run_name'          : run_name,
        'oa'                : float(oa),
        'macro_f1'          : float(macro_f1),
        'macro_f1_no_bg'    : float(macro_f1_no_bg),
        'macro_dice'        : float(macro_dice),
        'macro_dice_no_bg'  : float(macro_dice_no_bg),
        'sensitivity'       : sensitivity,
        'specificity'       : specificity,
        'dice'              : dice,
        'confusion_matrix'  : cm,
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """
    Aggregate per-fold metric dicts into median ± std summary.
    Follows Fabelo et al. (2023) reporting convention.

    Args:
        fold_metrics : list of dicts from compute_metrics(), one per fold

    Returns dict with median and std for all key metrics.
    """
    if not fold_metrics:
        return {}

    def _collect(key):
        return [m[key] for m in fold_metrics]

    def _collect_class(key, idx):
        return [m[key][idx] for m in fold_metrics]

    f1_all   = _collect('macro_f1')
    f1_no_bg = _collect('macro_f1_no_bg')
    dice_all = _collect('macro_dice')
    dice_nbg = _collect('macro_dice_no_bg')
    oas      = _collect('oa')

    classes = ['NT', 'TT', 'BV', 'BG']
    sens = {c: _collect_class('sensitivity', i) for i, c in enumerate(classes)}
    spec = {c: _collect_class('specificity', i) for i, c in enumerate(classes)}
    dice = {c: _collect_class('dice',        i) for i, c in enumerate(classes)}

    return {
        'n_folds'                  : len(fold_metrics),
        # ── Global metrics ────────────────────────────────────────────────────
        'oa_median'                : float(np.median(oas)),
        'oa_std'                   : float(np.std(oas)),
        'macro_f1_median'          : float(np.median(f1_all)),
        'macro_f1_std'             : float(np.std(f1_all)),
        'macro_f1_no_bg_median'    : float(np.median(f1_no_bg)),
        'macro_f1_no_bg_std'       : float(np.std(f1_no_bg)),
        'macro_dice_median'        : float(np.median(dice_all)),
        'macro_dice_std'           : float(np.std(dice_all)),
        'macro_dice_no_bg_median'  : float(np.median(dice_nbg)),
        'macro_dice_no_bg_std'     : float(np.std(dice_nbg)),
        # ── Per-class medians ─────────────────────────────────────────────────
        'per_class_sens_median'    : {c: float(np.median(v))
                                      for c, v in sens.items()},
        'per_class_sens_std'       : {c: float(np.std(v))
                                      for c, v in sens.items()},
        'per_class_spec_median'    : {c: float(np.median(v))
                                      for c, v in spec.items()},
        'per_class_spec_std'       : {c: float(np.std(v))
                                      for c, v in spec.items()},
        'per_class_dice_median'    : {c: float(np.median(v))
                                      for c, v in dice.items()},
        'per_class_dice_std'       : {c: float(np.std(v))
                                      for c, v in dice.items()},
        # ── Per-fold raw values for distribution analysis ─────────────────────
        'per_fold_f1_no_bg'        : f1_no_bg,
        'per_fold_f1_all'          : f1_all,
        'per_fold_dice_no_bg'      : dice_nbg,
        'per_fold_oa'              : oas,
    }


def print_metrics(metrics: dict):
    """Pretty-print a single fold's test metrics."""
    print(f"\n{'═'*75}")
    print(f"  TEST RESULTS — {metrics.get('run_name', '')}")
    print(f"{'═'*75}")
    print(f"  Overall Accuracy (OA)      : {metrics['oa']:.4f}")
    print(f"  Macro F1  (all classes)    : {metrics['macro_f1']:.4f}")
    print(f"  Macro F1  (no BG)          : {metrics['macro_f1_no_bg']:.4f}"
          f"  ← Fabelo benchmark metric")
    print(f"  Macro Dice (all classes)   : {metrics['macro_dice']:.4f}")
    print(f"  Macro Dice (no BG)         : {metrics['macro_dice_no_bg']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"\n  {'Class':<25} {'Sensitivity':>12} {'Specificity':>12}"
          f" {'Dice':>8} {'F1':>8}")
    print(f"  {'─'*68}")

    for i in range(N_CLASSES):
        TP   = cm[i, i]
        FN   = cm[i, :].sum() - TP
        FP   = cm[:, i].sum() - TP
        f1_i = (2 * TP) / (2 * TP + FP + FN + 1e-6)
        marker = "  ← clinical priority" if i == 1 else ""
        print(f"  {CLASS_NAMES[i+1]:<25} "
              f"{metrics['sensitivity'][i]:>12.4f} "
              f"{metrics['specificity'][i]:>12.4f} "
              f"{metrics['dice'][i]:>8.4f} "
              f"{f1_i:>8.4f}"
              f"{marker}")

    print(f"  {'─'*68}")
    print(f"  {'Mean':<25} "
          f"{metrics['sensitivity'].mean():>12.4f} "
          f"{metrics['specificity'].mean():>12.4f} "
          f"{metrics['macro_dice']:>8.4f} "
          f"{metrics['macro_f1']:>8.4f}")


def print_aggregate(aggregate: dict, model: str = "",
                    loss_fn: str = "", strategy: str = ""):
    """Pretty-print aggregated fold metrics."""
    print(f"\n{'═'*65}")
    print(f"  AGGREGATE — {model} × {loss_fn} × {strategy}")
    print(f"  {aggregate['n_folds']} folds")
    print(f"{'═'*65}")
    print(f"  Macro F1  (all)    : "
          f"{aggregate['macro_f1_median']*100:.1f} ± "
          f"{aggregate['macro_f1_std']*100:.1f}%")
    print(f"  Macro F1  (no BG)  : "
          f"{aggregate['macro_f1_no_bg_median']*100:.1f} ± "
          f"{aggregate['macro_f1_no_bg_std']*100:.1f}%"
          f"  ← vs Fabelo 70.2 ± 7.9%")
    print(f"  Macro Dice (all)   : "
          f"{aggregate['macro_dice_median']*100:.1f} ± "
          f"{aggregate['macro_dice_std']*100:.1f}%")
    print(f"  Macro Dice (no BG) : "
          f"{aggregate['macro_dice_no_bg_median']*100:.1f} ± "
          f"{aggregate['macro_dice_no_bg_std']*100:.1f}%")
    print(f"  OA                 : "
          f"{aggregate['oa_median']*100:.1f} ± "
          f"{aggregate['oa_std']*100:.1f}%")

    print(f"\n  {'Class':<6} {'Sens':>8} {'±':>5} {'Spec':>8} {'±':>5}"
          f"  {'Dice':>8} {'±':>5}")
    print(f"  {'─'*55}")
    for cls in ['NT', 'TT', 'BV', 'BG']:
        marker = "  ← clinical priority" if cls == 'TT' else ""
        print(f"  {cls:<6} "
              f"{aggregate['per_class_sens_median'][cls]*100:>7.1f}% "
              f"{'±':>3}{aggregate['per_class_sens_std'][cls]*100:>4.1f}%  "
              f"{aggregate['per_class_spec_median'][cls]*100:>7.1f}% "
              f"{'±':>3}{aggregate['per_class_spec_std'][cls]*100:>4.1f}%  "
              f"{aggregate['per_class_dice_median'][cls]*100:>7.1f}% "
              f"{'±':>3}{aggregate['per_class_dice_std'][cls]*100:>4.1f}%"
              f"{marker}")

    print(f"\n  Per-fold F1 (no BG)  : "
          f"{[f'{v*100:.1f}%' for v in aggregate['per_fold_f1_no_bg']]}")
    print(f"  Per-fold Dice (no BG): "
          f"{[f'{v*100:.1f}%' for v in aggregate['per_fold_dice_no_bg']]}")