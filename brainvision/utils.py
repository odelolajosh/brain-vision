import os
import torch
import random
import numpy as np

def build_run_name(model:          str,
                    loss_fn:       str,
                    strategy:      str,
                    fold:          int | None,
                    reduce_pixels: bool,
                    sf_mode:       str | None = None) -> str:
    """
    Construct a canonical run name encoding all experimental conditions.

    Format:
      {model}_{loss}_{balance}_{fold_or_strategy}
      {model}_{sf_mode}_{loss}_{balance}_{fold_or_strategy}   ← SpectralFormer only

    Examples:
      1dcnn_ce_bal_fold1_vpfabelo
      1dnn_fl_nobal_vp1
      hybridsn_ufl_bal_fold3_vpfabelo
      2dcnn_ce_bal_vp2
      1dnnfabelo_ce_bal_fold1_lopo
      spectralformer_vit_ce_bal_fold1_vpfabelo
      spectralformer_caf_ufl_bal_fold3_vpfabelo

    Args:
        model         : MODEL config value e.g. '1D-CNN', 'HybridSN', 'SpectralFormer'
        loss_fn       : LOSS_FN config value e.g. 'CE', 'UFL'
        strategy      : STRATEGY config value e.g. 'vp_fabelo', 'vp1', 'lopo'
        fold          : fold number (int) for fold-based strategies,
                        None for single-split strategies (vp1, vp2)
        reduce_pixels : REDUCE_PIXELS flag — True → 'bal', False → 'nobal'
        sf_mode       : SpectralFormer variant — 'ViT' or 'CAF'. Required when
                        model == 'SpectralFormer'; ignored (must be None) for
                        all other models.

    Returns:
        run_name string e.g. 'spectralformer_vit_ce_bal_fold1_vpfabelo'

    Raises:
        ValueError: if model is SpectralFormer and sf_mode is not provided,
                    or if sf_mode is provided for a non-SpectralFormer model.
    """
    model_tag    = model.lower().replace('-', '')
    loss_tag     = loss_fn.lower()
    balance_tag  = "bal" if reduce_pixels else "nobal"
    strategy_tag = strategy.replace("_", "")   # vp_fabelo → vpfabelo

    is_spectralformer = model_tag == "spectralformer"

    if is_spectralformer and sf_mode is None:
        raise ValueError(
            "sf_mode ('ViT' or 'CAF') is required when model == 'SpectralFormer'"
        )

    if is_spectralformer:
        sf_mode = sf_mode if sf_mode is not None else ""
        model_tag = f"{model_tag}_{sf_mode.lower()}"

    if fold is not None:
        fold_label = f"fold{fold}_{strategy_tag}"
    else:
        fold_label = strategy_tag

    return f"{model_tag}_{loss_tag}_{balance_tag}_{fold_label}"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
