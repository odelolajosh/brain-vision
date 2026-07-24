def build_run_name(model:          str,
                   loss_fn:        str,
                   strategy:       str,
                   fold:           int  | None,
                   reduce_pixels:  bool) -> str:
    """
    Construct a canonical run name encoding all experimental conditions.

    Format:
      {model}_{loss}_{balance}_{fold_or_strategy}

    Examples:
      1dcnn_ce_bal_fold1_vpfabelo
      1dnn_fl_nobal_vp1
      hybridsn_ufl_bal_fold3_vpfabelo
      2dcnn_ce_bal_vp2
      1dnnfabelo_ce_bal_fold1_lopo

    Args:
        model         : MODEL config value e.g. '1D-CNN', 'HybridSN'
        loss_fn       : LOSS_FN config value e.g. 'CE', 'UFL'
        strategy      : STRATEGY config value e.g. 'vp_fabelo', 'vp1', 'lopo'
        fold          : fold number (int) for fold-based strategies,
                        None for single-split strategies (vp1, vp2)
        reduce_pixels : REDUCE_PIXELS flag — True → 'bal', False → 'nobal'

    Returns:
        run_name string e.g. '1dcnn_ce_bal_fold1_vpfabelo'
    """
    model_tag    = model.lower().replace('-', '')
    loss_tag     = loss_fn.lower()
    balance_tag  = "bal" if reduce_pixels else "nobal"
    strategy_tag = strategy.replace("_", "")   # vp_fabelo → vpfabelo

    if fold is not None:
        fold_label = f"fold{fold}_{strategy_tag}"
    else:
        fold_label = strategy_tag

    return f"{model_tag}_{loss_tag}_{balance_tag}_{fold_label}"