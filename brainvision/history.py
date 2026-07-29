import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from brainvision.utils import build_run_name

def plot_history_curves(history: dict, run_name: str, out_dir: str | Path):
    """
    Plot loss, Val F1 (all + no-BG), Dice (no-BG),
    sensitivity and specificity across training epochs.
    """
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    epochs  = range(1, len(history['train_loss']) + 1)
    best_ep = history.get('best_epoch', 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Training Curves — {run_name}", fontsize=13)

    def _vline(ax):
        ax.axvline(best_ep, color='gray', linestyle='--',
                   alpha=0.6, label=f'Best epoch ({best_ep})')

    def _scatter_best(ax, series, color):
        if series and best_ep <= len(series):
            ax.scatter([best_ep], [series[best_ep - 1]],
                       color=color, zorder=5, s=60)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train', color='#378ADD')
    ax.plot(epochs, history['val_loss'],   label='Val',   color='#D85A30')
    _vline(ax)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()

    # ── Val F1 (all classes) ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history['val_f1'], color='#5DCAA5')
    best_f1_all = history.get('best_f1_all', history['val_f1'][best_ep - 1])
    _scatter_best(ax, history['val_f1'], '#5DCAA5')
    _vline(ax)
    ax.set_title(f"Val Macro F1 — all  (best={best_f1_all:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.set_ylim(0, 1)

    # ── Val F1 (no BG) ────────────────────────────────────────────────────────
    ax = axes[0, 2]
    if history.get('val_f1_no_bg'):
        best_f1_no_bg = history.get('best_f1',
                                     history['val_f1_no_bg'][best_ep - 1])
        ax.plot(epochs, history['val_f1_no_bg'], color='#7F77DD')
        _scatter_best(ax, history['val_f1_no_bg'], '#7F77DD')
        _vline(ax)
        ax.set_title(f"Val F1 no-BG  (best={best_f1_no_bg:.4f})")
    else:
        ax.text(0.5, 0.5, 'val_f1_no_bg\nnot recorded',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=11)
        ax.set_title("Val F1 no-BG")
        best_f1_no_bg = float('nan')
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.set_ylim(0, 1)

    # ── Val Dice (no BG) — new panel ──────────────────────────────────────────
    ax = axes[1, 0]
    if history.get('val_dice_no_bg'):
        best_dice_no_bg = history.get('best_dice_no_bg',
                                       history['val_dice_no_bg'][best_ep - 1])
        ax.plot(epochs, history['val_dice_no_bg'], color='#D85A30')
        _scatter_best(ax, history['val_dice_no_bg'], '#D85A30')
        _vline(ax)
        ax.set_title(f"Val Dice no-BG  (best={best_dice_no_bg:.4f})")
    else:
        ax.text(0.5, 0.5, 'val_dice_no_bg\nnot recorded',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=11)
        ax.set_title("Val Dice no-BG")
        best_dice_no_bg = float('nan')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1)

    # ── Sensitivity ───────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, history['val_sens'], color='#EF9F27')
    best_sens = history.get('best_sens', history['val_sens'][best_ep - 1])
    _scatter_best(ax, history['val_sens'], '#EF9F27')
    _vline(ax)
    ax.set_title(f"Val Sensitivity  (best={best_sens:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Sensitivity"); ax.set_ylim(0, 1)

    # ── Specificity ───────────────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.plot(epochs, history['val_spec'], color='#639922')
    best_spec = history.get('best_spec', history['val_spec'][best_ep - 1])
    _scatter_best(ax, history['val_spec'], '#639922')
    _vline(ax)
    ax.set_title(f"Val Specificity  (best={best_spec:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Specificity"); ax.set_ylim(0, 1)

    # ── Summary text ──────────────────────────────────────────────────────────
    balance_str = 'bal' if history.get('balance', True) else 'nobal'
    summary = (
        f"Run      : {run_name}\n\n"
        f"Balanced : {balance_str}\n\n"
        f"Best epoch      : {best_ep}\n"
        f"Val F1  (no BG) : {best_f1_no_bg:.4f}\n"
        f"Val Dice(no BG) : {best_dice_no_bg:.4f}\n"
        f"Val Sensitivity : {best_sens:.4f}\n"
        f"Val Specificity : {best_spec:.4f}\n"
        f"Total epochs    : {len(history['train_loss'])}"
    )

    # Repurpose the text panel — move it to a separate figure annotation
    fig.text(0.99, 0.01, summary, ha='right', va='bottom',
             fontsize=8, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{run_name}_curves.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved → {out_dir}/{run_name}_curves.png")



def load_history(cfg: dict, result_dir: str | Path) -> dict | None:
    result_dir = Path(result_dir) if isinstance(result_dir, str) else result_dir
    run_name = build_run_name(
        model         = cfg['model'],
        loss_fn       = cfg['loss'],
        strategy      = cfg['strategy'],
        fold          = cfg['fold'],
        reduce_pixels = cfg['reduce_pixels'],
    )
    path = result_dir / f"{run_name}_history.npy"
    if not path.exists():
        print(f"  ⚠️  Not found: {path.name}")
        return None
    history             = np.load(path, allow_pickle=True).item()
    history['run_name'] = run_name
    return history