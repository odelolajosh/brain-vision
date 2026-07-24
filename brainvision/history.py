import matplotlib.pyplot as plt
from pathlib import Path


def plot_history_curves(history: dict, run_name: str, out_dir: Path):
    """
    Plot loss, Val F1 (all + no-BG), sensitivity and specificity
    across training epochs. Marks the best epoch.
    """
    epochs  = range(1, len(history['train_loss']) + 1)
    best_ep = history.get('best_epoch', 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Training Curves — {run_name}", fontsize=12)

    def _vline(ax):
        ax.axvline(best_ep, color='gray', linestyle='--', alpha=0.6,
                   label=f'Best ({best_ep})')

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color='#378ADD', label='Train')
    ax.plot(epochs, history['val_loss'],   color='#D85A30', label='Val')
    _vline(ax); ax.set_title("Loss"); ax.legend()
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

    ax = axes[0, 1]
    ax.plot(epochs, history['val_f1'], color='#5DCAA5')
    best_f1_all = history.get('best_f1_all', history['val_f1'][best_ep-1])
    ax.scatter([best_ep], [best_f1_all], color='#5DCAA5', zorder=5, s=60)
    _vline(ax)
    ax.set_title(f"Val F1 all  (best={best_f1_all:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.set_ylim(0, 1)

    ax = axes[0, 2]
    if history.get('val_f1_no_bg'):
        best_f1_no_bg = history.get('best_f1', history['val_f1_no_bg'][best_ep-1])
        ax.plot(epochs, history['val_f1_no_bg'], color='#7F77DD')
        ax.scatter([best_ep], [best_f1_no_bg], color='#7F77DD', zorder=5, s=60)
        _vline(ax)
        ax.set_title(f"Val F1 no-BG  (best={best_f1_no_bg:.4f})")
    else:
        ax.set_title("Val F1 no-BG  (not recorded)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.set_ylim(0, 1)

    ax = axes[1, 0]
    ax.plot(epochs, history['val_sens'], color='#EF9F27')
    best_sens = history.get('best_sens', history['val_sens'][best_ep-1])
    ax.scatter([best_ep], [best_sens], color='#EF9F27', zorder=5, s=60)
    _vline(ax)
    ax.set_title(f"Val Sensitivity  (best={best_sens:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Sensitivity"); ax.set_ylim(0, 1)

    ax = axes[1, 1]
    ax.plot(epochs, history['val_spec'], color='#639922')
    best_spec = history.get('best_spec', history['val_spec'][best_ep-1])
    ax.scatter([best_ep], [best_spec], color='#639922', zorder=5, s=60)
    _vline(ax)
    ax.set_title(f"Val Specificity  (best={best_spec:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Specificity"); ax.set_ylim(0, 1)

    ax = axes[1, 2]
    ax.axis('off')

    text = (
        f"Run      : {run_name}\n\n"
        f"Best epoch     : {best_ep}\n"
        f"Val F1 (all)   : {best_f1_all:.4f}\n"
        f"Val F1 (no BG) : {history.get('best_f1', float('nan')):.4f}\n"
        f"Val Sensitivity: {best_sens:.4f}\n"
        f"Val Specificity: {best_spec:.4f}\n"
        f"Total epochs   : {len(history['train_loss'])}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / f"{run_name}_curves.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path