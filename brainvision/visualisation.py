# brainvision/visualisation.py
"""
Hypercube and ground truth visualisation utilities.

Functions
---------
make_pseudo_rgb        : Extract a 3-band pseudo-RGB array from a cube.
show_hypercube         : Display pseudo-RGB image, optionally overlaying a label map.
show_ground_truth      : Display a ground-truth or prediction label map.
show_patient           : Convenience wrapper — RGB + GT side by side.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from pathlib import Path
from PIL import Image
from typing import Optional

from brainvision.constants import (
    N_DECIMATED_BANDS,
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_NAMES_SHORT,
)

# ── Colour configuration ──────────────────────────────────────────────────────
#: Pseudo-RGB wavelengths (nm) used by Fabelo et al. for SRGB rendering.
#: Camera range 440.5–909.1 nm across 128 decimated bands.
PSEUDO_RGB_WAVELENGTHS: tuple[float, float, float] = (708.97, 539.0, 479.0)

#: Colourmap for label maps: 0=unlabelled(white), 1=NT, 2=TT, 3=BV, 4=BG
_LABEL_CMAP = ListedColormap([
    "white",               # 0 = unlabelled
    CLASS_COLORS[1],       # 1 = NT
    CLASS_COLORS[2],       # 2 = TT
    CLASS_COLORS[3],       # 3 = BV
    CLASS_COLORS[4],       # 4 = BG
])


# ── Spectral range constants ───────────────────────────────────────────────────
RAW_N_BANDS   = 826
RAW_WL_MIN    = 400.0
RAW_WL_MAX    = 1000.0

PROC_N_BANDS  = N_DECIMATED_BANDS   # 128
PROC_WL_MIN   = 440.5
PROC_WL_MAX   = 909.1


def _wavelength_to_band(
    wavelength_nm : float,
    n_bands       : int   = RAW_N_BANDS,
    wl_min        : float = RAW_WL_MIN,
    wl_max        : float = RAW_WL_MAX,
) -> int:
    """Convert a wavelength (nm) to the nearest band index."""
    fraction = (wavelength_nm - wl_min) / (wl_max - wl_min)
    idx      = int(round(fraction * (n_bands - 1)))
    return max(0, min(n_bands - 1, idx))


def _resolve_bands(
    bands  : tuple[int | float, int | float, int | float],
    n_bands: int = RAW_N_BANDS,
    wl_min : float = RAW_WL_MIN,
    wl_max : float = RAW_WL_MAX,
) -> tuple[int, int, int]:
    """
    Accept band indices (int) or wavelengths in nm (float) and return
    three integer band indices appropriate for a cube with n_bands bands.

    Floats are treated as wavelengths in nm and converted to indices.
    Ints are used directly — no conversion applied.
    """
    resolved = []
    for b in bands:
        if isinstance(b, float):
            resolved.append(_wavelength_to_band(b, n_bands, wl_min, wl_max))
        else:
            resolved.append(max(0, min(n_bands - 1, int(b))))
    return tuple(resolved)


# ── Core extraction ───────────────────────────────────────────────────────────
def make_pseudo_rgb(
    cube : np.ndarray,
    bands: tuple[int | float, int | float, int | float] = PSEUDO_RGB_WAVELENGTHS,
) -> np.ndarray:
    """
    Extract a pseudo-RGB uint8 array from a hyperspectral cube.

    Automatically detects whether the cube is raw (826 bands) or
    processed (128 bands) and resolves wavelength inputs accordingly.

    Parameters
    ----------
    cube :
        (H, W, B) float array — raw DN, reflectance, or normalised.
    bands :
        Three band indices (int) or wavelengths in nm (float).
        Defaults to Fabelo et al. SRGB wavelengths (709nm, 539nm, 479nm).
    """
    n_bands = cube.shape[2]

    # Select spectral range based on band count
    if n_bands > 200:
        # Raw cube — 826 bands, 400–1000nm
        wl_min, wl_max = RAW_WL_MIN, RAW_WL_MAX
    else:
        # Processed cube — 128 bands, 440.5–909.1nm
        wl_min, wl_max = PROC_WL_MIN, PROC_WL_MAX

    r_idx, g_idx, b_idx = _resolve_bands(bands, n_bands, wl_min, wl_max)

    rgb = np.stack(
        [cube[:, :, r_idx],
         cube[:, :, g_idx],
         cube[:, :, b_idx]],
        axis=-1
    ).astype(np.float32)

    mn, mx = rgb.min(), rgb.max()
    if mx > mn:
        rgb = (rgb - mn) / (mx - mn)

    return (rgb * 255).clip(0, 255).astype(np.uint8)


# ── Legend helper ─────────────────────────────────────────────────────────────
def _class_legend(present_labels: set[int]) -> list[mpatches.Patch]:
    patches = []
    for label_id in [1, 2, 3, 4]:          # 1=NT, 2=TT, 3=BV, 4=BG
        if label_id in present_labels:
            patches.append(mpatches.Patch(
                color=CLASS_COLORS[label_id],
                label=CLASS_NAMES[label_id]
            ))
    if 0 in present_labels:
        patches.append(mpatches.Patch(
            facecolor="white",
            edgecolor="gray",
            linewidth=0.5,
            label="Unlabelled"
        ))
    return patches


# ── Public API ────────────────────────────────────────────────────────────────
def show_hypercube(
    cube: np.ndarray,
    bands: tuple[int | float, int | float, int | float] = PSEUDO_RGB_WAVELENGTHS,
    label_map: Optional[np.ndarray] = None,
    alpha: float = 0.4,
    title: str = "Pseudo-RGB",
    ax: Optional[Axes] = None,
    show_legend: bool = True,
) -> Axes:
    """
    Display a hypercube as a pseudo-RGB image, with an optional label overlay.

    Parameters
    ----------
    cube :
        (H, W, B) float array — raw or normalised reflectance.
    bands :
        Three band indices or wavelengths (nm) for R, G, B channels.
    label_map :
        Optional (H, W) int array with values 0 (unlabelled) or 1–4 (classes).
        When provided the label map is blended over the RGB image.
    alpha :
        Opacity of the label overlay (0 = invisible, 1 = opaque).
    title :
        Axes title.
    ax :
        Existing ``matplotlib.Axes`` to draw into. A new figure is created
        if *None*.
    show_legend :
        Whether to add a class legend when a label map is provided.

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    rgb = make_pseudo_rgb(cube, bands)
    ax.imshow(rgb)

    if label_map is not None:
        ax.imshow(
            label_map,
            cmap=_LABEL_CMAP,
            vmin=0,
            vmax=4,
            alpha=alpha,
            interpolation="nearest",
        )
        if show_legend:
            ax.legend(
                handles=_class_legend(set(np.unique(label_map))),
                loc="lower right",
                fontsize=7,
                framealpha=0.8,
            )

    ax.set_title(title)
    ax.axis("off")
    return ax


def show_ground_truth(
    label_map: np.ndarray,
    title: str = "Ground Truth",
    background_image: Optional[np.ndarray] = None,
    alpha: float = 0.7,
    ax: Optional[Axes] = None,
    show_legend: bool = True,
) -> Axes:
    """
    Display a ground-truth or model-prediction label map.

    Parameters
    ----------
    label_map :
        (H, W) int array. Values:
          0 = unlabelled / background
          1 = Normal Tissue (NT)
          2 = Tumour Tissue (TT)
          3 = Blood Vessel (BV)
          4 = Background (BG)
    title :
        Axes title.
    background_image :
        Optional (H, W, 3) uint8 RGB to render under the label map.
        Useful for a faint anatomical reference.
    alpha :
        Opacity of the label map when a background image is provided.
        Ignored when no background image is given.
    ax :
        Existing ``matplotlib.Axes`` to draw into.
    show_legend :
        Whether to add a class legend.

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if background_image is not None:
        ax.imshow(background_image)
        ax.imshow(
            label_map,
            cmap=_LABEL_CMAP,
            vmin=0,
            vmax=4,
            alpha=alpha,
            interpolation="nearest",
        )
    else:
        ax.imshow(
            label_map,
            cmap=_LABEL_CMAP,
            vmin=0,
            vmax=4,
            interpolation="nearest",
        )

    if show_legend:
        ax.legend(
            handles=_class_legend(set(np.unique(label_map))),
            loc="lower right",
            fontsize=7,
            framealpha=0.8,
        )

    ax.set_title(title)
    ax.axis("off")
    return ax


def show_patient(
    cube: np.ndarray,
    label_map: np.ndarray,
    patient_id: str = "",
    bands: tuple[int | float, int | float, int | float] = PSEUDO_RGB_WAVELENGTHS,
    overlay_alpha: float = 0.4,
    figsize: tuple[int, int] = (13, 5),
) -> tuple[plt.Figure, tuple[Axes, Axes, Axes]]:
    """
    Convenience wrapper: pseudo-RGB | GT overlay | GT alone — side by side.

    Parameters
    ----------
    cube :
        (H, W, B) hyperspectral cube.
    label_map :
        (H, W) int label map.
    patient_id :
        String identifier used in the figure title.
    bands :
        RGB band selection passed to ``show_hypercube``.
    overlay_alpha :
        Opacity of the GT overlay in the centre panel.
    figsize :
        Overall figure size.

    Returns
    -------
    fig, (ax_rgb, ax_overlay, ax_gt)
    """
    fig, (ax_rgb, ax_overlay, ax_gt) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"Patient {patient_id}" if patient_id else "Patient", fontsize=12)

    show_hypercube(
        cube, bands,
        title="Pseudo-RGB",
        ax=ax_rgb,
        show_legend=False,
    )
    show_hypercube(
        cube, bands,
        label_map=label_map,
        alpha=overlay_alpha,
        title="RGB + GT overlay",
        ax=ax_overlay,
        show_legend=True,
    )
    show_ground_truth(
        label_map,
        title="Ground Truth",
        ax=ax_gt,
        show_legend=True,
    )

    plt.tight_layout()
    return fig, (ax_rgb, ax_overlay, ax_gt)


def show_prediction_disparity(
    label_map: np.ndarray,
    pred_map: np.ndarray,
    title: str = "Prediction Disparity",
    background_image: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    show_legend: bool = True,
    show_stats: bool = True,
) -> Axes:
    """
    Display a disparity map between ground truth and model prediction.

    Each labelled pixel is coloured by its outcome:
      ✅ Correct   — predicted class matches ground truth
      ❌ Incorrect — predicted class does not match ground truth
      ⬜ Unlabelled — pixel has no ground truth label (label_map == 0)

    Per-class correct/incorrect breakdown is shown in the legend
    when show_stats=True.

    Parameters
    ----------
    label_map :
        (H, W) int array — ground truth. 0=unlabelled, 1–4=classes.
    pred_map :
        (H, W) int array — model predictions. Same convention as label_map.
        Unlabelled pixels (label_map == 0) are ignored regardless of pred_map.
    title :
        Axes title.
    background_image :
        Optional (H, W, 3) uint8 RGB rendered under the disparity map.
    ax :
        Existing ``matplotlib.Axes`` to draw into.
    show_legend :
        Whether to add a disparity legend.
    show_stats :
        Whether to annotate the axes with per-class accuracy statistics.

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    H, W     = label_map.shape
    mask     = label_map > 0          # labelled pixels

    # ── Build disparity map ───────────────────────────────────────────────────
    # Values:
    #   0 = unlabelled (white)
    #   1 = correct    (green  #5DCAA5)
    #   2 = incorrect  (red    #D85A30)
    disparity = np.zeros((H, W), dtype=np.int32)
    disparity[mask & (pred_map == label_map)] = 1   # correct
    disparity[mask & (pred_map != label_map)] = 2   # incorrect

    disp_cmap = ListedColormap(["white", "#5DCAA5", "#D85A30"])

    if background_image is not None:
        ax.imshow(background_image)
        ax.imshow(disparity, cmap=disp_cmap, vmin=0, vmax=2,
                  alpha=0.75, interpolation="nearest")
    else:
        ax.imshow(disparity, cmap=disp_cmap, vmin=0, vmax=2,
                  interpolation="nearest")

    # ── Legend ────────────────────────────────────────────────────────────────
    if show_legend:
        total     = mask.sum()
        correct   = (disparity == 1).sum()
        incorrect = (disparity == 2).sum()
        oa        = correct / total * 100 if total > 0 else 0.0

        legend_patches = [
            mpatches.Patch(color="#5DCAA5",
                           label=f"Correct   ({correct:,} px, {oa:.1f}%)"),
            mpatches.Patch(color="#D85A30",
                           label=f"Incorrect ({incorrect:,} px, "
                                 f"{100 - oa:.1f}%)"),
            mpatches.Patch(
                facecolor="white",
                edgecolor="gray",
                linewidth=0.5,
                label="Unlabelled"
            ),
        ]
        ax.legend(handles=legend_patches, loc="lower right",
                  fontsize=7, framealpha=0.8)

    # ── Per-class accuracy annotation ─────────────────────────────────────────
    if show_stats:
        lines = []
        for i, name in enumerate(CLASS_NAMES_SHORT.values()):
            cls_mask    = label_map == (i + 1)
            cls_total   = cls_mask.sum()
            if cls_total == 0:
                continue
            cls_correct = (cls_mask & (pred_map == label_map)).sum()
            cls_acc     = cls_correct / cls_total * 100
            marker      = " ←" if i == 1 else ""   # highlight TT
            lines.append(f"{name:<4}: {cls_acc:5.1f}%{marker}")

        stats_text = "\n".join(lines)
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )

    ax.set_title(title)
    ax.axis("off")
    return ax


def show_patient_full(
    cube: np.ndarray,
    label_map: np.ndarray,
    pred_map: np.ndarray,
    patient_id: str = "",
    bands: tuple[int | float, int | float, int | float] = PSEUDO_RGB_WAVELENGTHS,
    overlay_alpha: float = 0.4,
    figsize: tuple[int, int] = (18, 5),
) -> tuple[plt.Figure, tuple[Axes, Axes, Axes, Axes]]:
    """
    Four-panel view: pseudo-RGB | GT | Prediction | Disparity.

    Parameters
    ----------
    cube :
        (H, W, B) hyperspectral cube.
    label_map :
        (H, W) int ground truth label map.
    pred_map :
        (H, W) int model prediction map. Same label convention as label_map.
    patient_id :
        String identifier used in the figure title.
    bands :
        RGB band selection.
    overlay_alpha :
        Opacity of label overlays.
    figsize :
        Overall figure size.

    Returns
    -------
    fig, (ax_rgb, ax_gt, ax_pred, ax_disp)
    """
    fig, (ax_rgb, ax_gt, ax_pred, ax_disp) = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(
        f"Patient {patient_id}" if patient_id else "Patient",
        fontsize=12
    )

    rgb = make_pseudo_rgb(cube, bands)

    show_hypercube(
        cube, bands,
        title="Pseudo-RGB",
        ax=ax_rgb,
        show_legend=False,
    )
    show_ground_truth(
        label_map,
        background_image=rgb,
        alpha=overlay_alpha,
        title="Ground Truth",
        ax=ax_gt,
        show_legend=True,
    )
    show_ground_truth(
        pred_map,
        background_image=rgb,
        alpha=overlay_alpha,
        title="Prediction",
        ax=ax_pred,
        show_legend=True,
    )
    show_prediction_disparity(
        label_map,
        pred_map,
        background_image=rgb,
        title="Disparity",
        ax=ax_disp,
        show_legend=True,
        show_stats=True,
    )

    plt.tight_layout()
    return fig, (ax_rgb, ax_gt, ax_pred, ax_disp)


def load_jpeg(jpg_path: str | Path) -> np.ndarray:
    """Load a JPEG preview into a uint8 numpy array."""
    return np.array(Image.open(jpg_path).convert("RGB"), dtype=np.uint8)