#!/usr/bin/env python3
"""
generate_prediction_figure.py — Ground truth vs prediction, overlaid on RGB.

Renders the qualitative comparison figure for Chapter 4 / defence slides: a
pseudo-RGB rendering of a hyperspectral image with the ground truth and the
model's prediction overlaid as semi-transparent tissue-class tints, using the
recommended deployment configuration.

Usage (from repo root):
    python scripts/generate_prediction_figure.py

    # Override the model, fold, or image explicitly
    python scripts/generate_prediction_figure.py \
        --run 1dcnn_ce_nobal --fold 2 --image 037-01

    # Flat colour-block style instead of an RGB overlay
    python scripts/generate_prediction_figure.py --style flat

    # Tint the model's prediction across the WHOLE image, not just where
    # ground truth exists (useful for a "this runs on the full field" slide)
    python scripts/generate_prediction_figure.py --full-pred

Default configuration: 1D-CNN-Hu, cross-entropy, natural (unbalanced) class
distribution, fold 2 — the best-performing checkpoint identified in Chapter 4
(test macro F1 excluding background: 84.7%; see Table 4.12).

Image selection: if --image is not given, every image in the fixed 15-image
test partition is scanned and the one with the largest tumour-tissue area is
selected automatically. The scan prints every candidate's composition so the
choice is auditable rather than silent.

Overlay convention: the base image is a percentile-stretched pseudo-RGB
rendering of three representative bands (~709/539/479 nm by default, matching
the project's demo application). Each classified pixel is tinted with its
class colour at a fixed transparency; UNLABELLED pixels are left as plain
anatomy with no tint, since sparse annotation is not a model failure and
should not visually read as one. The prediction overlay is, by default,
restricted to the same footprint as the ground-truth annotations, so the two
panels compare identical pixels; pass --full-pred to tint everywhere instead.

Output: TWO SEPARATE figure files, not a combined panel:
    figures/gt_overlay_<image>.png
    figures/pred_overlay_<run>_fold<N>_<image>.png
(or gt_map_/pred_map_ if --style flat is used, matching the original script)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from brainvision.constants import CLASS_COLORS, CLASS_NAMES, N_DECIMATED_BANDS, N_CLASSES
from brainvision.models.hu_1dcnn import HuEtAl1DCNN
from brainvision.models.fabelo_dnn import FabeloDNN

# Extend this if you want to render a 2D/3D model instead — infer_full_image
# below currently only implements pixel-wise inference.
MODEL_REGISTRY = {
    "1dcnn": (HuEtAl1DCNN, "pixel"),
    "1dnnfabelo": (FabeloDNN, "pixel"),
}

NPZ_CUBE_KEY = "processed"
NPZ_GT_KEY = "labels"

# Matches demo/app.py: ~709 nm (R), ~539 nm (G), ~479 nm (B) in the 128-band
# decimated space.
DEFAULT_RGB_BANDS = (88, 42, 7)

# The fixed 15-image test partition (Chapter 4, Section 4.6.1) — never
# trained or validated on by any fold, so any of these is a fair choice for
# a "the model has genuinely never seen this" qualitative figure.
TEST_IMAGES = [
    "012-01", "012-02", "019-01", "022-01", "022-02", "022-03",
    "037-01", "037-02", "037-03", "037-04", "038-01",
    "042-01", "042-02", "042-03", "053-01",
]

CAMPAIGNS = ["first_campaign", "second_campaign", "third_campaign"]


# ── Image resolution ───────────────────────────────────────────────────────────
def resolve_image_path(patient_id: str, processed_dir: Path) -> Path:
    for campaign in CAMPAIGNS:
        p = processed_dir / campaign / f"{patient_id}.npz"
        if p.exists():
            return p
    p = processed_dir / f"{patient_id}.npz"  # flat layout fallback
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not locate {patient_id}.npz under {processed_dir}")


def class_composition(labels: np.ndarray) -> dict:
    return {int(l): int((labels == l).sum()) for l in np.unique(labels)}


def pick_representative_image(processed_dir: Path) -> Path:
    """Scan the fixed test set and pick the image with the largest TT area."""
    print("Scanning test partition for a representative image...")
    print(f"  {'Patient':<10} {'NT':>8} {'TT':>8} {'BV':>8} {'BG':>8} {'unl.':>8}")
    best_path, best_tt = None, -1
    for pid in TEST_IMAGES:
        try:
            path = resolve_image_path(pid, processed_dir)
        except FileNotFoundError:
            print(f"  {pid:<10} (not found - skipped)")
            continue
        labels = np.load(path)[NPZ_GT_KEY]
        comp = class_composition(labels)
        tt = comp.get(2, 0)
        print(f"  {pid:<10} {comp.get(1, 0):>8,} {tt:>8,} {comp.get(3, 0):>8,} "
              f"{comp.get(4, 0):>8,} {comp.get(0, 0):>8,}")
        if tt > best_tt:
            best_tt, best_path = tt, path
    if best_path is None:
        raise FileNotFoundError("No test images found under the given directory.")
    print(f"\nSelected: {best_path.name} (largest tumour area, {best_tt:,} px)\n")
    return best_path


# ── Model and inference ────────────────────────────────────────────────────────
def load_model(model_class, ckpt_path: Path):
    model = model_class(input_channels=N_DECIMATED_BANDS, n_classes=N_CLASSES)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def infer_full_image(model, cube: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Whole-image pixel-wise inference. Returns an (H, W) label map, 1-indexed."""
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float32)
    preds = np.empty(len(flat), dtype=np.int64)
    for s in range(0, len(flat), batch_size):
        e = min(s + batch_size, len(flat))
        t = torch.from_numpy(flat[s:e])
        preds[s:e] = model(t).argmax(1).numpy() + 1
    return preds.reshape(H, W)


# ── Pseudo-RGB rendering ────────────────────────────────────────────────────────
def make_pseudo_rgb(cube: np.ndarray, bands=DEFAULT_RGB_BANDS,
                     low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """
    Percentile-stretched pseudo-RGB for DISPLAY only — independent of whatever
    normalisation the cube already has for inference. A simple min/max stretch
    is dominated by outlier pixels on real HSI data, so each of the three
    bands is contrast-stretched between its own 2nd and 98th percentile.
    """
    rgb = cube[:, :, list(bands)].astype(np.float32)
    out = np.zeros_like(rgb)
    bounds = []
    for i in range(3):
        lo, hi = np.percentile(rgb[:, :, i], [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1e-6
        out[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo), 0.0, 1.0)
        bounds.append((lo, hi))
    print(f"  pseudo-RGB stretch (bands {bands}, p{low_pct:.0f}-p{high_pct:.0f}): "
          + ", ".join(f"[{lo:.3f}, {hi:.3f}]" for lo, hi in bounds))
    return (out * 255).astype(np.uint8)


# ── Colour mapping ──────────────────────────────────────────────────────────────
def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def build_colour_map() -> dict:
    return {k: hex_to_rgb(v) for k, v in CLASS_COLORS.items()}


def labels_to_rgb(labels: np.ndarray, cmap: dict) -> np.ndarray:
    """Flat colour-block rendering (the original --style flat behaviour)."""
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl, colour in cmap.items():
        out[labels == lbl] = colour
    unlabelled_colour = cmap.get(4, (128, 128, 128))
    out[labels == 0] = unlabelled_colour
    return out


def overlay_labels(rgb: np.ndarray, labels: np.ndarray, cmap: dict,
                    alpha: float, footprint_mask: np.ndarray | None = None
                    ) -> np.ndarray:
    """
    Alpha-blend class-colour tints onto a pseudo-RGB base. Label 0
    (unlabelled) is never tinted — those pixels show plain anatomy. If
    footprint_mask is given, only pixels within it are tinted at all (used to
    restrict a prediction overlay to the ground-truth annotated footprint).
    """
    out = rgb.astype(np.float32).copy()
    for lbl, colour in cmap.items():
        m = labels == lbl
        if footprint_mask is not None:
            m = m & footprint_mask
        if not m.any():
            continue
        colour_arr = np.array(colour, dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + colour_arr * alpha
    return out.clip(0, 255).astype(np.uint8)


def add_legend(ax, cmap: dict):
    order = [1, 2, 3, 4]  # NT, TT, BV, BG
    handles = [
        Patch(facecolor=np.array(cmap[l]) / 255, edgecolor="black",
              linewidth=0.5, label=CLASS_NAMES.get(l, str(l)))
        for l in order if l in cmap
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.85, facecolor="white", edgecolor="none")


def save_map(rgb: np.ndarray, title: str, out_path: Path, dpi: int,
             cmap: dict | None = None, legend: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if legend and cmap is not None:
        add_legend(ax, cmap)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  written -> {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Ground truth vs prediction, overlaid on RGB, for the "
                    "best configuration")
    ap.add_argument("--run", default="1dcnn_ce_nobal",
                    help="Run prefix (default: best configuration overall — "
                         "1D-CNN-Hu / cross-entropy / unbalanced)")
    ap.add_argument("--model", default="1dcnn", choices=MODEL_REGISTRY,
                    help="Model key (default: 1dcnn = 1D-CNN-Hu)")
    ap.add_argument("--fold", type=int, default=2,
                    help="Fold to load (default: 2 — the best-performing "
                         "checkpoint for this configuration, Table 4.12)")
    ap.add_argument("--strategy", default="vpfabelo")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--processed-dir", default="processed")
    ap.add_argument("--image", default=None,
                    help="Patient ID (e.g. 037-01) or path to a .npz file. "
                         "If omitted, the test partition is scanned and the "
                         "image with the largest tumour area is used.")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--model-label", default="1D-CNN-Hu",
                    help="Display name used in the prediction title")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default="overlay", choices=["overlay", "flat"],
                    help="'overlay' tints class colours onto the pseudo-RGB "
                         "image (default); 'flat' renders solid colour "
                         "blocks with no anatomical background")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="Overlay tint opacity, 0 (invisible) to 1 (opaque). "
                         "Default 0.55 — visible colour, anatomy still shows "
                         "through")
    ap.add_argument("--full-pred", action="store_true",
                    help="Tint the prediction overlay across the WHOLE "
                         "image. Default restricts it to the same footprint "
                         "as the ground-truth annotations, so the two panels "
                         "compare identical pixels")
    ap.add_argument("--rgb-bands", type=int, nargs=3, default=DEFAULT_RGB_BANDS,
                    metavar=("R", "G", "B"),
                    help="Band indices for the pseudo-RGB base image "
                         f"(default {DEFAULT_RGB_BANDS}, ~709/539/479 nm)")
    ap.add_argument("--no-legend", action="store_true",
                    help="Omit the class-colour legend")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)

    # ── Resolve the image ───────────────────────────────────────────────────
    if args.image is None:
        image_path = pick_representative_image(processed_dir)
    elif args.image.endswith(".npz"):
        image_path = Path(args.image)
    else:
        image_path = resolve_image_path(args.image, processed_dir)

    data = np.load(image_path)
    cube, labels = data[NPZ_CUBE_KEY], data[NPZ_GT_KEY]
    print(f"Image: {image_path.name}  {cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}")

    comp = class_composition(labels)
    total = labels.size
    print("Composition:")
    for lbl, name in [(1, "NT"), (2, "TT"), (3, "BV"), (4, "BG"), (0, "unlabelled")]:
        n = comp.get(lbl, 0)
        print(f"  {name:<12} {n:>9,} px  ({100 * n / total:5.1f}%)")

    # ── Load model ────────────────────────────────────────────────────────────
    model_class, model_type = MODEL_REGISTRY[args.model]
    if model_type != "pixel":
        raise NotImplementedError(
            "This script currently renders pixel-wise (1D) models only. "
            "Extend infer_full_image with patch extraction (see "
            "demo/measure_latency.py:run_patch) to support 2D/3D models."
        )
    ckpt_name = f"{args.run}_fold{args.fold}_{args.strategy}.pt"
    ckpt_path = Path(args.ckpt_dir) / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"\nLoading checkpoint: {ckpt_path.name}")
    model = load_model(model_class, ckpt_path)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running whole-image inference...")
    pred = infer_full_image(model, cube.astype(np.float32))

    cmap = build_colour_map()
    stem = image_path.stem
    out_dir = Path(args.out_dir)
    legend = not args.no_legend

    if args.style == "flat":
        print("\nRendering flat colour-block maps...")
        gt_rgb = labels_to_rgb(labels, cmap)
        pred_rgb = labels_to_rgb(pred, cmap)
        save_map(gt_rgb, "Histopathological Ground Truth",
                  out_dir / f"gt_map_{stem}.png", args.dpi, cmap, legend)
        save_map(pred_rgb, f"{args.model_label} Prediction Map",
                  out_dir / f"pred_map_{args.run}_fold{args.fold}_{stem}.png",
                  args.dpi, cmap, legend)
    else:
        print("\nBuilding pseudo-RGB base image...")
        base_rgb = make_pseudo_rgb(cube, bands=tuple(args.rgb_bands))

        print(f"Overlaying ground truth (alpha={args.alpha})...")
        gt_overlay = overlay_labels(base_rgb, labels, cmap, args.alpha)

        gt_footprint = labels > 0
        pred_mask = None if args.full_pred else gt_footprint
        scope = "whole image" if args.full_pred else "ground-truth footprint only"
        print(f"Overlaying prediction (alpha={args.alpha}, scope: {scope})...")
        pred_overlay = overlay_labels(base_rgb, pred, cmap, args.alpha, pred_mask)

        save_map(gt_overlay, "Histopathological Ground Truth",
                  out_dir / f"gt_overlay_{stem}.png", args.dpi, cmap, legend)
        save_map(pred_overlay, f"{args.model_label} Prediction Map",
                  out_dir / f"pred_overlay_{args.run}_fold{args.fold}_{stem}.png",
                  args.dpi, cmap, legend)

    # ── Agreement summary (sanity check, not a substitute for Chapter 4) ────
    labelled = labels > 0
    if labelled.any():
        acc = (pred == labels)[labelled].sum() / labelled.sum() * 100
        print(f"\nPixel-level agreement on labelled pixels: {acc:.1f}%")
    print("(This is a qualitative full-image rendering for illustration. "
          "The reported test metrics use the pooled confusion matrix over "
          "the whole 15-image partition — see Chapter 4, Table 4.9.)")


if __name__ == "__main__":
    main()