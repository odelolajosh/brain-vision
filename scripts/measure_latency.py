#!/usr/bin/env python3
"""
measure_latency.py — Per-image inference latency for the Pareto analysis.

Runs identically on the development machine (MPS/CUDA) and on the
Raspberry Pi 5 (CPU). Produces the latency axis for Chapter 4, Figure 4.2.

Usage (from repo root on Mac):
    python demo/measure_latency.py --image processed/first_campaign/008-02.npz

Usage (on Pi, after ./scripts/pi_sync_code.sh):
    cd ~/brainvision_demo
    source .venv/bin/activate
    PYTHONPATH=. python demo/measure_latency.py \
        --image processed/first_campaign/008-02.npz

Protocol:
  - Latency is measured per WHOLE IMAGE (every pixel classified), because
    that is the clinically meaningful unit: a surgeon images a field of
    view, not a set of annotated pixels.
  - Each configuration gets warm-up iterations (discarded) followed by N
    timed repetitions; the MEDIAN is reported, with min/max for spread.
  - Patch models are extremely slow on CPU. Use --max-pixels to time a
    random subset and extrapolate linearly; extrapolated rows are marked
    as such in the output and must be labelled as estimates in the report.

Outputs:
  - Console table
  - results/latency_<device>.md    (markdown for the report)
  - results/latency_<device>.json  (machine-readable, for the Pareto plot)
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

# ── Direct module imports ─────────────────────────────────────────────────────
# NOTE: import from the module files, NOT the package __init__, so this works
# on the Pi where brainvision/__init__.py and models/__init__.py are stubs.
from brainvision.constants import N_DECIMATED_BANDS, N_CLASSES

from brainvision.models.fabelo_dnn import FabeloDNN

# 2D models are absent in a "fast mode" Pi deployment — import defensively.
_OPTIONAL = {}
for _key, _mod, _cls in [
    ("baseline_dnn",  "brainvision.models.baseline_dnn",  "Baseline1DDNN"),
    ("hu_1dcnn",      "brainvision.models.hu_1dcnn",      "HuEtAl1DCNN"),
    ("fabelo_2dcnn",  "brainvision.models.fabelo_2dcnn",  "Fabelo2DCNN"),
    ("simple_2dcnn",  "brainvision.models.simple_2dcnn",  "Simple2DCNN"),
    ("lee_2dcnn",     "brainvision.models.lee_2dcnn",     "LeeEtAl2DCNN"),
]:
    try:
        _module = __import__(_mod, fromlist=[_cls])
        _OPTIONAL[_key] = getattr(_module, _cls)
    except (ImportError, AttributeError):
        pass   # not deployed on this machine — skipped in the run

# ── Configuration matrix ──────────────────────────────────────────────────────
# Add a row here to time another configuration. `ckpt` is the filename inside
# --ckpt-dir; use the fold you report as the median fold in the test tables.
CONFIGS = [
    {
        "label":  "1D-NN-Fabelo (CE)",
        "model":  "fabelo_dnn",
        "params": 4_936,
        "type":   "pixel",
        "ckpt":   "1dnnfabelo_ce_bal_fold3_vpfabelo.pt",
    },
    {
        "label":  "1D-CNN Hu (CE)",
        "model":  "hu_1dcnn",
        "params": 76_824,
        "type":   "pixel",
        "ckpt":   "1dcnn_ce_bal_fold3_vpfabelo.pt",
    },
    {
        "label":  "2D-CNN-Simple (CE)",
        "model":  "simple_2dcnn",
        "params": 19_644,
        "type":   "patch",
        "ckpt":   "2dcnnsimple_ce_bal_fold3_vpfabelo.pt",
    },
    {
        "label":  "2D-CNN-Fabelo (CE)",
        "model":  "fabelo_2dcnn",
        "params": 142_052,
        "type":   "patch",
        "ckpt":   "2dcnnfabelo_ce_bal_fold3_vpfabelo.pt",
    },
]

PATCH_SIZE   = 11          # must match training
NPZ_CUBE_KEY = "processed"

CLINICAL_LIMIT_S = 60.0    # sub-minute constraint from the project aim


# ── Device ────────────────────────────────────────────────────────────────────
def pick_device(force: str | None) -> str:
    if force:
        return force
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_label(device: str) -> str:
    """Human-readable hardware description for the report."""
    machine = platform.machine()
    node    = platform.node()
    if device == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return f"MPS (Apple Silicon, {machine})"
    # Distinguish the Pi from any other CPU host
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00").strip()
        if model:
            return f"CPU ({model})"
    except OSError:
        pass
    return f"CPU ({machine}, {node})"


# ── Model loading ─────────────────────────────────────────────────────────────
def resolve_class(model_key: str):
    if model_key == "fabelo_dnn":
        return FabeloDNN
    return _OPTIONAL.get(model_key)


def load_model(model_class, ckpt_path: Path, device: str):
    model = model_class(input_channels=N_DECIMATED_BANDS, n_classes=N_CLASSES)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.eval().to(device)


# ── Inference (timed sections) ────────────────────────────────────────────────
@torch.no_grad()
def run_pixel(model, cube, device, batch_size, coords=None):
    """Classify every pixel (or the given coords) of an (H, W, B) cube."""
    H, W, B = cube.shape
    if coords is None:
        flat = cube.reshape(-1, B)
    else:
        flat = cube[coords[:, 0], coords[:, 1], :]
    flat = np.ascontiguousarray(flat, dtype=np.float32)

    n = len(flat)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        t = torch.from_numpy(flat[s:e]).to(device)
        _ = model(t).argmax(1)
    return n


@torch.no_grad()
def run_patch(model, cube, device, batch_size, patch_size, coords=None):
    """Classify every pixel (or the given coords) using P×P patches."""
    H, W, B = cube.shape
    P, pad  = patch_size, patch_size // 2
    padded  = np.pad(
        cube.transpose(2, 0, 1),
        ((0, 0), (pad, pad), (pad, pad)),
        mode="reflect",
    )
    if coords is None:
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coords = np.stack([rows.ravel(), cols.ravel()], axis=1)

    n = len(coords)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = np.stack([
            padded[:, r:r + P, c:c + P] for r, c in coords[s:e]
        ]).astype(np.float32)
        t = torch.from_numpy(batch).to(device)
        _ = model(t).argmax(1)
    return n


def synchronise(device: str):
    """Ensure async GPU work has finished before stopping the clock."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Per-image inference latency")
    ap.add_argument("--image", default="processed/first_campaign/008-02.npz",
                    help="Representative .npz cube to time against")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--reps", type=int, default=5,
                    help="Timed repetitions per configuration (default 5)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warm-up iterations, discarded (default 1)")
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="Pixel-model batch size (patch models use /8)")
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="Time a random subset of this many pixels and "
                         "extrapolate to the full image. Use on the Pi for "
                         "patch models. Results are marked as estimates.")
    ap.add_argument("--device", default=None,
                    choices=["cpu", "mps", "cuda"],
                    help="Override device auto-detection")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only time configs whose label contains one of these "
                         "substrings, e.g. --only 1D")
    args = ap.parse_args()

    device = pick_device(args.device)
    dev_label = device_label(device)

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    cube = np.load(image_path, allow_pickle=False)[NPZ_CUBE_KEY]
    H, W, B = cube.shape
    total_px = H * W

    print(f"Device      : {dev_label}")
    print(f"Image       : {image_path.name}  ({H}×{W}×{B} = {total_px:,} px)")
    print(f"Repetitions : {args.reps} (+{args.warmup} warm-up)")
    print(f"Patch size  : {args.patch_size}")
    if args.max_pixels:
        print(f"Subsampling : {args.max_pixels:,} px, extrapolated to full image")
    print("=" * 78)

    # Fixed random subset, shared across configs so comparisons are fair
    coords = None
    if args.max_pixels and args.max_pixels < total_px:
        rng = np.random.default_rng(0)
        idx = rng.choice(total_px, size=args.max_pixels, replace=False)
        coords = np.stack([idx // W, idx % W], axis=1)

    results = []

    for cfg in CONFIGS:
        if args.only and not any(s.lower() in cfg["label"].lower() for s in args.only):
            continue

        model_class = resolve_class(cfg["model"])
        if model_class is None:
            print(f"\n{cfg['label']}: model module not available here — skipped")
            continue

        ckpt = Path(args.ckpt_dir) / cfg["ckpt"]
        if not ckpt.exists():
            print(f"\n{cfg['label']}: checkpoint not found ({ckpt.name}) — skipped")
            continue

        print(f"\n── {cfg['label']} ──")
        model = load_model(model_class, ckpt, device)

        is_patch   = cfg["type"] == "patch"
        batch_size = args.batch_size // 8 if is_patch else args.batch_size

        def one_pass():
            if is_patch:
                return run_patch(model, cube, device, batch_size,
                                 args.patch_size, coords)
            return run_pixel(model, cube, device, batch_size, coords)

        for _ in range(args.warmup):
            one_pass()
            synchronise(device)

        times = []
        for r in range(args.reps):
            t0 = time.perf_counter()
            n_px = one_pass()
            synchronise(device)
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"  rep {r + 1}: {dt:.3f}s  ({n_px / dt:,.0f} px/s)")

        measured = statistics.median(times)
        extrapolated = coords is not None
        scale = (total_px / len(coords)) if extrapolated else 1.0
        per_image = measured * scale

        results.append({
            "label":        cfg["label"],
            "model":        cfg["model"],
            "params":       cfg["params"],
            "type":         cfg["type"],
            "checkpoint":   cfg["ckpt"],
            "median_s":     round(per_image, 4),
            "min_s":        round(min(times) * scale, 4),
            "max_s":        round(max(times) * scale, 4),
            "px_per_s":     round(total_px / per_image, 1),
            "extrapolated": extrapolated,
            "meets_limit":  per_image < CLINICAL_LIMIT_S,
        })

        flag = " (extrapolated)" if extrapolated else ""
        ok   = "PASS" if per_image < CLINICAL_LIMIT_S else "FAIL"
        print(f"  → per image: {per_image:.3f}s{flag}   "
              f"{total_px / per_image:,.0f} px/s   [{ok} 60s limit]")

    if not results:
        sys.exit("\nNo configurations were timed — check checkpoints and imports.")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  LATENCY SUMMARY — {dev_label}")
    print("=" * 78)
    print(f"  {'Configuration':<24} {'Params':>9} {'Per image':>11} "
          f"{'px/s':>12}  {'60s':>5}")
    print("  " + "-" * 72)
    for r in results:
        note = "*" if r["extrapolated"] else " "
        print(f"  {r['label']:<24} {r['params']:>9,} "
              f"{r['median_s']:>10.3f}s{note} {r['px_per_s']:>12,.0f}  "
              f"{'PASS' if r['meets_limit'] else 'FAIL':>5}")
    if any(r["extrapolated"] for r in results):
        print("\n  * extrapolated from a subsample — report as an estimate")

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tag = device if device != "cpu" else "pi" if "Raspberry" in dev_label else "cpu"

    payload = {
        "device":        dev_label,
        "device_tag":    tag,
        "image":         image_path.name,
        "image_shape":   [H, W, B],
        "total_pixels":  total_px,
        "patch_size":    args.patch_size,
        "reps":          args.reps,
        "subsampled_px": len(coords) if coords is not None else None,
        "date":          date.today().isoformat(),
        "results":       results,
    }
    json_path = out_dir / f"latency_{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md_path = out_dir / f"latency_{tag}.md"
    with open(md_path, "w") as f:
        f.write(f"# Inference latency — {dev_label}\n\n")
        f.write("| field | value |\n|---|---|\n")
        f.write(f"| Hardware | {dev_label} |\n")
        f.write(f"| Image | {image_path.name} ({H}×{W}, {total_px:,} px) |\n")
        f.write(f"| Patch size | {args.patch_size} |\n")
        f.write(f"| Repetitions | {args.reps} (+{args.warmup} warm-up) |\n")
        if coords is not None:
            f.write(f"| Subsample | {len(coords):,} px, linearly extrapolated |\n")
        f.write(f"| Date | {date.today().isoformat()} |\n\n")
        f.write("Median of timed repetitions; whole-image inference "
                "(every pixel classified).\n\n")
        f.write("| Configuration | Params | Type | Per image (s) | px/s | <60 s |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            note = " \\*" if r["extrapolated"] else ""
            f.write(f"| {r['label']} | {r['params']:,} | {r['type']} | "
                    f"{r['median_s']:.3f}{note} | {r['px_per_s']:,.0f} | "
                    f"{'yes' if r['meets_limit'] else '**no**'} |\n")
        if any(r["extrapolated"] for r in results):
            f.write("\n\\* extrapolated from a random subsample — estimate.\n")

    print(f"\n  Markdown → {md_path}")
    print(f"  JSON     → {json_path}")


if __name__ == "__main__":
    main()