#!/usr/bin/env python3
"""
measure_latency.py — Per-image inference latency and memory.

Runs identically on the development machine (MPS/CUDA) and on the
Raspberry Pi 5 (CPU). Produces the latency and memory axes for Chapter 4.

Usage (from repo root on Mac):
    python demo/measure_latency.py --image processed/first_campaign/008-02.npz

Usage (on Pi, after ./scripts/pi_sync_code.sh --mode full):
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
  - Models with no checkpoint (e.g. HybridSN, never trained) can still be
    timed with randomly initialised weights: latency depends on the
    architecture, not the weight values. Use --allow-random-init.

TWO DIFFERENT MEMORY FIGURES ARE REPORTED — do not conflate them:

  1. Parameter memory = n_params x 4 bytes (float32 weights).
     A property of the ARCHITECTURE. Distinguishes a 4,936-parameter model
     from a 4.17M-parameter one. Sub-megabyte for everything except the
     17M baseline and HybridSN.

  2. Peak process RSS = resident-set high-water mark during inference.
     Dominated by the DATA, not the model: run_patch builds a reflection-
     padded copy of the whole cube (~140 MB) on top of the loaded cube
     (~135 MB), against which a 142K-parameter model's weights are noise.
     This answers "does whole-image inference fit on a 4 GB Pi?", and must
     NOT be presented as an architectural footprint — two models with very
     different parameter counts will show near-identical RSS.

  RSS caveat: ru_maxrss is a per-process high-water mark that cannot be
  reset mid-run, so with several configs in one invocation only the FIRST
  config's figure is attributable. For clean per-config memory, re-exec
  one config per process:

      for cfg in "1D-NN-Fabelo" "2D-CNN-Fabelo" "HybridSN"; do
          python demo/measure_latency.py --only "$cfg" --out-suffix "$cfg"
      done

Outputs:
  - Console table
  - results/latency_<device>.md    (markdown for the report)
  - results/latency_<device>.json  (machine-readable, for the Pareto plot)
"""

import argparse
import json
import platform
import resource
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

# Some models are absent in a "fast mode" Pi deployment, or depend on packages
# not installed there (SpectralFormer needs einops). Import defensively, but
# record WHY each one failed so a silent skip cannot be mistaken for a result.
_OPTIONAL = {}
_IMPORT_ERRORS = {}
for _key, _mod, _cls in [
    ("baseline_dnn",   "brainvision.models.baseline_dnn",   "Baseline1DDNN"),
    ("hu_1dcnn",       "brainvision.models.hu_1dcnn",       "HuEtAl1DCNN"),
    ("fabelo_2dcnn",   "brainvision.models.fabelo_2dcnn",   "Fabelo2DCNN"),
    ("simple_2dcnn",   "brainvision.models.simple_2dcnn",   "Simple2DCNN"),
    ("lee_2dcnn",      "brainvision.models.lee_2dcnn",      "LeeEtAl2DCNN"),
    # --- ADJUST THESE TWO LINES to match your actual module/class names ---
    ("hybridsn",       "brainvision.models.hybridsn",       "HybridSN"),
    ("spectralformer", "brainvision.models.spectralformer", "SpectralFormer"),
]:
    try:
        _module = __import__(_mod, fromlist=[_cls])
        _OPTIONAL[_key] = getattr(_module, _cls)
    except (ImportError, AttributeError) as _exc:
        _IMPORT_ERRORS[_key] = f"{type(_exc).__name__}: {_exc}"

# ── Configuration matrix ──────────────────────────────────────────────────────
# Add a row here to time another configuration. `ckpt` is the filename inside
# --ckpt-dir; use the fold you report as the median fold in the test tables.
#
# `ckpt: None` means "no trained checkpoint exists" — the model is timed with
# random weights (requires --allow-random-init). Latency is a property of the
# architecture, so this is valid for the feasibility pre-screen, but such rows
# are flagged in the output and MUST be reported as architecture-only timings.
#
# `kwargs` are passed to the model constructor on top of the input_channels /
# n_classes defaults — use for models needing patch_size, mode, etc.
CONFIGS = [
    {
        "label":  "1D-NN-Fabelo (CE)",
        "model":  "fabelo_dnn",
        "params": 4_936,
        "type":   "pixel",
        "ckpt":   "1dnnfabelo_ce_bal_fold3_vpfabelo.pt",
    },
    {
        "label":  "1D-NN-Baseline (CE)",
        "model":  "baseline_dnn",
        "params": 17_000_000,
        "type":   "pixel",
        "ckpt":   "1dnn_ce_bal_fold3_vpfabelo.pt",
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
    {
        "label":  "2D-CNN-LeeEtAl (CE)",
        "model":  "lee_2dcnn",
        "params": 296_580,
        "type":   "patch",
        "ckpt":   "2dcnn_ce_bal_fold3_vpfabelo.pt",
    },
    {
        # Attention builds (batch x heads x tokens x tokens) intermediates, so
        # the default pixel batch of 4096 exhausts Pi RAM and the process is
        # OOM-killed. batch_size caps it per configuration.
        "label":  "SpectralFormer-ViT (CE)",
        "model":  "spectralformer",
        "params": 198_197,
        "type":   "pixel",
        "ckpt":   "spectralformer_vit_ce_bal_fold3_vpfabelo.pt",
        "kwargs": {"mode": "ViT"},
        "batch_size": 256,
    },
    {
        "label":  "SpectralFormer-CAF (CE)",
        "model":  "spectralformer",
        "params": 198_197,
        "type":   "pixel",
        "ckpt":   "spectralformer_caf_ce_bal_fold3_vpfabelo.pt",
        "kwargs": {"mode": "CAF"},
        "batch_size": 256,
    },
    {
        # Never trained — excluded by the latency feasibility pre-screen.
        # Timed with random weights to make that exclusion empirical rather
        # than predicted. Requires --allow-random-init.
        "label":  "HybridSN (untrained)",
        "model":  "hybridsn",
        "params": 4_174_452,
        "type":   "patch",
        "ckpt":   None,
    },
]

PATCH_SIZE   = 11          # must match training
NPZ_CUBE_KEY = "processed"

CLINICAL_LIMIT_S = 60.0    # sub-minute constraint from the project aim
BYTES_PER_PARAM  = 4       # float32 weights


# ── Device ────────────────────────────────────────────────────────────────────
def pick_device(force):
    if force:
        return force
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_label(device):
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


# ── Memory ────────────────────────────────────────────────────────────────────
def peak_rss_mb():
    """
    Peak resident set size for this process, in MB.

    ru_maxrss is reported in KB on Linux (incl. Raspberry Pi OS) but in
    BYTES on macOS — normalise so development-machine runs are not off by
    1024x. Only the Pi figures should be reported in the dissertation.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def param_memory_mb(n_params):
    """Weight storage for the architecture, in MB (float32)."""
    return n_params * BYTES_PER_PARAM / (1024.0 * 1024.0)


# ── Model loading ─────────────────────────────────────────────────────────────
def resolve_class(model_key):
    if model_key == "fabelo_dnn":
        return FabeloDNN
    return _OPTIONAL.get(model_key)


def build_model(model_class, kwargs, device):
    return model_class(
        input_channels=N_DECIMATED_BANDS,
        n_classes=N_CLASSES,
        **kwargs,
    )


def load_model(model_class, ckpt_path, device, kwargs=None):
    """Build the model; load weights if a checkpoint is given."""
    model = build_model(model_class, kwargs or {}, device)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    return model.eval().to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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
    """Classify every pixel (or the given coords) using P x P patches."""
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


def synchronise(device):
    """Ensure async GPU work has finished before stopping the clock."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ── Markdown writer ───────────────────────────────────────────────────────────
def write_markdown(md_path, rows, dev_label, image_path, H, W, total_px,
                   args, coords, baseline_rss):
    """Render the results table. Called after every configuration."""
    with open(md_path, "w") as f:
        f.write(f"# Inference latency and memory — {dev_label}\n\n")
        f.write("| field | value |\n|---|---|\n")
        f.write(f"| Hardware | {dev_label} |\n")
        f.write(f"| Image | {image_path.name} ({H}x{W}, {total_px:,} px) |\n")
        f.write(f"| Patch size | {args.patch_size} |\n")
        f.write(f"| Repetitions | {args.reps} (+{args.warmup} warm-up) |\n")
        f.write(f"| Baseline RSS | {baseline_rss:.1f} MB "
                f"(interpreter + cube loaded) |\n")
        if coords is not None:
            f.write(f"| Subsample | {len(coords):,} px, linearly extrapolated |\n")
        f.write(f"| Date | {date.today().isoformat()} |\n\n")
        f.write("Median of timed repetitions; whole-image inference "
                "(every pixel classified).\n\n")
        f.write("**Parameter memory** is the float32 weight storage — a "
                "property of the architecture. **Peak RSS above baseline** is "
                "the additional resident memory used during inference, which "
                "for patch models is dominated by the padded cube rather than "
                "the model, and so is a deployment-feasibility figure rather "
                "than an architectural one.\n\n")
        f.write("| Configuration | Params | Param mem (MB) | Type | "
                "Per image (s) | px/s | Peak RSS (MB) | RSS over base (MB) | "
                "<60 s |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            note = ""
            if r["extrapolated"]:
                note += " \\*"
            if r["random_init"]:
                note += " †"
            if r["rss_clean"]:
                rss_abs = f"{r['peak_rss_mb']:.0f}"
                rss_del = f"{r['rss_over_base']:.0f}"
            else:
                rss_abs = rss_del = "--"
            f.write(f"| {r['label']} | {r['params']:,} | "
                    f"{r['param_mem_mb']:.2f} | {r['type']} | "
                    f"{r['median_s']:.3f}{note} | {r['px_per_s']:,.0f} | "
                    f"{rss_abs} | {rss_del} | "
                    f"{'yes' if r['meets_limit'] else '**no**'} |\n")
        if any(r["extrapolated"] for r in rows):
            f.write("\n\\* extrapolated from a random subsample — estimate.\n")
        if any(r["random_init"] for r in rows):
            f.write("\n† randomly initialised weights — architecture-only "
                    "timing, valid for latency but not for accuracy.\n")
        if len(rows) > 1:
            f.write("\nPeak RSS is attributable only to the first "
                    "configuration timed in a process (`ru_maxrss` is a "
                    "non-resettable high-water mark); re-run one "
                    "configuration per process for per-config memory.\n")
        if _IMPORT_ERRORS:
            f.write("\n**Models not measured on this machine:**\n\n")
            for k, v in sorted(_IMPORT_ERRORS.items()):
                f.write(f"- `{k}` — {v}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Per-image inference latency and memory")
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
    ap.add_argument("--allow-random-init", action="store_true",
                    help="Time configs that have no checkpoint, using randomly "
                         "initialised weights. Valid for latency (which does "
                         "not depend on weight values) but such rows are "
                         "flagged and must be reported as architecture-only.")
    ap.add_argument("--out-suffix", default=None,
                    help="Appended to output filenames — use when running one "
                         "config per process for clean per-config memory.")
    ap.add_argument("--no-merge", action="store_true",
                    help="Overwrite the output files instead of merging with "
                         "any results already recorded for this device. By "
                         "default a partial run (e.g. --only SpectralFormer) "
                         "ADDS TO the existing file rather than replacing it, "
                         "so measurements taken in batches accumulate.")
    args = ap.parse_args()

    device = pick_device(args.device)
    dev_label = device_label(device)

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    cube = np.load(image_path, allow_pickle=False)[NPZ_CUBE_KEY]
    H, W, B = cube.shape
    total_px = H * W

    baseline_rss = peak_rss_mb()

    print(f"Device      : {dev_label}")
    print(f"Image       : {image_path.name}  ({H}x{W}x{B} = {total_px:,} px)")
    print(f"Repetitions : {args.reps} (+{args.warmup} warm-up)")
    print(f"Patch size  : {args.patch_size}")
    print(f"Baseline RSS: {baseline_rss:.1f} MB (interpreter + cube loaded)")
    if args.max_pixels:
        print(f"Subsampling : {args.max_pixels:,} px, extrapolated to full image")

    # Surface any model that could not be imported, so a skipped row is never
    # mistaken for a measured absence.
    if _IMPORT_ERRORS:
        print()
        print("Models that failed to import on this machine:")
        for k, v in sorted(_IMPORT_ERRORS.items()):
            print(f"  {k:16} {v}")
        print("  (install the missing package, or sync with --mode full)")

    print("=" * 78)

    # Fixed random subset, shared across configs so comparisons are fair
    coords = None
    if args.max_pixels and args.max_pixels < total_px:
        rng = np.random.default_rng(0)
        idx = rng.choice(total_px, size=args.max_pixels, replace=False)
        coords = np.stack([idx // W, idx % W], axis=1)

    # ── Output paths (resolved before the loop so results can be saved
    #     incrementally — a long run that is OOM-killed or interrupted must
    #     not lose the configurations already measured) ──────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tag = device if device != "cpu" else "pi" if "Raspberry" in dev_label else "cpu"
    if args.out_suffix:
        tag = f"{tag}_{args.out_suffix}"
    json_path = out_dir / f"latency_{tag}.json"
    md_path   = out_dir / f"latency_{tag}.md"

    # Load anything previously recorded for this device so partial runs
    # accumulate rather than replace.
    prior = []
    if json_path.exists() and not args.no_merge:
        try:
            prior = json.loads(json_path.read_text()).get("results", [])
            if prior:
                print(f"Existing    : {len(prior)} result(s) in {json_path.name} "
                      f"(will be merged)")
        except (json.JSONDecodeError, KeyError):
            print(f"Existing    : {json_path.name} unreadable — will overwrite")
            prior = []

    results = []
    n_timed = 0

    def merge_results():
        """Union of prior and current results, keyed by label, in CONFIGS order."""
        new_labels = {r["label"] for r in results}
        carried = [r for r in prior if r["label"] not in new_labels]
        order = {c["label"]: i for i, c in enumerate(CONFIGS)}
        return sorted(carried + results, key=lambda r: order.get(r["label"], 999))

    def save(final=False):
        """Write JSON and markdown. Called after every config, not just at the end."""
        merged = merge_results()
        payload = {
            "device":          dev_label,
            "device_tag":      tag,
            "image":           image_path.name,
            "image_shape":     [H, W, B],
            "total_pixels":    total_px,
            "patch_size":      args.patch_size,
            "reps":            args.reps,
            "subsampled_px":   len(coords) if coords is not None else None,
            "baseline_rss_mb": round(baseline_rss, 1),
            "import_errors":   _IMPORT_ERRORS,
            "complete":        final,
            "date":            date.today().isoformat(),
            "results":         merged,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        write_markdown(md_path, merged, dev_label, image_path, H, W, total_px,
                       args, coords, baseline_rss)
        return merged

    for cfg in CONFIGS:
        if args.only and not any(s.lower() in cfg["label"].lower()
                                 for s in args.only):
            continue

        model_class = resolve_class(cfg["model"])
        if model_class is None:
            why = _IMPORT_ERRORS.get(cfg["model"], "not available")
            print(f"\n{cfg['label']}: SKIPPED — {why}")
            continue

        ckpt_name = cfg.get("ckpt")
        random_init = ckpt_name is None
        ckpt = None

        if random_init:
            if not args.allow_random_init:
                print(f"\n{cfg['label']}: no checkpoint (needs "
                      f"--allow-random-init) — skipped")
                continue
        else:
            ckpt = Path(args.ckpt_dir) / ckpt_name
            if not ckpt.exists():
                print(f"\n{cfg['label']}: checkpoint not found "
                      f"({ckpt.name}) — skipped")
                continue

        print(f"\n-- {cfg['label']} --")
        try:
            model = load_model(model_class, ckpt, device, cfg.get("kwargs"))
        except TypeError as exc:
            print(f"  constructor mismatch ({exc}) — skipped")
            continue

        actual_params = count_params(model)
        param_mb = param_memory_mb(actual_params)
        if actual_params != cfg["params"]:
            print(f"  NOTE: measured {actual_params:,} params, "
                  f"config table says {cfg['params']:,} — using measured")
        if random_init:
            print("  weights: RANDOM (architecture-only timing)")
        print(f"  parameter memory: {param_mb:.2f} MB "
              f"({actual_params:,} x {BYTES_PER_PARAM} B)")

        is_patch = cfg["type"] == "patch"
        # Per-config override wins; otherwise patch models get a smaller batch
        # than pixel models because each sample is P*P times larger.
        if cfg.get("batch_size"):
            batch_size = cfg["batch_size"]
            print(f"  batch size: {batch_size} (per-config override)")
        else:
            batch_size = args.batch_size // 8 if is_patch else args.batch_size

        def one_pass():
            if is_patch:
                return run_patch(model, cube, device, batch_size,
                                 args.patch_size, coords)
            return run_pixel(model, cube, device, batch_size, coords)

        try:
            for _ in range(args.warmup):
                one_pass()
                synchronise(device)
        except Exception as exc:
            print(f"  inference failed ({type(exc).__name__}: {exc}) — skipped")
            print(f"  check the 'type' field for this config "
                  f"(currently '{cfg['type']}')")
            continue

        times = []
        for r in range(args.reps):
            t0 = time.perf_counter()
            n_px = one_pass()
            synchronise(device)
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"  rep {r + 1}: {dt:.3f}s  ({n_px / dt:,.0f} px/s)")

        rss_after = peak_rss_mb()
        # Only the first config in a process yields a clean attribution,
        # since ru_maxrss is a non-resettable high-water mark.
        mem_clean = (n_timed == 0)
        n_timed += 1

        measured = statistics.median(times)
        extrapolated = coords is not None
        scale = (total_px / len(coords)) if extrapolated else 1.0
        per_image = measured * scale

        results.append({
            "label":          cfg["label"],
            "model":          cfg["model"],
            "params":         actual_params,
            "param_mem_mb":   round(param_mb, 3),
            "type":           cfg["type"],
            "checkpoint":     ckpt_name,
            "random_init":    random_init,
            "median_s":       round(per_image, 4),
            "min_s":          round(min(times) * scale, 4),
            "max_s":          round(max(times) * scale, 4),
            "px_per_s":       round(total_px / per_image, 1),
            "peak_rss_mb":    round(rss_after, 1),
            "rss_over_base":  round(rss_after - baseline_rss, 1),
            "rss_clean":      mem_clean,
            "extrapolated":   extrapolated,
            "meets_limit":    per_image < CLINICAL_LIMIT_S,
        })

        flags = []
        if extrapolated:
            flags.append("extrapolated")
        if random_init:
            flags.append("random weights")
        flag = f" ({', '.join(flags)})" if flags else ""
        ok = "PASS" if per_image < CLINICAL_LIMIT_S else "FAIL"
        mem_note = "" if mem_clean else " [not attributable]"
        print(f"  -> per image: {per_image:.3f}s{flag}   "
              f"{total_px / per_image:,.0f} px/s   [{ok} 60s limit]")
        print(f"  -> peak RSS : {rss_after:.1f} MB "
              f"(+{rss_after - baseline_rss:.1f} over baseline){mem_note}")

        # Persist immediately — the next configuration may be OOM-killed.
        save()
        print(f"  -> saved    : {json_path.name}")

    if not results:
        sys.exit("\nNo configurations were timed — check checkpoints and imports.")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  LATENCY AND MEMORY SUMMARY — {dev_label}")
    print("=" * 78)
    print(f"  {'Configuration':<26} {'Params':>10} {'Wt MB':>7} "
          f"{'Per image':>11} {'px/s':>11} {'dRSS':>7}  {'60s':>5}")
    print("  " + "-" * 84)
    for r in results:
        note = "*" if r["extrapolated"] else "+" if r["random_init"] else " "
        drss = f"{r['rss_over_base']:.0f}" if r["rss_clean"] else "--"
        print(f"  {r['label']:<26} {r['params']:>10,} "
              f"{r['param_mem_mb']:>7.2f} "
              f"{r['median_s']:>10.3f}s{note} {r['px_per_s']:>11,.0f} "
              f"{drss:>7}  {'PASS' if r['meets_limit'] else 'FAIL':>5}")
    print()
    print("  Wt MB = parameter memory (architecture property)")
    print("  dRSS  = peak process RSS above baseline, MB (data-dominated)")
    if any(r["extrapolated"] for r in results):
        print("  *     = extrapolated from a subsample — report as an estimate")
    if any(r["random_init"] for r in results):
        print("  +     = randomly initialised — architecture-only timing")
    if len(results) > 1:
        print("  --    = RSS not attributable; re-run one config per process")
        print("          (--only ... --out-suffix ...) for per-config memory")

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tag = device if device != "cpu" else "pi" if "Raspberry" in dev_label else "cpu"
    if args.out_suffix:
        tag = f"{tag}_{args.out_suffix}"

    # ── Final write ──────────────────────────────────────────────────────────
    merged = save(final=True)

    if len(merged) > len(results):
        print()
        print(f"Merged with earlier results in {json_path.name}:")
        print(f"  measured this run : {len(results)}")
        print(f"  carried over      : {len(merged) - len(results)}")
        print(f"  total in output   : {len(merged)}")
        print("  (use --no-merge to overwrite instead)")

    print(f"\n  Markdown -> {md_path}")
    print(f"  JSON     -> {json_path}")


if __name__ == "__main__":
    main()