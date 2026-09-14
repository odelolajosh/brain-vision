"""
brainvision — Intraoperative HSI Brain Tumour Classification
Defence touchscreen demo.

Four-screen flow, touch-only, built for an 800x480 7" Raspberry Pi 5 panel:

    Welcome/Idle --"Begin Demo"--> Choose Mode --+--> Static Comparison
                       ^                          |
                       +---------- Home (⌂) ------+--> Realtime Demo

Everything a presenter tunes for a specific defence lives in the CASES list
and the MODELS registry below. Import everything possible from the
`brainvision` package — no duplicated model or preprocessing code.

Run from the repo root:
    streamlit run demo/app.py
"""

import random
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

from brainvision.constants import CLASS_NAMES, N_DECIMATED_BANDS, N_CLASSES
from brainvision.device import get_device
from brainvision.preprocessing import minmax_normalise
from brainvision.models.fabelo_dnn import FabeloDNN
from brainvision.models.hu_1dcnn import HuEtAl1DCNN

DEVICE = get_device()   # CUDA > MPS > CPU — resolves to CPU on the Pi 5 itself

# ═════════════════════════════════════════════════════════════════════════════
#  PRESENTER CONFIG 1 — the three preset images, shared by Static & Realtime
# ═════════════════════════════════════════════════════════════════════════════
# `crop` is (row0, col0, size) in pixels, or None for the whole image. These
# three were chosen from the HELICoiD set already on disk to match the three
# required narrative roles (spec §4.1) — not synthesised or arbitrary.

CASES = [
    {
        "key":     "clear_match",
        "title":   "Clear Match",
        "role":    "A clean success — prediction and labels line up.",
        "image":   "processed/first_campaign/008-02.npz",
        "crop":    None,
    },
    {
        "key":     "spatial_sensitivity",
        "title":   "Spatial Sensitivity Gain",
        "role":    "The sensitivity-prioritised model catches more tumour tissue.",
        "image":   "processed/second_campaign/038-01.npz",
        "crop":    (170, 285, 165),
    },
    {
        "key":     "honest_near_miss",
        "title":   "Honest Near-Miss",
        "role":    "A case the model gets partly wrong — shown on purpose.",
        "image":   "processed/first_campaign/012-01.npz",
        "crop":    None,
    },
]

# Static Comparison screen's own preset set — 7 images, shuffled between via
# the top-bar button. Deliberately separate from CASES (Realtime's 3): each
# screen indexes its own list, so they can't cache-collide or drift together.
# Picked for a spread of class mixes (checked against each image's labelled
# pixel counts) rather than reusing the same 3 narrative cases.
STATIC_CASES = [
    {"key": "clear_match",     "title": "Clear Match",
     "image": "processed/first_campaign/008-02.npz",  "crop": None},
    {"key": "vessel_tumour",   "title": "Vessel & Tumour",
     "image": "processed/first_campaign/012-01.npz",  "crop": None},
    {"key": "balanced_mix",    "title": "Balanced Mix",
     "image": "processed/second_campaign/038-01.npz", "crop": None},
    {"key": "tumour_dominant", "title": "Tumour Dominant",
     "image": "processed/first_campaign/020-01.npz",  "crop": None},
    {"key": "vessel_heavy",    "title": "Vessel Heavy",
     "image": "processed/first_campaign/015-01.npz",  "crop": None},
    {"key": "rich_mix",        "title": "Rich Mix",
     "image": "processed/first_campaign/012-02.npz",  "crop": None},
    {"key": "compact_mix",     "title": "Compact Mix",
     "image": "processed/second_campaign/040-02.npz", "crop": None},
]

# ═════════════════════════════════════════════════════════════════════════════
#  PRESENTER CONFIG 2 — model registry
# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint filenames follow the run-naming convention
# {model}_{loss}_{balance}_{fold}_{strategy}. Fold is the MEDIAN fold from
# each model's results/test_eval_*.md, matching thesis reporting convention.

MODELS = {
    "1D-DNN-Fabelo": dict(
        cls=FabeloDNN, kind="pixel", kwargs={},
        checkpoint="checkpoints/1dnnfabelo_ce_bal_fold3_vpfabelo.pt",
        params=4_936,
        arch_label="1D-DNN-Fabelo (Fabelo et al., 2019) — 4,936 params",
    ),
    "1D-CNN-Hu": dict(
        cls=HuEtAl1DCNN, kind="pixel", kwargs={},
        # See STATIC_MODEL_IS_FALLBACK below — the CE+no-balancing checkpoint
        # this points at when available is fold 3; the fallback used here is
        # fold 4 (CE+balanced's own median fold).
        checkpoint="checkpoints/1dcnn_ce_bal_fold4_vpfabelo.pt",
        params=76_824,
        arch_label="1D-CNN-Hu, CE (Hu et al., 2015) — 76,824 params",
    ),
}

MODEL_A_KEY = "1D-DNN-Fabelo"   # used by the Realtime screen only

# ── Static Comparison screen — single fixed (model, image) pair ────────────
# The intended checkpoint (CE loss, no class-balancing, median fold 3 per
# results/test_eval_1dcnn_ce_nobal_vpfabelo.md) was evaluated but its .pt
# weights were not kept in this checkout's checkpoints/ directory — the same
# situation the old Model B (SpectralFormer-CAF-CE) was in. MODELS["1D-CNN-Hu"]
# above already points at the nearest available fallback (CE, balanced,
# fold 4); this flag just decides whether the screen shows that it's not
# running the exact preferred variant.
STATIC_MODEL_KEY             = "1D-CNN-Hu"
STATIC_PREFERRED_CHECKPOINT  = "checkpoints/1dcnn_ce_nobal_fold3_vpfabelo.pt"
STATIC_MODEL_IS_FALLBACK     = not Path(STATIC_PREFERRED_CHECKPOINT).exists()

NPZ_CUBE_KEY = "processed"
NPZ_GT_KEY   = "labels"

# ═════════════════════════════════════════════════════════════════════════════
#  Design tokens — sampled directly from the reference mockup's pixels.
#  Single source of truth: emitted as CSS custom properties (see the <style>
#  block below) and reused in the few places Python builds inline styles, so
#  a future palette change only requires editing this block.
# ═════════════════════════════════════════════════════════════════════════════
BG_APP             = "#001B33"
BG_PANEL_RAISED    = "#113A5E"

ACCENT_PRIMARY     = "#0A7CF5"   # top-level CTAs (Begin Demo, Static Image tile)
ACCENT_SECONDARY   = "#2A3FA0"   # Realtime Demo tile
ACCENT_ICON_CHIP   = "#3798FC"

STATE_SELECTED     = "#0568C7"   # active segment in an in-screen toggle
STATE_UNSELECTED   = "#022E56"   # inactive segment in the same toggle

STATUS_READY_GREEN = "#05C832"
STATUS_LIVE_RED    = "#FF1020"
STATUS_OFF         = "#4F6C88"   # not sampled — muted neutral for a paused/off dot

TEXT_HEADING       = "#FFFFFF"
TEXT_SECONDARY     = "#A9C2DA"

PROGRESS_TRACK     = "#04447A"
PROGRESS_FILL      = "#0074B0"

# Demo-only tissue-class skin for the kiosk UI. brainvision.constants.
# CLASS_COLORS (used by notebooks/thesis figures) is intentionally untouched —
# this repaints the same 4 classes for this screen only.
DEMO_CLASS_COLORS = {1: "#22C55E", 2: "#FF1521", 3: "#FAF404", 4: "#87A5C7"}

# Static screen's classification-map + legend palette — matches the
# presenter's reference slide (blue/red/green/grey), independent of
# DEMO_CLASS_COLORS (the Realtime screen's palette).
STATIC_CLASS_COLORS = {1: "#2F6FDE", 2: "#C41E3A", 3: "#3CA55C", 4: "#A9B0B8"}

SCAN_MARKER   = TEXT_HEADING   # minimap border / viewport-rectangle colour — bright,
                                # high-contrast, and not used as a fill anywhere else
SCAN_N_STEPS  = 240
SPEED_CONFIG  = {   # label -> (path steps advanced per tick, seconds between ticks)
    "Slow":   (1, 0.55),
    "Normal": (2, 0.32),
    "Fast":   (4, 0.16),
}

# Realtime viewport — the small window actually being classified. The primary
# display shows this crop enlarged; the minimap shows where it sits in the
# full source image. Fixed pixel-size box (clamped to the image), not a
# shrinking crop, so the minimap rectangle's size reads consistently.
VIEWPORT_FRAC   = 0.34
VIEWPORT_ASPECT = 1.45          # viewport width / height
VIEWPORT_MIN_H  = 56
VIEWPORT_MAX_H  = 260
MINIMAP_SIZE    = 130           # minimap thumbnail width, px
OVERLAY_ALPHA   = 0.55
OVERLAY_CLASSES = (1, 2, 3)     # NT, TT, BV tinted; Background left as raw tissue colour


# ═════════════════════════════════════════════════════════════════════════════
#  Page setup + kiosk styling — tuned for 800x480 (7" touchscreen)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="brainvision — Intraoperative HSI Classification",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_TOKENS_CSS = f"""
:root {{
    --bg-app: {BG_APP};
    --bg-panel-raised: {BG_PANEL_RAISED};
    --accent-primary: {ACCENT_PRIMARY};
    --accent-secondary: {ACCENT_SECONDARY};
    --accent-icon-chip: {ACCENT_ICON_CHIP};
    --state-selected: {STATE_SELECTED};
    --state-unselected: {STATE_UNSELECTED};
    --status-ready-green: {STATUS_READY_GREEN};
    --status-live-red: {STATUS_LIVE_RED};
    --status-off: {STATUS_OFF};
    --text-heading: {TEXT_HEADING};
    --text-secondary: {TEXT_SECONDARY};
    --progress-track: {PROGRESS_TRACK};
    --progress-fill: {PROGRESS_FILL};
    --tissue-nt: {DEMO_CLASS_COLORS[1]};
    --tissue-tt: {DEMO_CLASS_COLORS[2]};
    --tissue-bv: {DEMO_CLASS_COLORS[3]};
    --tissue-bg: {DEMO_CLASS_COLORS[4]};
}}
"""

_REST_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]
    { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stAppViewContainer"] { background: var(--bg-app); }
.block-container { max-width: 800px; padding: 8px 12px 6px 12px; }

/* Global Background and Font */
.stApp {
    background-color: #04162a; 
    color: #ffffff;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

.bv-screen-title { font-size: 20px; font-weight: 800; color: var(--text-heading); margin: 0; }
.bv-screen-sub   { font-size: 12px; color: var(--text-secondary); margin: 1px 0 0 0; }

.bv-chip {
    display: inline-block; background: var(--bg-panel-raised); border-radius: 9px;
    padding: 7px 11px; margin: 3px 4px 3px 0; border-left: 4px solid var(--state-unselected);
}
.bv-chip .k { font-size: 10px; letter-spacing: .05em; text-transform: uppercase;
              color: var(--text-secondary); margin-bottom: 2px; }
.bv-chip .v { font-size: 24px; font-weight: 800; color: var(--text-heading); line-height: 1.05; }
.bv-chip .p { font-size: 11px; color: var(--text-secondary); margin-top: 2px; max-width: 260px; }
.bv-chip.tt   { border-left-color: var(--tissue-tt); }
.bv-chip.good { border-left-color: var(--status-ready-green); }

.bv-sparsenote {
    background: var(--bg-panel-raised); border-radius: 8px;
    padding: 7px 10px 7px 8px; font-size: 12px; color: var(--text-secondary); line-height: 1.3;
    display: flex; align-items: flex-start; gap: 7px;
}
.bv-sparsenote .ic { color: var(--status-ready-green); flex-shrink: 0; }
.bv-modelnote {
    background: var(--bg-panel-raised); border-radius: 8px; padding: 6px 10px;
    font-size: 12px; color: var(--text-secondary); margin-top: 4px;
}
.bv-warn {
    background: #3A2A12; border: 1px solid #7A5A22; color: #F0C987;
    border-radius: 8px; padding: 7px 10px; font-size: 12px; font-weight: 600;
    margin-bottom: 6px;
}

.bv-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--bg-panel-raised); border-radius: 18px; padding: 5px 12px;
    font-size: 13px; color: var(--text-heading); font-variant-numeric: tabular-nums;
}
.bv-badge .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--status-ready-green); }
.bv-badge .dot.live { background: var(--status-live-red); }
.bv-badge .dot.off  { background: var(--status-off); }

.bv-panel-label { font-size: 12px; letter-spacing: .05em; text-transform: uppercase;
                  color: var(--text-secondary); margin: 0 0 3px 0; }

.stButton > button {
    width: 100%; min-height: 46px; font-size: 14px; font-weight: 700;
    border-radius: 14px; border: none; background: var(--bg-panel-raised);
    color: var(--text-heading);
}
.stButton > button:hover { filter: brightness(1.18); }
.stButton > button:focus { box-shadow: none; }
.stButton > button[kind="primary"] {
    background: var(--accent-primary); border: none; color: var(--text-heading);
}

/* Mode-select tiles (Choose Mode screen) — fixed fills, not a toggle */
.st-key-tile_static button   { background: var(--accent-primary) !important; }
.st-key-tile_realtime button { background: var(--accent-secondary) !important; }

/* Segmented toggles: A/B model switch, Ground Truth/RGB/Agreement, Speed */
.st-key-toggle_model button[kind="secondary"],
.st-key-toggle_view button[kind="secondary"],
.st-key-toggle_speed button[kind="secondary"] {
    background: var(--state-unselected) !important; color: var(--text-secondary) !important;
}
.st-key-toggle_model button[kind="primary"],
.st-key-toggle_view button[kind="primary"],
.st-key-toggle_speed button[kind="primary"] {
    background: var(--state-selected) !important; color: var(--text-heading) !important;
}

/* Realtime Pause/Play — an always-on in-screen control, not a navigation button */
.st-key-rt_playpause button { background: var(--state-selected) !important; }

[data-testid="stImage"] img {
    max-height: 190px; width: auto !important; display: block; margin: 0 auto;
    object-fit: contain; border-radius: 6px;
}

.bv-legend { display: inline-flex; align-items: center; gap: 5px; margin-right: 12px;
             font-size: 12px; color: var(--text-secondary); }
.bv-legend .sw { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }

[data-testid="stExpander"] { border: 1px solid var(--bg-panel-raised); border-radius: 10px;
                              background: var(--bg-app); }

[data-testid="stProgressBarTrack"] {
    background-color: var(--progress-track) !important;
    border-radius: 999px !important; overflow: hidden;
}
[data-testid="stProgressBarTrack"] > div {
    background-color: var(--progress-fill) !important; border-radius: 999px !important;
}

/* Static screen's main section: big classification map + legend, filling
   most of the viewport so only the metrics below need a scroll. */
.st-key-static_main { display: flex; align-items: center; min-height: 62vh; }
.st-key-static_main [data-testid="stHorizontalBlock"] { width: 100%; align-items: center; }
.st-key-static_main [data-testid="stImage"] img {
    max-height: 64vh !important; width: auto !important; max-width: 100%;
    object-fit: contain; border-radius: 10px; display: block; margin: 0 auto;
}

/* Realtime screen's main section: big viewport (+ minimap inset) + a right
   panel (legend, Pause/Play, Speed) — leaves room below for the progress bar. */
.st-key-realtime_main { display: flex; align-items: center; min-height: 52vh; }
.st-key-realtime_main [data-testid="stHorizontalBlock"] { width: 100%; align-items: center; }
.st-key-realtime_main [data-testid="stImage"] img {
    max-height: 56vh !important; width: auto !important; max-width: 100%;
    object-fit: contain; border-radius: 10px; display: block; margin: 0 auto;
}
"""

st.markdown(f"<style>{_TOKENS_CSS}{_REST_CSS}</style>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Data / model helpers
# ═════════════════════════════════════════════════════════════════════════════
def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


CLASS_RGB = {k: _hex_to_rgb(v) for k, v in DEMO_CLASS_COLORS.items()}
CLASS_RGB[0] = (24, 26, 34)   # unlabelled -> near-background

STATIC_CLASS_RGB = {k: _hex_to_rgb(v) for k, v in STATIC_CLASS_COLORS.items()}
STATIC_CLASS_RGB[0] = (24, 26, 34)

PSEUDO_RGB_BANDS = (88, 42, 7)   # ~709 R, ~539 G, ~479 B

# Realtime indexes CASES, Static indexes its own STATIC_CASES — keyed by name
# (not by passing the list itself) so each screen's cache entries can never
# collide even though both use plain integer indices.
CASE_LISTS = {"realtime": CASES, "static": STATIC_CASES}


@st.cache_resource(show_spinner=False)
def load_case_cube(list_name: str, case_idx: int) -> tuple:
    """Load + normalise + crop one preset image. Cached — the file never changes."""
    case = CASE_LISTS[list_name][case_idx]
    data = np.load(case["image"], allow_pickle=False)
    cube = minmax_normalise(data[NPZ_CUBE_KEY].astype(np.float32))
    gt   = data[NPZ_GT_KEY]

    if case["crop"]:
        r0, c0, sz = case["crop"]
        H, W = cube.shape[:2]
        r0, c0 = max(0, min(r0, H - sz)), max(0, min(c0, W - sz))
        cube, gt = cube[r0:r0 + sz, c0:c0 + sz], gt[r0:r0 + sz, c0:c0 + sz]
    return cube, gt


def make_pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    r, g, b = (cube[:, :, i] for i in PSEUDO_RGB_BANDS)
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def make_label_map(labels: np.ndarray, class_rgb: dict = CLASS_RGB) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl, color in class_rgb.items():
        out[labels == lbl] = color
    return out


def make_overlay(rgb: np.ndarray, pred: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Tint predicted NT/TT/BV pixels over the pseudo-RGB base; Background is
    left as raw tissue colour so the overlay reads like an intraoperative
    classification aid, not a flat colour map."""
    out = rgb.astype(np.float32).copy()
    for lbl in OVERLAY_CLASSES:
        m = pred == lbl
        out[m] = (1 - alpha) * out[m] + alpha * np.array(CLASS_RGB[lbl], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_live_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    F1-noBG, TT sensitivity, and agreement fraction — computed directly from
    the confusion counts over LABELLED pixels only, following the Fabelo et
    al. (2023) convention used throughout the thesis (see
    brainvision/metrics.py). sklearn is deliberately not imported here: the
    Pi build strips it out (see demo/requirements_demo.txt), so this
    replicates macro-F1-excl-BG by hand from a 4x4 confusion matrix.
    """
    labelled = gt > 0
    n_lab = int(labelled.sum())
    if n_lab == 0:
        return dict(n_labelled=0, n_unlabelled=int((~labelled).sum()),
                    n_correct=0, n_incorrect=0, agree_frac=float("nan"),
                    tt_total=0, tt_sens=float("nan"), f1_no_bg=float("nan"))

    t = gt[labelled].astype(int) - 1     # 0=NT,1=TT,2=BV,3=BG
    p = pred[labelled].astype(int) - 1
    n = N_CLASSES
    cm = np.zeros((n, n), dtype=np.int64)
    for ti in range(n):
        row = t == ti
        for pi in range(n):
            cm[ti, pi] = int((row & (p == pi)).sum())

    f1 = np.zeros(n)
    for i in range(n):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        f1[i] = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    f1_no_bg = float(f1[:3].mean())      # NT, TT, BV — exclude BG

    tt_total = int((t == 1).sum())
    tt_hit   = int(((t == 1) & (p == 1)).sum())
    n_correct = int((t == p).sum())

    return dict(
        n_labelled=n_lab, n_unlabelled=int((~labelled).sum()),
        n_correct=n_correct, n_incorrect=n_lab - n_correct,
        agree_frac=n_correct / n_lab,
        tt_total=tt_total,
        tt_sens=(tt_hit / tt_total) if tt_total else float("nan"),
        f1_no_bg=f1_no_bg,
    )


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    spec = MODELS[model_key]
    model = spec["cls"](input_channels=N_DECIMATED_BANDS, n_classes=N_CLASSES,
                         **spec["kwargs"])
    state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model


def infer(model, crop: np.ndarray, kind: str, patch_size: int = 11) -> tuple:
    """Run inference on a full crop. Returns (pred_map[H,W], latency_ms)."""
    t0 = time.perf_counter()
    if kind == "pixel":
        H, W, B = crop.shape
        flat = crop.reshape(-1, B).astype(np.float32)
        preds = np.empty(flat.shape[0], dtype=np.int64)
        batch_size = 4096
        with torch.no_grad():
            for s in range(0, flat.shape[0], batch_size):
                e = min(s + batch_size, flat.shape[0])
                x = torch.from_numpy(flat[s:e]).to(DEVICE)
                preds[s:e] = model(x).argmax(1).cpu().numpy() + 1
        pred = preds.reshape(H, W)
    else:
        H, W, B = crop.shape
        P, pad = patch_size, patch_size // 2
        padded = np.pad(crop.transpose(2, 0, 1), ((0, 0), (pad, pad), (pad, pad)),
                         mode="reflect")
        preds, idx, coords, batch_size = np.empty(H * W, dtype=np.int64), 0, [], 512
        with torch.no_grad():
            for r in range(H):
                for c in range(W):
                    coords.append((r, c))
                    if len(coords) == batch_size or (r == H - 1 and c == W - 1):
                        patches = np.stack([padded[:, rr:rr + P, cc:cc + P]
                                            for rr, cc in coords]).astype(np.float32)
                        x = torch.from_numpy(patches).to(DEVICE)
                        out = model(x).argmax(1).cpu().numpy() + 1
                        preds[idx:idx + len(coords)] = out
                        idx += len(coords)
                        coords = []
        pred = preds.reshape(H, W)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return pred, latency_ms


# ═════════════════════════════════════════════════════════════════════════════
#  Static Comparison — one fixed model, shuffled across STATIC_CASES
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def compute_static(case_idx: int) -> dict:
    cube, gt = load_case_cube("static", case_idx)
    spec = MODELS[STATIC_MODEL_KEY]
    model = load_model(STATIC_MODEL_KEY)
    pred, latency_ms = infer(model, cube, spec["kind"],
                              spec["kwargs"].get("patch_size", 11))

    return {
        "pred_map":   make_label_map(pred, STATIC_CLASS_RGB),
        "latency_ms": latency_ms,
        "metrics":    compute_live_metrics(pred, gt),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Realtime Demo — deterministic scan path + one-shot Model A prediction
# ═════════════════════════════════════════════════════════════════════════════
def build_scan_path(h: int, w: int, n_steps: int = SCAN_N_STEPS) -> np.ndarray:
    """
    Deterministic serpentine sweep across an HxW crop, mimicking a hand-held
    probe passing over exposed tissue. Pure function of (h, w) — no RNG, no
    clock — so the path is byte-identical on every run and every rehearsal.
    """
    t = np.linspace(0.0, 1.0, n_steps)
    margin_r, margin_c = h * 0.16, w * 0.10
    col = margin_c + t * (w - 2 * margin_c)
    row = margin_r + (h - 2 * margin_r) * (0.5 - 0.5 * np.cos(2 * np.pi * 1.5 * t))
    return np.stack([row, col], axis=1)


def compute_viewport_size(h: int, w: int) -> tuple:
    """Fixed pixel-size viewport box, clamped to the crop — never shrinks near
    the edges (the path margins in build_scan_path already keep it in-bounds)."""
    edge = int(np.clip(min(h, w) * VIEWPORT_FRAC, VIEWPORT_MIN_H, VIEWPORT_MAX_H))
    vp_h = min(edge, h)
    vp_w = min(int(edge * VIEWPORT_ASPECT), w)
    return vp_h, vp_w


@st.cache_resource(show_spinner=False)
def compute_realtime(case_idx: int) -> dict:
    """
    Precompute for the Realtime screen: the scan path is a pure function of
    the image size (build_scan_path — fixed forever, no RNG). For every step
    along it we run a REAL inference pass over just that step's
    viewport-sized crop, so the latency badge reflects per-viewport timing,
    never a one-shot full-frame number. Model A (FabeloDNN) classifies each
    pixel independently of its neighbours, so a viewport crop's predictions
    are identical to slicing the equivalent region out of a full-frame
    prediction — precomputing every step here means playback (even at
    "Fast") never has to wait on inference during the animation loop, which
    matters on the Pi 5's CPU.
    """
    cube, gt = load_case_cube("realtime", case_idx)
    h, w = cube.shape[:2]
    model = load_model(MODEL_A_KEY)

    # Plain pseudo-RGB of the full scene — the minimap's job is spatial
    # orientation ("where is the viewport"), not a second copy of the
    # classified output already shown at full size in the primary display.
    rgb_full = make_pseudo_rgb(cube)

    path = build_scan_path(h, w)
    vp_h, vp_w = compute_viewport_size(h, w)

    frames = []
    for row, col in path:
        r0 = int(np.clip(row - vp_h / 2, 0, h - vp_h))
        c0 = int(np.clip(col - vp_w / 2, 0, w - vp_w))
        crop = cube[r0:r0 + vp_h, c0:c0 + vp_w]
        pred, latency_ms = infer(model, crop, MODELS[MODEL_A_KEY]["kind"])
        frames.append({
            "overlay":    make_overlay(make_pseudo_rgb(crop), pred),
            "latency_ms": latency_ms,
            "box":        (r0, c0, r0 + vp_h, c0 + vp_w),
        })

    return {"frames": frames, "rgb_full": rgb_full, "shape": (h, w)}


def compose_viewport_frame(frame: dict, rgb_full: np.ndarray, full_shape: tuple,
                            display_w: int = 760) -> np.ndarray:
    """Enlarge the current viewport crop's live classified prediction to fill
    the primary display, and paste a minimap — the plain pseudo-RGB scene
    (not classification colours — the minimap is for orientation, the
    primary view already shows the model's output) with a rectangle over
    the current viewport — into its top-right corner. The rectangle's
    position is derived from the same box used to build this frame's crop,
    so the two views can never disagree about where the camera currently is."""
    overlay = frame["overlay"]
    h_sub, w_sub = overlay.shape[:2]
    display_h = max(1, int(display_w * h_sub / w_sub))
    main = Image.fromarray(overlay).resize((display_w, display_h), Image.NEAREST)

    H, W = full_shape
    mini_w = MINIMAP_SIZE
    mini_h = max(1, int(mini_w * H / W))
    mini = Image.fromarray(rgb_full).resize((mini_w, mini_h), Image.BILINEAR)
    draw = ImageDraw.Draw(mini)
    r0, c0, r1, c1 = frame["box"]
    sx, sy = mini_w / W, mini_h / H
    draw.rectangle([c0 * sx, r0 * sy, c1 * sx, r1 * sy], outline=SCAN_MARKER, width=2)

    border = 3
    framed = Image.new("RGB", (mini_w + 2 * border, mini_h + 2 * border), SCAN_MARKER)
    framed.paste(mini, (border, border))
    pad = 10
    main.paste(framed, (display_w - framed.width - pad, pad))
    return np.array(main)


# ═════════════════════════════════════════════════════════════════════════════
#  System-ready check (real, not hardcoded)
# ═════════════════════════════════════════════════════════════════════════════
def system_ready() -> tuple:
    model_a_ok      = Path(MODELS[MODEL_A_KEY]["checkpoint"]).exists()
    static_model_ok = Path(MODELS[STATIC_MODEL_KEY]["checkpoint"]).exists()
    data_ok         = all(Path(c["image"]).exists() for c in CASES + STATIC_CASES)
    ready = model_a_ok and static_model_ok and data_ok
    if ready:
        return True, "Raspberry Pi 5 — System Ready"
    missing = []
    if not model_a_ok:      missing.append("Realtime model checkpoint")
    if not static_model_ok: missing.append("Static model checkpoint")
    if not data_ok:         missing.append("preset images")
    return False, "System Degraded — missing " + ", ".join(missing)


# ═════════════════════════════════════════════════════════════════════════════
#  Small UI atoms
# ═════════════════════════════════════════════════════════════════════════════
def goto(screen: str) -> None:
    st.session_state.screen = screen


def top_bar(title: str, show_home: bool = True) -> None:
    left, mid = st.columns([1, 8])
    with left:
        if show_home and st.button("⌂", key=f"home_{st.session_state.screen}",
                                    help="Home"):
            goto("welcome")
            st.rerun()
    with mid:
        st.markdown(f"<div class='bv-screen-title'>{title}</div>",
                    unsafe_allow_html=True)


def latency_badge(ms: float, live: bool | None = None) -> str:
    dot_cls = "dot"
    if live is True:  dot_cls = "dot live"
    if live is False: dot_cls = "dot off"
    if ms < 1000:
        return f"<div class='bv-badge'><span class='{dot_cls}'></span>{ms:.0f} ms</div>"
    return f"<div class='bv-badge'><span class='{dot_cls}'></span>{ms / 1000:.1f} s</div>"


def classification_legend(class_colors: dict) -> str:
    """Legend shared by the Static and Realtime right panels — one row per
    class: a solid swatch plus its plain name, sized to read from across a
    room. Each screen passes its own palette (STATIC_CLASS_COLORS /
    DEMO_CLASS_COLORS) so the swatches always match what's on screen."""
    rows = "".join(
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:18px'>"
        f"<span style='width:28px;height:28px;border-radius:6px;flex-shrink:0;"
        f"background:{class_colors[l]}'></span>"
        f"<span style='font-size:17px;font-weight:600;color:{TEXT_HEADING}'>"
        f"{CLASS_NAMES[l].split(' (')[0]}</span></div>"
        for l in sorted(class_colors)
    )
    return (
        f"<div style='display:flex;flex-direction:column;justify-content:center'>"
        f"{rows}</div>"
    )


def chip(label: str, value: str, caption: str, css: str = "") -> str:
    return (f"<div class='bv-chip {css}'><div class='k'>{label}</div>"
            f"<div class='v'>{value}</div><div class='p'>{caption}</div></div>")


# ═════════════════════════════════════════════════════════════════════════════
#  Screen 1 — Welcome / Idle
# ═════════════════════════════════════════════════════════════════════════════
def screen_welcome() -> None:
    ready, status_text = system_ready()
    dot_color = STATUS_READY_GREEN if ready else STATUS_LIVE_RED

    # Compact header matching the UI design
    st.markdown("""
    <div style='display: flex; align-items: center; border-bottom: 1px solid var(--bg-panel-raised); padding-bottom: 8px; margin-bottom: 15px;'>
        <div style='font-size: 28px; color: var(--accent-primary); margin-right: 12px;'>🧠</div>
        <div>
            <p style='font-size: 18px; font-weight: 800; margin: 0; line-height: 1.1; color: var(--text-heading);'>Brain Tumour Classification</p>
            <p style='font-size: 12px; color: var(--text-secondary); margin: 0;'>Hyperspectral Imaging Demo</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Split layout optimized for landscape 7-inch LCD
    col1, col2 = st.columns([1, 1.1], gap="medium")

    with col1:
        try:
            # Referenced verbatim as requested
            st.image("image_7ae572.png", use_container_width=True)
        except Exception:
            st.info("ℹ️ Place 'image_7ae572.png' here")

    with col2:
        # Hero text scaled down for 800x480 
        st.markdown(
            "<div style='font-size: 26px; font-weight: 800; line-height: 1.2; margin-top: 5px; margin-bottom: 12px; color: var(--text-heading);'>"
            "Real-time AI for<br>Safer Surgery</div>"
            "<div style='font-size: 14px; color: var(--text-secondary); line-height: 1.4; margin-bottom: 25px; max-width: 95%;'>"
            "Classifying brain tissue during surgery using hyperspectral imaging and deep learning.</div>",
            unsafe_allow_html=True
        )
        
        # Original functioning button (styled automatically by your existing _REST_CSS)
        if st.button("Begin Demo →", key="begin", type="primary"):
            goto("mode")
            st.rerun()

    # Footer containing dynamic system check
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color: var(--text-heading); font-size: 12px; display: flex; align-items: center;'>"
        f"<span style='color: {dot_color}; font-size: 14px; margin-right: 8px;'>●</span> {status_text}</div>",
        unsafe_allow_html=True,
    )
    
    # Preserve original missing-files caption
    if not ready:
        st.caption("Not independently verified as running on Pi 5 hardware — "
                   "this check only confirms model + data files are present.")

# ═════════════════════════════════════════════════════════════════════════════
#  Screen 2 — Choose Mode
# ═════════════════════════════════════════════════════════════════════════════
def screen_mode() -> None:
    # Inject CSS for the rich tiles and overlay technique tailored to 800x480
    st.markdown("""
    <style>
    /* Turn the containers into relative bounding boxes */
    .st-key-tile_static, .st-key-tile_realtime {
        position: relative;
        height: 96px;
        margin-bottom: 8px;
    }

    .st-key-mode_static, .st-key-mode_realtime {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        height: 96px !important;
        right: 0 !important;
        z-index: 10 !important;
        opacity: 1 !important;
    }
    
    /* Stretch invisible Streamlit buttons perfectly over the custom HTML cards */
    .st-key-tile_static .stButton, .st-key-tile_realtime .stButton {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        bottom: 0 !important;
        right: 0 !important;
        z-index: 10 !important;
        opacity: 0 !important;
    }
    
    .st-key-tile_static .stButton button, .st-key-tile_realtime .stButton button {
        width: 100% !important;
        height: 100% !important;
        padding: 0 !important;
        cursor: pointer !important;
    }
    
    /* Card visual styling */
    .demo-card {
        border-radius: 12px; 
        padding: 0 16px; 
        display: flex; 
        align-items: center; 
        height: 96px; /* Sized for 800x480 layout */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .card-static { background: var(--accent-primary); }
    .card-realtime { background: var(--accent-secondary); }
    
    .icon-box {
        width: 60px; 
        height: 60px; 
        border-radius: 12px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 28px; 
        margin-right: 20px;
        background: rgba(255,255,255,0.15);
        color: white;
    }
    
    .text-box { flex-grow: 1; }
    .card-title { font-size: 22px; font-weight: 700; color: white; margin-bottom: 4px; line-height: 1.1; }
    .card-desc { font-size: 15px; color: rgba(255,255,255,0.85); line-height: 1.3; margin: 0; }
    .chevron { font-size: 30px; font-weight: 800; color: white; opacity: 0.9; }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    c_head, c_home = st.columns([7, 1])
    with c_head:
        st.markdown("""
        <div style='display: flex; align-items: center; height: 100%; margin-top: 5px;'>
            <div style='font-size: 26px; color: var(--accent-primary); margin-right: 12px;'>🧠</div>
            <div style='font-size: 18px; font-weight: 700; color: var(--text-heading);'>Brain Tumour Classification</div>
        </div>
        """, unsafe_allow_html=True)
    with c_home:
        if st.button("⌂", key="home_mode", help="Home"):
            goto("welcome")
            st.rerun()
            
    st.markdown("<div style='border-bottom: 1px solid var(--bg-panel-raised); margin-bottom: 25px; margin-top: 10px;'></div>", unsafe_allow_html=True)

    # Title
    st.markdown("<div style='font-size:24px; font-weight:bold; text-align:center; margin-bottom: 25px; color: var(--text-heading);'>Choose a Demo Mode</div>", unsafe_allow_html=True)

    # Tile 1: Static Image
    with st.container(key="tile_static"):
        st.markdown("""
        <div class="demo-card card-static">
            <div class="icon-box">🖼️</div>
            <div class="text-box">
                <div class="card-title">Static Image</div>
                <div class="card-desc">Compare prediction against fixed ground truth.</div>
            </div>
            <div class="chevron">❯</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Invisible overlay button handling the routing
        if st.button("hidden_static", key="mode_static", use_container_width=True):
            goto("static")
            st.rerun()

    # Tile 2: Realtime Demo
    with st.container(key="tile_realtime"):
        st.markdown("""
        <div class="demo-card card-realtime">
            <div class="icon-box">〰️</div>
            <div class="text-box">
                <div class="card-title">Realtime Demo</div>
                <div class="card-desc">Continuous inference along a moving scan path.</div>
            </div>
            <div class="chevron">❯</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Invisible overlay button handling the routing
        if st.button("hidden_realtime", key="mode_realtime", use_container_width=True):
            goto("realtime")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  Screen 3 — Static Comparison
# ═════════════════════════════════════════════════════════════════════════════
def screen_static() -> None:
    if STATIC_MODEL_IS_FALLBACK:
        st.markdown(
            "<div class='bv-warn'>1D-CNN-Hu (CE, no class-balancing) checkpoint not "
            "found — using the CE + balanced checkpoint instead.</div>",
            unsafe_allow_html=True,
        )

    case_idx = st.session_state.setdefault("static_case_idx", 0)

    # ── Top bar: home | title | shuffle | latency ──────────────────────────
    c_home, c_title, c_shuffle, c_lat = st.columns([0.6, 3.4, 0.6, 1.4])
    with c_home:
        if st.button("⌂", key="home_static", help="Home"):
            goto("welcome")
            st.rerun()
    with c_shuffle:
        if st.button("🔀", key="static_shuffle", help="Shuffle image"):
            if len(STATIC_CASES) > 1:
                case_idx = random.choice(
                    [i for i in range(len(STATIC_CASES)) if i != case_idx])
            st.session_state.static_case_idx = case_idx
            st.rerun()

    result = compute_static(case_idx)

    with c_title:
        st.markdown("<div class='bv-screen-title' style='padding-top:8px'>Static Comparison</div>",
                    unsafe_allow_html=True)
    with c_lat:
        st.markdown(f"<div style='padding-top:5px'>{latency_badge(result['latency_ms'])}</div>",
                    unsafe_allow_html=True)

    # ── Main: big classification map + legend, fills most of the screen ────
    with st.container(key="static_main"):
        col_img, col_legend = st.columns([3.4, 1], gap="medium")
        with col_img:
            st.image(result["pred_map"], use_container_width=True)
        with col_legend:
            st.markdown(classification_legend(STATIC_CLASS_COLORS), unsafe_allow_html=True)

    # ── Metrics — below the fold ────────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    m = result["metrics"]
    st.markdown(
        chip("F1-noBG", f"{m['f1_no_bg'] * 100:.0f}%", "Overall tissue agreement", "good")
        + chip("TT Sensitivity",
               "n/a" if m["tt_total"] == 0 else f"{m['tt_sens'] * 100:.0f}%",
               "Tumour tissue detected", "tt")
        + chip("Agreement (labelled px)", f"{m['agree_frac'] * 100:.0f}%",
               "Accuracy on the pixels that were actually scored"),
        unsafe_allow_html=True,
    )

    st.caption(f"{m['n_labelled']:,} labelled px scored · {m['n_incorrect']:,} disagree "
               f"· {m['n_unlabelled']:,} unlabelled px not scored")

# ═════════════════════════════════════════════════════════════════════════════
#  Screen 4 — Realtime Demo
# ═════════════════════════════════════════════════════════════════════════════
def screen_realtime() -> None:
    playing  = st.session_state.setdefault("rt_playing", True)
    speed    = st.session_state.setdefault("rt_speed", "Normal")
    case_idx = st.session_state.setdefault("rt_case_idx", 0)

    # ── Top bar: home | title | shuffle | latency | live ───────────────────
    c_home, c_title, c_shuffle, c_lat, c_live = st.columns(
        [0.6, 3.0, 0.6, 1.2, 1.0]
    )
    with c_home:
        if st.button("⌂", key="home_realtime", help="Home"):
            goto("welcome")
            st.rerun()
    with c_shuffle:
        if st.button("🔀", key="rt_shuffle", help="Shuffle image"):
            if len(CASES) > 1:
                case_idx = random.choice(
                    [i for i in range(len(CASES)) if i != case_idx])
            st.session_state.rt_case_idx = case_idx
            st.rerun()

    # Image choice, play state and scan progress are independent: only the
    # case selection resolved above feeds this lookup.
    rt = compute_realtime(case_idx)
    n_steps = len(rt["frames"])
    pos_key = f"rt_pos_{case_idx}"
    pos = st.session_state.setdefault(pos_key, 0) % n_steps
    frame = rt["frames"][pos]

    with c_title:
        st.markdown("<div class='bv-screen-title' style='padding-top:8px'>Realtime Demo</div>",
                    unsafe_allow_html=True)
    with c_lat:
        st.markdown(f"<div style='padding-top:5px'>{latency_badge(frame['latency_ms'])}</div>",
                    unsafe_allow_html=True)
    with c_live:
        live_color = STATUS_LIVE_RED if playing else STATUS_OFF
        live_text  = "Live" if playing else "Paused"
        st.markdown(
            f"<div style='text-align:right;padding-top:9px;color:{TEXT_HEADING};font-size:13px;"
            f"font-weight:700'><span style='color:{live_color}'>●</span> {live_text}</div>",
            unsafe_allow_html=True)

    # ── Main: big viewport (+ minimap) + right panel (legend, Pause/Play, Speed) ──
    with st.container(key="realtime_main"):
        col_img, col_panel = st.columns([3.4, 1], gap="medium")
        with col_img:
            composed = compose_viewport_frame(frame, rt["rgb_full"], rt["shape"])
            st.image(composed, use_container_width=True)
        with col_panel:
            st.markdown(classification_legend(DEMO_CLASS_COLORS), unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            with st.container(key="rt_playpause"):
                if st.button("⏸ Pause" if playing else "▶ Play", key="rt_playpause_btn"):
                    playing = not playing
                    st.session_state.rt_playing = playing
            st.markdown(f"<div style='color:{TEXT_SECONDARY};font-size:11px;margin-top:10px'>"
                        f"Speed</div>", unsafe_allow_html=True)
            with st.container(key="toggle_speed"):
                for label in ("Slow", "Normal", "Fast"):
                    if st.button(label, key=f"speed_{label}",
                                 type="primary" if speed == label else "secondary"):
                        speed = label
            st.session_state.rt_speed = speed

    # ── Progress: plain filled bar, no handle — not a scrubber ──
    pct = pos / (n_steps - 1) if n_steps > 1 else 0.0
    st.progress(pct)
    st.caption(f"Scanning… {pct * 100:.0f}% along path.")

    if playing:
        step, tick_s = SPEED_CONFIG[speed]
        time.sleep(tick_s)
        st.session_state[pos_key] = (pos + step) % n_steps
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  Router
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.session_state.setdefault("screen", "welcome")
    screen = st.session_state.screen
    if screen == "welcome":
        screen_welcome()
    elif screen == "mode":
        screen_mode()
    elif screen == "static":
        screen_static()
    elif screen == "realtime":
        screen_realtime()
    else:
        goto("welcome")
        st.rerun()


if __name__ == "__main__":
    main()
