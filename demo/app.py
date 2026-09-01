"""
brainvision — Intraoperative HSI Brain Tumour Classification
Defence touchscreen demo.

Implements the merged conceptual UI spec: a five-screen, touch-driven flow
(Idle -> Case Select -> Live Inference -> Summary, plus a flagged Free Pan/Zoom
bonus mode). It reorganises the existing pan/zoom simulator, minimap,
live-inference badge and three-panel display — it does not rebuild them.

Run from the repo root:
    streamlit run demo/app.py

Everything a presenter tunes for a specific defence lives in the two config
blocks below: CASES (the three fixed cases) and PARETO_POINTS (the summary
scatter). Numbers marked REHEARSAL should be replaced with values measured on
the actual defence hardware.
"""

import io
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

# ── brainvision package imports (no duplicated model / preprocessing code) ─────
from brainvision.constants import (
    CLASS_COLORS, CLASS_NAMES, N_DECIMATED_BANDS, N_CLASSES,
)
from brainvision.models.fabelo_dnn   import FabeloDNN
from brainvision.models.baseline_dnn import Baseline1DDNN
from brainvision.models.hu_1dcnn     import HuEtAl1DCNN
from brainvision.models.fabelo_2dcnn import Fabelo2DCNN
from brainvision.models.lee_2dcnn    import LeeEtAl2DCNN
from brainvision.models.simple_2dcnn import Simple2DCNN

# ═════════════════════════════════════════════════════════════════════════════
#  PRESENTER CONFIG 1 — the three fixed cases (spec §0, §2b, §2c)
# ═════════════════════════════════════════════════════════════════════════════
# Each case is one screenful of the Live Inference view. `crop` is
# (row0, col0, size) in pixels, or None for the whole image. Keep 2D (patch)
# cases cropped — patch inference is slow. `caption` is the plain-language
# narrative line shown in the caption band; retune it in rehearsal so it
# matches what the chosen crop actually shows.

CASES = [
    {
        "key":        "clear_match",
        "title":      "Clear Match",
        "role":       "A clean success — prediction and labels line up.",
        "image":      "processed/first_campaign/008-02.npz",
        "crop":       None,
        "model":      "1D-NN-Fabelo",
        "checkpoint": "checkpoints/1dnnfabelo_ce_bal_fold2_vpfabelo.pt",
        "patch_size": None,
        "caption":    "The spectral model separates healthy tissue, vessel and "
                      "background cleanly — most labelled pixels agree.",
    },
    {
        "key":        "spatial_sensitivity",
        "title":      "Spatial Sensitivity Gain",
        "role":       "Spatial context catches more tumour tissue.",
        "image":      "processed/second_campaign/038-01.npz",
        "crop":       (170, 285, 165),   # centred on this image's tumour region
        "model":      "2D-CNN-Fabelo",
        "checkpoint": "checkpoints/2dcnnfabelo_ufl_bal_fold2_vpfabelo.pt",
        "patch_size": 11,
        "caption":    "The spatial model catches most tumour tissue present here — "
                      "it prioritises not missing cancer over precision.",
    },
    {
        "key":        "honest_near_miss",
        "title":      "Honest Near-Miss",
        "role":       "A case the model gets partly wrong — shown on purpose.",
        "image":      "processed/first_campaign/012-01.npz",
        "crop":       None,
        "model":      "1D-NN-Fabelo",
        "checkpoint": "checkpoints/1dnnfabelo_ce_bal_fold2_vpfabelo.pt",
        "patch_size": None,
        "caption":    "The prediction disagrees with part of the labelled region. "
                      "No model is perfect on this dataset — this is a real result.",
    },
]

# ═════════════════════════════════════════════════════════════════════════════
#  PRESENTER CONFIG 2 — the Summary Pareto scatter (spec §2d)
# ═════════════════════════════════════════════════════════════════════════════
# x = inference latency for a full image (seconds, log axis)
# y = macro F1-noBG (the primary metric)
# `tt_sens` is annotated only on the two starred viable-region points.
# LATENCY VALUES ARE REHEARSAL PLACEHOLDERS — measure on the defence hardware.

PARETO_POINTS = [
    # label,                 latency_s, f1_no_bg, params,      viable, tt_sens
    ("1D-DNN (spectral)",        0.02,    0.684,     4_936,     True,   0.644),
    ("1D-CNN",                   0.05,    0.703,    76_824,     False,  None),
    ("2D-CNN (spatial)",        22.0,     0.759,   142_052,     True,   0.669),
    ("HybridSN",                90.0,     0.751, 2_601_588,     False,  None),
    ("SpectralFormer",         120.0,     0.742,   198_197,     False,  None),
]
VIABILITY_LINE_S = 60.0   # clinical-viability latency ceiling (spec §2d)

# Rotating one-liners under the scatter — presenter taps to cycle (spec §2d).
SUMMARY_FINDINGS = [
    "A 4,936-parameter spectral model matches a model 3,400x larger on F1 — "
    "parameter count is not the bottleneck.",
    "Spatial context buys tumour sensitivity, not median F1: about +2.55 pp "
    "tumour sensitivity for roughly no change in F1-noBG.",
    "Unified Focal Loss lifts tumour sensitivity by up to 18.5 pp over "
    "cross-entropy — but the size of the gain depends on the architecture.",
]

# ── Plain-language metric translation (spec §5) — one consequence sentence each
METRIC_PLAIN = {
    "f1_no_bg": ("Overall tissue agreement",
                 "How well the model tells tissue types apart, excluding background."),
    "tt_sens":  ("Tumour detection",
                 "Of the tumour that is really there, how much the model catches."),
    "latency":  ("Inference time",
                 "How long the surgeon would wait for this to update."),
    "params":   ("Model size",
                 "How small and fast the model is to run on this hardware."),
    "agree":    ("Agreement on labelled pixels",
                 "Where ground truth exists, how often the prediction matches it."),
}

# ── Model registry (reused from the simulator) ───────────────────────────────
MODEL_REGISTRY = {
    "1D-NN-Fabelo":   (FabeloDNN,     "pixel"),
    "1D-NN-Baseline": (Baseline1DDNN, "pixel"),
    "1D-CNN":         (HuEtAl1DCNN,   "pixel"),
    "2D-CNN-Fabelo":  (Fabelo2DCNN,   "patch"),
    "2D-CNN-Simple":  (Simple2DCNN,   "patch"),
    "2D-CNN-LeeEtAl": (LeeEtAl2DCNN,  "patch"),
}

PSEUDO_RGB_BANDS = (88, 42, 7)          # ~709 R, ~539 G, ~479 B
NPZ_CUBE_KEY     = "processed"
NPZ_GT_KEY       = "labels"

# ── Agreement overlay palette (spec §4, §8) ──────────────────────────────────
# A separate three-state scheme, deliberately NOT the tissue-class colours, so
# it can never be read as a tissue class. Always shown with the ✓ / ✕ / · glyph
# and word — never colour alone.
AGREE_COLORS = {
    "correct":    ("#3C6E47", "✓", "Correct"),      # muted green — calm
    "incorrect":  ("#E8A33D", "✗", "Incorrect"),    # amber — warning
    "unlabelled": ("#3A3A3A", "·", "Unlabelled"),   # grey hatch — neutral
}

SPARSE_GT_NOTE = (
    "Ground truth is sparse: unlabelled pixels are NOT counted as errors. "
    "They were never annotated."
)


# ═════════════════════════════════════════════════════════════════════════════
#  Page setup + kiosk styling (spec §0: 1024x600 landscape; §8: type / contrast)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="brainvision — Intraoperative HSI Classification",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/* Kiosk: hide Streamlit chrome, tighten the frame to a fixed panel feel */
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]
    { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stAppViewContainer"] { background: #0C0E14; }
.block-container {
    max-width: 1024px;
    padding: 12px 18px 10px 18px;
}

/* ── Two-tier type scale (spec §8) ───────────────────────────────────────── */
html, body, [class*="css"] { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; }

.bv-screen-title { font-size: 26px; font-weight: 800; color: #F4F6FB; margin: 0; }
.bv-screen-sub   { font-size: 15px; color: #9AA3B2; margin: 2px 0 0 0; }

/* Hero number always on a solid contrasting chip — never dark-on-dark (§8) */
.bv-chip {
    display: inline-block;
    background: #1B2130;
    border-radius: 10px;
    padding: 10px 16px;
    margin: 4px 6px 4px 0;
    border-left: 5px solid #4C5A72;
}
.bv-chip .k { font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
              color: #9AA3B2; margin-bottom: 3px; }
.bv-chip .v { font-size: 40px; font-weight: 800; color: #FFFFFF; line-height: 1.05; }
.bv-chip .p { font-size: 13px; color: #B9C1CE; margin-top: 4px; max-width: 320px; }
.bv-chip.tt   { border-left-color: #D85A30; }
.bv-chip.good { border-left-color: #3C6E47; }

/* Persistent sparse-GT caption, anchored under the reference column (§2c, §4) */
.bv-sparsenote {
    background: #17202E;
    border: 1px solid #2A3547;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    color: #C7CEDA;
    line-height: 1.35;
}

/* Plain-language caption band (§2c) */
.bv-captionband {
    background: #14351F;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 17px;
    color: #EAF3EC;
    line-height: 1.4;
}

/* Static latency badge — no animation, no colour change on a slow frame (§6, §9) */
.bv-badge {
    display: inline-flex; align-items: center; gap: 9px;
    background: #1B2130; border-radius: 20px; padding: 7px 15px;
    font-size: 14px; color: #F4F6FB; font-variant-numeric: tabular-nums;
}
.bv-badge .dot { width: 9px; height: 9px; border-radius: 50%; background: #5DCAA5; }

.bv-unscripted {
    background: #3A2A12; border: 1px solid #7A5A22; color: #F0C987;
    border-radius: 8px; padding: 8px 14px; font-size: 14px; font-weight: 600;
}

.bv-panel-label { font-size: 14px; letter-spacing: .06em; text-transform: uppercase;
                  color: #9AA3B2; margin: 0 0 4px 0; }

/* Touch targets — every button large (spec §7) */
.stButton > button {
    width: 100%;
    min-height: 58px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 12px;
    border: 1px solid #2A3547;
    background: #1B2130;
    color: #F4F6FB;
}
.stButton > button:hover { border-color: #4C5A72; background: #232B3C; }
.stButton > button:focus { box-shadow: none; }

/* Legend chips */
.bv-legend { display: inline-flex; align-items: center; gap: 6px; margin-right: 16px;
             font-size: 14px; color: #E6EAF1; }
.bv-legend .sw { width: 14px; height: 14px; border-radius: 3px; display: inline-block;
                 border: 1px solid #3A4557; }
</style>
""",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers reused from the simulator
# ═════════════════════════════════════════════════════════════════════════════
def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


CLASS_RGB = {k: _hex_to_rgb(v) for k, v in CLASS_COLORS.items()}
CLASS_RGB[0] = (24, 26, 34)   # unlabelled -> near-background


def load_npz(source) -> tuple:
    data = np.load(source, allow_pickle=False)
    cube = data[NPZ_CUBE_KEY]
    gt   = data[NPZ_GT_KEY] if NPZ_GT_KEY in data else None
    return cube, gt


def normalise(cube: np.ndarray) -> np.ndarray:
    H, W, B = cube.shape
    flat  = cube.reshape(-1, B).astype(np.float32)
    mn    = flat.min(1, keepdims=True)
    mx    = flat.max(1, keepdims=True)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    return ((flat - mn) / denom).reshape(H, W, B)


def make_pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    r, g, b = (cube[:, :, i] for i in PSEUDO_RGB_BANDS)
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def make_label_map(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl, color in CLASS_RGB.items():
        out[labels == lbl] = color
    return out


@st.cache_resource(show_spinner=False)
def load_model(model_key: str, ckpt_path: str):
    model_class, _ = MODEL_REGISTRY[model_key]
    model = model_class(input_channels=N_DECIMATED_BANDS, n_classes=N_CLASSES)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def infer_pixel(model, crop: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    H, W, B = crop.shape
    flat  = crop.reshape(-1, B).astype(np.float32)
    preds = np.empty(flat.shape[0], dtype=np.int64)
    with torch.no_grad():
        for s in range(0, flat.shape[0], batch_size):
            e = min(s + batch_size, flat.shape[0])
            preds[s:e] = model(torch.from_numpy(flat[s:e])).argmax(1).numpy() + 1
    return preds.reshape(H, W)


def infer_patch(model, crop: np.ndarray, patch_size: int,
                batch_size: int = 512) -> np.ndarray:
    H, W, B = crop.shape
    P, pad = patch_size, patch_size // 2
    padded = np.pad(crop.transpose(2, 0, 1),
                    ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    preds, idx, coords = np.empty(H * W, dtype=np.int64), 0, []
    with torch.no_grad():
        for r in range(H):
            for c in range(W):
                coords.append((r, c))
                if len(coords) == batch_size or (r == H - 1 and c == W - 1):
                    patches = np.stack([padded[:, rr:rr + P, cc:cc + P]
                                        for rr, cc in coords]).astype(np.float32)
                    out = model(torch.from_numpy(patches)).argmax(1).numpy() + 1
                    preds[idx:idx + len(coords)] = out
                    idx += len(coords)
                    coords = []
    return preds.reshape(H, W)


def make_minimap(pseudo_rgb: np.ndarray, r0: int, c0: int,
                 vp_h: int, vp_w: int) -> io.BytesIO:
    H, W = pseudo_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(4, 4 * H / W))
    ax.imshow(pseudo_rgb)
    ax.add_patch(mpatches.Rectangle((c0, r0), vp_w, vp_h,
                 linewidth=2, edgecolor="#D85A30", facecolor="none"))
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="#0C0E14")
    plt.close(fig)
    buf.seek(0)
    return buf


# ═════════════════════════════════════════════════════════════════════════════
#  Agreement overlay + metrics (spec §4, §5)
# ═════════════════════════════════════════════════════════════════════════════
def make_agreement_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Recolour the prediction to three states: correct / incorrect / unlabelled.

    Unlabelled pixels get a hatched grey texture so 'no ground truth here' never
    reads as either an error or as simply absent (spec §4).
    """
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    labelled = gt > 0

    out[labelled & (pred == gt)] = _hex_to_rgb(AGREE_COLORS["correct"][0])
    out[labelled & (pred != gt)] = _hex_to_rgb(AGREE_COLORS["incorrect"][0])

    base = np.array(_hex_to_rgb(AGREE_COLORS["unlabelled"][0]), dtype=np.uint8)
    out[~labelled] = base
    rr, cc = np.indices(pred.shape)
    hatch = (~labelled) & (((rr + cc) % 8) < 2)          # diagonal stripes
    out[hatch] = np.clip(base.astype(int) + 26, 0, 255).astype(np.uint8)
    return out


def agreement_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Confusion-free summary over LABELLED pixels only (spec: sparse GT)."""
    labelled = gt > 0
    n_lab = int(labelled.sum())
    n_correct = int((labelled & (pred == gt)).sum())

    tt = gt == 2                                          # 2 = Tumour Tissue
    tt_total = int(tt.sum())
    tt_hit = int((tt & (pred == 2)).sum())

    return {
        "n_labelled":   n_lab,
        "n_unlabelled": int((~labelled).sum()),
        "n_correct":    n_correct,
        "n_incorrect":  n_lab - n_correct,
        "agree_frac":   (n_correct / n_lab) if n_lab else float("nan"),
        "tt_total":     tt_total,
        "tt_sens":      (tt_hit / tt_total) if tt_total else float("nan"),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Per-case compute (cached in session_state so navigation stays instant)
# ═════════════════════════════════════════════════════════════════════════════
def compute_case(case: dict) -> dict | None:
    store = st.session_state.setdefault("case_results", {})
    if case["key"] in store:
        return store[case["key"]]

    img_path = Path(case["image"])
    ckpt_path = Path(case["checkpoint"])
    if not img_path.exists():
        st.error(f"Case image not found: `{img_path}`")
        return None
    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: `{ckpt_path}`")
        return None

    with st.spinner(f"Running {case['title']}…"):
        cube, gt = load_npz(img_path)
        cube = normalise(cube)
        if gt is None:
            st.error(f"Case image `{img_path.name}` has no `labels` — needed for "
                     f"the agreement view.")
            return None

        if case["crop"]:
            r0, c0, sz = case["crop"]
            H, W = cube.shape[:2]
            r0, c0 = min(r0, H - sz), min(c0, W - sz)
            r0, c0 = max(0, r0), max(0, c0)
            cube = cube[r0:r0 + sz, c0:c0 + sz]
            gt   = gt[r0:r0 + sz, c0:c0 + sz]

        _, model_type = MODEL_REGISTRY[case["model"]]
        model = load_model(case["model"], str(ckpt_path))

        t0 = time.perf_counter()
        if model_type == "pixel":
            pred = infer_pixel(model, cube)
        else:
            pred = infer_patch(model, cube, case["patch_size"] or 11)
        latency_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "rgb":        make_pseudo_rgb(cube),
        "gt_map":     make_label_map(gt),
        "pred_map":   make_label_map(pred),
        "agree_map":  make_agreement_map(pred, gt),
        "latency_ms": latency_ms,
        "shape":      cube.shape,
        "model_type": model_type,
        "stats":      agreement_stats(pred, gt),
    }
    store[case["key"]] = result
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  Small UI atoms
# ═════════════════════════════════════════════════════════════════════════════
def goto(screen: str) -> None:
    st.session_state.screen = screen


def home_row(subtitle: str = "") -> None:
    """Fixed home control, same corner on every screen (spec §1, §7)."""
    left, right = st.columns([1, 9])
    with left:
        if st.button("⌂", key=f"home_{st.session_state.screen}",
                     help="Back to Case Select"):
            goto("cases")
            st.rerun()
    with right:
        if subtitle:
            st.markdown(f"<div class='bv-screen-sub'>{subtitle}</div>",
                        unsafe_allow_html=True)


def chip(kind: str, value: str, css: str = "") -> str:
    label, plain = METRIC_PLAIN[kind]
    return (f"<div class='bv-chip {css}'><div class='k'>{label}</div>"
            f"<div class='v'>{value}</div><div class='p'>{plain}</div></div>")


def latency_badge(ms: float) -> str:
    # Static, unremarkable number — reads as a measurement, not an error state
    # (spec §6, §9). No colour change or animation on a slow frame.
    if ms < 1000:
        rate = f"&nbsp;&nbsp;/&nbsp;&nbsp;{1000.0 / ms:.0f} fps" if ms > 0 else ""
        return (f"<div class='bv-badge'><span class='dot'></span>"
                f"{ms:.0f} ms / frame{rate}</div>")
    return (f"<div class='bv-badge'><span class='dot'></span>"
            f"{ms / 1000:.1f} s / frame</div>")


def tissue_legend() -> str:
    return "".join(
        f"<span class='bv-legend'><span class='sw' style='background:{CLASS_COLORS[l]}'>"
        f"</span>{CLASS_NAMES[l]}</span>"
        for l in sorted(CLASS_COLORS)
    )


def agreement_legend() -> str:
    return "".join(
        f"<span class='bv-legend'><span class='sw' style='background:{hexc}'></span>"
        f"{glyph} {word}</span>"
        for hexc, glyph, word in AGREE_COLORS.values()
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Screen 2a — Title / Idle (spec §2a)
# ═════════════════════════════════════════════════════════════════════════════
def screen_idle() -> None:
    st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center'>"
        "<div class='bv-screen-title' style='font-size:34px'>"
        "Real-Time Intraoperative Tissue Classification</div>"
        "<div class='bv-screen-sub' style='font-size:17px;margin-top:8px'>"
        "Hyperspectral imaging + deep learning</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 3, 2])
    with mid:
        if st.button("Begin Demo", key="begin"):
            goto("cases")
            st.rerun()
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;color:#8A93A3;font-size:14px'>"
        "●&nbsp; Raspberry Pi 5 — System Ready</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Screen 2b — Case Select (spec §2b) — no ordinal step counter
# ═════════════════════════════════════════════════════════════════════════════
def screen_cases() -> None:
    home_row()
    st.markdown("<div class='bv-screen-title'>Choose a Case</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    for col, (i, case) in zip(cols, enumerate(CASES)):
        with col:
            thumb = _case_thumb(case["key"], case["image"])
            if thumb is not None:
                st.image(thumb, use_container_width=True)
            st.markdown(f"<div class='bv-panel-label'>{case['role']}</div>",
                        unsafe_allow_html=True)
            if st.button(case["title"], key=f"case_{case['key']}"):
                st.session_state.case_idx = i
                goto("live")
                st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    _, right = st.columns([2, 1])
    with right:
        if st.button("Live Mode (bonus)", key="cases_livemode"):
            st.session_state.freepan_return = "cases"
            goto("freepan")
            st.rerun()


@st.cache_data(show_spinner=False)
def _case_thumb(case_key: str, image_path: str) -> np.ndarray | None:
    path = Path(image_path)
    if not path.exists():
        return None
    cube, _ = load_npz(path)
    return make_pseudo_rgb(normalise(cube))


# ═════════════════════════════════════════════════════════════════════════════
#  Screen 2c — Live Inference (the hero screen)
# ═════════════════════════════════════════════════════════════════════════════
def screen_live() -> None:
    case = CASES[st.session_state.case_idx]
    res  = compute_case(case)
    home_row()
    if res is None:
        return

    # Header: case title + latency badge (badge is secondary — spec §3, §6)
    h_left, h_right = st.columns([3, 2])
    with h_left:
        st.markdown(f"<div class='bv-screen-title'>{case['title']}</div>",
                    unsafe_allow_html=True)
    with h_right:
        st.markdown(f"<div style='text-align:right'>{latency_badge(res['latency_ms'])}</div>",
                    unsafe_allow_html=True)

    primary = st.session_state.setdefault("primary_view", "prediction")
    show_agree = st.session_state.setdefault("show_agreement", False)

    # ── Primary (large) + reference thumbnails (spec §2c asymmetric layout) ──
    big, side = st.columns([7, 3], gap="medium")

    with big:
        if primary == "prediction":
            tog_l, tog_r = st.columns([3, 2])
            with tog_l:
                st.markdown("<div class='bv-panel-label'>Model prediction</div>",
                            unsafe_allow_html=True)
            with tog_r:
                if st.button(("Show: Agreement" if not show_agree else "Show: Prediction"),
                             key="agree_toggle"):
                    st.session_state.show_agreement = not show_agree
                    st.rerun()
            st.image(res["agree_map"] if show_agree else res["pred_map"],
                     use_container_width=True)
            st.markdown(
                (agreement_legend() if show_agree else tissue_legend()),
                unsafe_allow_html=True,
            )
        elif primary == "gt":
            st.markdown("<div class='bv-panel-label'>Ground truth (labelled pixels)</div>",
                        unsafe_allow_html=True)
            st.image(res["gt_map"], use_container_width=True)
            st.markdown(tissue_legend(), unsafe_allow_html=True)
        else:  # rgb
            st.markdown("<div class='bv-panel-label'>Pseudo-RGB</div>",
                        unsafe_allow_html=True)
            st.image(res["rgb"], use_container_width=True)

    with side:
        # tap a thumbnail to promote it into the primary slot (spec §2c, §7)
        st.markdown("<div class='bv-panel-label'>Ground truth</div>",
                    unsafe_allow_html=True)
        st.image(res["gt_map"], use_container_width=True)
        if st.button("View ground truth", key="promote_gt"):
            st.session_state.primary_view = "gt" if primary != "gt" else "prediction"
            st.rerun()

        st.markdown("<div class='bv-panel-label'>Pseudo-RGB</div>",
                    unsafe_allow_html=True)
        st.image(res["rgb"], use_container_width=True)
        if st.button("View pseudo-RGB", key="promote_rgb"):
            st.session_state.primary_view = "rgb" if primary != "rgb" else "prediction"
            st.rerun()

        # Sparse-GT note anchored under the reference column (spec §2c, §4)
        st.markdown(f"<div class='bv-sparsenote'>{SPARSE_GT_NOTE}</div>",
                    unsafe_allow_html=True)

    # ── Plain-language caption band (spec §2c) ──────────────────────────────
    st.markdown(f"<div class='bv-captionband'>{case['caption']}</div>",
                unsafe_allow_html=True)

    # ── Metric chips — every number paired with one consequence line (§5) ───
    s = res["stats"]
    st.markdown(
        chip("agree", f"{s['agree_frac'] * 100:.0f}%", css="good")
        + chip("tt_sens",
               ("n/a" if s["tt_total"] == 0 else f"{s['tt_sens'] * 100:.0f}%"),
               css="tt")
        + chip("latency", f"{res['latency_ms']:.0f} ms"),
        unsafe_allow_html=True,
    )
    st.caption(
        f"{s['n_labelled']:,} labelled px scored · {s['n_incorrect']:,} disagree "
        f"· {s['n_unlabelled']:,} unlabelled px not scored · "
        f"{res['shape'][0]}×{res['shape'][1]}×{N_DECIMATED_BANDS}"
    )

    # ── Footer nav (spec §2c) — Prev / Compare All / Next, no counter ───────
    f1, f2, f3 = st.columns(3, gap="small")
    with f1:
        if st.button("◀  Prev case", key="prev_case"):
            _step_case(-1)
            st.rerun()
    with f2:
        if st.button("Compare All Cases", key="to_summary"):
            goto("summary")
            st.rerun()
    with f3:
        if st.button("Next case  ▶", key="next_case"):
            _step_case(+1)
            st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    _, r = st.columns([3, 1])
    with r:
        if st.button("Live Mode (bonus)", key="live_to_freepan"):
            st.session_state.freepan_return = "live"
            goto("freepan")
            st.rerun()


def _step_case(delta: int) -> None:
    st.session_state.case_idx = (st.session_state.case_idx + delta) % len(CASES)
    st.session_state.primary_view = "prediction"
    st.session_state.show_agreement = False


# ═════════════════════════════════════════════════════════════════════════════
#  Screen 2d — Summary / Pareto scatter (spec §2d)
# ═════════════════════════════════════════════════════════════════════════════
def screen_summary() -> None:
    home_row()
    st.markdown("<div class='bv-screen-title'>Accuracy vs. Speed Trade-off</div>",
                unsafe_allow_html=True)

    fig = _pareto_figure()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    idx = st.session_state.setdefault("finding_idx", 0)
    st.markdown(f"<div class='bv-captionband'>{SUMMARY_FINDINGS[idx]}</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("◀  Back to Cases", key="summary_back"):
            goto("cases")
            st.rerun()
    with c2:
        if st.button("Next finding  ▶", key="cycle_finding"):
            st.session_state.finding_idx = (idx + 1) % len(SUMMARY_FINDINGS)
            st.rerun()


def _pareto_figure():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    fig.patch.set_facecolor("#0C0E14")
    ax.set_facecolor("#0C0E14")

    # viable region: latency <= viability line
    ax.axvspan(1e-3, VIABILITY_LINE_S, color="#1C3A2A", alpha=0.55, zorder=0)
    ax.axvline(VIABILITY_LINE_S, color="#5DCAA5", lw=1.2, ls="--")
    ax.text(VIABILITY_LINE_S, 0.5, "  60 s clinical-viability line",
            rotation=90, va="bottom", ha="left", color="#5DCAA5", fontsize=9)

    for label, lat, f1, params, viable, tt in PARETO_POINTS:
        if viable:
            ax.scatter(lat, f1, s=260, marker="*", color="#F0C987",
                       edgecolor="#FFFFFF", linewidth=0.8, zorder=5)
            note = f"{label}\n{params:,} params"
            if tt is not None:
                note += f"\nTT sens {tt * 100:.0f}%"
            ax.annotate(note, (lat, f1), textcoords="offset points", xytext=(10, -6),
                        color="#F4F6FB", fontsize=8.5, va="top")
        else:
            ax.scatter(lat, f1, s=90, color="#7F8BA3", zorder=4)
            ax.annotate(f"{label}", (lat, f1), textcoords="offset points",
                        xytext=(8, 4), color="#9AA3B2", fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlim(0.01, 200)
    ax.set_xlabel("Inference latency per image  (s, log scale)  →",
                  color="#B9C1CE", fontsize=10)
    ax.set_ylabel("Macro F1-noBG  →", color="#B9C1CE", fontsize=10)
    ax.set_ylim(0.4, 0.85)
    ax.tick_params(colors="#8A93A3", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2A3547")
    ax.grid(alpha=0.12, color="#3A4557")
    ax.set_title("Star = viable-region pick (spectral + spatial)",
                 color="#9AA3B2", fontsize=9)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Screen 2e — Free Pan/Zoom (bonus, flagged unscripted) — spec §2e
# ═════════════════════════════════════════════════════════════════════════════
def screen_freepan() -> None:
    home_row()
    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.markdown("<div class='bv-screen-title'>Live Mode</div>",
                    unsafe_allow_html=True)
        st.markdown("<span class='bv-unscripted'>UNSCRIPTED · free exploration</span>",
                    unsafe_allow_html=True)

    # Uses the first pixel-model case as the substrate for the pan/zoom sim.
    case = next((c for c in CASES if MODEL_REGISTRY[c["model"]][1] == "pixel"), CASES[0])
    img_path = Path(case["image"])
    ckpt_path = Path(case["checkpoint"])
    if not img_path.exists() or not ckpt_path.exists():
        st.error("Live Mode assets missing — check the first pixel-model case.")
        _freepan_exit()
        return

    fp = st.session_state.setdefault("freepan", {})
    if fp.get("img") != str(img_path):
        cube, gt = load_npz(img_path)
        cube = normalise(cube)
        fp.update(img=str(img_path), cube=cube, gt=gt, rgb=make_pseudo_rgb(cube))
    cube, gt, rgb = fp["cube"], fp["gt"], fp["rgb"]
    H, W = cube.shape[:2]

    # Coarse drag targets — not thin precision sliders (spec §2e, §7)
    st.session_state.setdefault("fp_zoom", 160)
    zoom = st.select_slider("Zoom — viewport size (px)",
                            options=[80, 120, 160, 220, 300, 400],
                            key="fp_zoom")
    zoom = min(zoom, H, W)
    max_r, max_c = max(1, H - zoom), max(1, W - zoom)
    # seed pan positions once, then keep them inside the current zoom's range
    st.session_state.setdefault("fp_r0", max_r // 2)
    st.session_state.setdefault("fp_c0", max_c // 2)
    st.session_state["fp_r0"] = min(st.session_state["fp_r0"], max_r)
    st.session_state["fp_c0"] = min(st.session_state["fp_c0"], max_c)
    r0 = st.slider("Pan up / down", 0, max_r, step=max(1, zoom // 8), key="fp_r0")
    c0 = st.slider("Pan left / right", 0, max_c, step=max(1, zoom // 8), key="fp_c0")

    crop = cube[r0:r0 + zoom, c0:c0 + zoom]
    model = load_model(case["model"], str(ckpt_path))
    with st.spinner("Inferring…"):
        t0 = time.perf_counter()
        pred = infer_pixel(model, crop)
        ms = (time.perf_counter() - t0) * 1000.0

    st.markdown(f"<div style='text-align:right'>{latency_badge(ms)}</div>",
                unsafe_allow_html=True)

    big, side = st.columns([7, 3], gap="medium")
    with big:
        st.markdown("<div class='bv-panel-label'>Prediction</div>",
                    unsafe_allow_html=True)
        st.image(make_label_map(pred), use_container_width=True)
        st.markdown(tissue_legend(), unsafe_allow_html=True)
    with side:
        st.markdown("<div class='bv-panel-label'>Minimap</div>",
                    unsafe_allow_html=True)
        st.image(make_minimap(rgb, r0, c0, zoom, zoom), use_container_width=True)
        st.markdown("<div class='bv-panel-label'>Pseudo-RGB</div>",
                    unsafe_allow_html=True)
        st.image(rgb[r0:r0 + zoom, c0:c0 + zoom], use_container_width=True)
        if gt is not None:
            st.markdown("<div class='bv-panel-label'>Ground truth</div>",
                        unsafe_allow_html=True)
            st.image(make_label_map(gt[r0:r0 + zoom, c0:c0 + zoom]),
                     use_container_width=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    _, r = st.columns([3, 1])
    with r:
        if st.button("Exit Live Mode", key="fp_exit"):
            _freepan_exit()
            st.rerun()


def _freepan_exit() -> None:
    goto(st.session_state.get("freepan_return", "live"))


# ═════════════════════════════════════════════════════════════════════════════
#  Router
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.session_state.setdefault("screen", "idle")
    st.session_state.setdefault("case_idx", 0)

    screen = st.session_state.screen
    if screen == "idle":
        screen_idle()
    elif screen == "cases":
        screen_cases()
    elif screen == "live":
        screen_live()
    elif screen == "summary":
        screen_summary()
    elif screen == "freepan":
        screen_freepan()
    else:
        goto("idle")
        st.rerun()


if __name__ == "__main__":
    main()
