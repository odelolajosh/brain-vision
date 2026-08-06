"""
brainvision — Intraoperative HSI Brain Tumour Classification Demo
Real-time pan/zoom inference simulator.

Run from the repo root:
    streamlit run demo/app.py
"""

import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
import torch

# ── brainvision package imports ───────────────────────────────────────────────
from brainvision.constants import CLASS_COLORS, CLASS_NAMES, N_DECIMATED_BANDS, N_CLASSES
from brainvision.models.fabelo_dnn   import FabeloDNN
from brainvision.models.baseline_dnn import Baseline1DDNN
from brainvision.models.hu_1dcnn     import HuEtAl1DCNN
from brainvision.models.fabelo_2dcnn import Fabelo2DCNN
from brainvision.models.lee_2dcnn    import LeeEtAl2DCNN
from brainvision.models.simple_2dcnn import Simple2DCNN

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="brainvision — Live HSI Inference",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark surgical aesthetic */
[data-testid="stAppViewContainer"] { background: #0F1117; }
[data-testid="stSidebar"]          { background: #161923; }
[data-testid="stHeader"]           { background: transparent; }

/* Metric tiles */
.metric-tile {
    border-radius: 8px;
    padding: 14px 16px;
    background: #1E2130;
    margin-bottom: 8px;
}
.metric-tile .label {
    font-size: 10px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #888780;
    margin-bottom: 4px;
}
.metric-tile .value {
    font-size: 26px;
    font-weight: 700;
    color: #FAFAFA;
    line-height: 1;
}
.metric-tile .sub {
    font-size: 12px;
    color: #888780;
    margin-top: 3px;
}
.metric-tile.tt-highlight {
    border: 1.5px solid #D85A30;
    box-shadow: 0 0 12px rgba(216,90,48,0.25);
}

/* Inference badge */
.infer-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1E2130;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: #FAFAFA;
}
.infer-badge .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #5DCAA5;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.3; }
}

/* Section label */
.section-label {
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #888780;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PSEUDO_RGB_BANDS    = (88, 42, 7)        # ~709nm R, ~539nm G, ~479nm B
DEFAULT_DEMO_IMAGE  = Path("processed/first_campaign/008-02.npz")
NPZ_CUBE_KEY        = "processed"
NPZ_GT_KEY          = "labels"

MIN_VIEWPORT = 50
MAX_VIEWPORT = 400

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "1D-NN-Fabelo  (4,936 params)":   (FabeloDNN,     "pixel"),
    "1D-NN-Baseline (17M params)":    (Baseline1DDNN, "pixel"),
    "1D-CNN  (76,824 params)":        (HuEtAl1DCNN,   "pixel"),
    "2D-CNN-Fabelo (142K params)":    (Fabelo2DCNN,   "patch"),
    "2D-CNN-Simple (19K params)":     (Simple2DCNN,   "patch"),
    "2D-CNN-LeeEtAl (296K params)":   (LeeEtAl2DCNN,  "patch"),
}

# ── Colour helpers ────────────────────────────────────────────────────────────
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

CLASS_RGB = {k: _hex_to_rgb(v) for k, v in CLASS_COLORS.items()}
CLASS_RGB[0] = (30, 30, 30)   # unlabelled → dark

# ── Data helpers ──────────────────────────────────────────────────────────────
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
    r, g, b = [cube[:, :, i] for i in PSEUDO_RGB_BANDS]
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)

def make_label_map(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl, color in CLASS_RGB.items():
        out[labels == lbl] = color
    return out

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_class, ckpt_path: str):
    model = model_class(input_channels=N_DECIMATED_BANDS, n_classes=N_CLASSES)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def infer_pixel(model, crop: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    H, W, B = crop.shape
    flat    = crop.reshape(-1, B).astype(np.float32)
    preds   = np.empty(flat.shape[0], dtype=np.int64)
    with torch.no_grad():
        for s in range(0, flat.shape[0], batch_size):
            e = min(s + batch_size, flat.shape[0])
            preds[s:e] = model(torch.from_numpy(flat[s:e])).argmax(1).numpy() + 1
    return preds.reshape(H, W)

def infer_patch(model, crop: np.ndarray, patch_size: int,
                batch_size: int = 512) -> np.ndarray:
    H, W, B = crop.shape
    P, pad  = patch_size, patch_size // 2
    padded  = np.pad(crop.transpose(2,0,1),
                     ((0,0),(pad,pad),(pad,pad)), mode="reflect")
    preds, idx, coords = np.empty(H*W, dtype=np.int64), 0, []
    with torch.no_grad():
        for r in range(H):
            for c in range(W):
                coords.append((r, c))
                if len(coords) == batch_size or (r==H-1 and c==W-1):
                    patches = np.stack([padded[:, rr:rr+P, cc:cc+P]
                                        for rr, cc in coords]).astype(np.float32)
                    out = model(torch.from_numpy(patches)).argmax(1).numpy() + 1
                    preds[idx:idx+len(coords)] = out
                    idx += len(coords)
                    coords = []
    return preds.reshape(H, W)

# ── Minimap with viewport rectangle ──────────────────────────────────────────
def make_minimap(pseudo_rgb: np.ndarray, r0: int, c0: int,
                 vp_h: int, vp_w: int) -> np.ndarray:
    """
    Draw the full pseudo-RGB image with a red rectangle showing
    the current viewport position.
    """
    H, W = pseudo_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(4, 4 * H/W))
    ax.imshow(pseudo_rgb)
    rect = mpatches.Rectangle(
        (c0, r0), vp_w, vp_h,
        linewidth=2, edgecolor="#D85A30", facecolor="none"
    )
    ax.add_patch(rect)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                facecolor="#0F1117")
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Metric tile ───────────────────────────────────────────────────────────────
def metric_tile(name: str, pct: float, n: int, color: str,
                highlight: bool = False):
    cls = "metric-tile tt-highlight" if highlight else "metric-tile"
    st.markdown(
        f"<div class='{cls}' style='border-left:4px solid {color}'>"
        f"<div class='label'>{name}</div>"
        f"<div class='value'>{pct:.1f}%</div>"
        f"<div class='sub'>{n:,} px</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Class legend ──────────────────────────────────────────────────────────────
def class_legend():
    html = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px">'
        f'<span style="width:10px;height:10px;border-radius:2px;'
        f'background:{CLASS_COLORS[lbl]};display:inline-block;margin-right:5px">'
        f'</span><span style="font-size:12px;color:#FAFAFA">{name}</span></span>'
        for lbl, name in CLASS_NAMES.items() if lbl in CLASS_COLORS
    )
    st.markdown(html, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
        <h1 style='margin-bottom:0;color:#FAFAFA'>🧠 brainvision</h1>
        <p style='color:#888780;margin-top:2px;font-size:14px'>
        Real-time intraoperative HSI brain tumour classification ·
        UNILAG Systems Engineering · Joshua Adeniji (190407028)
        </p>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Sidebar — configuration (set once) ────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        image_source = st.radio(
            "Image source",
            ["Demo — 008-02 (C1)", "Upload .npz"],
            label_visibility="collapsed",
        )

        st.markdown("**Model**")
        model_name = st.selectbox("Architecture", list(MODEL_REGISTRY),
                                  label_visibility="collapsed")
        model_class, model_type = MODEL_REGISTRY[model_name]

        ckpt_path = st.text_input(
            "Checkpoint (.pt)",
            value="checkpoints/1dnnfabelo_ce_bal_fold2_vpfabelo.pt",
        )

        with st.expander("Advanced"):
            batch_size = st.select_slider(
                "Batch size", options=[256,512,1024,2048,4096,8192], value=4096
            )
            patch_size = st.select_slider(
                "Patch size (2D only)", options=[3,5,7,9], value=5
            )
            is_raw = st.toggle(
                "Raw cube (826 bands)", value=False,
                help="Enable if uploading an unprocessed cube."
            )

        st.divider()
        st.caption("HELICoiD · Fabelo et al. 2023")

    # ── Load and preprocess cube (cached in session_state) ────────────────────
    if "cube" not in st.session_state or \
       st.session_state.get("loaded_source") != (image_source, ckpt_path):

        if image_source == "Demo — 008-02 (C1)":
            if not DEFAULT_DEMO_IMAGE.exists():
                st.error(f"Demo image not found at `{DEFAULT_DEMO_IMAGE}`.")
                return
            with st.spinner("Loading demo image…"):
                cube, gt = load_npz(DEFAULT_DEMO_IMAGE)
            source_label = "008-02.npz · Patient 008-02, Campaign 1"
        else:
            uploaded = st.file_uploader(
                "Upload .npz (with 'processed' and 'labels' keys)",
                type=["npz", "npy"],
            )
            if uploaded is None:
                st.info("Upload an HSI cube to begin.")
                return
            with st.spinner("Loading…"):
                cube, gt = load_npz(uploaded) if uploaded.name.endswith(".npz") \
                           else (np.load(uploaded), None)
            source_label = uploaded.name

        with st.spinner("Preprocessing cube…"):
            if is_raw:
                cube = normalise(cube)  # raw pipeline not needed — .npz already processed
            else:
                cube = normalise(cube)

        st.session_state["cube"]          = cube
        st.session_state["gt"]            = gt
        st.session_state["pseudo_rgb"]    = make_pseudo_rgb(cube)
        st.session_state["source_label"]  = source_label
        st.session_state["loaded_source"] = (image_source, ckpt_path)

    cube       = st.session_state["cube"]
    gt         = st.session_state["gt"]
    pseudo_rgb = st.session_state["pseudo_rgb"]
    H, W, _    = cube.shape

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        st.error(f"Checkpoint not found: `{ckpt_path}`")
        return

    try:
        model = load_model(model_class, str(ckpt))
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return

    # ── Pan / zoom controls ───────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Viewport controls</div>",
                unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        viewport_size = st.slider(
            "🔍 Zoom (viewport size px)",
            min_value=MIN_VIEWPORT,
            max_value=min(MAX_VIEWPORT, H, W),
            value=min(150, H, W),
            step=10,
        )
    with ctrl2:
        max_row = max(0, H - viewport_size)
        row_start = st.slider(
            "↕ Pan row",
            min_value=0, max_value=max(1, max_row),
            value=max_row // 2, step=5,
        )
    with ctrl3:
        max_col = max(0, W - viewport_size)
        col_start = st.slider(
            "↔ Pan column",
            min_value=0, max_value=max(1, max_col),
            value=max_col // 2, step=5,
        )

    # Clamp viewport to image bounds
    r0 = min(row_start, H - viewport_size)
    c0 = min(col_start, W - viewport_size)
    r1 = r0 + viewport_size
    c1 = c0 + viewport_size

    # ── Crop ─────────────────────────────────────────────────────────────────
    crop_cube = cube[r0:r1, c0:c1, :]
    crop_rgb  = pseudo_rgb[r0:r1, c0:c1, :]
    crop_gt   = gt[r0:r1, c0:c1] if gt is not None else None

    # ── Inference on crop ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    pred = (infer_pixel(model, crop_cube, batch_size)
            if model_type == "pixel"
            else infer_patch(model, crop_cube, patch_size, batch_size))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_map = make_label_map(pred)

    # ── Inference badge + legend ──────────────────────────────────────────────
    badge_col, legend_col = st.columns([1, 3])
    with badge_col:
        st.markdown(
            f'<div class="infer-badge">'
            f'<span class="dot"></span>'
            f'<span>{elapsed_ms:.0f} ms · '
            f'{viewport_size*viewport_size/(elapsed_ms/1000):,.0f} px/s</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with legend_col:
        class_legend()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main viewport display ─────────────────────────────────────────────────
    n_panels = 3 if crop_gt is not None else 2
    panels   = st.columns(n_panels)

    with panels[0]:
        st.markdown("<div class='section-label'>Pseudo-RGB</div>",
                    unsafe_allow_html=True)
        st.image(crop_rgb, use_container_width=True)

    if crop_gt is not None:
        with panels[1]:
            st.markdown("<div class='section-label'>Ground truth</div>",
                        unsafe_allow_html=True)
            st.image(make_label_map(crop_gt), use_container_width=True)

    with panels[-1]:
        st.markdown("<div class='section-label'>Prediction map</div>",
                    unsafe_allow_html=True)
        st.image(pred_map, use_container_width=True)

    # ── Minimap + stats ───────────────────────────────────────────────────────
    st.divider()
    map_col, stats_col = st.columns([1, 2])

    with map_col:
        st.markdown("<div class='section-label'>Minimap</div>",
                    unsafe_allow_html=True)
        minimap_buf = make_minimap(pseudo_rgb, r0, c0, viewport_size, viewport_size)
        st.image(minimap_buf, use_container_width=True)
        st.caption(
            f"`{st.session_state['source_label']}` · "
            f"{H}×{W}×{N_DECIMATED_BANDS} · "
            f"Viewport: {viewport_size}×{viewport_size} @ ({r0},{c0})"
        )

    with stats_col:
        st.markdown("<div class='section-label'>Class distribution — visible region</div>",
                    unsafe_allow_html=True)

        total = pred.size
        tile_cols = st.columns(4)
        for i, (lbl, name) in enumerate(
            [(l, n) for l, n in CLASS_NAMES.items() if l in CLASS_COLORS]
        ):
            n_px = int((pred == lbl).sum())
            pct  = 100 * n_px / total
            with tile_cols[i]:
                metric_tile(name, pct, n_px, CLASS_COLORS[lbl],
                            highlight=(lbl == 2))  # TT highlighted

        # Model info
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Model</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#1E2130;border-radius:8px;padding:12px 16px;"
            f"font-size:13px;color:#FAFAFA;line-height:2'>"
            f"<b>{model_name}</b><br>"
            f"Checkpoint: <code>{ckpt.name}</code><br>"
            f"Type: {'Pixel (spectral)' if model_type == 'pixel' else 'Patch (spatial)'}<br>"
            f"Inference: <b>{elapsed_ms:.1f} ms</b> on "
            f"{'GPU' if torch.cuda.is_available() else 'CPU'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Spectral viewer ───────────────────────────────────────────────────────
    with st.expander("🔬 Spectral signature viewer"):
        sv1, sv2 = st.columns(2)
        with sv1:
            px_r = st.number_input("Row (within viewport)",
                                   0, viewport_size-1, viewport_size//2)
        with sv2:
            px_c = st.number_input("Col (within viewport)",
                                   0, viewport_size-1, viewport_size//2)

        spectrum   = crop_cube[int(px_r), int(px_c)]
        pred_lbl   = int(pred[int(px_r), int(px_c)])
        pred_name  = CLASS_NAMES.get(pred_lbl, "Unknown")
        pred_color = CLASS_COLORS.get(pred_lbl, "#888888")

        gt_str = ""
        if crop_gt is not None:
            gt_lbl   = int(crop_gt[int(px_r), int(px_c)])
            gt_name  = CLASS_NAMES.get(gt_lbl, "Unlabelled")
            gt_color = CLASS_COLORS.get(gt_lbl, "#888888")
            correct  = "✅" if gt_lbl == pred_lbl else "❌"
            gt_str   = (f" · GT: <span style='color:{gt_color};font-weight:600'>"
                        f"{gt_name}</span> {correct}")

        st.markdown(
            f"Predicted: <span style='color:{pred_color};font-weight:600'>"
            f"{pred_name}</span>{gt_str}",
            unsafe_allow_html=True,
        )

        wl  = np.linspace(440.5, 909.1, N_DECIMATED_BANDS)
        fig, ax = plt.subplots(figsize=(9, 2.5))
        fig.patch.set_facecolor("#1E2130")
        ax.set_facecolor("#1E2130")
        ax.plot(wl, spectrum, color=pred_color, linewidth=1.5)
        ax.axvspan(440, 580, alpha=0.08, color="violet",
                   label="HbO₂ region (~540/577 nm)")
        ax.axvspan(700, 910, alpha=0.08, color="darkred",
                   label="NIR region")
        ax.axvline(760, color="#378ADD", linewidth=0.8, linestyle=":",
                   label="deoxyHb peak (~760 nm)")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="#888780", labelsize=8)
        ax.set_xlabel("Wavelength (nm)", fontsize=9, color="#888780")
        ax.set_ylabel("Reflectance", fontsize=9, color="#888780")
        ax.set_title(
            f"Pixel ({r0+int(px_r)}, {c0+int(px_c)}) — {pred_name}",
            fontsize=9, color="#FAFAFA"
        )
        ax.legend(fontsize=7, facecolor="#1E2130", labelcolor="#888780",
                  edgecolor="#333")
        ax.grid(alpha=0.15, color="#333")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Export ────────────────────────────────────────────────────────────────
    with st.expander("⬇ Export"):
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "Download prediction labels (.npy)",
                data=pred.astype(np.uint8).tobytes(),
                file_name=f"brainvision_pred_{ckpt.stem}_r{r0}c{c0}.npy",
                mime="application/octet-stream",
            )
        with ex2:
            buf = io.BytesIO()
            plt.imsave(buf, pred_map, format="png")
            st.download_button(
                "Download prediction map (.png)",
                data=buf.getvalue(),
                file_name=f"brainvision_pred_{ckpt.stem}_r{r0}c{c0}.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()