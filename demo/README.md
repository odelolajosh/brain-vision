# brainvision — Defence Touchscreen Demo

Intraoperative HSI brain-tumour classification demonstrator, built to the
merged conceptual UI spec. Imports directly from the `brainvision` package —
no duplicated model or preprocessing code.

## Screen flow (spec §1)

```
Idle ──"Begin Demo"──▶ Case Select ──tap a case──▶ Live Inference
                            ▲                          │   │
                            │              ┌───────────┘   └── "Compare All Cases" ──▶ Summary
                            └── Home (⌂) ──┴── from any screen                            │
                                                                        "Back to Cases" ─┘
Bonus (flagged unscripted):  Live / Case Select ──"Live Mode (bonus)"──▶ Free Pan/Zoom ──"Exit"──▶ back
```

- **Idle** — title + one large *Begin Demo* target.
- **Case Select** — three fixed cases as large tiles. No "Case 1 of 3" counter;
  jump to any case in any order.
- **Live Inference** (hero) — one large prediction panel + tappable Ground-Truth
  / Pseudo-RGB thumbnails that promote into the primary slot. A
  **Prediction ⇄ Agreement** toggle recolours the primary panel to
  ✓ Correct / ✗ Incorrect / · Unlabelled. The sparse-GT caveat is anchored
  under the reference column, always visible. Plain-language caption band +
  metric chips beneath.
- **Summary** — Pareto scatter (F1-noBG vs. latency, viable region shaded, 60 s
  line, spectral + spatial picks starred). Tap to cycle the finding line.
- **Free Pan/Zoom** — the original simulator, demoted to a bonus mode with an
  "unscripted" banner and coarse drag controls.

## Tuning it for a specific defence

Everything a presenter changes lives in two config blocks at the top of
`demo/app.py`:

| Block | Controls |
|---|---|
| `CASES` | the three cases: image `.npz`, `crop` `(row0, col0, size)` or `None`, `model` key, `checkpoint` path, `patch_size`, and the narrative `caption` |
| `PARETO_POINTS` / `SUMMARY_FINDINGS` | the summary scatter points and the rotating finding lines |

Latency values in `PARETO_POINTS` are **rehearsal placeholders** — measure them
on the actual defence hardware and paste the real numbers in. Rehearse on the
exact hardware, weights, crops and tap sequence used live (spec §10).

## Run locally

```bash
# from the repo root
streamlit run demo/app.py
```

Kiosk / touchscreen:

```bash
streamlit run demo/app.py \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
# then open the browser full-screen (F11) at http://<host>:8501
```

## Deploy to Raspberry Pi 5

Copy only what the Pi needs:

```
brainvision/                        ← package (models, preprocessing, constants)
demo/app.py
demo/requirements_demo.txt
processed/first_campaign/008-02.npz     ← Case 1 image
processed/first_campaign/012-01.npz     ← Case 3 image
processed/second_campaign/038-01.npz    ← Case 2 image
checkpoints/1dnnfabelo_ce_bal_fold2_vpfabelo.pt      ← Cases 1 & 3 (~20 KB)
checkpoints/2dcnnfabelo_ufl_bal_fold2_vpfabelo.pt    ← Case 2 (patch)
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r demo/requirements_demo.txt
streamlit run demo/app.py --server.address 0.0.0.0 --server.headless true
```

**Notes**
- Cases 1 & 3 use the 4,936-param `FabeloDNN` — near-instant on the Pi.
- Case 2 uses the `Fabelo2DCNN` patch model on a small crop — the spec's
  deliberate "spatial trades speed" case; expect tens of seconds on the Pi.
  Shrink the `crop` size in `CASES` if it needs to be faster.
- Results per case are computed once and cached for the session, so navigating
  between screens is instant after the first visit.

## Import map

| app.py uses | comes from |
|---|---|
| `CLASS_COLORS`, `CLASS_NAMES`, `N_DECIMATED_BANDS`, `N_CLASSES` | `brainvision/constants.py` |
| `normalise` (local), pseudo-RGB, label maps | reused simulator helpers |
| `FabeloDNN`, `Baseline1DDNN`, `HuEtAl1DCNN`, `Fabelo2DCNN`, `LeeEtAl2DCNN`, `Simple2DCNN` | `brainvision/models/` |

If a class name doesn't match, update the import at the top of `demo/app.py`.
