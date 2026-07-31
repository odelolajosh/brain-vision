# brainvision — Training Log (Spatial & Spatio-spectral Models)

Student   : Odelola Oluwatolamise Joshua (190407028)\
Supervisor: Prof. Fashanu | UNILAG Systems Engineering\
Repo      : brainvision\
Updated   : 2026-07-31

**Scope**: Tier 3 (Spatial 2D), Tier 4 (Spatio-spectral 3D), Tier 5 (Vision Transformer).\
**Companion file**: `docs/TRAINING_LOG_1D.md` — Tier 1/2 spectral pixel models.\
**EDA findings**: `docs/LOGS.md`

---

## Overview

**Validation strategy** : vp_fabelo (5-fold CV, fixed test set)\
**Primary metric**       : Macro F1-Score excluding BG (Fabelo benchmark)\
**Clinical metric**      : TT sensitivity\
**Fabelo benchmark**     : 70.2 ± 7.9% macro F1-noBG (DNN + spatial KNN)

---

## Data Preparation — Patch Models

Patch models receive (C, P, P) spatial patches centred on each labelled pixel.

**Balancing**: Random centre undersampling to the minority class count (typically TT).
All classes undersampled to n_min → perfectly balanced 4 × n_min patch dataset.

**Augmentation** (training only):
```
Random horizontal flip  : p = 0.5
Random vertical flip    : p = 0.5
Random 90° rotation     : k ∈ {0, 1, 2, 3}
```

**Key constants:**
```python
PATCH_SIZE   = 5       # 5×5 spatial context
BATCH_SIZE   = 64
MAX_EPOCHS   = 150
EARLY_STOP_PATIENCE = 20
```

**Note**: Val loss is dominated by easy BG patches and is NOT informative
for 2D patch models. Monitor val F1-noBG only.

---

## Spectral 1D Baseline Reference

Best spectral-only results from `docs/TRAINING_LOG_1D.md`:

| Model | Loss | Median val F1-noBG | Std |
|---|---|---|---|
| 1D-NN-Fabelo | CE  | 0.7045 | 0.0638 |
| 1D-CNN       | CE  | 0.7032 | 0.0596 |
| 1D-NN-Fabelo | UFL | 0.6825 | 0.0860 |

Gate 2 condition: does best 2D model improve TT sensitivity over best 1D?

---

### Phase 3 — Spatial 2D (Tier 3)

| Run | Model | Loss | Bal | Status |
|---|---|---|---|---|
| 3.1 | 2D-CNN-Fabelo | CE  | bal | ✅ Complete |
| 3.2 | 2D-CNN-Simple | CE  | bal | ✅ Complete |
| 3.3 | 2D-CNN        | CE  | bal | ⏳ |
| 3.4 | Best 2D       | UFL | bal | ⏳ |

Gate 2 evaluated after Phase 3.

---

### Run 3.1 — 2D-CNN-Fabelo × CE × bal × vp_fabelo

**Configuration:**

```
Model       : 2D-CNN-Fabelo (AlexNet-inspired 2D spatial-spectral CNN)
Params      : 142,052
Loss        : CrossEntropyLoss (class weights = 0.25 each)
Balance     : Random centre undersampling → minority class count
Patch size  : 5×5 spatial context
Augmentation: H-flip (p=0.5), V-flip (p=0.5), 90° rotation
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
Epoch time  : ~11–13s per epoch (vs ~2–3s for pixel models)
```

**Per-fold training set sizes (balanced to TT count):**

| Fold | TT centres | Total patches | Val patches |
|---|---|---|---|
| 1 | 19,739 | 78,956 | 128,322 |
| 2 | 16,207 | 64,828 | 168,931 |
| 3 | 21,652 | 86,608 |  76,740 |
| 4 | 19,135 | 76,540 | 136,787 |
| 5 | 18,863 | 75,452 | 146,349 |

**Per-fold val results:**

| Fold | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|
| 1 | 0.6155 |  7 | 18 | Train loss plateau |
| 2 | **0.8558** | **1** | 16 | Val F1 patience |
| 3 | 0.7030 |  7 | 16 | Train loss plateau |
| 4 | 0.7002 |  7 | 22 | Val F1 patience |
| 5 | 0.8265 | 13 | 21 | Train loss plateau |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.7030 ± 0.0956
Val Sens     : best fold 2 = 0.9080, worst fold 4 = 0.7537
Val Spec     : 0.9526–0.9851 across folds
Best epoch   : 1–13 (very fast convergence)
```

**Notable — fold 2 best epoch = 1:**
The model achieves its highest val F1-noBG (0.8558) on the very
first epoch and never recovers that peak. Random initialisation
combined with balanced patch training immediately produces a
representation that generalises to fold 2's representative val
set. Progressive overfitting over subsequent epochs erodes the
initial generalisation. CE loss on the full val patch set is
dominated by easy BG patches and does not track F1-noBG —
only val F1-noBG is informative for 2D patch models.

**2D-CNN-Fabelo vs best 1D models (CE, val F1-noBG):**

| Fold | 1D-Fabelo CE | 1D-CNN CE | 2D-Fabelo CE | Best |
|---|---|---|---|---|
| 1 | 0.5748 | 0.5797 | 0.6155 | **2D** |
| 2 | 0.7633 | 0.7324 | **0.8558** | **2D** |
| 3 | 0.6970 | **0.7417** | 0.7030 | 1D-CNN |
| 4 | **0.7045** | 0.6716 | 0.7002 | 1D-Fabelo |
| 5 | 0.7245 | 0.7032 | **0.8265** | **2D** |
| **Median** | 0.7045 | 0.7032 | **0.7030** | three-way tie |

2D spatial context dramatically improves folds 2 and 5 (+9.25pp
and +10.2pp over 1D-Fabelo). Fold 3 and 4 favour 1D models.
The medians are essentially identical across all three models —
the key difference is variance: 2D has wider fold-level range
(0.6155–0.8558, std=0.0956) vs 1D-Fabelo (0.5748–0.7633,
std=0.0638). Spatial context is highly fold-composition dependent.

**Reproducibility confirmed:**
Run 3.1 was executed twice — both runs identical across all folds.

---

---

### Run 3.2 — 2D-CNN-Simple × CE × bal × vp_fabelo

**Configuration:**

```
Model       : 2D-CNN-Simple (lightweight 2D spatial CNN)
Params      : 19,644  ← smallest model in the entire experiment
Loss        : CrossEntropyLoss (class weights = 0.25 each)
Balance     : Random centre undersampling → minority class count
Patch size  : 5×5 spatial context
Augmentation: H-flip (p=0.5), V-flip (p=0.5), 90° rotation
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
Epoch time  : ~8–10s per epoch
```

**Per-fold val results:**

| Fold | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|
| 1 | 0.5986 |  4 | 19 | Val F1 patience |
| 2 | 0.8492 | 10 | 25 | Val F1 patience |
| 3 | 0.7116 |  8 | 23 | Val F1 patience |
| 4 | 0.5736 |  9 | 11 | Train loss plateau |
| 5 | **0.7796** | 22 | 27 | Train loss plateau |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.7116 ± 0.1076
Val Sens     : best fold 2 = 0.9005, worst fold 4 = 0.7152
Val Spec     : 0.9383–0.9799 across folds
Best epoch   : 4–22 (high variance)
```

⚠️ Fold 4 epoch 8 took 201.3s — MPS sleep event. Use `caffeinate -i`.

**2D-CNN-Simple vs 2D-CNN-Fabelo (CE, val F1-noBG):**

| Fold | 2D-Fabelo CE | 2D-Simple CE | Δ |
|---|---|---|---|
| 1 | **0.6155** | 0.5986 | −1.69pp |
| 2 | **0.8558** | 0.8492 | −0.66pp |
| 3 | 0.7030 | **0.7116** | +0.86pp |
| 4 | **0.7002** | 0.5736 | **−12.66pp** |
| 5 | **0.8265** | 0.7796 | −4.69pp |
| **Median** | **0.7030** | 0.7116 | +0.86pp |

2D-Simple surprisingly edges 2D-Fabelo on median (0.7116 vs 0.7030)
— driven by fold 3 (+0.86pp) while being comparable on folds 1 and 2.
However fold 4 is catastrophic (−12.66pp). The C3-heavy fold 4 val
set exposes the capacity limit of the Simple model — with only
19,644 params it cannot learn the C3 spectral distribution shift.

**All 2D models — CE comparison:**

| Fold | 2D-Fabelo | 2D-Simple | Best |
|---|---|---|---|
| 1 | **0.6155** | 0.5986 | 2D-Fabelo |
| 2 | **0.8558** | 0.8492 | 2D-Fabelo |
| 3 | 0.7030 | **0.7116** | 2D-Simple |
| 4 | **0.7002** | 0.5736 | 2D-Fabelo |
| 5 | **0.8265** | 0.7796 | 2D-Fabelo |
| **Median** | 0.7030 | **0.7116** | marginal tie |

2D-CNN-Fabelo is more robust — wins on 4 of 5 folds and avoids
the fold 4 collapse. 2D-CNN-Simple has higher capacity risk on
cross-campaign folds.

**Reproducibility confirmed:**
Run 3.2 was executed twice — both runs identical across all folds.

---

### Phase 4 — Spatio-spectral 3D (Tier 4)

| Run | Model | Loss | Bal | Status |
|---|---|---|---|---|
| 5.1 | SpectralFormer | CE  | bal | ⏳ (pending Gate 3) |
| 5.2 | SpectralFormer | UFL | bal | ⏳ (pending Gate 1 + 3) |

---

## Gate Decisions

| Gate | Condition | Decision | Rationale |
|---|---|---|---|
| 1 | UFL > CE by 10pp TT sens? | CONDITIONAL ✅ | UFL not uniformly +10pp — fold-composition dependent. Run UFL on all tiers, CE as primary baseline. |
| 2 | 2D > 1D on TT sens? | ⏳ | Pending Phase 3 |
| 3 | 3D > 2D by 5pp F1-noBG? | ⏳ | Pending Phase 4 |

---

## Key Constants

```python
MAX_EPOCHS           = 150
EARLY_STOP_PATIENCE  = 20
TRAIN_LOSS_DELTA_MIN = 1e-4
EARLY_STOP_MIN_EPOCHS = 10
LEARNING_RATE        = 1e-3
LR_DECAY_FACTOR      = 0.5
LR_DECAY_PATIENCE    = 5
BATCH_SIZE           = 64
N_PIXELS_PER_CLASS   = 1000    # K-Means balanced training
PATCH_SIZE           = 5       # for 2D/3D/HybridSN models
N_DECIMATED_BANDS    = 128
N_CLASSES            = 4       # NT, TT, BV, BG
```

---

## Notation

```
Run name format : {model}_{loss}_{bal/nobal}_{fold}_{strategy}
Example         : 1dnnfabelo_ce_bal_fold1_vpfabelo

Metrics:
  F1-noBG  : macro F1 excluding BG class — primary comparison metric
  TT Sens  : tumour tissue sensitivity — clinical priority metric
  OA       : overall pixel accuracy
  ±        : std across 5 folds (median ± std)

Fabelo benchmark: 70.2 ± 7.9% F1-noBG  (DNN + KNN spatial filtering)
```

---