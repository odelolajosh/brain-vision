# brainvision — Training Log (Spectral Models)

Student   : Odelola Oluwatolamise Joshua (190407028)\
Supervisor: Prof. Fashanu | UNILAG Systems Engineering\
Repo      : brainvision\
Updated   : 2026-07-31

**Scope**: Tier 1 (Spectral FC) and Tier 2 (Spectral Conv) — pixel-level models.\
**Companion file**: `docs/TRAINING_LOG_2D.md` — Tier 3/4/5 spatial and spatio-spectral models.\
**EDA findings**: `docs/LOGS.md`

---
---

## Overview

This log tracks all training experiments, decisions, and results
for the brainvision hyperspectral brain tumour classification project.
EDA findings are in `docs/LOGS.md`.

**Validation strategy** : vp_fabelo (5-fold CV, fixed test set)\
**Primary metric**       : Macro F1-Score excluding BG (Fabelo benchmark)\
**Clinical metric**      : TT sensitivity\
**Fabelo benchmark**     : 70.2 ± 7.9% macro F1-noBG (DNN + spatial KNN)

---

## Configuration Space

```
Models (9)  : 1D-NN-Fabelo, 1D-NN, 1D-CNN, SpectralFormer
              2D-CNN-Simple, 2D-CNN, 2D-CNN-Fabelo
              3D-CNN, HybridSN

Loss fns (4): CE, FL, DL, UFL

Balance (2) : bal, nobal

Strategy    : vp_fabelo — 5-fold CV, fixed 20% test set

Full space  : 9 × 4 × 2 × 5 folds = 360 runs
```

---

## Reduction Strategy

Full training of all 360 runs is not feasible. The following
principled rules reduce the space to ~110 runs while preserving
the ability to answer all research questions.

### Rule 1 — Fix balance

```
Primary : bal (balanced training via K-Means reduction)
nobal   : only where explicitly needed to quantify balancing contribution
```

Justification: Global TT=4.1%, NT:TT=8.2:1. Running CE without
balancing produces near-zero TT sensitivity (confirmed in VP1
preliminary experiments). Balanced training is a prerequisite
for meaningful TT classification.

### Rule 2 — Loss function funnel per architecture

Rather than running all 4 loss functions on every model:

```
Step 1 : CE + bal       ← baseline for all architectures
Step 2 : UFL + bal      ← best dynamic loss, run if CE TT sens warrants it
Step 3 : FL + bal       ← only if UFL shows meaningful gain over CE
Step 4 : DL             ← SKIPPED for all architectures
```

Justification for skipping DL: DL requires soft probability
targets to compute meaningful overlap — it is architecturally
mismatched with pixel-level classification where class boundaries
are sharp. DL treats the task as segmentation (region overlap)
when it is classification (per-pixel label). This makes DL
the weakest candidate a priori.

### Rule 3 — Architecture tier ordering

Train in order of increasing complexity. Each tier informs
whether to proceed:

```
Tier 1 — Spectral FC       : 1D-NN-Fabelo, 1D-NN
Tier 2 — Spectral Conv     : 1D-CNN, SpectralFormer
Tier 3 — Spatial 2D        : 2D-CNN-Simple, 2D-CNN-Fabelo, 2D-CNN
Tier 4 — Spatio-spectral   : 3D-CNN, HybridSN
```

### Rule 4 — Skip criteria (decision gates)

```
Gate 1 — After Tier 1:
  Does UFL improve TT sensitivity by > 10pp over CE?
  YES → run UFL on all subsequent tiers
  NO  → CE only, skip UFL

Gate 2 — After Tier 3:
  Does best 2D model improve TT sensitivity over best 1D model?
  YES → proceed to Tier 4 (3D models justified)
  NO  → stop at 2D, skip Tier 4

Gate 3 — After Tier 4:
  Does HybridSN or 3D-CNN beat 2D models by > 5pp F1-noBG?
  YES → SpectralFormer worth running
  NO  → SpectralFormer optional (time permitting)

Skip DL    : for all architectures — see Rule 2
Skip nobal : if bal CE already achieves TT sensitivity > 60%
Skip FL    : if UFL improvement over CE < 3pp TT sensitivity
```

---

## Training Data Preparation

Two different strategies are used depending on model type.
The split between pixel models and patch models is:

```
Pixel models : 1D-NN-Fabelo, 1D-NN, 1D-CNN, SpectralFormer
Patch models : 2D-CNN-Simple, 2D-CNN-Fabelo, 2D-CNN, 3D-CNN, HybridSN
```

---

### Pixel Models — K-Means Pixel Reduction

Follows Fabelo et al. (2023) exactly.

**Procedure:**

```
Input : all labelled pixels from training images
        e.g. NT=221,479 px  TT=20,931 px  BV=73,735 px  BG=238,017 px

Step 1 — K-Means clustering per class
  Apply K-Means independently to each class
  K = 100 clusters per class → 400 centroids total
  Each centroid = a prototype spectral signature for that class

Step 2 — SAM-based pixel selection
  For each centroid, find the n most similar real pixels
  using Spectral Angle Mapper (SAM) distance
  n = 10 → selects 1,000 pixels per class (100 × 10)

Step 3 — Output balanced training set
  NT: 1,000 px   TT: 1,000 px   BV: 1,000 px   BG: 1,000 px
  Total: 4,000 pixels — perfectly balanced (25% each)
```

**Why 1,000 per class:**
Fabelo tested n ∈ {10, 20, 40} → {1,000, 2,000, 4,000} px/class and
found no statistically significant difference in macro F1-Score.
1,000 was selected as the smallest size achieving equivalent
performance — giving ~48× training speedup and ~20% improvement
in TT accuracy vs the full unbalanced training set.

The improvement comes from balance not quantity — with 1,000
pixels per class, TT contributes 25% of gradient signal vs 3.7%
in the full unbalanced set (6.8× more TT gradient). Additional
pixels beyond 1,000 are near-duplicates of existing centroids
and contribute redundant gradients.

**Applied to:** training set only. Validation and test sets
always use all labelled pixels — no reduction applied.

**Class weights after reduction:** all equal to 0.25 (perfectly
balanced) — CE loss treats all classes equally after reduction.

---

### Patch Models — Centre-Pixel Balancing

Patch models receive (C, P, P) spatial patches centred on each
labelled pixel. Balancing is applied differently from pixel models
because patches overlap and cannot be independently K-Means clustered.

**Procedure:**

```
Step 1 — Collect all centre-pixel coordinates per class
  For each training image, identify all labelled pixel locations
  per class — these are candidate patch centres

Step 2 — Undersample to the minority class count
  Find the smallest class count across NT, TT, BV, BG
  in the training set (typically TT — e.g. 20,931 px)
  Randomly subsample all other classes to that count

Step 3 — Extract patches on-the-fly (lazy)
  HSIPatchDatasetLazy stores only the padded cube + centre
  coordinates — patches extracted during DataLoader iteration
  Avoids materialising the full patch tensor in memory

Output: balanced set of patch centres
  NT: n_min px   TT: n_min px   BV: n_min px   BG: n_min px
  Total: 4 × n_min patches — perfectly balanced
```

**Why random undersampling for patches:**
K-Means on patch centres would require clustering (C×P×P)-dimensional
vectors — prohibitively expensive for large P and many patches.
Random undersampling of centre coordinates is computationally
tractable and sufficient since spatial augmentation (applied
during training) introduces diversity.

---

### Patch Models — Spatial Augmentation

Applied to patch models during training only. Val and test
sets are never augmented.

**Augmentations applied:**

```
Random horizontal flip  : p = 0.5
Random vertical flip    : p = 0.5
Random 90° rotation     : k ∈ {0, 1, 2, 3}  (0°, 90°, 180°, 270°)
```

All three are applied independently per patch per batch.

**Physical justification:**
Brain tissue has no canonical orientation in the surgical field —
the HSI camera can be positioned at any angle relative to the
exposed brain surface. Horizontal flip, vertical flip, and 90°
rotation are all physically valid transformations that do not
alter the spectral content of the patch. They increase effective
training set diversity and reduce spatial overfitting.

**Why spectral augmentation is not applied:**
Spectral augmentation (band dropout, noise injection, spectral
shifting) risks introducing physically unrealistic spectra that
do not correspond to real tissue signatures. The K-Means
pixel reduction already provides spectral diversity for pixel
models. Spatial augmentation provides geometric diversity for
patch models without corrupting spectral integrity.

**Applied to:** training patches only via HSIPatchDatasetLazy
(augment=True). Val and test use augment=False.

---

### Summary — data preparation by model type

| Aspect | Pixel models | Patch models |
|---|---|---|
| Balancing method | K-Means + SAM selection | Random centre undersampling |
| Pixels per class | 1,000 (fixed) | min class count (varies) |
| Total training samples | 4,000 | 4 × n_min |
| Augmentation | None | Flip + rotate (train only) |
| Val/test reduction | None | None |
| Val/test augmentation | None | None |

---

## Training Plan

### Phase 1 — Spectral FC (Tier 1)

| Run | Model | Loss | Bal | Status |
|---|---|---|---|---|
| 1.1 | 1D-NN-Fabelo | CE  | bal | ✅ Complete |
| 1.2 | 1D-NN-Fabelo | UFL | bal | ✅ Complete |
| 1.3 | 1D-NN-Fabelo | FL  | bal | ✅ Complete |
| 1.4 | 1D-NN        | CE  | bal | ✅ Complete |
| 1.5 | 1D-NN        | UFL | bal | ✅ Complete |

Gate 1 evaluated after Phase 1.

---

### Run 1.1 — 1D-NN-Fabelo × CE × bal × vp_fabelo

**Configuration:**

```
Model       : 1D-NN-Fabelo (FabeloDNN)
Params      : 4,936
Loss        : CrossEntropyLoss (class weights = 0.25 each)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.5748 | 37 | 46 | Train loss plateau |
| 2 | 168,931 | 0.7633 | 10 | 12 | Train loss plateau |
| 3 |  76,740 | 0.6970 | 19 | 34 | Val F1 patience |
| 4 | 136,787 | 0.7045 | 34 | 49 | Val F1 patience |
| 5 | 146,349 | 0.7245 | 25 | 40 | Val F1 patience |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.7045 ± 0.0638
Val Sens     : best fold 2 = 0.8752, worst fold 5 = 0.7363
Val Spec     : 0.9448–0.9682 across folds
Best epoch   : 10–37 (high variance — fold-dependent)
```

**Per-fold training observations:**

Fold 1 — Slowest convergence (46 epochs). Val F1-noBG never broke
0.58. Consistent with fold complexity analysis — C1-heavy val,
lowest TT pixel count, smallest within-campaign NT/TT SAM (2.00°).

Fold 2 — Fastest convergence (12 epochs, stopped by train plateau).
Best val F1-noBG (0.7633) at epoch 10. Highest val sensitivity
(0.8752). Most balanced campaign composition. Confirms fold 2
as the most representative and learnable partition.

Fold 3 — Peaked at epoch 19, then gradual decline. High TT
intra-variance (0.1170) in val set causes F1 instability — model
finds a boundary for some TT subtypes but cannot sustain it.

Fold 4 — Long steady learning (49 epochs). Val F1-noBG climbed
from 0.45 to 0.70 monotonically. C3-heavy val set (5 patients)
presented consistent TT spectra (TT Var=0.0667) enabling stable
learning despite cross-campaign gap.

Fold 5 — Moderate convergence. Epoch 15 anomaly: 234s vs ~2.6s
typical — likely MPS memory event or system interruption.
Best at epoch 25 (0.7245). Despite highest NT:TT ratio (17.5),
C2-heavy composition provided compatible TT spectra.

**Notable pattern:**
Fold 2 stopped at epoch 12 via train loss plateau — converged
very fast on fold 2's learnable val set. All other folds ran
34–49 epochs, confirming fold 2's val patients are exceptionally
close to the training distribution.

**Replication check:**
Fabelo et al. (2023) reported 70.2 ± 7.9% macro F1-noBG using
DNN + KNN spatial filtering. This run (CE, no spatial filtering)
achieves median val F1-noBG of ~70.5% — on-track for replication
before test evaluation and spatial post-processing.

---

### Run 1.2 — 1D-NN-Fabelo × UFL × bal × vp_fabelo

**Configuration:**

```
Model       : 1D-NN-Fabelo (FabeloDNN)
Params      : 4,936
Loss        : UnifiedFocalLoss (alpha=class_weights, lambda=0.5, delta=0.6)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.5816 | 26 | 41 | Val F1 patience |
| 2 | 168,931 | **0.8540** | 22 | 31 | Train loss plateau |
| 3 |  76,740 | 0.6767 | 15 | 30 | Val F1 patience |
| 4 | 136,787 | 0.6197 | 21 | 23 | Train loss plateau |
| 5 | 146,349 | 0.6825 | 16 | 31 | Val F1 patience |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.6825 ± 0.0860
Val Sens     : best fold 2 = 0.8894, worst fold 1 = 0.7687
Val Spec     : 0.9403–0.9801 across folds
Best epoch   : 15–26 (faster convergence than CE)
```

**CE vs UFL comparison (val F1-noBG per fold):**

| Fold | CE | UFL | Δ |
|---|---|---|---|
| 1 | 0.5748 | 0.5816 | +0.0068 |
| 2 | 0.7633 | **0.8540** | **+0.0907** |
| 3 | 0.6970 | 0.6767 | −0.0203 |
| 4 | 0.7045 | 0.6197 | −0.0848 |
| 5 | 0.7245 | 0.6825 | −0.0420 |
| **Median** | **0.7045** | **0.6825** | **−0.0220** |

**Per-fold training observations:**

Fold 1 — Marginal improvement over CE (+0.0068). UFL provides
slightly higher sensitivity (0.7687 vs CE 0.7792 — actually lower)
but F1-noBG barely changes. Fold 1's C1-heavy composition limits
both loss functions equally.

Fold 2 — Dramatic improvement (+0.0907). UFL achieves 0.8540 vs
CE's 0.7633 — the largest UFL gain of any fold. Fold 2's balanced
campaign composition allows UFL's hard-example weighting to exploit
the full spectral diversity. Val sensitivity reaches 0.8894.

Fold 3 — UFL slightly worse than CE (−0.0203). High TT intra-
variance (0.1170) means UFL's hard example focus amplifies noise
from inconsistent TT pixels rather than improving TT sensitivity.

Fold 4 — UFL noticeably worse than CE (−0.0848). Fold 4's C3-heavy
val set (5 patients) with consistent but campaign-shifted TT
spectra responds poorly to UFL's focal weighting — UFL may be
over-penalising C3 TT pixels that are simply spectrally different
from training data, treating them as hard negatives.

Fold 5 — UFL worse than CE (−0.0420). MPS sleep events at epochs
18, 24, 27, 30 (each ~904s) disrupted training rhythm. Results
should be treated with caution — the LR scheduler may have
misfired due to timing anomalies.

⚠️ Note: Fold 5 had 4 MPS system sleep events (~904s each at
epochs 18, 24, 27, 30). Mac display sleep prevented continuous
training. Fix: use `caffeinate -i` before future runs.

**Key observation:**
UFL does not uniformly improve over CE across all folds. It
dramatically helps fold 2 (+9pp) but hurts folds 3, 4, 5. The
median F1-noBG is lower for UFL (0.6825) than CE (0.7045) — a
reversal of the expected direction.

This suggests UFL's benefit is fold-composition dependent. On
folds where the val set is spectrally compatible with training
(fold 2), UFL's hard-example weighting exploits the full signal.
On folds with campaign drift (fold 4, C3-heavy) or TT heterogeneity
(fold 3), UFL amplifies noise.

---

**Configuration:**

```
Model       : 1D-NN-Fabelo (FabeloDNN)
Params      : 4,936
Loss        : CrossEntropyLoss (class weights = 0.25 each)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.5748 | 37 | 46 | Train loss plateau |
| 2 | 168,931 | 0.7633 | 10 | 12 | Train loss plateau |
| 3 |  76,740 | 0.6970 | 19 | 34 | Val F1 patience |
| 4 | 136,787 | 0.7045 | 34 | 49 | Val F1 patience |
| 5 | 146,349 | 0.7245 | 25 | 40 | Val F1 patience |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.7045 ± 0.0638
Val Sens     : best fold 2 = 0.8752, worst fold 5 = 0.7363
Val Spec     : 0.9448–0.9682 across folds
Best epoch   : 10–37 (high variance — fold-dependent)
```

**Per-fold training observations:**

Fold 1 — Slowest convergence (46 epochs). Val F1-noBG never broke
0.58. Consistent with fold complexity analysis — C1-heavy val,
lowest TT pixel count, smallest within-campaign NT/TT SAM (2.00°).

Fold 2 — Fastest convergence (12 epochs, stopped by train plateau).
Best val F1-noBG (0.7633) at epoch 10. Highest val sensitivity
(0.8752). Most balanced campaign composition. Confirms fold 2
as the most representative and learnable partition.

Fold 3 — Peaked at epoch 19, then gradual decline. High TT
intra-variance (0.1170) in val set causes F1 instability — model
finds a boundary for some TT subtypes but cannot sustain it.

Fold 4 — Long steady learning (49 epochs). Val F1-noBG climbed
from 0.45 to 0.70 monotonically. C3-heavy val set (5 patients)
presented consistent TT spectra (TT Var=0.0667) enabling stable
learning despite cross-campaign gap.

Fold 5 — Moderate convergence. Epoch 15 anomaly: 234s vs ~2.6s
typical — likely MPS memory event or system interruption.
Best at epoch 25 (0.7245). Despite highest NT:TT ratio (17.5),
C2-heavy composition provided compatible TT spectra.

**Notable pattern:**
Fold 2 stopped at epoch 12 via train loss plateau — converged
very fast on fold 2's learnable val set. All other folds ran
34–49 epochs, confirming fold 2's val patients are exceptionally
close to the training distribution.

**Replication check:**
Fabelo et al. (2023) reported 70.2 ± 7.9% macro F1-noBG using
DNN + KNN spatial filtering. This run (CE, no spatial filtering)
achieves median val F1-noBG of ~70.5% — on-track for replication
before test evaluation and spatial post-processing.

**Reproducibility confirmed:**
Run 1.2 was executed twice (second run without MPS sleep events).
Both runs produced identical results across all 5 folds — every
epoch loss, F1-noBG, best epoch, and stop epoch matched exactly.
Confirms the pipeline is fully deterministic (fixed K-Means seed,
fixed model initialisation seed). Fold 5 sleep events in the first
run did not affect the final checkpoints.

---

### Run 1.3 — 1D-NN-Fabelo × FL × bal × vp_fabelo

**Configuration:**

```
Model       : 1D-NN-Fabelo (FabeloDNN)
Params      : 4,936
Loss        : FocalLoss (alpha=class_weights, gamma=2.0)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.4965 | 13 | 14 | Train loss plateau |
| 2 | 168,931 | 0.7910 | 15 | 17 | Train loss plateau |
| 3 |  76,740 | 0.6656 | 11 | 12 | Train loss plateau |
| 4 | 136,787 | 0.6659 | 11 | 15 | Train loss plateau |
| 5 | 146,349 | 0.6898 |  9 | 18 | Train loss plateau |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.6659 ± 0.0979
Val Sens     : best fold 2 = 0.8264, worst fold 1 = 0.7427
Val Spec     : 0.9127–0.9752 across folds
Best epoch   : 9–15 — fastest convergence of all three loss fns
```

**Notable training behaviour — FL loss scale:**
FL train loss values are in the range 0.005–0.027 vs CE's 0.08–1.26.
This is expected — Focal Loss applies a modulating factor (1-p)^γ
to each sample, down-weighting easy examples. The absolute loss
values are not comparable across loss functions.

**Train plateau criterion fires on every fold:**
All 5 folds stopped via train loss plateau (9–17 epochs) — 2–3×
faster than CE (34–49 epochs). FL's modulating factor suppresses
gradients from easy examples, causing train loss to plateau quickly
once the model handles easy cases. The model is not necessarily
converged on hard examples — early stopping may be cutting FL off
prematurely. Consider increasing EARLY_STOP_MIN_EPOCHS for FL in
future runs.

**Cross-loss comparison — 1D-NN-Fabelo, all three loss functions:**

| Fold | CE | FL | UFL | Best |
|---|---|---|---|---|
| 1 | 0.5748 | 0.4965 | 0.5816 | UFL |
| 2 | 0.7633 | 0.7910 | **0.8540** | UFL |
| 3 | **0.6970** | 0.6656 | 0.6767 | CE |
| 4 | **0.7045** | 0.6659 | 0.6197 | CE |
| 5 | **0.7245** | 0.6898 | 0.6825 | CE |
| **Median** | **0.7045** | 0.6659 | 0.6825 | **CE** |

CE has the highest median F1-noBG (0.7045) and wins on 3 of 5
folds. FL has the lowest median (0.6659) — early plateau stopping
limits convergence on hard examples. UFL wins dramatically on fold
2 (+9pp over CE) but underperforms CE on folds 3, 4, 5.

The fold-2 UFL result (0.8540) is exceptional and distribution-
dependent — UFL exploits fold 2's representative mixed-campaign
val set but amplifies noise on less representative folds.

**Reproducibility confirmed:**
Run 1.3 was executed twice — both runs identical across all folds.

---

### Run 1.4 — 1D-NN × CE × bal × vp_fabelo

**Configuration:**

```
Model       : 1D-NN (deep fully-connected, 4 hidden layers)
Params      : 17,055,748  ← 3,459× more than 1D-NN-Fabelo (4,936)
Loss        : CrossEntropyLoss (class weights = 0.25 each)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.6080 |  6 | 21 | Val F1 patience |
| 2 | 168,931 | **0.8672** | 30 | 45 | Val F1 patience |
| 3 |  76,740 | 0.6760 |  6 | 21 | Val F1 patience |
| 4 | 136,787 | 0.6727 | 13 | 28 | Val F1 patience |
| 5 | 146,349 | 0.7735 | 18 | 33 | Val F1 patience |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.6760 ± 0.0924
Val Sens     : best fold 2 = 0.9034, worst fold 1 = 0.7937
Val Spec     : 0.9288–0.9801 across folds
Best epoch   : 6–30 (high variance)
```

**Critical finding — overparameterisation:**

1D-NN has 17,055,748 parameters trained on 4,000 pixels.
That is a parameter-to-sample ratio of 4,264:1. The model is
severely overparameterised for this training set size.

```
1D-NN-Fabelo × CE  median val F1-noBG : 0.7045   (4,936 params)
1D-NN        × CE  median val F1-noBG : 0.6760  (17,055,748 params)
```

The larger model performs worse on median despite 3,459× more
parameters. Fold 2 is exceptional (1D-NN 0.8672 vs Fabelo 0.7633)
but folds 1, 3, 4 all underperform the tiny Fabelo model.

This is a textbook overparameterisation result — the model
memorises the 4,000 training pixels rather than learning
generalisable spectral signatures. The validation loss is
consistently much higher than training loss (e.g. fold 1:
train loss 0.20 at epoch 21 vs val loss 1.10 — a 5.5× gap).

**Best epoch instability:**
Folds 1 and 3 both peak at epoch 6 — very early. The model
finds a reasonable solution quickly then overfits, and the
val F1 never recovers to that early peak despite 15 more
epochs of training. This is characteristic of overfit models
on small training sets — early stopping catches the generalisation
peak before memorisation dominates.

**1D-NN-Fabelo vs 1D-NN — all losses (val F1-noBG):**

| Fold | Fabelo CE | 1D-NN CE | Δ |
|---|---|---|---|
| 1 | **0.5748** | 0.6080 | +0.0332 |
| 2 | 0.7633 | **0.8672** | +0.1039 |
| 3 | **0.6970** | 0.6760 | −0.0210 |
| 4 | **0.7045** | 0.6727 | −0.0318 |
| 5 | 0.7245 | **0.7735** | +0.0490 |
| **Median** | **0.7045** | 0.6760 | **−0.0285** |

1D-NN wins on folds 1, 2, 5 but loses on folds 3, 4.
Fabelo wins on median — the compact model generalises more
reliably across diverse fold compositions.

---

### Run 1.5 — 1D-NN × UFL × bal × vp_fabelo

**Configuration:**

```
Model       : 1D-NN (deep fully-connected, 4 hidden layers)
Params      : 17,055,748
Loss        : UnifiedFocalLoss (alpha=class_weights, lambda=0.5, delta=0.6)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.5869 |  5 | 15 | Train loss plateau |
| 2 | 168,931 | 0.8369 | 12 | 27 | Val F1 patience |
| 3 |  76,740 | 0.6725 | 12 | 27 | Val F1 patience |
| 4 | 136,787 | 0.6756 | 17 | 32 | Val F1 patience |
| 5 | 146,349 | 0.7304 | 23 | 33 | Train loss plateau |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.6756 ± 0.0876
Val Sens     : best fold 2 = 0.8960, worst fold 1 = 0.7680
Val Spec     : 0.9354–0.9775 across folds
Best epoch   : 5–23
```

**1D-NN CE vs UFL comparison:**

| Fold | CE | UFL | Δ |
|---|---|---|---|
| 1 | **0.6080** | 0.5869 | −0.0211 |
| 2 | **0.8672** | 0.8369 | −0.0303 |
| 3 | **0.6760** | 0.6725 | −0.0035 |
| 4 | 0.6727 | **0.6756** | +0.0029 |
| 5 | **0.7735** | 0.7304 | −0.0431 |
| **Median** | **0.6760** | 0.6756 | **−0.0004** |

Unlike 1D-NN-Fabelo where UFL dramatically helped fold 2, for 1D-NN
UFL consistently underperforms or matches CE. The overparameterised
model cannot benefit from UFL's hard-example weighting — it already
overfits regardless of loss function, so UFL's mechanism has no
meaningful effect.

**Reproducibility confirmed:**
Run 1.5 was executed twice — both runs identical across all folds.

---

### Phase 1 Complete — Summary and Gate 1 Decision

**All runs across both architectures and all loss functions:**

| Run | Model | Loss | Median F1-noBG | Std | Best fold | Worst fold |
|---|---|---|---|---|---|---|
| 1.1 | 1D-NN-Fabelo | CE  | **0.7045** | 0.0638 | F2: 0.7633 | F1: 0.5748 |
| 1.2 | 1D-NN-Fabelo | UFL | 0.6825 | 0.0860 | F2: **0.8540** | F1: 0.5816 |
| 1.3 | 1D-NN-Fabelo | FL  | 0.6659 | 0.0979 | F2: 0.7910 | F1: 0.4965 |
| 1.4 | 1D-NN        | CE  | 0.6760 | 0.0924 | F2: 0.8672 | F1: 0.6080 |
| 1.5 | 1D-NN        | UFL | 0.6756 | 0.0876 | F2: 0.8369 | F1: 0.5869 |

**Key Phase 1 findings:**

1. **1D-NN-Fabelo × CE is the strongest baseline** — highest median
   F1-noBG (0.7045), lowest std (0.0638). Compact model generalises
   more reliably than overparameterised 1D-NN.

2. **UFL is fold-composition dependent** — dramatically boosts fold 2
   on Fabelo (+9pp) but reduces median by −2.2pp. Not a uniform gain.

3. **FL consistently underperforms** — early train loss plateau at
   9–17 epochs cuts FL off prematurely. Worst median of all runs.

4. **1D-NN is overparameterised** — 17M params on 4,000 training
   pixels (4,264:1 ratio) causes overfitting. Neither CE nor UFL
   overcomes the capacity-data mismatch.

---

### Gate 1 Decision

**Condition**: Does UFL improve TT sensitivity by > 10pp over CE?

Per-fold TT sensitivity delta (UFL − CE) for 1D-NN-Fabelo:

```
Fold 1 : 0.7687 − 0.7792 = −0.0105
Fold 2 : 0.8894 − 0.8752 = +0.0142
Fold 3 : 0.7543 − 0.7753 = −0.0210
Fold 4 : 0.7713 − 0.7998 = −0.0285
Fold 5 : 0.7249 − 0.7363 = −0.0114
Median delta           : −0.0114  (UFL marginally worse overall)
```

**Gate 1 Decision: CONDITIONAL ✅**

UFL does NOT uniformly improve TT sensitivity by >10pp. Median
delta is slightly negative (−0.0114). However fold 2's UFL
result (F1-noBG=0.8540, Sens=0.8894) is exceptional — UFL CAN
work when the val set is representative of the training distribution.

**Decision: Run UFL alongside CE for subsequent tiers** but treat
CE as the primary comparison baseline. UFL's value will be assessed
per-architecture. This is a modification of the original binary gate
— the fold-composition dependency warrants a conditional approach.

---

### Phase 2 — Spectral Conv (Tier 2)

| Run | Model | Loss | Bal | Status |
|---|---|---|---|---|
| 2.1 | 1D-CNN | CE  | bal | ✅ Complete |
| 2.2 | 1D-CNN | UFL | bal | ✅ Complete |

SpectralFormer deferred to Gate 3 decision.

---

### Run 2.1 — 1D-CNN × CE × bal × vp_fabelo

**Configuration:**

```
Model       : 1D-CNN (1D convolutional spectral feature extractor)
Params      : 76,824
Loss        : CrossEntropyLoss (class weights = 0.25 each)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.5797 | 16 | 31 | Val F1 patience |
| 2 | 168,931 | 0.7324 | 29 | 30 | Train loss plateau |
| 3 |  76,740 | **0.7417** | 36 | 51 | Val F1 patience |
| 4 | 136,787 | 0.6716 | 15 | 30 | Val F1 patience |
| 5 | 146,349 | 0.7032 | 29 | 44 | Val F1 patience |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.7032 ± 0.0596
Val Sens     : best fold 3 = 0.8003, worst fold 1 = 0.7718
Val Spec     : 0.9512–0.9803 across folds
Best epoch   : 15–36 (moderate variance)
```

**1D-CNN vs 1D-NN-Fabelo comparison (CE, val F1-noBG):**

| Fold | Fabelo CE | 1D-CNN CE | Δ |
|---|---|---|---|
| 1 | **0.5748** | 0.5797 | +0.0049 |
| 2 | **0.7633** | 0.7324 | −0.0309 |
| 3 | 0.6970 | **0.7417** | +0.0447 |
| 4 | **0.7045** | 0.6716 | −0.0329 |
| 5 | **0.7245** | 0.7032 | −0.0213 |
| **Median** | **0.7045** | 0.7032 | **−0.0013** |

1D-CNN and 1D-NN-Fabelo are essentially tied on median (0.7032
vs 0.7045 — 0.13pp difference). 1D-CNN wins on folds 1 and 3
but loses on 2, 4, 5. Local spectral convolutions do not provide
a clear advantage over the FC Fabelo baseline on median.

**Notable — fold 3 improvement (+4.5pp):**
Fold 3 has the highest TT intra-class variance (0.1170). Local
spectral convolutions may better capture diverse TT spectral
patterns in this heterogeneous val set vs Fabelo's global FC
mapping.

**Reproducibility confirmed:**
Run 2.1 was executed twice — both runs identical across all folds.

---

### Run 2.2 — 1D-CNN × UFL × bal × vp_fabelo

**Configuration:**

```
Model       : 1D-CNN (1D convolutional spectral feature extractor)
Params      : 76,824
Loss        : UnifiedFocalLoss (alpha=class_weights, lambda=0.5, delta=0.6)
Balance     : bal (K-Means, 1,000 px/class, 4,000 total)
Strategy    : vp_fabelo (5-fold, fixed test set)
Hardware    : MPS (Apple Silicon)
```

**Per-fold val results:**

| Fold | Val px | Best F1-noBG | Best epoch | Stopped at | Reason |
|---|---|---|---|---|---|
| 1 | 128,322 | 0.5759 | 21 | 36 | Val F1 patience |
| 2 | 168,931 | 0.7535 | 36 | 39 | Train loss plateau |
| 3 |  76,740 | 0.7146 | 18 | 33 | Val F1 patience |
| 4 | 136,787 | 0.6821 | 39 | 54 | Val F1 patience |
| 5 | 146,349 | 0.6827 | 21 | 31 | Train loss plateau |

**Aggregate (val, median ± std across folds):**

```
Val F1-noBG  : 0.6827 ± 0.0667
Val Sens     : best fold 4 = 0.8301, worst fold 1 = 0.7810
Val Spec     : 0.9444–0.9735 across folds
Best epoch   : 18–39 (high variance)
```

**1D-CNN CE vs UFL comparison:**

| Fold | CE | UFL | Δ |
|---|---|---|---|
| 1 | **0.5797** | 0.5759 | −0.38pp |
| 2 | **0.7324** | 0.7535 | +2.11pp |
| 3 | **0.7417** | 0.7146 | −2.71pp |
| 4 | 0.6716 | **0.6821** | +1.05pp |
| 5 | **0.7032** | 0.6827 | −2.05pp |
| **Median** | **0.7032** | 0.6827 | **−2.05pp** |

CE has the higher median for 1D-CNN (0.7032 vs 0.6827). UFL
improves fold 2 (+2.11pp) and fold 4 (+1.05pp) but hurts folds
3 and 5. The same fold-composition dependency seen in the Fabelo
model applies to 1D-CNN — UFL benefits folds with representative
mixed-campaign val sets.

**Longest run — fold 4 runs 54 epochs:**
The longest single fold training in the entire experiment so far.
UFL's focal weighting keeps finding marginal improvements in fold
4's C3-heavy val set through epoch 39, before plateauing. This
contrasts with CE's fold 4 which peaked at epoch 15. UFL explores
more of the loss landscape on C3 patients.

**Reproducibility confirmed:**
Run 2.2 was executed twice — both runs identical across all folds.

---

### Phase 2 Complete — Summary

**All 1D-CNN runs:**

| Run | Model | Loss | Median F1-noBG | Std | Best fold | Worst fold |
|---|---|---|---|---|---|---|
| 2.1 | 1D-CNN | CE  | 0.7032 | 0.0596 | F3: 0.7417 | F1: 0.5797 |
| 2.2 | 1D-CNN | UFL | 0.6827 | 0.0667 | F2: 0.7535 | F1: 0.5759 |

**Cross-tier comparison (CE, all spectral models):**

| Run | Model | Median F1-noBG | Best fold | Worst fold |
|---|---|---|---|---|
| 1.1 | 1D-NN-Fabelo | **0.7045** | F2: 0.7633 | F1: 0.5748 |
| 1.4 | 1D-NN        | 0.6760 | F2: 0.8672 | F1: 0.6080 |
| 2.1 | 1D-CNN       | 0.7032 | F3: 0.7417 | F1: 0.5797 |

**Key Phase 2 findings:**

1. **1D-CNN ties Fabelo on median** — 0.7032 vs 0.7045 (0.13pp
   gap). Local spectral convolution provides no clear advantage
   over the global FC Fabelo baseline on median performance.

2. **1D-CNN wins on fold 3** — +4.47pp over Fabelo. Local
   convolution better handles high TT intra-class variance
   (fold 3 TT Var=0.1170, highest of any fold).

3. **UFL hurts 1D-CNN on median** — −2.05pp vs CE. The same
   fold-composition dependency applies across all spectral models.

4. **No clear spectral Conv advantage over spectral FC** —
   the Fabelo model with 4,936 parameters matches a 76,824
   parameter CNN on median. For pixel-level classification,
   inductive bias toward simple global spectral mappings
   is as effective as local spectral convolutions.

---