# Dynamic Loss-Driven Hyperspectral Brain Tumour Classification  

**Student**: Joshua Odelola (190407028)  
**Supervisor**: Prof. Fashanu  
**Institution**: University of Lagos, Systems Engineering  
**Last updated**: 13 July 2026

---

## Overview

This project benchmarks five deep learning architectures against four loss
functions for pixel-level brain tissue classification using the HELICoiD
in-vivo hyperspectral brain database. The primary research question is whether
dynamic loss functions (Focal Loss, Dice Loss, Unified Focal Loss) improve
tumour tissue sensitivity over Cross-Entropy in the presence of severe class
imbalance. A secondary question is whether spatial architectures outperform
purely spectral ones for the clinically critical NT/TT discrimination task.

---

## Training Design

### Architectures
Five architectures are benchmarked:

| ID | Model | Type | Input | Params |
|---|---|---|---|---|
| 1D-NN | Baseline1DDNN | Spectral FC | (B, 128) | 4,936 |
| 1D-CNN | HuEtAl1DCNN | Spectral Conv | (B, 128) | 76,824 |
| 2D-CNN | LeeEtAl2DCNN | Spatial Conv | (B, 128, 5, 5) | 296,580 |
| 3D-CNN | HamidaEtAl3DCNN | Spatio-spectral Conv | (B, 128, 5, 5) | 33,004 |
| HybridSN | HybridSN | 3D+2D Hybrid | (B, 128, 5, 5) | 2,601,588 |
| SF | SpectralFormer | Transformer | (B, 128) | ~110,000 |

### Loss Functions
Four loss functions are evaluated per architecture:

| ID | Loss | Reference |
|---|---|---|
| CE | Cross-Entropy | Baseline control |
| FL | Focal Loss (γ=2.0) | Lin et al. (2018) |
| DL | Dice Loss (ε=1e-6) | Standard |
| UFL | Unified Focal Loss (λ=0.5, δ=0.6) | Yeung et al. (2022) |

Total experiments per validation strategy: 6 architectures × 4 loss
functions = **24 runs**.

---

## Validation Strategies

Four validation procedures are implemented to evaluate robustness across
different data partitioning schemes. All splits are performed at patient
level — never at pixel level — to prevent data leakage between patients.

### VP1 — Campaign-level split
```
Train + Val : Campaign 1 + Campaign 2  (80/20 patient-level split)
Test        : Campaign 3 (held out entirely)
```
Rationale: simulates clinical deployment across acquisition setups.
Training on one set of acquisition conditions and testing on an entirely
different campaign (2.5 years later, different equipment calibration) is
the strictest generalisation test. If a model performs well under VP1 it
is genuinely robust to acquisition variability.

### VP2 — Stratified random split
```
Train : 60% of patients (sampled equally from all 3 campaigns)
Val   : 20% of patients (sampled equally from all 3 campaigns)
Test  : 20% of patients (sampled equally from all 3 campaigns)
```
Rationale: maximises training data by including all campaigns. Equal
sampling per campaign ensures no campaign dominates any split. Single
run — no fold repetition.

### VP3 — K-Fold cross-validation (campaign-separated)
```
Train + Val : Campaign 1 + Campaign 2 (K=5 patient-level folds)
Test        : Campaign 3 (fixed, held out across all folds)
```
Rationale: provides variance estimate on train/val performance while
maintaining the strict campaign-3 test holdout. Results reported as
median ± std across 5 folds.

### VP_Fabelo — Replication of Fabelo et al. (2023)
```
All campaigns pooled → random 60/20/20 patient-level split
5 independent folds with different random seeds per fold
Results reported as median ± std across 5 folds
```
Rationale: directly replicates the validation methodology of Leon et al.
(2023), npj Precision Oncology, enabling direct numerical comparison of
our results against their published benchmark (70.2 ± 7.9% median macro
F1 using DNN + spatial/spectral approach). Each fold uses an independent
random seed (SPLIT_SEED + fold_index) to produce genuinely independent
partitions rather than rotating folds from a fixed shuffle.

Reference:
> Leon, R., Fabelo, H., et al. (2023). Hyperspectral imaging benchmark
> based on machine learning for intraoperative brain tumour detection.
> npj Precision Oncology, 7, 119.
> https://doi.org/10.1038/s41698-023-00475-9

---

## Training Data Reduction for Pixel-Based Models

### Motivation
A review of Fabelo et al. (2023) revealed that training data were not
used in full. The raw training set contains severe class imbalance:

```
NT  : ~193,476 pixels  (≈ 43%)
TT  :  ~19,966 pixels  (≈  4%)   ← clinically critical
BV  :  ~80,158 pixels  (≈ 18%)
BG  : ~157,332 pixels  (≈ 35%)
```

Training on this imbalanced full set causes the model to systematically
underfit TT — the most clinically important class. Initial experiments
confirmed near-zero TT sensitivity (0.0076–0.1585) across all 1D-NN
configurations trained on the full pixel set.

Fabelo et al. report that TT accuracy improved by up to 20% when using
their K-Means reduced balanced training set. This motivated adopting
their reduction procedure as a standard preprocessing step for all
pixel-based models.

### Method
Following Fabelo et al. (2023), for each class independently:

1. Apply K-Means clustering (K=100) to all labelled training pixels
   of that class to find 100 spectral prototypes (centroids).
2. For each centroid, identify the `n` most spectrally similar real
   pixels using the Spectral Angle Mapper (SAM) distance metric.
   SAM is preferred over Euclidean distance as it is invariant to
   illumination-induced amplitude differences.
3. Retain only those `n × 100` pixels per class.

With `n=10`: **1,000 pixels per class → 4,000 total training pixels**,
perfectly balanced across NT, TT, BV, BG.

### Scope
- ✅ Applied to pixel-based models: **1D-NN, 1D-CNN, SpectralFormer**
- ❌ Not applied to patch-based models: **2D-CNN, 3D-CNN, HybridSN**

Patch-based models extract 5×5 spatial neighbourhoods around each
labelled pixel. Reducing the labelled pixel set before patch extraction
would discard valid spatial context. Since each patch already encodes
unique spatial information, pixel-level redundancy is lower. This is a
deliberate methodological difference, noted as a limitation.

Reduction is applied to the **training set only**. Validation and test
sets always use the full labelled pixel set for representative evaluation.

### Configuration
```python
REDUCE_PIXELS      = True
N_PIXELS_PER_CLASS = 1000   # n × 100 clusters = 1000 pixels per class
```

---

## Model Selection Criterion — No-BG Macro F1

### Motivation
Fabelo et al. (2023) explicitly exclude the Background class from their
macro F1 computation when selecting optimal hyperparameters:

> "The optimal hyperparameters were selected using the best macro
> F1-Score result of each fold **without considering the BG class**."

Background pixels are easy to classify (non-tissue, instruments,
draping) and including them inflates the macro F1 metric, masking poor
performance on the clinically relevant tissue classes (NT, TT, BV).

### Implementation
Two F1 values are computed at every validation epoch:

```python
# All 4 classes — for internal cross-architecture comparison
macro_f1       = f1_score(targets, preds, average='macro')

# NT, TT, BV only — matches Fabelo's benchmark metric
macro_f1_no_bg = f1_score(targets, preds, labels=[0, 1, 2], average='macro')
```

The following decisions are all based on `macro_f1_no_bg`:
- **LR scheduler** — `ReduceLROnPlateau.step(val_f1_no_bg)`
- **Early stopping** — fires when `val_f1_no_bg` does not improve
- **Checkpointing** — saves model at best `val_f1_no_bg` epoch
- **Reported results** — primary metric for comparison vs Fabelo

`macro_f1` (all classes) is also logged at every epoch and stored in
history for internal analysis and cross-architecture comparison where
consistency across all four classes is needed.

---

## Optimiser and Scheduler

```
Optimiser    : Adam (lr=1e-3, weight_decay=0)
Scheduler    : ReduceLROnPlateau
                 mode     = 'max'      (monitors val_f1_no_bg)
                 factor   = 0.5        (halve LR on plateau)
                 patience = 5 epochs   (before decaying LR)
Early stop   : patience = 15 epochs   (monitors val_f1_no_bg)
Max epochs   : 100
Batch size   : 64
```

Note: Fabelo et al. use LR=0.1 and 300 epochs for their DNN, trained
with MATLAB's Deep Learning Toolbox. We use Adam with LR=1e-3 for all
architectures for consistency across the benchmark. The 1D-NN Fabelo
exact replication uses SGD with LR=0.1 and 45 epochs (multiclass
setting) to match the original paper as closely as possible.

---

## Class Weights

Class weights are computed from the **training set only** using inverse
frequency weighting:

```python
weight_c = 1 / (count_c + 1e-6)
weights  = weights / weights.sum()   # normalise to sum to 1
```

Weights are passed to CE and FL loss functions. DL is naturally robust
to imbalance and does not use explicit weights. UFL uses weights in its
Focal component only.

Weights are recomputed per fold since training set composition changes
across folds.

---

## Key Findings (updated as experiments complete)

| Run | Strategy | Best Val F1 (no BG) | TT Sens | Notes |
|---|---|---|---|---|
| 1dnn_ce_vp1 | VP1 | 0.6976 | 0.1585 (test) | Overfit epoch 3 |
| 1dnn_fl_vp1 | VP1 | 0.3778 | — | Overfit epoch 1 |
| 1dnn_dl_vp1 | VP1 | 0.0071 | — | Complete failure |
| 1dnn_ufl_vp1 | VP1 | 0.7179 | 0.0076 (test) | Overfit epoch 1 |
| 1dnn_ce_vp1 (Fabelo exact) | VP1 | 0.7327 | — | Overfit epoch 1 |
| 1dcnn_ufl_vp1 | VP1 | 0.7871 | — | Stable training to epoch 31 |

---

## Open Questions / To-Do

- [ ] Run all 24 experiments under VP_Fabelo strategy
- [ ] Apply K-Means pixel reduction and re-run 1D-NN and 1D-CNN
- [ ] Evaluate all completed runs on test set (per-class metrics)
- [ ] Implement LOPO for strict 1D-DNN Fabelo replication
- [ ] Add inference latency measurement for Pareto analysis
- [ ] Confirm whether VP1 test patients include TT-labelled images

---

## References

1. Leon, R., Fabelo, H., et al. (2023). Hyperspectral imaging benchmark
   based on machine learning for intraoperative brain tumour detection.
   *npj Precision Oncology*, 7, 119.
   https://doi.org/10.1038/s41698-023-00475-9

2. Fabelo, H., et al. (2019). Deep learning-based framework for in vivo
   identification of glioblastoma tumor using hyperspectral images of
   human brain. *Sensors*, 19, 920.
   https://doi.org/10.3390/s19040920

3. Lin, T.-Y., et al. (2018). Focal loss for dense object detection.
   *IEEE TPAMI*, 42(2), 318–327.
   https://doi.org/10.1109/TPAMI.2018.2858826

4. Yeung, M., et al. (2022). Unified focal loss: Generalising dice and
   cross entropy-based losses to handle class imbalanced medical image
   segmentation. *Neural Networks*, 154, 1–9.
   https://doi.org/10.1016/j.neunet.2021.11.015

5. Roy, S. K., et al. (2020). HybridSN: Exploring 3D-2D CNN feature
   hierarchy for hyperspectral image classification.
   *IEEE Geoscience and Remote Sensing Letters*, 17(2), 277–281.
   https://doi.org/10.1109/LGRS.2019.2918719

6. Hong, D., et al. (2021). SpectralFormer: Rethinking hyperspectral
   image classification with transformers.
   *IEEE TGRS*, 60, 1–14.
   https://doi.org/10.1109/TGRS.2021.3130716