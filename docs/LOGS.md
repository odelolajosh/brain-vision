## July 2026 — Dataset & Preprocessing Decisions

### HELICoiD dataset access
Obtained access to all three HELICoiD campaigns:
- Campaign 1: 36 images, 22 patients (March 2015 – June 2016)
- Campaign 2: additional patients (October 2016 – April 2017)
- Campaign 3: additional patients (July 2019 – October 2019)

**Decision**: Use benchmark subset of Campaign 1 only (images 004-02 to
022-03) as specified by Fabelo to replicate published benchmark results.
Campaigns 2 and 3 used in full.

### Preprocessing pipeline — follows Fabelo et al. (2023) exactly
Five-step pipeline implemented in `02_preprocessing.ipynb`:

1. Calibration: `R = (raw - dark) / (white - dark)`
2. Smoothing: moving average, window=5
3. Band removal: drop first 56 + last 126 bands → 644 retained
4. Decimation: uniform sampling → 128 bands (3.61nm interval)
5. Normalisation: per-pixel Min-Max to [0, 1]

**Decision**: Chose spectral decimation over PCA for dimensionality
reduction. Rationale: (a) matches Fabelo's benchmark preprocessing
exactly for comparable results; (b) preserves physical wavelength meaning
of each band; (c) no data leakage risk (no fitted parameters); (d) PCA
is common in HybridSN/SpectralFormer papers but benchmark comparability
takes priority since loss functions are the research variable, not
preprocessing.

### Processed data saved as .npz
Preprocessed cubes saved to disk as compressed `.npz` files to avoid
re-running the expensive preprocessing pipeline each Kaggle session.
Structure: `data/processed/{campaign_name}/{patient_id}.npz`

Each file contains:
- `processed`: `(H, W, 128)` float32
- `labels`: `(H, W)` int32

---

## July 2026 — Architecture & Model Decisions

### Architecture selection — DeepHyperX as primary source
The initial set of deep learning architectures was drawn from the
DeepHyperX repository (Acquarelli et al.), a well-established open-source
benchmark suite for hyperspectral image classification. DeepHyperX
provided implementations across the major architecture families needed
for this benchmark:

- **1D-CNN**: HuEtAl (Hu et al., 2015) — spectral-only convolutions
- **2D-CNN**: LeeEtAl (Lee & Kwon, 2016) — spatial Inception + residual
- **3D-CNN**: HamidaEtAl (Hamida et al., 2018) — spatio-spectral convolutions
- **HybridSN**: Roy et al. (2020) — 3D-CNN followed by 2D-CNN

These architectures were selected because they span the full complexity
spectrum from lightweight spectral classifiers to heavy spatio-spectral
models, enabling a fair evaluation of whether dynamic loss functions
provide consistent benefit regardless of architectural complexity.

SpectralFormer (Hong et al., 2021) was added separately as it was not
part of DeepHyperX at the time of selection, representing the
transformer-based direction in HSI classification.

### Fabelo's papers inspired simpler baseline architectures
Following a detailed review of Fabelo et al. (2019) and Fabelo et al.
(2023) — the primary benchmark papers for this project — two additional
simpler architectures were added to enable faithful replication of their
published results:

**FabeloDNN** — from Fabelo et al. (2019), Sensors:
A minimal two-hidden-layer 1D-DNN (28→40 nodes, ReLU, no BatchNorm,
no dropout) trained with SGD LR=0.1 for 45 epochs using LOPO
cross-validation. This is the simplest spectral classifier in the
benchmark and serves as the weakest baseline.

**FabeloCNN** — from Fabelo et al. (2019), Sensors:
An AlexNet-inspired 2D-CNN operating on 11×11 spatial patches, trained
with AdaDelta LR=1.0 for 50 epochs. Fabelo used this to demonstrate that
spatial context improves over purely spectral classification.

These two architectures are evaluated separately under LOPO
cross-validation to produce results directly comparable to Fabelo's
published numbers. The main 24-run benchmark uses the DeepHyperX-inspired
architectures.

### Full model set

| Model | Type | Params | Source |
|---|---|---|---|
| Baseline1DDNN | Spectral FC | 17,055,748 | Custom expanded |
| FabeloDNN | Spectral FC | 4,936 | Fabelo et al. (2019) |
| HuEtAl1DCNN | Spectral Conv | 76,824 | DeepHyperX / Hu et al. (2015) |
| LeeEtAl2DCNN | Spatial Conv | 296,580 | DeepHyperX / Lee & Kwon (2016) |
| FabeloCNN | Spatial Conv | — | Fabelo et al. (2019) |
| HamidaEtAl3DCNN | Spatio-spectral | 33,004 | DeepHyperX / Hamida et al. (2018) |
| HybridSN | 3D+2D Hybrid | 2,601,588 | DeepHyperX / Roy et al. (2020) |
| SpectralFormer | Transformer | ~110,000 | Hong et al. (2021) |

### HybridSN padding adjustment
HybridSN padding changed from original (no spatial padding) to
`padding=(0,1,1)` on all 3D conv layers to preserve spatial dimensions
with `PATCH_SIZE=5`. Original architecture collapses spatial dims to zero
after three 3D conv layers at this patch size.

### SpectralFormer integration
Implemented `SpectralFormer` wrapper around the original `ViT` backbone
from Hong et al. (2021). Key design decision: use `unfold` with
`near_band=3` to create overlapping spectral token groups. SpectralFormer
uses `HSIPixelDataset` — not a patch model.

---

## July 2026 — Validation Strategy Decisions

### Five validation procedures implemented
After reviewing Fabelo et al. (2023) methodology in detail:

**VP1 — Campaign-level split**
Train+Val: Campaign 1+2, Test: Campaign 3 (held out entirely).
Strictest generalisation test — different acquisition periods.

**VP2 — Stratified random split**
60/20/20 across all campaigns, equal per-campaign sampling, single run.

**VP3 — K-Fold cross-validation**
K=5 folds on Campaign 1+2, Campaign 3 fixed test.

**VP_Fabelo — Replication of Fabelo et al. (2023)**
All campaigns pooled, 60/20/20 patient-level split, 5 independent folds
with different seeds per fold. Results reported as median ± std.
Primary comparison strategy against Fabelo's 70.2 ± 7.9%.

**LOPO — Leave-One-Patient-Out**
Follows Fabelo et al. (2019). One patient held out per fold, Campaign 3
fixed test. Used specifically for FabeloDNN and FabeloCNN replication.

**Decision**: All splits are patient-level — never pixel-level.

---

## July 2026 — Training Decisions

### Macro F1 excluding BG — follows Fabelo (2023)
Fabelo et al. (2023) explicitly exclude Background class from macro F1:

> "The optimal hyperparameters were selected using the best macro
> F1-Score result of each fold without considering the BG class."

**Decision**: Checkpointing, LR scheduling, and early stopping all
monitor `val_f1_no_bg` (NT, TT, BV only). Both metrics logged per epoch.
`macro_f1_no_bg` is the primary reported metric vs Fabelo's benchmark.

### Optimiser and scheduler configuration
```
Optimiser   : Adam (lr=1e-3)              — all models except Fabelo replications
Scheduler   : ReduceLROnPlateau
                mode='max', factor=0.5, patience=5 epochs
Early stop  : patience=15 epochs
Max epochs  : 100
Batch size  : 64

FabeloDNN   : SGD lr=0.1, max 45 epochs
FabeloCNN   : AdaDelta lr=1.0, max 50 epochs
```

### Class weights
Computed from training set only using inverse frequency weighting.
Recomputed per fold. Applied to CE and FL. DL uses no weights. UFL
applies weights to Focal component only.

---

## July 2026 — Early Experimental Results

### 1D-NN results (VP1)
All four loss functions showed catastrophic overfitting — best epoch
always 1 or 3, val loss exploding immediately.

| Run | Best Epoch | Best Val F1 | TT Sens (test) |
|---|---|---|---|
| 1dnn_ce_vp1 | 3 | 0.6976 | 0.1585 |
| 1dnn_fl_vp1 | 1 | 0.3778 | — |
| 1dnn_dl_vp1 | 1 | 0.0071 | — (complete failure) |
| 1dnn_ufl_vp1 | 1 | 0.7179 | 0.0076 |
| 1dnn_ce_vp1 (FabeloDNN) | 1 | 0.7327 | — |

**Key findings**:
- DL (Dice Loss) completely failed — never exceeded val F1 of 0.0071
- Even the 4,936-parameter FabeloDNN overfit at epoch 1, identical to
  the 17M parameter version — confirms overfitting is caused by campaign
  generalisation gap, not parameter count
- CE outperformed UFL on TT sensitivity (0.1585 vs 0.0076) — UFL traded
  TT detection for BG accuracy, inflating OA while abandoning the
  clinically critical class

### 1D-CNN result (VP1, HuEtAl architecture)

| Run | Best Epoch | Best Val F1 | Training behaviour |
|---|---|---|---|
| 1dcnn_ufl_vp1 | 16 | 0.7871 | Stable — LR decayed 3× |

Significantly healthier training than all 1D-NN runs. 76K parameters
outperformed 17M parameter 1D-NN — confirms architectural inductive bias
(spectral convolutions) matters more than raw capacity.

---

## July 2026 — Dynamic Loss Findings & Balancing Decision

### Dynamic loss functions did not consistently improve training

Initial experiments across 1D-NN and 1D-CNN under VP1 showed that
dynamic loss functions (Focal Loss, Dice Loss, Unified Focal Loss) did
not provide consistent improvement over Cross-Entropy:

- **Dice Loss** failed entirely on the 1D-NN (val F1 = 0.0071) — the
  loss surface was too flat to provide useful gradient signal when input
  features are purely spectral and classes are highly overlapping
- **Focal Loss** peaked at epoch 1 (val F1 = 0.3778) then degraded —
  amplifying hard examples backfired when the entire dataset is hard
  due to NT/TT spectral similarity
- **UFL** achieved the highest val F1 (0.7179) but collapsed TT
  sensitivity to 0.0076 on the test set — worse than CE's 0.1585
- The improvement pattern was inconsistent across architectures —
  1D-CNN with UFL trained stably to epoch 16, but 1D-NN with UFL
  was best at epoch 1

The working hypothesis is that dynamic loss functions cannot compensate
for the fundamental class imbalance problem when the training set itself
is severely skewed (~10:1 NT:TT ratio). The loss function sees the
imbalance in every batch and the weighting alone is insufficient.

### Adopted downsampling balancing for equal class ratios

Following Fabelo et al. (2023) and (2019), downsampling was adopted as
the primary strategy to ensure equal class representation during training:

**For pixel-based models (1D-NN, 1D-CNN, SpectralFormer)**:
K-Means based pixel reduction — 100 clusters per class, SAM-based
selection of the n most representative real pixels per centroid.
Default: 1000 pixels per class → 4000 total training pixels, perfectly
balanced across NT, TT, BV, BG.

**For patch-based models (2D-CNN, 3D-CNN, HybridSN)**:
Random undersampling of centre pixels to the minority class (TT) count
before patch extraction. Applied inside `HSIPatchDatasetLazy.__init__`
before patches are extracted — only patches for selected centres are
materialised, keeping memory usage proportional to the balanced count.

**Rationale**: Balancing fixes the frequency problem — how often the
model sees each class per batch. Dynamic loss functions fix the difficulty
problem — how much attention to hard examples. Both address class
imbalance but from different angles. With a balanced training set, CE
loss may suffice for simpler architectures. The 2×2 experimental design
(balanced × loss function) will determine whether dynamic losses provide
additional benefit on top of balancing.

**Decision**: `REDUCE_PIXELS=True` is now the default for all models.
Run names encode the balancing decision:
e.g. `1dnn_ce_bal_fold1_vp_fabelo` vs `1dnn_ce_nobal_fold1_vp_fabelo`.

Fabelo reports up to 20% TT accuracy improvement with the balanced
reduced set — expected to address the near-zero TT sensitivity observed
in initial experiments.

### Data augmentation for patch-based models

To increase robustness of patch-based models and reduce overfitting,
on-the-fly spatial augmentation was added to `HSIPatchDatasetLazy`
following Fabelo et al. (2019):

> "The 2D-CNN was trained with a batch size of 12 patches, which were
> augmented to 96 patches during training by applying rotations and
> vertical mirroring to produce 800% augmentation."

Three augmentation operations are applied randomly during training:
- **Horizontal flip** — 50% probability
- **Vertical flip** — 50% probability
- **90° rotation** — random k ∈ {0, 1, 2, 3}

These operations are biologically valid — brain tissue patches are
equally representative at any rotation since the HSI camera can be
positioned at any angle relative to the surgical field. The label is
invariant to all spatial transforms.

Augmentation is applied in `__getitem__` at batch-fetch time —
no additional memory cost. Applied to training patches only. Val and
test datasets use `augment=False`.

**Additional motivation**: Augmentation indirectly helps with TT
imbalance — even after balancing, the model benefits from seeing each
TT patch in multiple orientations, effectively multiplying the diversity
of TT representations in training.

---

## July 2026 — Infrastructure Decisions

### Project structure
```
dylos-hsi/
├── constants.py
├── utils.py
├── NOTES.md
├── LOGS.md
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_training.ipynb
├── 04_evaluation.ipynb
├── brainvision/
│   ├── models/
│   ├── data/
│   └── validation/
├── data/
│   ├── first_campaign/
│   ├── second_campaign/
│   ├── third_campaign/
│   └── processed/
├── checkpoints/
└── results/
```

### Kaggle workflow
1. Preprocess once → `.npz` files promoted to Kaggle Dataset
   `brain-hsi-processed`
2. Code uploaded as Kaggle Dataset `brain-hsi-utils`
3. Training sessions load from `/kaggle/input/brain-hsi-processed/`
4. Campaigns dict cached to `/kaggle/working/cache/campaigns.pkl`
5. Splits cached per strategy — training and evaluation notebooks use
   identical partitions
6. Checkpoints promoted to Kaggle Dataset `brain-hsi-checkpoints`

### Notebook separation rationale

| Notebook | Owns | Writes |
|---|---|---|
| 01_eda.ipynb | Raw data analysis | Plots only |
| 02_preprocessing.ipynb | Preprocessing pipeline | `.npz` files |
| 03_training.ipynb | Splits, training loop | `.pt`, `_history.npy` |
| 04_evaluation.ipynb | Metrics, plots | `_test_metrics.npy`, plots |

---

## 14 July 2026 — Memory fix for patch-based models

Running 2D-CNN with Cross-Entropy and `vp_fabelo` validation strategy
caused Kaggle container to restart due to memory exhaustion.

**Root cause**: `HSIPatchDataset` pre-extracts all patches into a single
tensor at init time. With VP_Fabelo's large training set (~35 images) and
no balancing, this requires ~56 GB which crashes the container.

**Fix**: Replaced with `HSIPatchDatasetLazy` which stores only padded
cubes (~2.4 GB) and a list of centre pixel coordinates. Patches are
extracted on-the-fly in `__getitem__`. Memory drops from ~56 GB to
~2.4 GB.

| | Pre-extraction | Lazy extraction |
|---|---|---|
| Memory (unbalanced) | ~56 GB | ~2.4 GB |
| Memory (balanced) | ~1 GB | ~2.4 GB |
| Per-batch cost | None | One array slice per sample |

**Decision**: `HSIPatchDatasetLazy` is now the default for all patch-based
models. Switched 2D-CNN to `vp1` as an interim measure — will revert to
`vp_fabelo` once lazy dataset is confirmed stable.


## 15 July 2026 - 2D CNN is bad omen

The val loss increases during the training period.

### Validation loss keeps sky-rocketing for HyperDeep 1D-NN, VP1 and CE

Now, with a bit of clear conscious, I blinding training all possible configuration combination.

Next training plan, train all models using a constant configuration:

- No balancing
- Using vp_fabelo's validation strategy
- Using Cross Entropy loss function


---

## Open Items

- [ ] Complete all 24 runs (6 models × 4 loss functions) under VP_Fabelo
- [ ] Re-run 1D-NN and 1D-CNN with K-Means balancing enabled
- [ ] Run 2D-CNN under VP_Fabelo now that lazy dataset is implemented
- [ ] Run 3D-CNN, HybridSN under VP1 and VP_Fabelo
- [ ] Run SpectralFormer under VP_Fabelo
- [ ] Run FabeloDNN under LOPO for direct benchmark comparison
- [ ] Run FabeloCNN under LOPO for direct benchmark comparison
- [ ] Implement and run 2×2 balancing experiment (BALANCE flag)
- [ ] Add inference latency measurement for Pareto analysis
- [ ] Aggregate fold results and build master comparison table
- [ ] Confirm which VP1 test patients have TT-labelled pixels
- [ ] Begin Chapter 3 (Methodology) write-up

---

## References

1. Leon, R., Fabelo, H., et al. (2023). npj Precision Oncology, 7, 119.
   https://doi.org/10.1038/s41698-023-00475-9

2. Fabelo, H., Halicek, M., et al. (2019). Sensors, 19(4), 920.
   https://doi.org/10.3390/s19040920

3. Hu, W., et al. (2015). Deep Convolutional Neural Networks for
   Hyperspectral Image Classification. Journal of Sensors, 2015.
   https://doi.org/10.1155/2015/258619

4. Lee, H., & Kwon, H. (2016). Contextual Deep CNN Based Hyperspectral
   Classification. IGARSS 2016.

5. Ben Hamida, A., et al. (2018). 3-D Deep Learning Approach for Remote
   Sensing Image Classification. IEEE TGRS, 2018.
   https://doi.org/10.1109/TGRS.2018.2818945

6. Roy, S. K., et al. (2020). HybridSN. IEEE GRSL, 17(2), 277–281.
   https://doi.org/10.1109/LGRS.2019.2918719

7. Hong, D., et al. (2021). SpectralFormer. IEEE TGRS, 60, 1–14.
   https://doi.org/10.1109/TGRS.2021.3130716

8. Lin, T.-Y., et al. (2018). Focal Loss. IEEE TPAMI, 42(2), 318–327.
   https://doi.org/10.1109/TPAMI.2018.2858826

9. Yeung, M., et al. (2022). Unified Focal Loss. Neural Networks, 154.
   https://doi.org/10.1016/j.neunet.2021.11.015