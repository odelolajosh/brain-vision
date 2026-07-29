# Summary: "Hyperspectral Imaging Benchmark Based on Machine Learning for Intraoperative Brain Tumour Detection" (Leon et al., 2023, *npj Precision Oncology*)

## Aim
Neurosurgeons removing brain tumours must maximize resection while sparing healthy tissue, but visually distinguishing tumour margins is difficult. The paper builds on the **HELICoiD project** and extends prior proof-of-concept work by benchmarking multiple machine learning (ML) and deep learning (DL) approaches on a larger in-vivo hyperspectral imaging (HSI) dataset, using a more rigorous k-fold cross-validation, to support real-time intraoperative decision-making.

## Methods
Here's a structured outline of the Methods section:

### 1. Study Population

- Patients over 18 years with primary or secondary brain tumours undergoing surgery at University Hospital of Gran Canaria Doctor Negrín, Spain
- Three data acquisition campaigns:
    - Campaign 1: March 2015 – June 2016
    - Campaign 2: October 2016 – April 2017
    - Campaign 3: July 2019 – October 2019
- Written informed consent obtained from all participants
- Ethics approved by Research Ethics Committee (Ref 130069 for C1+C2, Ref 2019-001-1 for C3)

### 2. Study Procedure

```
Step 1: Craniotomy performed using IGS neuronavigation
Step 2: Durotomy to expose brain surface
Step 3: HS acquisition system positioned over exposed brain
Step 4: Rubber ring markers placed on tumour and normal tissue areas
         (guided by IGS neuronavigation information)
Step 5: Tumour tissue resected for neuropathological evaluation
Step 6: Additional HS images captured during resection where possible
```


### 3. In-vivo Hyperspectral Brain Database

- HS images manually cropped to select region of interest (parenchymal area)
- Labelled using neuropathologist information + operating surgeon knowledge
- Semi-automatic labelling tool based on SAM algorithm
- Four classes established:
    - TT — Tumour Tissue (red)
    - NT — Normal Tissue (green)
    - BV — Blood Vessel (blue)
    - BG — Background (black)
- White pixels = non-labelled (only high-confidence pixels labelled)
- 61 HS images from 34 patients included after excluding inadequate acquisitions



### 4. Clinical Data Collection

- 61 HS images from 34 adult patients
- Age range: 30–73 years, median 51.5 years
- 21 males, 13 females
- Primary tumours: 28 patients (82.4%)
    - G4 most frequent (44.1%, n=15)
    - G1 and G2 (14.7% each, n=5)
    - G3 (8.8%, n=3)
- Secondary tumours: 6 patients (17.6%)
    - Breast carcinoma (n=3)
    - Lung adenocarcinoma and carcinoma (n=2)
    - Renal carcinoma (n=1)
- Most common tumour location: right temporal lobe (23.5%, n=8)



### 5. Intraoperative Hyperspectral Acquisition System

- VNIR pushbroom camera (Hyperspec® VNIR A-Series, Headwall Photonics)
- Spectral range: 400–1000nm, 826 spectral channels
- Spectral resolution: 2–3nm
- Maximum spatial resolution: 741 × 1004 pixels
- Illumination: 150W QTH lamp coupled to fibre optic cold light illuminator
- Working distance: 40cm from lens to brain surface
- Pixel size: 128.7µm
- Maximum acquisition time: 60 seconds



### 6. HS Data Pre-processing

Five sequential steps:

```
Step 1 — Calibration
  CI = (RI - DI) / (WI - DI)
  CI = calibrated image
  RI = raw image
  WI = white reference (99% reflectance Spectralon tile)
  DI = dark reference (shutter closed)
  Purpose: remove ambient illumination and sensor dark currents

Step 2 — Noise smoothing
  Moving average filter, window = 5 data points
  Purpose: reduce high-frequency sensor noise

Step 3 — Extreme band removal
  Remove first 56 bands + last 126 bands
  Retains 644 channels: 440.5–909.1nm operating bandwidth
  Purpose: remove low-capability sensor edges with high noise

Step 4 — Spectral decimation
  Optimal sampling interval: 3.61nm
  Reduces 644 channels → 128 channels
  Purpose: reduce redundancy and computational cost
           without losing diagnostic performance

Step 5 — Normalisation
  Per-pixel min-max normalisation to [0, 1]
  Purpose: remove intensity differences due to non-flat
           brain surface — classify on spectral shape only
```



### 7. Supervised Classification Algorithms

Eight algorithms evaluated across two categories:

**ML and DL algorithms:**

| Algorithm | Key hyperparameter | Implementation |
|---|---|---|
| SVM-L | Cost C | LIBSVM library |
| SVM-RBF | Cost C + gamma γ | LIBSVM library |
| RF | Number of trees T | MATLAB Statistics Toolbox |
| KNN-E | Number of neighbours N (Euclidean) | MATLAB Statistics Toolbox |
| KNN-C | Number of neighbours N (Cosine) | MATLAB Statistics Toolbox |
| DNN | Hidden layer output size L | MATLAB Deep Learning Toolbox |

**DNN architecture specifically:**
- Two hidden layers + batch normalisation
- ReLU activation
- Learning rate: 0.1
- Training: 300 epochs
- Hidden layer size L: optimised as hyperparameter

**Unmixing-based algorithms:**

| Algorithm | Type |
|---|---|
| EBEAE | Linear blind end-member and abundance extraction |
| NEBEAE | Nonlinear version of EBEAE |

EBEAE/NEBEAE settings: 2 endmembers for NT and TT, 1 for BV, 3 for BG; ρ=0.3 for NT, 0.2 for TT, 0.01 for BG.



### 8. Three-way Data Partition and K-Fold Cross-validation

```
Patient-level split:
  Training   : 60% of patients
  Validation : 20% of patients
  Test       : 20% of patients

Cross-validation:
  5 independent folds created
  Each fold has a different random 60/20/20 partition
  Patients used as instances (one patient may have multiple images)

Hyperparameter optimisation:
  Performed independently per fold using validation set
  Metric: macro F1-Score excluding BG class
  Coarse search strategy

Final evaluation:
  Test set used for quantitative results only
  Never seen during training or hyperparameter optimisation
```



### 9. Training Data Reduction

```
Method: K-Means based pixel reduction (from Martinez et al. 2019)

Step 1: Apply K-Means clustering independently per class
        K = 100 clusters per class (400 total across 4 classes)

Step 2: For each of the 100 centroids per class:
        Identify n most similar real pixels using SAM distance

Step 3: Three training set sizes evaluated:
        n=10  → 1,000 pixels per class  (4,000 total)
        n=20  → 2,000 pixels per class  (8,000 total)
        n=40  → 4,000 pixels per class  (16,000 total)

Result: 1,000 pixels per class selected as optimal
        — no significant performance difference at larger sizes
        — ~20% improvement in TT accuracy vs full training set
        — ~48× speedup in training time

Validation and test sets: full labelled pixel sets used
                          (no reduction applied)
```



### 10. Proposed Processing Framework for TMD Map Generation

Five-stage spatial-spectral pipeline:

```
Stage 1 — Dimensionality reduction
  PCA applied to pre-processed HS image
  Output: 1-band grayscale representation

Stage 2 — Supervised spectral classification
  Any of the 8 algorithms applied to 128-band spectra
  Output: per-pixel class probability maps + classification map

Stage 3 — KNN spatial filtering (Spatial/Spectral stage)
  Parameters: λ=1, K=40 neighbours, window=8 rows, Euclidean distance
  Inputs: probability maps from Stage 2 + PCA map from Stage 1
  Output: smoothed KNN-Filtered classification map
  Purpose: reduce granularity, incorporate spatial continuity

Stage 4 — Unsupervised segmentation
  Hierarchical K-Means (HKM), K=24 clusters
  Output: unsupervised segmentation map of 24 spectral regions

Stage 5 — Majority Voting (MV)
  Merges KNN-Filtered map (Stage 3) with HKM map (Stage 4)
  Each cluster assigned the majority class within it
  Output: TMD (Three Maximum Density) map

TMD map visualisation:
  R channel = % TT pixels per cluster
  G channel = % NT pixels per cluster
  B channel = % BV pixels per cluster
  BG always rendered black
  Pure red = confident TT, pure green = confident NT
  Purple = TT+BV mixture (hypervascularised tumour)
```



### 11. Performance Metrics

Four metrics computed:

```
Macro F1-Score = mean of per-class F1 across classes
                 BG class EXCLUDED from computation
                 Primary reported metric

Overall Accuracy (OA) = (TP + TN) / (TP + TN + FP + FN)

Sensitivity = TP / (TP + FN)
              per-class and average reported

Specificity = TN / (TN + FP)
              per-class and average reported

Statistical testing:
  Paired two-sided Wilcoxon Rank Sum test
  5% significance level
  Used for spectral characterisation comparisons
```


## Results
- **Spectral characterization:** Statistically significant differences (Wilcoxon rank-sum, p<0.05) were found between TT vs. NT and TT vs. BV spectral signatures across channels, though high inter-patient variability was noted.
- **Spectral-only classification (validation set):** SVM-based and DNN methods gave the best macro F1-scores; unmixing methods (EBEAE/NEBEAE) performed worse and produced more false positives/negatives. SVM-RBF had the highest overall accuracy (~91.5%), while DNN had the best tumour sensitivity (~65.9%). Specificity exceeded 90% for ML/DL methods.
- **Adding spatial information:** Improved macro F1-Score medians (by 0.4–7.7%) and reduced variability for most classifiers (though not statistically significant vs. spectral-only). SVM-RBF achieved the highest overall accuracy (92.3±4.6%), while DNN again had the best tumour sensitivity (68.9±14.3%), closely followed by linear SVM.
- **Final test-set performance:** The DNN-based spatial/spectral framework achieved the study's headline result: **median macro F1-Score of 70.2 ± 7.9%**, about 3.6% lower than validation performance (expected generalization drop).
- **Qualitative results:** The system successfully highlighted high-grade gliomas, low-grade tumours, and secondary tumours, and clearly delineated vascular structures on the brain surface.

## Discussion
- The framework demonstrates HSI's promise as a **real-time, non-invasive intraoperative decision-support tool** capable of identifying diverse tumour types (not just glioblastoma, as in earlier work), using a more realistic and robust cross-validation than most prior studies.
- The authors contrast their results with other published HSI brain-tumour studies (e.g., Fabelo et al., Urbanos et al., Mühle et al., Puustinen et al.), noting that many prior works reported unrealistically high accuracies (96–99%) due to very small datasets, single-image proof-of-concepts, or leave-one-patient-out/intra-patient validation schemes that overstate generalizability. This paper's larger, more diverse dataset and stricter patient-level 5-fold validation give a more realistic, if lower, benchmark (~70% macro F1).
- **Limitations acknowledged:** Suboptimal acquisition conditions (poor focus/illumination, non-flat brain surface, pushbroom sensor constraints) degraded spectral signatures in some images (e.g., deep-layer tumours), though the DNN was relatively robust to some of these issues. High inter-patient and inter-tumour-type variability also limits performance.
- The paper positions itself as an important **benchmark reference** for future HSI-based brain tumour detection research, with the DNN plus spatial-spectral fusion (KNN filtering + HKM + majority voting) representing the best-performing, most clinically realistic pipeline to date.