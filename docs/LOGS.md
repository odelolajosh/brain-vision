# brainvision — Project Log

Student   : Odelola Oluwatolamise Joshua (190407028)\
Supervisor: Prof. Fashanu | UNILAG Systems Engineering\
Repo      : brainvision\
Updated   : 2026-07-29

---

## Methodology Progress

| Step | Notebook | Status |
|---|---|---|
| 01 Raw EDA | 01_eda_raw.ipynb | 🔄 In progress |
| 02 Preprocessing | 02_preprocessing.ipynb | ⏳ Pending |
| 03 Processed EDA | 03_eda_processed.ipynb | ⏳ Pending |
| 04 Campaign EDA | 04_eda_campaign.ipynb | ⏳ Pending |
| 05 Fold EDA | 05_eda_folds.ipynb | ⏳ Pending |
| 06 Training | 06_training.ipynb | ⏳ Pending |
| 07 Evaluation | 07_evaluation.ipynb | ⏳ Pending |

---

## 01 — Raw EDA (`01_eda_raw.ipynb`)

### Dataset Summary

| Campaign | Images | Patients | Date range |
|---|---|---|---|
| Campaign 1 | 27 | 16 | Mar 2015 – Jun 2016 |
| Campaign 2 | 24 | 10 | Oct 2016 – Apr 2017 |
| Campaign 3 | 10 | 8 | Jul 2019 – Oct 2019 |
| **Total** | **61** | **34** | |

Patient IDs:
- C1: 004, 005, 007, 008, 010, 012, 013, 014, 015, 016, 017, 018, 019, 020, 021, 022
- C2: 034, 035, 036, 037, 038, 039, 040, 041, 042, 043
- C3: 050, 051, 053, 054, 055, 056, 057, 058

---

### Finding 01-A — Data integrity ✅

All 61 patients pass raw integrity checks:
- Zero NaN values across all images
- Zero Inf values across all images
- All raw cubes have exactly 826 spectral bands (B dimension consistent)

**Implication**: No corrupted or incomplete ENVI files. Safe to proceed to preprocessing.

---

### Finding 01-B — Spatial dimensions vary by image (expected)

All 61 images have shape `(H, W, 826)` where H and W vary per image.
H range: 280–721 pixels | W range: 244–752 pixels.

Largest image : 058-02 (721×752) — Campaign 3
Smallest image: 034-03 (290×301) — Campaign 2

**Implication**: This is expected — each image is manually cropped to the
exposed parenchymal area. No fixed spatial resolution is required since
all preprocessing and training operations are pixel-wise. The variable
spatial size does NOT affect the preprocessing pipeline.

---

### Finding 01-C — Two anomalous acquisitions flagged

Patients 055-01 and 055-02 (Campaign 3) have unusually low DN Max values:

| Patient | DN Max | Expected |
|---|---|---|
| 055-01 | 1,286 | ~4,095 |
| 055-02 | 1,002 | ~4,095 |
| All others | 1,542–4,095 | ~4,095 |

This matches the acquisition problems described by Fabelo et al. (2023,
Discussion): *"Op55C1 and Op55C2... decrease in reflectance values in
the infrared region (700–900nm)"* attributed to sensor sensitivity
limitations and possible lens misalignment.

**Decision**: These images are retained in the dataset (Fabelo also
retained them) but flagged. Their anomalous spectral signatures in the
NIR range may cause misclassification. Results involving these patients
should be interpreted with caution.

---

### Finding 01-D — Severe TT label sparsity

Only 28 of 61 images (46%) contain any TT-labelled pixels.

**Images with TT**: 008-01, 008-02, 012-01, 012-02, 015-01, 020-01,
021-01, 021-02, 036-02, 038-01, 039-01, 039-02, 040-01, 040-02,
041-01, 041-02, 042-02, 042-03, 043-01, 043-02, 050-01, 053-01,
055-02, 056-01, 056-02, 057-01, 058-02, 034-02, 034-03, 035-02,
037-03, 037-04, 022-03

**Images without TT** (33 images): Pure normal brain acquisitions where
the tumour was either not exposed, was in a deep layer, or the patient
underwent surgery for another pathology (blood clot, epilepsy).

Missing class patterns:
- TT missing only          : 18 images — normal brain surface acquisitions
- NT + TT missing          : 4 images  — background/post-resection only
- BV missing               : 5 images  — blood vessels not labelled
- BG missing               : 3 images  — entire FOV is labelled tissue

**Implication**: The 33 TT-free images still contribute NT, BV, and BG
training signal. The K-Means pixel reduction (1000 px/class) at training
time mitigates the impact of these images having no TT contribution.
This finding directly motivates the use of dynamic loss functions and
class balancing as methodological choices.

**Notable sparse TT cases**:
- 008-02: only 138 TT pixels (very small tumour region)
- 021-02: only 32 TT pixels
- 021-01: only 240 TT pixels
- 041-01: only 69 TT pixels
- 041-02: only 158 TT pixels

These patients contribute minimal TT signal even when TT is present.

---

### Finding 01-E — NT:TT class imbalance

From the label coverage table, the global imbalance is severe.
Quantitative analysis pending `plot_class_distribution` output.

Preliminary estimate from row sums:
- NT is the dominant labelled tissue class
- TT is consistently the minority class
- This imbalance is the primary motivation for the loss function
  comparison experiment (CE vs FL vs DL vs UFL)

---

### ENVI Warning (non-critical)

```
UserWarning: Parameters with non-lowercase names encountered and
converted to lowercase.
```
Triggered by `spectral` library when reading ENVI headers. Non-critical —
data loads correctly. Can be suppressed with:
```python
import spectral
spectral.settings.envi_support_nonlowercase_params = True
```

---

*Log continues as EDA progresses...*

---

### Finding 01-F — Raw DN band profiles confirm band removal decision

Plot: `plot_raw_band_profiles`

All 61 patients show a smooth, consistent bell-shaped DN curve across
the retained spectral region (bands 56–700), peaking around band
400–500 (~640–720nm NIR range). This is physically consistent with
biological tissue reflectance under QTH illumination.

**Removed bands (0–56)**: Near-zero DN with a sharp spike at bands 5–10.
This is CCD sensor noise at the UV/blue spectral edge — band removal
is confirmed correct.

**Removed bands (700–826)**: Rapid drop-off to near-zero. CCD sensitivity
falling off at the NIR edge — band removal confirmed correct.

**Bimodal shape in retained region**: A visible shoulder at bands 200–250
(~540–560nm) before the main NIR peak. This corresponds to the
haemoglobin absorption region (HbO2 peaks at ~540nm and ~575nm,
deoxyHb at ~555nm) — consistent with Fabelo et al. (2023) Fig. 3
spectral characterisation. This feature is the key discriminative
signal between tissue classes.

**Outliers identified**:
- 035-01: highest DN values across all bands (~830 peak) — largest
  image with extensive tissue labelling
- 055-01, 055-02: lowest, flattest curves — confirms anomalous
  acquisition identified in Finding 01-C

---

### Finding 01-G — Dark reference reveals inter-campaign sensor variation

Plot: `plot_reference_profiles` (left panel)

The dark reference is NOT near-zero as expected for an ideal sensor.
Two distinct clusters are present:

| Cluster | DN range | Likely source |
|---|---|---|
| Upper | ~20–22 DN | Campaign 1 and/or Campaign 2 |
| Lower | ~15–16 DN | Campaign 3 subset |

**Interpretation**: The dark current (electronic baseline noise) differs
across campaigns, suggesting the sensor was operated at different
temperatures, gain settings, or was recalibrated between campaigns.

**Impact**: The calibration formula `R = (raw - dark) / (white - dark)`
explicitly subtracts the dark reference from each raw acquisition,
correcting for this campaign-level offset. The variable dark current
does NOT propagate into the preprocessed data — but it is additional
evidence that inter-campaign acquisition conditions differed
systematically, supporting the illumination drift hypothesis
established in the campaign EDA.

**Sharp downward spike at bands 0–10** visible in dark reference for
several patients — additional justification for removing the first 56
bands.

---

### Finding 01-H — White reference confirms inter-campaign illumination differences

Plot: `plot_reference_profiles` (right panel)

Two distinct amplitude groups visible in the white reference:

| Group | Peak DN | Interpretation |
|---|---|---|
| Upper | ~1,400–1,600 DN | Higher lamp intensity or better fibre alignment |
| Lower | ~800–1,000 DN | Lower lamp intensity or different operating setup |

**Shape is consistent** across all patients — smooth bell curve peaking
at band ~200–250. The QTH lamp spectral output shape is stable across
campaigns even if its absolute intensity varies.

**Impact**: The calibration step normalises these amplitude differences
out completely — `(raw - dark) / (white - dark)` divides by the white
reference amplitude, making all calibrated reflectance values
campaign-comparable in theory.

**Residual risk**: If the white reference tile was not perfectly flat,
had dust, or was positioned at a slightly different angle between
campaigns, small spectral shape differences may remain after
calibration. This is the most plausible mechanism for the
visible-range TT spectral drift observed between campaigns in the
campaign EDA (Finding 04-C, to be logged).

---

*Awaiting: plot_raw_class_spectra, plot_class_distribution outputs*

---

### Finding 01-I — Raw class spectral signatures confirm tissue separability

Plot: `plot_raw_class_spectra`

All four tissue classes are spectrally separable in the raw DN domain,
with the following ordering in the NIR peak region (bands 300–600):

| Class | Peak DN (approx) | Rank |
|---|---|---|
| NT | ~600 | Highest |
| TT | ~470 | 2nd |
| BG | ~420 | 3rd |
| BV | ~360 | Lowest |

**Physical interpretation**: BV has the highest haemoglobin content →
most light absorbed → lowest reflectance. NT has lowest haemoglobin
density → highest reflectance. TT sits below NT due to deoxyhaemoglobin
content from tumour hypoxia (Fabelo et al. 2023).

**Haemoglobin shoulder visible** at bands 200–250 (~540–560nm) in all
four classes as a distinct bump before the NIR peak. This corresponds
to HbO2 absorption peaks at ~540nm and ~575nm and deoxyHb at ~555nm.
LIME interpretability analysis (Fabelo 2023) confirmed these bands as
among the most discriminative for classification.

**NT vs TT overlap is substantial**: Despite clear mean separation,
the ±1 std bands of NT and TT cross throughout the full spectral range.
This inter-patient variability is the fundamental challenge for
spectral classification — the same tissue type looks different across
patients due to:
  - Different tumour grades (G1–G4) affecting vascularity
  - Acquisition illumination differences across campaigns
  - Non-flat brain surface affecting local reflectance

**BG has the widest variance** of all classes — reflects the
heterogeneous nature of background pixels (surgical drapes, instruments,
skull bone, dura, rubber markers) with fundamentally different spectral
signatures all labelled as one class.

**Removed bands confirm correct removal**: Both red zones (bands 0–56
and 700–826) show near-zero DN for all four classes. No class-
discriminative information is lost by removing these bands.

---

### Finding 01-J — Severe global class imbalance quantified

Plot: `plot_class_distribution`

Global pixel counts across all 61 patients:

| Class | Pixels | Percentage |
|---|---|---|
| Background (BG) | 445,540 | 49.3% |
| Normal Tissue (NT) | 308,328 | 34.1% |
| Blood Vessel (BV) | 112,407 | 12.4% |
| Tumour Tissue (TT) | 37,399 | 4.1% |

**Key imbalance ratios**:
- NT:TT = 8.2:1 — for every TT pixel there are 8.2 NT pixels
- BG:TT = 11.9:1 — background alone outnumbers TT by 12x
- Total:TT = 24.4:1 — TT represents only 1 in 24 labelled pixels

**This is the primary motivation for the experimental design**:
- Standard Cross-Entropy loss treats each pixel equally → gradient
  dominated by BG and NT → TT sensitivity near zero
- K-Means pixel reduction balances training to 1000 px/class →
  mitigates imbalance at the data level
- Dynamic loss functions (FL, DL, UFL) mitigate imbalance at the
  objective level
- Macro F1 excluding BG is the primary metric → prevents BG from
  inflating reported performance

**BG excluded from primary metrics**: Following Fabelo et al. (2023),
BG is excluded from macro F1-Score computation. BG pixels are
clinically irrelevant — neurosurgeons only need to distinguish NT,
TT, and BV in the surgical field.

---

### Raw EDA — Summary of Key Findings

| Finding | Category | Implication |
|---|---|---|
| 01-A | Integrity | Dataset clean — safe to preprocess |
| 01-B | Spatial | Variable H×W expected — no issue |
| 01-C | Acquisition | 055-01/02 anomalous — flag in results |
| 01-D | Labels | 33/61 images TT-free — motivates balancing |
| 01-E | Imbalance | NT:TT=8.2:1 — motivates dynamic losses |
| 01-F | Spectral | Band removal confirmed correct |
| 01-G | Sensor | Dark current varies by campaign — illumination drift evidence |
| 01-H | Sensor | White reference amplitude varies — calibration essential |
| 01-I | Class | NT/TT separable in mean but overlap in std — hard classification task |
| 01-J | Imbalance | TT=4.1% globally — CE loss clinically unsafe without compensation |

**Raw EDA status: COMPLETE ✅**
Next step: 02_preprocessing.ipynb

---

---

## 02 — Preprocessing (`02_preprocessing.ipynb`)

### Status: COMPLETE ✅

All 61 patients preprocessed and verified across all three campaigns.
Pipeline follows Fabelo et al. (2023) exactly — confirmed in previous
session.

---

### Finding 02-A — All 61 files pass verification ✅

| Campaign | Files | Saved | Skipped | All ✅ |
|---|---|---|---|---|
| Campaign 1 | 27 | 0 | 27 | ✅ |
| Campaign 2 | 24 | 0 | 24 | ✅ |
| Campaign 3 | 10 | 0 | 10 | ✅ |
| **Total**  | **61** | | | **✅** |

All files were previously saved (skipped count = total count) —
preprocessing was run correctly in a prior session. Verification
confirms all outputs are valid.

---

### Finding 02-B — Output shape: all (H, W, 128) float32 ✅

Every patient has exactly 128 spectral bands after decimation.
Spatial dimensions (H, W) vary per image as expected — each image
is manually cropped to the exposed parenchymal area.

**Spatial dimension range across all 61 patients:**
- H range : 280 (034-03) → 721 (058-02)
- W range : 244 (014-01) → 752 (058-02)
- Largest image : 058-02  (721×752) — Campaign 3
- Smallest image: 034-03  (290×301) — Campaign 2

dtype = float32 for all patients — correct for downstream
PyTorch DataLoader consumption.

---

### Finding 02-C — Value range: all [0.000, 1.000] ✅

Every patient has Min=0.000 and Max=1.000 — per-pixel min-max
normalisation worked correctly for all 61 patients including the
two anomalous acquisitions (055-01, 055-02) flagged in raw EDA.

**Significance**: 055-01 and 055-02 had anomalously low raw DN
values (max ~1000 vs ~4095 for normal patients). After calibration
and normalisation their output range is [0, 1] identical to all
other patients. The pipeline handles these acquisitions without
special-casing — the per-pixel min-max normalisation absorbs the
amplitude difference. However their spectral *shape* in the NIR
range remains anomalous after normalisation, as confirmed by
Fabelo et al. (2023).

---

### Finding 02-D — Pipeline shape progression confirmed

For a representative patient (008-02, 480×553 pixels):

| Step | Operation | Shape | Value range |
|---|---|---|---|
| 0 | Raw DN | (480, 553, 826) | [0, 4095] |
| 1 | Calibration | (480, 553, 826) | [0, 1] |
| 2 | Smoothing | (480, 553, 826) | [0, 1] |
| 3 | Band removal | (480, 553, 644) | [0, 1] |
| 4 | Decimation | (480, 553, 128) | [0, 1] |
| 5 | Normalisation | (480, 553, 128) | [0, 1] |

Shape change occurs at steps 3 (826→644) and 4 (644→128).
Value range change occurs at step 1 (DN→reflectance). Steps 2,
3, 4 preserve the [0,1] range. Step 5 re-normalises per-pixel
to ensure exactly [0,1] after decimation may introduce floating
point drift.

---

### Preprocessing — Summary

| Check | Result |
|---|---|
| All 61 files present | ✅ |
| Shape = (H, W, 128) | ✅ all 61 |
| dtype = float32 | ✅ all 61 |
| Min = 0.000 | ✅ all 61 |
| Max = 1.000 | ✅ all 61 |
| Anomalous patients (055-01/02) normalised | ✅ (shape correct, spectral shape flagged) |
| Pipeline matches Fabelo et al. (2023) | ✅ confirmed in prior session |

**Preprocessing status: COMPLETE ✅**
Next step: 03_eda_processed.ipynb

---

---

## 03 — Processed EDA (`03_eda_processed.ipynb`)

### Status: 🔄 In progress

---

### Finding 03-A — All 61 patients pass integrity checks ✅

Check 1 output:
  Shape  (H, W, 128) : 61/61 ✅
  Dtype  float32     : 61/61 ✅
  Range  [0.000, 1.000] : 61/61 ✅

Confirms preprocessing output is valid for all patients.
No corruption on disk-to-memory load. Safe to proceed with
all downstream training and evaluation.

---

### Finding 03-B — Normalisation successful: amplitude outliers eliminated

Plot: `plot_processed_band_profiles`

All 61 patient curves fall within [0, 1] after preprocessing.
Raw amplitude outliers identified in Finding 01-C (035-01 with
DN max ~830, 055-01/02 with DN max ~1000) are no longer
visually distinguishable — calibration and per-pixel min-max
normalisation absorbed all amplitude differences.

**The haemoglobin absorption dip is preserved** at ~540–580nm
in virtually every patient curve. The characteristic valley
before the steep NIR rise is the key spectral feature for
NT/TT discrimination — preprocessing preserved it correctly.

---

### Finding 03-C — Residual campaign-level spectral shape differences persist

Plot: `plot_processed_band_profiles` (colour-coded by campaign)

Despite normalisation removing amplitude differences, campaign-
level separation in spectral SHAPE remains visible in the
processed data:

| Campaign | NIR region (800–900nm) | Characteristic |
|---|---|---|
| C1 (blue) | 0.80–0.92 | Tight cluster, high NIR |
| C2 (orange) | Wide spread | Highest variance |
| C3 (green) | 0.50–0.75 | Lower NIR plateau |

**Key implication**: Per-pixel min-max normalisation removes
illumination-level amplitude differences but cannot correct
for spectral shape differences caused by different tissue
composition, tumour grade distribution, or residual
acquisition artefacts across campaigns. This confirms that
inter-campaign generalisation is a genuine challenge that
cannot be solved by preprocessing alone — it persists into
the feature space seen by the models.

---

### Finding 03-D — One anomalous Campaign 2 outlier identified

Plot: `plot_processed_band_profiles`

One C2 (orange) patient shows a dramatically different spectral
profile — NIR reflectance drops steeply from ~0.65 at 700nm
to ~0.12 at 900nm, where all other patients plateau or rise.
This is the most anomalous curve in the processed dataset.

**Candidate patients**: 034-01 (only BG labels), 034-02
(only TT and BG), 034-03 (only TT and BG). All three are
Campaign 2 patients with unusual label compositions — the
outlier curve is likely one of these.

**Action**: Identify the specific patient ID in
`03_eda_processed.ipynb` by colouring curves by patient ID
and cross-referencing with label coverage from Finding 01-D.

---

*Awaiting: plot_processed_class_spectra, plot_nt_tt_separability,
patient gallery, pipeline steps outputs*

---

### Finding 03-F — 108/128 bands statistically significant (NT vs TT)

Plot: `plot_nt_tt_separability`

Wilcoxon rank-sum test (5% significance level, n=5000 subsampled pixels):

  Significant bands (p < 0.05): 108 / 128 (84.4%)
  Non-significant bands        :  20 / 128 (15.6%)

**Replicates Fabelo et al. (2023) Fig. 2a** — paper reported
significant differences across all spectral channels between TT and NT.
Our result (108/128) is consistent — the 20 non-significant bands
cluster in the ~600–700nm transition zone where both classes share
similar reflectance values during the steep rise from the visible
haemoglobin region to the NIR scattering plateau.

**Implication**: The spectral data contains strong discriminative
signal for NT/TT classification across the vast majority of bands.
The classification challenge is not lack of signal but inter-patient
variability (wide ±1 std overlap) and the subtle mean separation
between classes (~5–10% reflectance difference in NIR).

NT:  pixels = [logged separately in 03-G]
TT:  pixels = [logged separately in 03-G]

---

### Finding 03-G — Patient gallery confirms spatial coherence ✅

Plot: `plot_patient_gallery` (first 9 patients, Campaign 1)

All 9 pseudo-RGB images render anatomically coherent brain surface
views using Fabelo SRGB wavelengths (709nm R, 539nm G, 479nm B).
The dominant red channel is expected — brain tissue has high NIR
reflectance and the 709nm band falls in the NIR.

**Spatial observations**:
- Rubber ring surgical markers clearly visible as black circles
  in RGB views — correctly excluded (unlabelled/white) in GT maps
- GT maps confirm label sparsity — sparse scattered regions against
  large unlabelled backgrounds consistent with pixel count statistics
- 008-01 and 008-02 show TT (red/orange) patches — confirmed as the
  highest TT-pixel patients from Finding 01-D
- Surgical instrument artefacts visible in 005-01 (blue drape) and
  010-03 (brown retractor) — correctly labelled as BG

**Spatial coherence confirmed**: tissue regions are spatially
contiguous and anatomically plausible — no noise artefacts or
checkerboard patterns indicating preprocessing corruption.

---

### Processed EDA — Summary of key findings so far

| Finding | Result |
|---|---|
| 03-A | All 61 patients pass in-memory integrity checks ✅ |
| 03-B | Amplitude outliers eliminated — all curves in [0,1] |
| 03-C | Campaign spectral shape differences persist after normalisation |
| 03-D | One anomalous C2 patient identified — NIR drops to ~0.12 at 900nm |
| 03-E | Class ordering preserved — NT > TT > BV in NIR ✅ |
| 03-F | 108/128 bands significantly different NT vs TT (p<0.05) |
| 03-G | Spatial coherence confirmed — tissue regions anatomically plausible |

*Awaiting: pipeline step effects (plot_pipeline_steps), label statistics*

---

### Finding 03-E — Class spectral ordering explained by haemoglobin physics ✅

Plot: `plot_processed_class_spectra`
Reference: Fabelo et al. (2023) Section "Spectral characterisation", Fig. 3

**Reflectance ordering (NT > TT > BV across full spectrum):**

All four tissue classes maintain consistent ordering in the processed
domain — NT (green) sits above TT (red) which sits above BV (blue)
across the entire 440–909nm range. This ordering is preserved in both
raw DN and normalised reflectance domains.

**Biological explanation — from Fabelo et al. (2023):**

Note: absorbance = -log(reflectance). Higher absorbance = lower
reflectance. The paper reports absorbance values; your plots show
reflectance. The orderings are inverse.

Paper reports absorbance order: BV > TT > NT
Your plots show reflectance order: NT > TT > BV  ← consistent ✅

| Class | Biological state | Haemoglobin | Reflectance |
|---|---|---|---|
| NT | Well oxygenated | High HbO₂, minimal deoxyHb | Highest — healthy cell scattering |
| TT | Hypoxic (O₂ starved) | Elevated deoxyHb | Below NT — more absorption |
| BV | Mixed veins + arteries | Highest total Hb | Lowest — most absorption |

**The haemoglobin absorption shoulder at 500–600nm:**

All classes show a retardation in reflectance rise at ~500–580nm —
the characteristic shoulder shape before the NIR peak. This
corresponds to HbO₂ Q-band absorption peaks at ~542nm and ~577nm,
plus deoxyHb absorption at ~555nm. The shoulder is NOT a drop below
baseline — it is a slowing of the reflectance rise caused by
haemoglobin absorbing incoming light at those wavelengths.

BV shows the most pronounced shoulder — highest total haemoglobin
concentration of all classes. NT and TT show similar shoulder shapes
but NT sits higher throughout because TT has elevated deoxyHb from
tumour hypoxia.

**The deoxyHb peak at ~760nm:**

Per Fabelo et al. (2023), an absorbance peak at ~760nm related to
deoxyHb is present in:
  - BV  : strongest contribution (veins carry deoxygenated blood)
  - TT  : weaker contribution (tumour hypoxia)
  - NT  : absent (well-oxygenated normal tissue)

This manifests in your processed class spectra as a subtle plateau
or inflection around 750–760nm visible in BV and TT but not in NT.

**Why BV shows no clear HbO₂ double-peak:**

BV was labelled without distinguishing arteries (HbO₂-rich) from
veins (deoxyHb-rich). The mixed label produces a blended signal
where the HbO₂ peaks at 542/577nm cancel out, leaving only the
deoxyHb absorption at 760nm as the dominant BV spectral feature.

**Key discriminators between NT and TT:**
  1. ~540–580nm shoulder depth: TT absorbs more (elevated deoxyHb)
  2. NIR level (700–900nm): NT reflects more (healthier cell scattering)
  3. ~760nm deoxyHb peak: present in TT, absent in NT

These three features are confirmed as the most discriminative bands
by LIME analysis in Fabelo et al. (2023) and are consistent with the
108/128 statistically significant bands found in Finding 03-F.

**Dissertation citation:**
*"Fabelo et al. (2023) confirmed that TT exhibits higher absorbance
than NT across 500–600nm, attributable to elevated deoxyhaemoglobin
content from tumour hypoxia. This manifests in the reflectance domain
as NT sitting above TT across the full VNIR spectrum (440–909nm),
with significant separation at the haemoglobin Q-band region
(~540–580nm) and NIR scattering plateau (700–900nm)."*


---

### Finding 03-H — Pipeline step effects confirmed correct

Plot: `plot_pipeline_steps` — Patient 008-02 (480×553, TT-containing)

**Shape progression:**

| Step | Operation | Shape | Range |
|---|---|---|---|
| 0 | Raw DN | (480, 553, 826) | [1, 4095] |
| 1 | Calibration | (480, 553, 826) | [0, 1] |
| 2 | Smoothing | (480, 553, 826) | [0, 1] |
| 3 | Band removal | (480, 553, 644) | [0, 1] |
| 4 | Decimation | (480, 553, 128) | [0, 1] |
| 5 | Normalisation | (480, 553, 128) | [0, 1] |

**Step-by-step observations:**

Step 0→1 (Calibration): DN [1,4095] → reflectance [0,1]. Spectral
shape preserved — haemoglobin shoulder and NIR peak both visible.
Std band narrows as illumination-driven variance is removed.

Step 1→2 (Smoothing): Mean curve unchanged — smoothing does not
alter mean, only reduces band-to-band noise. Std band narrows
noticeably. Moving average filter working correctly.

Step 2→3 (Band removal): Pure crop — shape identical to retained
region of previous step. Edge noise (bands 0–56, 700–826) removed.
No effect on intensity values.

Step 3→4 (Decimation): Shape perfectly preserved at lower resolution.
Haemoglobin shoulder now at bands ~20–40, NIR peak at ~60–80 in
128-band space. Uniform 3.61nm subsampling introduces no distortion.

Step 4→5 (Normalisation): Most dramatic visual change. Mean curve
stretches from [0, ~0.45] to [0, ~0.95]. Per-pixel min-max
normalisation stretches each pixel independently — mean rises because
previously low NIR pixels now map to high normalised values.
Haemoglobin dip at bands 20–40 becomes most prominent feature.

**Histogram observations:**

Raw DN: right-skewed, peak near 0, tail to 4095.

Calibration: bimodal — dark pixels spike at 0, tissue pixels form
bump at 0.3–0.6. Tiny spike at exactly 1.0 = saturated pixels
clipped by np.clip(R, 0, 1).

Smoothing → Band removal → Decimation: distribution shape unchanged
across all three steps — confirms these are pure structural
operations with no effect on intensity values.

Normalisation: bimodal with peaks at BOTH 0 and 1. This is the
expected signature of per-pixel min-max — every pixel's minimum
maps to 0, maximum maps to 1. Large masses at extremes, flat
middle region. Distribution is correct and expected.

---

### Finding 03-I — Label pixel counts confirm campaign structural differences

Plot: `plot_label_statistics`

Per-patient pixel count distributions (median values):

| Class | Campaign 1 | Campaign 2 | Campaign 3 |
|---|---|---|---|
| NT | ~3,000 px | ~4,000 px | ~1,000 px |
| TT | ~100 px | ~300 px | ~800 px |
| BV | ~1,000 px | ~1,500 px | ~500 px |
| BG | ~3,000 px | ~5,000 px | ~16,000 px |

**Campaign 1 — TT almost invisible per patient**
TT median ~100 px per patient — pulled down by 18 TT-free images.
Outliers in TT box represent the 9 images that do have TT.
NT and BG are roughly equal in median pixel count.

**Campaign 2 — Most BV-rich campaign**
Largest BV pixel counts of any campaign — consistent with C2
patients having highly vascularised GBM tumours. NT has widest
spread — heterogeneous image compositions across patients.

**Campaign 3 — BG completely dominates**
BG median ~16,000 px per patient — box extends to 42,000 px.
Dwarfs all other classes. Confirms 80% BG finding from
campaign class distribution table.
Most balanced TT per patient (~800 px) — TT:NT ratio approaching
1:1 at patient level, consistent with NT:TT=1.4 globally.
C3 is the most TT-representative campaign per image despite
having fewest total images (10).

**Key implication for training**:
K-Means pixel reduction (1000 px/class) is essential for C1 — many
C1 patients contribute fewer than 1000 TT pixels total, meaning
the K-Means pool for TT selection is very shallow in C1-heavy
training folds. This is a direct structural explanation for why
C1-heavy folds underperform — the model sees less diverse TT
signal from C1 patients during training.

---

### Processed EDA — Complete summary

| Finding | Category | Result |
|---|---|---|
| 03-A | Integrity | All 61 patients pass in-memory checks ✅ |
| 03-B | Normalisation | Amplitude outliers eliminated — all in [0,1] |
| 03-C | Campaign drift | Spectral shape differences persist after normalisation |
| 03-D | Outlier | One anomalous C2 patient — NIR drops to ~0.12 at 900nm |
| 03-E | Physics | NT > TT > BV ordering explained by haemoglobin physics |
| 03-F | Statistics | 108/128 bands significantly different NT vs TT (p<0.05) |
| 03-G | Spatial | Patient gallery confirms spatial coherence ✅ |
| 03-H | Pipeline | All 5 steps verified correct on patient 008-02 |
| 03-I | Labels | C1 TT sparse (~100 px/patient), C3 BG dominant (~16,000 px) |

**Processed EDA status: COMPLETE ✅**
Next step: 04_eda_campaign.ipynb

---

---

## 04 — Campaign EDA (`04_eda_campaign.ipynb`)

### Status: 🔄 In progress

---

### Finding 04-A — NT is the most acquisition-stable class

Plot: `plot_mean_spectra_per_campaign` (NT panel)

C1, C2, C3 NT mean curves overlap tightly across the full spectrum.
Visible range (bands 0–40): all three nearly identical.
NIR range (bands 60–128): C3 slightly lower, C1/C2 nearly identical.

Mean inter-campaign drift (from Finding 04-C):
  C1 vs C2 : 0.0198  ← lowest drift of any class pair
  C1 vs C3 : 0.0244
  C2 vs C3 : 0.0102  ← closest campaign pair for NT

**Implication**: NT features learned in one campaign transfer well to
another. NT is not the source of the cross-campaign generalisation
challenge — the classification difficulty is concentrated in TT.

---

### Finding 04-B — TT shows clinically significant inter-campaign drift

Plot: `plot_mean_spectra_per_campaign` (TT panel)

C1 TT has a systematically different spectral shape from C2 and C3:
  Visible range (bands 0–40)  : C1 sits ABOVE C2 and C3
  NIR range    (bands 60–128) : C1 sits BELOW C2 and C3

This means C1 TT has higher visible reflectance AND lower NIR
reflectance than C2/C3 TT — a shape difference that cannot be
corrected by per-pixel normalisation (which only adjusts amplitude).

C3 TT std band is very wide — TT in C3 is highly heterogeneous,
reflecting diverse tumour grades (G1–G4) and secondary tumours
in the C3 cohort.

**Direct link to fold performance**: Fold 1 has 8 C1 val patients.
When a model trained on C2/C3-dominated training sets evaluates on
C1 TT pixels, it encounters TT spectra that don't match the TT
signature it learned — causing TT misclassification and low
fold 1 F1-noBG (59.6% for 1D-CNN × UFL).

---

### Finding 04-C — Inter-campaign spectral drift quantified per class

Plot: `plot_spectral_drift`

Mean absolute spectral drift |mean_Ca - mean_Cb| per band, averaged
across all 128 bands:

| Class | C1 vs C2 | C1 vs C3 | C2 vs C3 | Dominant pair |
|---|---|---|---|---|
| NT | 0.0198 | 0.0244 | 0.0102 | C1 vs C3 |
| TT | 0.0222 | **0.0757** | 0.0627 | C1 vs C3 |
| BV | 0.0496 | 0.0489 | 0.0529 | All similar |
| BG | 0.0762 | 0.1146 | **0.1372** | C2 vs C3 |

**TT C1 vs C3 drift = 0.0757** — 3.4× larger than C1 vs C2 (0.0222).
This is the largest tissue-class drift in the dataset. Concentrated
at bands 40–60 (~580–650nm) — the transition from haemoglobin
visible absorption to NIR scattering plateau — exactly where the
model needs to discriminate TT from NT.

**BG has the highest drift overall** — C2 vs C3 = 0.1372. BG is
the most acquisition-sensitive class due to its heterogeneous
composition (surgical drapes, instruments, skull bone) varying
across surgical setups in different campaigns.

**NT has the lowest drift** — confirms NT as the most reliable
class for cross-campaign generalisation.

**BV drift is moderate and uniform** — ~0.049–0.053 across all
pairs. Concentrated in NIR onset (band ~60) and NIR region,
consistent with artery/vein proportion varying across campaigns.

---

### Finding 04-D — Campaign 1 NT/TT most spectrally similar (SAM=2.00°)

Plot: `plot_nt_tt_separability_per_campaign`

Per-campaign NT vs TT spectral angle (SAM):

| Campaign | SAM° | NT pixels | TT pixels | Classification difficulty |
|---|---|---|---|---|
| C1 | **2.00°** | 4,625 | 1,376 | Hardest — most similar |
| C2 | 3.47° | 4,160 | 3,284 | Moderate |
| C3 | 4.06° | 4,361 | 3,372 | Easiest — most distinct |

C1 has the smallest SAM — NT and TT mean spectra are most angularly
similar within C1. The C1 plot shows NT and TT curves nearly
indistinguishable with heavily overlapping std bands throughout.

C2 and C3 show progressively clearer NT/TT separation — the gap
between mean curves is visible in NIR (bands 60–100).

**Note on fold-level vs campaign-level SAM**: Fold 1 (8 C1 val
patients) had SAM=6.88° in the fold EDA — apparently contradicting
C1's campaign SAM=2.00°. The fold-level SAM is computed across all
tissue types in the val set (not just NT vs TT), while campaign SAM
is specifically NT vs TT within-campaign. These measure different
things and are both valid.

**Key implication**: C1 presents the hardest NT/TT classification
challenge because its NT and TT spectra are most similar. Models
trained predominantly on C2/C3 data (where NT and TT are more
distinct) struggle when evaluated on C1 patients where the
discrimination boundary is narrower.

---

### Campaign EDA — Summary of findings so far

| Finding | Class | Key result |
|---|---|---|
| 04-A | NT | Most stable — drift 0.010–0.024 across campaigns |
| 04-B | TT | C1 TT elevated visible, depressed NIR vs C2/C3 |
| 04-C | All | TT C1 vs C3 drift = 0.0757 (largest tissue drift) |
| 04-D | TT | C1 SAM=2.00° — NT/TT hardest to separate in C1 |

*Awaiting: PCA clustering, Kruskal-Wallis test, image statistics outputs*

---

### Finding 04-E — PCA of TT spectra shows partial campaign clustering

Plot: `plot_pca_campaign_clustering` (TT class)

PCA variance explained: PC1=41.6%, PC2=27.3%, PC3=15.7% (total=84.6%)

Three campaigns show partial overlap in PCA space — NOT cleanly
separated but NOT randomly mixed either. C3 (green) tends toward
lower-left, C1 (blue) toward upper region, C2 (orange) spread
throughout. Consistent with moderate drift values (0.022–0.076).

One extreme outlier at PC1≈55 — almost certainly from anomalous
patient 055-01 or 055-02 (Campaign 3, anomalous NIR behaviour
identified in Finding 01-C). Its extreme PCA position confirms it
is a genuine acquisition outlier, not a labelling error.

**Implication**: TT campaign drift is real but not catastrophic —
models can learn some cross-campaign TT features but will struggle
with the campaign-specific offsets, particularly C1 vs C3.

---

### Finding 04-F — PCA of NT spectra shows unexpected C2 isolation

Plot: `plot_pca_campaign_clustering` (NT class)

PCA variance explained: PC1=45.2%, PC2=31.4%, PC3=9.3% (total=85.9%)

Surprising finding: C2 (orange) NT pixels form a completely isolated
cluster at PC1≈-40 to -45, separate from C1 and C3 which overlap
heavily in the centre.

This appears to contradict Finding 04-A (NT lowest drift). Resolution:
PCA captures variance structure (spread pattern), not just mean
differences. The isolated C2 cluster represents a subset of C2
patients with unusual NT spectral variance — likely the patients
driving the wide NT spread in the C2 label statistics box plot
(Finding 03-I). These are high-NT-pixel C2 patients with unusual
brain surface characteristics.

C1 and C3 NT overlap substantially — their NT spectra are the most
compatible in PCA space, despite C1 and C3 being temporally furthest
apart (2015–2019). C2 is the outlier for NT in PCA space.

---

### Finding 04-G — Image-level statistics quantify campaign labelling differences

Plot: `plot_image_level_stats`
Table: campaign summary output

**Image size** (median H×W pixels):
  C1: ~180,000 px  C2: ~185,000 px  C3: ~260,000 px
C3 images are systematically larger — later acquisition period
possibly reflecting larger craniotomies or improved surgical exposure.

**Labelled pixels per image** (median):
  C1:  8,803 px  C2: 14,100 px  C3: 21,480 px
C3 has 2.4× more labelled pixels per image than C1 — more
comprehensive annotations in the most recent campaign.

**TT pixels per image** (median):
  C1:      0 px  ← majority of images have ZERO TT
  C2:    345 px
  C3:    908 px  ← most TT-rich per image

C1's median TT is literally zero — confirmed by 33% TT image rate.
C3 has the highest TT density per image despite fewest total images.

---

### Finding 04-H — Campaign summary statistics

Full campaign characterisation:

| Metric | Campaign 1 | Campaign 2 | Campaign 3 |
|---|---|---|---|
| Images | 27 | 24 | 10 |
| Patients | 16 | 10 | 8 |
| Images with TT | 9 (33%) | 17 (71%) | 7 (70%) |
| TT pixel % | 4.22% | 2.99% | **5.80%** |
| NT:TT ratio | 8.9 | 16.3 | **1.4** |
| NT vs TT SAM° | **2.00°** | 3.47° | 4.06° |
| TT intra-var | 0.1040 | 0.0958 | **0.0711** |
| Median TT px/image | 0 | 345 | 908 |

**Campaign 3 is the most TT-favourable campaign**:
- Highest TT pixel % (5.80%)
- Most balanced NT:TT ratio (1.4:1)
- Most TT-rich per image (908 px median)
- Lowest TT intra-class variance (0.0711) — most consistent TT
- Most distinct NT vs TT separation (SAM=4.06°)

**Campaign 1 is the most TT-challenging campaign**:
- Only 33% images have TT
- Median TT per image = 0
- Most similar NT/TT spectra (SAM=2.00°)
- Highest TT intra-class variance (0.1040) — least consistent TT
- Largest TT spectral drift vs C3 (0.0757)

**Campaign 2 has the most severe NT:TT imbalance** (16.3:1) —
despite 71% of images having TT, those TT regions are very small
(median 345 px) relative to large NT regions.

**For training**: The fixed test set (20% of patients) draws from
all three campaigns. Because C3 is the most TT-favourable and
C1 the most challenging, the test set composition critically
affects reported TT sensitivity. Models evaluated on C1-heavy
test partitions will appear worse than those evaluated on
C3-heavy partitions — not due to model quality but due to
the inherent difficulty difference between campaigns.

---

### Campaign EDA — Complete summary

| Finding | Key result |
|---|---|
| 04-A | NT most stable — drift 0.010–0.024 |
| 04-B | C1 TT elevated visible, depressed NIR vs C2/C3 |
| 04-C | TT C1 vs C3 drift=0.0757 (largest tissue drift) |
| 04-D | C1 SAM=2.00° — NT/TT hardest to separate in C1 |
| 04-E | TT PCA: partial campaign clustering (real but not catastrophic) |
| 04-F | NT PCA: C2 isolated cluster — unexpected variance structure |
| 04-G | C3 most labelled per image; C1 median TT=0 px/image |
| 04-H | C3 most TT-favourable; C1 most challenging; C2 most imbalanced |

**Campaign EDA status: COMPLETE ✅**
Next step: 05_eda_folds.ipynb

---

---

## 05 — Fold EDA (`05_eda_folds.ipynb`)

### Status: COMPLETE ✅

---

### Finding 05-A — Fixed test set confirmed identical across all folds

Printed output and Image 2 (class distribution bars):

Test set is identical for all 5 folds:
  TT%     : 5.5%   NT:TT : 8.1   SAM° : 3.97°
  Images  : 15     (fixed 20% of 61 total images)

build_splits_fabelo K-Fold implementation confirmed correct ✅
Test bars are visually identical across all five fold columns.

---

### Finding 05-B — Training TT% consistent across folds (~3.7%)

All five training sets have TT approximately 3.7% — K-Fold
partitioning distributes TT-containing patients roughly evenly.
Training conditions are comparable across folds.

Val TT% varies (2.9%–4.6%) — this variability is the primary
source of fold-level metric instability.

---

### Finding 05-C — Fold complexity table

Full fold analysis output:

| Fold | Val TT% | Val NT:TT | Val SAM° | Val TT Var | C1 | C2 | C3 |
|---|---|---|---|---|---|---|---|
| 1 | 3.2% | 4.5 | **6.88** | 0.0663 | **8** | 3 | 1 |
| 2 | **4.6%** | 5.3 | **2.97** | 0.0835 | 5 | 4 | 2 |
| 3 | 2.9% | 13.9 | 3.79 | **0.1170** | 4 | 2 | 1 |
| 4 | 3.5% | 4.4 | 3.42 | 0.0667 | 0 | 2 | **5** |
| 5 | 3.4% | **17.5** | 5.42 | 0.0747 | 4 | **5** | 0 |

Fold 1: most C1-heavy (8 C1 val patients), highest SAM°,
         lowest TT pixel count → consistently weakest fold
Fold 2: most TT-rich val (4.6%), lowest SAM°, mixed campaigns
         → consistently strongest fold
Fold 3: highest TT intra-variance (0.1170), most imbalanced
         (NT:TT=13.9) → moderate difficulty
Fold 4: most C3-heavy (5 C3 val patients), low SAM°, near-
         identical NT/TT curves → moderate difficulty
Fold 5: most C2-heavy (5 C2 val patients), highest NT:TT (17.5),
         moderate SAM → moderate-good performance

---

### Finding 05-D — Fold 1 NT/TT visible-range separation explained

Plot: `plot_nt_tt_spectra_per_fold` (fold 1)

Fold 1 shows the most dramatic NT/TT separation in bands 0–40
(visible range, ~440–580nm) of any fold. NT (green) sits
significantly above TT (red) in the visible range. This is NOT
a sign of easier classification — it reflects C1 TT spectral
drift identified in Finding 04-B:

C1 TT has systematically LOWER visible reflectance than C1 NT.
Fold 1's 8 C1 val patients expose this drift — the TT pixels
the model encounters in validation have a different visible-range
signature from the TT signature learned during training
(which is a mix of C1/C2/C3 training patients).

The large SAM=6.88° for fold 1 is CAUSED BY this campaign-level
drift, not by genuine NT/TT biological separability. This is the
key distinction: fold 1 has high SAM° because C1's TT spectra
are offset from the training distribution, not because C1's
tissue types are more biologically distinct.

---

### Finding 05-E — Fold complexity vs F1 correlations

Plot: `plot_complexity_vs_performance`
Model: 1D-CNN × UFL × bal × vp_fabelo

| Metric | Correlation with F1-noBG | Direction | Interpretation |
|---|---|---|---|
| SAM° (NT vs TT) | r = −0.758 | Negative | Higher SAM → lower F1 |
| TT pixel % | r = +0.476 | Positive | More TT → higher F1 |
| NT:TT ratio | r = +0.430 | Positive | More imbalanced → higher F1 (paradox) |

**SAM° is the strongest predictor** of fold difficulty (r=−0.758).
But as established in Finding 05-D, fold 1's high SAM is caused
by campaign-level acquisition drift, not biological separability.
The true predictor is campaign composition — folds dominated by
C1 val patients are harder because C1 TT spectra are offset from
the training distribution.

**TT pixel % correlation (r=0.476)** — more TT pixels in val
means more stable F1 metric. With very few TT pixels, individual
misclassifications swing the metric dramatically. Fold 1's
near-zero median TT makes its F1 estimate highly unreliable.

**NT:TT ratio paradox (r=0.430)** — higher imbalance correlates
with better F1. Explained by the nature of TT regions: folds with
high NT:TT (folds 3 and 5) have small, well-defined tumour regions
that the model classifies reliably. Folds with low NT:TT have
larger, more heterogeneous TT regions that are harder to classify
consistently. Tumour size and spatial consistency matter as much
as pixel count.

---

### Finding 05-F — Fold 3 TT intra-variance highest (0.1170)

Fold 3 has the widest TT std band of any fold — TT pixels in
fold 3's val set are the most spectrally heterogeneous (0.1170
vs 0.0663–0.0835 for other folds). This is consistent with fold
3 having one C3 val patient — C3 has the most diverse tumour
types (multiple grades, secondary tumours). A single C3 patient
with highly variable TT pixels can dramatically increase the
fold-level TT intra-variance.

---

### Fold EDA — Complete summary

| Finding | Key result |
|---|---|
| 05-A | Fixed test set confirmed identical across all folds ✅ |
| 05-B | Train TT% consistent ~3.7% across all folds |
| 05-C | Full complexity table — fold 1 C1-heavy, fold 4 C3-heavy |
| 05-D | Fold 1 high SAM=6.88° caused by C1 TT drift, not separability |
| 05-E | SAM r=−0.758 strongest predictor; NT:TT paradox explained |
| 05-F | Fold 3 TT intra-var=0.1170 highest — C3 tumour heterogeneity |

**Unified fold difficulty explanation**:
Fold performance variability (F1 range: 59.6%–73.9%) is primarily
driven by campaign composition of the val set, not class imbalance
or raw spectral separability. Folds dominated by C1 val patients
encounter TT spectra offset from the training distribution
(campaign-level acquisition drift). Folds with more C2/C3 val
patients encounter TT spectra closer to the training distribution
and achieve higher F1.

**Fold EDA status: COMPLETE ✅**

---

### Finding 05-G — Per-fold conclusions

**Fold 1 — Hardest fold, for the wrong reason**
```
SAM=6.88°  NT:TT=4.5  TT%=3.2%  C1=8  C2=3  C3=1
F1-noBG: 59.6% (1D-CNN × UFL)
```
Consistently underperforms across every model and loss function.
High SAM is caused by C1 TT acquisition drift, not biological
separability — 8 C1 val patients bring TT spectra systematically
offset from the training distribution (elevated visible, depressed
NIR). Low TT pixel count makes the F1 metric itself unreliable.
Fold 1 is not just the hardest fold — it is also the noisiest
estimate of true model performance.

**Fold 2 — Best fold, for the right reason**
```
SAM=2.97°  NT:TT=5.3  TT%=4.6%  C1=5  C2=4  C3=2
F1-noBG: 73.9% (1D-CNN × UFL)
```
Most balanced across every dimension — campaign composition, TT
pixel count, and NT:TT ratio. Mixed campaign val set (5 C1, 4 C2,
2 C3) means val patients are representative of the training
distribution — no systematic spectral offset. Low SAM (2.97°)
with narrow std bands gives a consistent and learnable decision
boundary. Fold 2 gives the most reliable estimate of true model
generalisation.

**Fold 3 — Hard for a different reason**
```
SAM=3.79°  NT:TT=13.9  TT%=2.9%  C1=4  C2=2  C3=1
F1-noBG: 70.3% (1D-CNN × UFL)  best epoch=3
```
Difficulty comes from TT heterogeneity rather than campaign drift.
Highest TT intra-class variance of any fold (0.1170). Model peaks
at epoch 3 and never recovers — finds a decision boundary that
works for some TT subtypes but cannot generalise across the full
TT spectral range in this val set. High NT:TT ratio (13.9) means
very few TT pixels, compounding instability.

**Fold 4 — The cross-campaign test**
```
SAM=3.42°  NT:TT=4.4  TT%=3.5%  C1=0  C2=2  C3=5
F1-noBG: 69.6% (1D-CNN × UFL)
```
Only fold with zero C1 val patients. Five C3 val patients make it
the most cross-campaign test — C3 images are 2.5–4 years newer
with different surgical setups. Despite this the model achieves
69.6% — C3 TT is more compatible with the training distribution
than C1 TT. Nearly identical NT/TT curves (low SAM) mean the
model works from very subtle spectral differences.

**Fold 5 — The imbalance outlier**
```
SAM=5.42°  NT:TT=17.5  TT%=3.4%  C1=4  C2=5  C3=0
F1-noBG: 72.8% (1D-CNN × UFL)
```
Most severe NT:TT imbalance (17.5:1) yet second best result.
Explains the NT:TT paradox from Finding 05-E — TT regions are
small and well-defined (compact C2 tumour regions), classified
reliably despite scarcity. Five C2 val patients bring consistent
TT signatures compatible with the training distribution. No C3
patients means no C3-specific acquisition variability.

**Unified conclusion**:
Fold performance ranking is almost entirely explained by campaign
composition rather than class imbalance or spectral separability:

| Rank | Fold | F1-noBG | Campaign profile |
|---|---|---|---|
| 1st | 2 | 73.9% | Mixed — all campaigns represented |
| 2nd | 5 | 72.8% | C2-heavy — spectrally compatible |
| 3rd | 3 | 70.3% | Mostly C1/C2 — TT heterogeneity |
| 4th | 4 | 69.6% | C3-heavy — cross-campaign gap |
| 5th | 1 | 59.6% | C1-heavy — acquisition drift |

Random patient-level K-Fold on this dataset does NOT produce IID
folds — campaign of origin introduces systematic bias making some
folds structurally harder independent of model quality or loss
function. This argues for campaign-stratified cross-validation
as a future work recommendation.

---

## EDA Progress Summary

| Notebook | Status | Key finding |
|---|---|---|
| 01_eda_raw | ✅ COMPLETE | TT=4.1% globally, 33/61 images TT-free |
| 02_preprocessing | ✅ COMPLETE | All 61 patients verified [0,1] float32 |
| 03_eda_processed | ✅ COMPLETE | 108/128 bands significant, NT>TT throughout |
| 04_eda_campaign | ✅ COMPLETE | C1 TT drift, C3 most TT-favourable |
| 05_eda_folds | ✅ COMPLETE | Campaign composition drives fold difficulty |

**All EDA notebooks complete. Ready for training experiments.**

---