# constants.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for all constants used across notebooks.
# Import in any notebook with: from brainvision.constants import *
# ─────────────────────────────────────────────────────────────────────────────

# ── SENSOR ────────────────────────────────────────────────────────────────────
NUMBER_OF_BANDS    = 826      # raw bands from HELICoiD VNIR camera
SPECTRAL_RANGE_NM  = (400, 1000)   # nm

# ── BAND REMOVAL (Fabelo et al., 2023) ────────────────────────────────────────
BANDS_REMOVE_START = 56       # drop first 56 bands  (400–440nm)
BANDS_REMOVE_END   = 126      # drop last 126 bands  (902–1000nm)
BAND_START_IDX     = BANDS_REMOVE_START                    # 56
BAND_END_IDX       = NUMBER_OF_BANDS - BANDS_REMOVE_END    # 700
RETAINED_BANDS     = BAND_END_IDX - BAND_START_IDX         # 644

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
SMOOTH_WINDOW      = 5        # moving average window (Fabelo et al., 2023)
N_DECIMATED_BANDS  = 128      # spectral channels after decimation
CALIBRATION_EPS    = 1e-6     # avoid division by zero in reflectance formula

# ── DATASET ───────────────────────────────────────────────────────────────────
N_CLASSES          = 4
CLASS_NAMES        = {
    0: 'Unlabelled',
    1: 'Normal Tissue (NT)',
    2: 'Tumour Tissue (TT)',
    3: 'Blood Vessel (BV)',
    4: 'Background (BG)',
}
CLASS_NAMES_SHORT  = {0: 'Unlabelled', 1: 'NT', 2: 'TT', 3: 'BV', 4: 'BG'}
CLASS_COLORS       = {
    1: '#5DCAA5',   # NT — green
    2: '#D85A30',   # TT — red
    3: '#378ADD',   # BV — blue
    4: '#888780',   # BG — grey
}

# ── TRAINING DATA REDUCTION ──────────────────────────────
REDUCE_PIXELS    = True    # apply K-Means reduction to training set
N_PIXELS_PER_CLASS = 1000  # pixels per class after reduction

# ── SPLITS ────────────────────────────────────────────────────────────────────
SPLIT_SEED         = 42       # fix once, never change
TRAIN_RATIO        = 15       # patients
VAL_RATIO          = 4        # patients
TEST_RATIO         = 3        # patients

# ── SPECTRALFORMER ────────────────────────────────────────────────────────────
SF_NEAR_BAND   = 3      # adjacent bands per spectral token
SF_DIM         = 64     # transformer embedding dimension
SF_DEPTH       = 5      # transformer layers
SF_HEADS       = 4      # attention heads
SF_DIM_HEAD    = 16     # dimension per head
SF_MLP_DIM     = 8      # feedforward hidden dimension
SF_DROPOUT     = 0.1
SF_EMB_DROPOUT = 0.1
SF_MODE        = 'ViT'  # 'ViT' or 'CAF'

# ── TRAINING ──────────────────────────────────────────────────────────────────
BATCH_SIZE         = 64
MAX_EPOCHS         = 100
LEARNING_RATE      = 1e-3
LR_DECAY_FACTOR    = 0.5
LR_DECAY_PATIENCE  = 5        # epochs before LR decay
EARLY_STOP_PATIENCE   = 15       # epochs without val F1-noBG improvement
EARLY_STOP_MIN_EPOCHS = 10       # don't fire train loss criterion before this
TRAIN_LOSS_DELTA_MIN  = 1e-4     # minimum meaningful train loss change
DROPOUT_RATE       = 0.5
PATCH_SIZE = 11

# ── LOSS FUNCTIONS ────────────────────────────────────────────────────────────
FOCAL_GAMMA        = 2.0      # Lin et al. (2018)
UFL_LAMBDA         = 0.5      # Yeung et al. (2022)
UFL_DELTA          = 0.6      # Yeung et al. (2022)
DICE_EPS           = 1e-6


# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA_PROCESSED_DIR = "../processed"
CHECKPOINTS_DIR    = "../checkpoints"
RESULTS_DIR        = "../results"

CAMPAIGN_DIRS = {
    1: "../datasets/first_campaign",
    2: "../datasets/second_campaign",
    3: "../datasets/third_campaign",
}

PROCESSED_DIRS = {
    1: "../processed/first_campaign",
    2: "../processed/second_campaign",
    3: "../processed/third_campaign",
}
