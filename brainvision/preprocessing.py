"""
brainvision.preprocessing — Five-step HSI preprocessing pipeline.

Follows Fabelo et al. (2023) exactly:
  1. Calibrate       R = (raw - dark) / (white - dark)
  2. Smooth spectra  moving average, window=5
  3. Remove bands    drop first 56 + last 126
  4. Decimate        644 → 128 bands (3.61nm interval)
  5. Normalise       per-pixel min-max [0, 1]

Usage:
    from brainvision.preprocessing import preprocess
    patient = preprocess(patient)
"""

import numpy as np

from brainvision.constants import (
    SMOOTH_WINDOW,
    BAND_START_IDX,
    BAND_END_IDX,
    N_DECIMATED_BANDS,
)


def calibrate(raw:   np.ndarray,
              dark:  np.ndarray,
              white: np.ndarray,
              eps:   float = 1e-6) -> np.ndarray:
    """
    Convert raw DN to reflectance: R = (raw - dark) / (white - dark)
    Fabelo et al. (2023) Eq. 1.

    Parameters
    ----------
    raw   : (H, W, 826) float — raw digital numbers
    dark  : (1, W, 826) float — dark reference (shutter closed)
    white : (1, W, 826) float — white reference (99% Spectralon tile)
    eps   : numerical stability guard against division by zero

    Returns
    -------
    (H, W, 826) float32 clipped to [0, 1]
    """
    dark_mean  = dark.mean(axis=0,  keepdims=True)
    white_mean = white.mean(axis=0, keepdims=True)
    R = (raw - dark_mean) / (white_mean - dark_mean + eps)
    return np.clip(R, 0, 1).astype(np.float32)


def smooth_spectra(cube:   np.ndarray,
                   window: int = SMOOTH_WINDOW) -> np.ndarray:
    """
    Moving average filter along the spectral axis (axis=2).
    Window=5 as per Fabelo et al. (2023).

    Input / Output: (H, W, B) float32
    """
    from scipy.ndimage import uniform_filter1d  # lazy: only this raw-pipeline
    # step needs scipy — importing it at module level would force every
    # submodule-direct import (e.g. `from brainvision.preprocessing import
    # minmax_normalise`, which the Pi demo does) to pull in scipy too, even
    # though minmax_normalise itself is pure numpy. See
    # demo/requirements_demo.txt and scripts/pi_sync_code.sh, both of which
    # assume scipy is NOT required for the demo's own imports.
    return uniform_filter1d(cube, size=window, axis=2).astype(np.float32)


def remove_noisy_bands(cube: np.ndarray) -> np.ndarray:
    """
    Remove first 56 and last 126 spectral bands.
    Retains 644 channels (440.5–909.1 nm operating bandwidth).
    Fabelo et al. (2023).

    (H, W, 826) → (H, W, 644)
    """
    return cube[:, :, BAND_START_IDX:BAND_END_IDX].astype(np.float32)


def decimate_spectral_channels(cube:     np.ndarray,
                                n_output: int = N_DECIMATED_BANDS) -> np.ndarray:
    """
    Uniform spectral subsampling to n_output bands.
    Optimal sampling interval 3.61 nm → 128 bands.
    Fabelo et al. (2023).

    (H, W, 644) → (H, W, 128)
    """
    B       = cube.shape[2]
    indices = np.linspace(0, B - 1, n_output, dtype=int)
    return cube[:, :, indices].astype(np.float32)


def minmax_normalise(cube: np.ndarray) -> np.ndarray:
    """
    Per-pixel min-max normalisation to [0, 1] across the spectral axis.
    Fabelo et al. (2023).

    Input / Output: (H, W, B) float32
    """
    H, W, B  = cube.shape
    pixels   = cube.reshape(-1, B)
    mn       = pixels.min(axis=1, keepdims=True)
    mx       = pixels.max(axis=1, keepdims=True)
    normed   = (pixels - mn) / (mx - mn + 1e-6)
    return normed.reshape(H, W, B).astype(np.float32)


def preprocess(patient: dict,
               verbose: bool = True) -> dict:
    """
    Full five-step pipeline — Fabelo et al. (2023).

    Steps applied in order:
      1. Calibration   (H, W, 826) → reflectance [0, 1]
      2. Smoothing     moving average window=5
      3. Band removal  (H, W, 826) → (H, W, 644)
      4. Decimation    (H, W, 644) → (H, W, 128)
      5. Normalisation per-pixel min-max [0, 1]

    Parameters
    ----------
    patient : dict with keys 'id', 'raw', 'dark', 'white', 'labels'
    verbose : if True, print shape and range after processing

    Returns
    -------
    patient dict with 'processed' key added — (H, W, 128) float32
    """
    if verbose:
        print(f"  Processing {patient['id']}...", end=" ")

    cube = calibrate(patient['raw'], patient['dark'], patient['white'])
    cube = smooth_spectra(cube)
    cube = remove_noisy_bands(cube)
    cube = decimate_spectral_channels(cube)
    cube = minmax_normalise(cube)

    if verbose:
        print(f"✅  {cube.shape}  [{cube.min():.3f}, {cube.max():.3f}]")

    patient['processed'] = cube
    return patient