"""
brainvision.data.io — ENVI file loading, zip handling, and campaign utilities.

Functions for reading raw HELICoiD hyperspectral data from ZIP archives,
loading preprocessed patient cubes from .npz files, and summarising
campaign directories.
"""

import shutil
import zipfile
from pathlib import Path

import numpy as np
from spectral import envi


def unzip_patient(zip_path: str) -> Path:
    """Unzip a single patient archive and return the folder containing the ENVI files."""
    zip_path = Path(zip_path)

    extract_dir = zip_path.parent
    target = zip_path.parent / zip_path.stem

    if target.exists():
        return target

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return target


def find_envi_pair(folder: Path, stem: str) -> tuple[Path, Path]:
    """Return (data_file, header_file) for a given ENVI stem name."""
    hdr = folder / f"{stem}.hdr"
    data = folder / stem
    if not hdr.exists():
        raise FileNotFoundError(f"Header not found: {hdr}")
    if not data.exists():
        raise FileNotFoundError(f"Data file not found: {data}")
    return data, hdr


def load_envi_cube(folder: Path, stem: str) -> np.ndarray:
    """Load an ENVI binary file and return it as a float32 numpy array."""
    data_file, hdr_file = find_envi_pair(folder, stem)
    img = envi.open(str(hdr_file), str(data_file))
    return img.load().astype(np.float32)


def load_labels(folder: Path) -> np.ndarray:
    """Load pixel-level ground-truth label map. Returns (H, W) int array."""
    data_file, hdr_file = find_envi_pair(folder, "gtMap")
    gt = envi.open(str(hdr_file), str(data_file)).load()
    return gt.squeeze().astype(np.int32)


def load_patient(zip_path: str, cleanup: bool = False) -> dict:
    """
    Load raw data from a patient ZIP — no preprocessing.
    Returns raw cube, labels, and JPEG preview paths.

    Returns:
    {
        'id':      str,          # e.g. '012-02'
        'raw':     np.ndarray,   # (H, W, 826)  float32  — raw DN values
        'dark':    np.ndarray,   # (1, W, 826)  float32  — dark reference
        'white':   np.ndarray,   # (1, W, 826)  float32  — white reference
        'labels':  np.ndarray,   # (H, W)        int32
        'image':   Path,         # RGB preview
        'gt_map':  Path,         # GT label preview
    }
    """
    zip_path = Path(zip_path)
    patient_id = zip_path.stem
    folder = unzip_patient(zip_path)

    raw = load_envi_cube(folder, "raw")
    dark = load_envi_cube(folder, "darkReference")
    white = load_envi_cube(folder, "whiteReference")
    labels = load_labels(folder)

    if cleanup:
        shutil.rmtree(folder)

    return {
        "id": patient_id,
        "raw": raw,
        "dark": dark,
        "white": white,
        "labels": labels,
        "image": folder / "image.jpg",
        "gt_map": folder / "gtMap.jpg",
    }


def list_patient_zips(data_dir: str = "data/first_campaign") -> list[Path]:
    """Return sorted list of all patient ZIP paths in the given directory."""
    return sorted(Path(data_dir).glob("*.zip"))


def campaign_summary(data_dir: str) -> dict:
    """
    Count the number of HSI images and unique patients in a campaign folder.

    Args:
        data_dir: path to campaign folder containing .zip files

    Returns:
        {
            'data_dir'  : str,
            'n_images'  : int,   # total number of zip files
            'n_patients': int,   # number of unique patients
            'images'    : list[str],   # all image IDs e.g. ['004-02', ...]
            'patients'  : list[str],   # unique patient IDs e.g. ['004', ...]
        }
    """
    zips = sorted(Path(data_dir).glob("*.zip"))
    images = [z.stem for z in zips]
    patients = sorted(set(z.stem.split("-")[0] for z in zips))

    return {
        "data_dir": data_dir,
        "n_images": len(images),
        "n_patients": len(patients),
        "images": images,
        "patients": patients,
    }


def print_campaign_summary(data_dir: str):
    """Pretty-print the campaign summary."""
    s = campaign_summary(data_dir)

    print(f"Campaign : {s['data_dir']}")
    print(f"  Images   (n) : {s['n_images']}")
    print(f"  Patients (m) : {s['n_patients']}")
    print(f"  Patient IDs  : {s['patients']}")
    print(f"  Image IDs    : {s['images']}")


def all_campaigns_summary(campaign_dirs: dict[int, str]):
    """Print summary for all campaigns and combined totals."""
    total_images, total_patients = 0, set()

    for campaign, data_dir in campaign_dirs.items():
        print(f"\n── Campaign {campaign} {'─' * 40}")
        s = campaign_summary(data_dir)
        print_campaign_summary(data_dir)
        total_images += s["n_images"]
        total_patients |= set(s["patients"])

    print(f"\n── Combined {'─' * 43}")
    print(f"  Total images   : {total_images}")
    print(f"  Total patients : {len(total_patients)}")


def load_processed_patients(data_dir: str) -> list[dict]:
    """
    Load all preprocessed patient cubes from a directory of .npz files.

    Each .npz file should contain 'processed' and 'labels' arrays.

    Returns:
        List of dicts with keys: 'id', 'processed', 'labels', 'campaign'.
    """
    paths = sorted(Path(data_dir).glob("*.npz"))
    patients = []
    for p in paths:
        d = np.load(p)
        patients.append({
            'id':        p.stem,
            'processed': d['processed'],
            'labels':    d['labels'],
            'campaign':  data_dir,
        })
    print(f"  Loaded {len(patients):>3} patients from {data_dir}")
    return patients


_campaigns_cache: None | dict = None

def load_all_campaigns(processed_dirs: dict, force_reload: bool = False) -> dict:
    """
    Load preprocessed patients for all three campaigns.
    Cached in memory — only loads from disk once per session.
    Set force_reload=True to reload from disk.
    """
    global _campaigns_cache

    if _campaigns_cache is not None and not force_reload:
        total = sum(len(v) for v in _campaigns_cache.values())
        print(f"Campaigns loaded from memory cache "
              f"({total} patients across {len(_campaigns_cache)} campaigns)")
        return _campaigns_cache

    print("Loading all campaigns from disk...")
    _campaigns_cache = {
        c: load_processed_patients(processed_dirs[c])
        for c in [1, 2, 3]
    }
    total = sum(len(v) for v in _campaigns_cache.values())
    print(f"Loaded {total} patients across "
          f"{len(_campaigns_cache)} campaigns\n")
    return _campaigns_cache
