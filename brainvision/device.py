"""
brainvision.device — Device selection and memory management utilities.

Supports:
  - CUDA  : NVIDIA GPUs (Kaggle, Colab, Linux workstations)
  - MPS   : Apple Silicon unified memory (macOS M1/M2/M3/M4)
  - CPU   : fallback

Usage:
    from brainvision.device import get_device, empty_cache, device_info

    device = get_device()
    # ... training ...
    empty_device_cache(device)
"""

import torch


def get_device() -> torch.device:
    """
    Return the best available device in priority order:
      1. CUDA  — NVIDIA GPU
      2. MPS   — Apple Silicon GPU (macOS)
      3. CPU   — fallback

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def empty_device_cache(device: torch.device | None = None):
    """
    Free unused memory from the current device's cache.
    Safe to call regardless of device type.

    Args:
        device : torch.device to clear. If None, uses get_device().
    """
    if device is None:
        device = get_device()

    d = str(device).lower()

    if d.startswith('cuda'):
        torch.cuda.empty_cache()
    elif d.startswith('mps'):
        torch.mps.empty_cache()
    # CPU has no cache to clear


def device_info(device: torch.device | None = None) -> dict:
    """
    Return a dict of device properties useful for logging and debugging.

    Args:
        device : torch.device to inspect. If None, uses get_device().

    Returns dict with:
        device_str     : str  e.g. 'cuda', 'mps', 'cpu'
        device_name    : str  e.g. 'NVIDIA Tesla T4', 'Apple M2', 'CPU'
        memory_total   : int  total memory in bytes  (0 if unknown)
        memory_free    : int  free  memory in bytes  (0 if unknown)
        memory_used    : int  used  memory in bytes  (0 if unknown)
    """
    if device is None:
        device = get_device()

    d          = str(device).lower()
    info       = {
        'device_str'   : d,
        'device_name'  : 'Unknown',
        'memory_total' : 0,
        'memory_free'  : 0,
        'memory_used'  : 0,
    }

    if d.startswith('cuda'):
        props                = torch.cuda.get_device_properties(device)
        mem_free, mem_total  = torch.cuda.mem_get_info(device)
        info['device_name']  = props.name
        info['memory_total'] = mem_total
        info['memory_free']  = mem_free
        info['memory_used']  = mem_total - mem_free

    elif d.startswith('mps'):
        info['device_name']  = 'Apple Silicon (MPS)'
        # MPS does not expose memory stats via PyTorch API yet
        # recommended_max is available from macOS 14+
        try:
            info['memory_total'] = (
                torch.mps.recommended_max_memory()
            )
        except AttributeError:
            pass    # older PyTorch — leave as 0

    else:
        info['device_name'] = 'CPU'

    return info


def print_device_info(device: torch.device | None = None):
    """Print a formatted summary of the current device."""
    if device is None:
        device = get_device()

    info = device_info(device)

    def _mb(b: int) -> str:
        return f"{b / 1e6:,.0f} MB" if b > 0 else "N/A"

    print(f"  Device     : {info['device_str'].upper()}")
    print(f"  Name       : {info['device_name']}")
    print(f"  Memory     : {_mb(info['memory_total'])} total  "
          f"| {_mb(info['memory_free'])} free  "
          f"| {_mb(info['memory_used'])} used")