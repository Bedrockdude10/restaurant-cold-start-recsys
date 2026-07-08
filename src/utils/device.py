"""Single source of truth for compute-device selection.

Priority: CUDA > MPS > CPU. Import `get_device()` everywhere instead of
duplicating `torch.device(...)` logic, so changing the policy here moves the
whole pipeline (training and evaluation).
"""

import torch


def get_device(verbose: bool = False) -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"Device: {dev}")
    return dev
