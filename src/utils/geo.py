"""Geographic utilities for distance computation."""

import math

import numpy as np
import torch


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Compute haversine distance between two GPS points in kilometers."""
    R = 6371.0  # Earth radius in km

    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def haversine_distance_vectorized(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Haversine distance in km. Works on both scalars and arrays."""
    R = 6371.0

    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def haversine_torch(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    """Haversine distance in km on GPU. Works on torch tensors.

    Identical formula to the numpy/scalar versions, but operates on
    torch tensors for GPU-accelerated batch computation. Uses float32
    throughout for MPS compatibility (MPS doesn't support float64).
    Precision loss is negligible at per-batch scale (~4K elements).
    """
    R = 6371.0
    deg2rad = math.pi / 180.0

    lat1_r = lat1 * deg2rad
    lat2_r = lat2 * deg2rad
    dlat = (lat2 - lat1) * deg2rad
    dlon = (lon2 - lon1) * deg2rad

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c