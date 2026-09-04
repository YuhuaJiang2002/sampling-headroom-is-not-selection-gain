"""The common CVA-Select readout used across the released experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _finite_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] < 2:
        raise ValueError(f"{name} must have at least two candidates on its last axis")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def within_pool_zscore(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Standardize each candidate pool along the last axis.

    Constant components become exact zeros, so they cannot change a pick.
    """

    array = _finite_array("values", values)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    centered = array - array.mean(axis=-1, keepdims=True)
    scale = array.std(axis=-1, keepdims=True)
    return np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=scale > epsilon,
    )


@dataclass(frozen=True)
class CVASelection:
    """Frozen CVA-Select utilities and picks for one or more pools."""

    picks: np.ndarray
    utility: np.ndarray
    standardized_global: np.ndarray
    standardized_motion: np.ndarray


def cva_select(
    global_evidence: np.ndarray,
    motion_evidence: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> CVASelection:
    """Apply ``z(G) + z(M)`` and select the first maximum in each pool.

    Both inputs must already be oriented so larger values are better. For a
    motion penalty, pass its negative as ``motion_evidence``.
    """

    global_array = _finite_array("global_evidence", global_evidence)
    motion_array = _finite_array("motion_evidence", motion_evidence)
    if global_array.shape != motion_array.shape:
        raise ValueError("global and motion evidence must have identical shapes")
    standardized_global = within_pool_zscore(global_array, epsilon)
    standardized_motion = within_pool_zscore(motion_array, epsilon)
    utility = standardized_global + standardized_motion
    picks = np.argmax(utility, axis=-1).astype(np.int64)
    return CVASelection(
        picks=picks,
        utility=utility,
        standardized_global=standardized_global,
        standardized_motion=standardized_motion,
    )


def full_factorial_utilities(
    global_evidence: np.ndarray,
    native_evidence: np.ndarray,
    motion_evidence: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> Mapping[str, np.ndarray]:
    """Return all seven non-empty combinations of standardized G, N, and M."""

    global_array = _finite_array("global_evidence", global_evidence)
    native_array = _finite_array("native_evidence", native_evidence)
    motion_array = _finite_array("motion_evidence", motion_evidence)
    if not (global_array.shape == native_array.shape == motion_array.shape):
        raise ValueError("G, N, and M must have identical shapes")
    g = within_pool_zscore(global_array, epsilon)
    n = within_pool_zscore(native_array, epsilon)
    m = within_pool_zscore(motion_array, epsilon)
    return {
        "G": g,
        "N": n,
        "M": m,
        "G+N": g + n,
        "CVA-Select": g + m,
        "N+M": n + m,
        "G+N+M": g + n + m,
    }
