"""Core quantities for the four-stage Compute-Value Audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


@dataclass(frozen=True)
class AuditReport:
    """Measured CVA quantities and sequential gate decisions."""

    oracle_headroom: float
    recovered_fraction: float
    action_gain: float
    entry_fee: float
    net_value: float
    opportunity_passed: bool
    state_passed: bool
    action_passed: bool
    fee_passed: bool

    @property
    def complete_chain_passed(self) -> bool:
        return (
            self.opportunity_passed
            and self.state_passed
            and self.action_passed
            and self.fee_passed
        )

    def as_dict(self) -> dict:
        return {**asdict(self), "complete_chain_passed": self.complete_chain_passed}


def compute_value_audit(
    reference_quality: np.ndarray,
    selected_quality: np.ndarray,
    oracle_quality: np.ndarray,
    *,
    state_passed: bool,
    entry_fee: float = 0.0,
    minimum_headroom: float = 0.0,
    tolerance: float = 1e-12,
) -> AuditReport:
    """Compute opportunity, action, and fee values for matched examples.

    The state decision is supplied by the experiment's predeclared statistical
    test; CVA does not infer it from outcome labels.
    """

    reference = _vector("reference_quality", reference_quality)
    selected = _vector("selected_quality", selected_quality)
    oracle = _vector("oracle_quality", oracle_quality)
    if not (reference.shape == selected.shape == oracle.shape):
        raise ValueError("reference, selected, and oracle arrays must match")
    if not np.isfinite(entry_fee) or entry_fee < 0.0:
        raise ValueError("entry_fee must be finite and non-negative")
    if not np.isfinite(minimum_headroom) or minimum_headroom < 0.0:
        raise ValueError("minimum_headroom must be finite and non-negative")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    oracle_headroom = float(np.mean(oracle - reference))
    action_gain = float(np.mean(selected - reference))
    recovered_fraction = (
        action_gain / oracle_headroom if oracle_headroom > tolerance else float("nan")
    )
    net_value = action_gain - entry_fee
    opportunity_passed = oracle_headroom > minimum_headroom + tolerance
    action_passed = opportunity_passed and state_passed and action_gain > tolerance
    fee_passed = action_passed and net_value > tolerance
    return AuditReport(
        oracle_headroom=oracle_headroom,
        recovered_fraction=recovered_fraction,
        action_gain=action_gain,
        entry_fee=float(entry_fee),
        net_value=net_value,
        opportunity_passed=opportunity_passed,
        state_passed=bool(state_passed),
        action_passed=action_passed,
        fee_passed=fee_passed,
    )


def verifier_alignment(scores: np.ndarray, quality: np.ndarray) -> float:
    """Average within-pool Pearson alignment, excluding degenerate pools."""

    score_array = np.asarray(scores, dtype=np.float64)
    quality_array = np.asarray(quality, dtype=np.float64)
    if score_array.shape != quality_array.shape or score_array.ndim != 2:
        raise ValueError("scores and quality must share shape [examples, candidates]")
    if score_array.shape[1] < 2:
        raise ValueError("each pool must contain at least two candidates")
    if not np.isfinite(score_array).all() or not np.isfinite(quality_array).all():
        raise ValueError("scores and quality must be finite")
    correlations = []
    for score_row, quality_row in zip(score_array, quality_array):
        if score_row.std() <= 1e-12 or quality_row.std() <= 1e-12:
            continue
        correlations.append(float(np.corrcoef(score_row, quality_row)[0, 1]))
    if not correlations:
        raise ValueError("all candidate pools are degenerate")
    return float(np.mean(correlations))
