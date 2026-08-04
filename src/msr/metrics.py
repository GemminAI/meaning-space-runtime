"""EXP-Ubuntu004 verification metrics: Runtime Stability and Drift Convergence.

These are pure functions over already-recorded :class:`~msr.abi.MeaningState`
history or plain coordinate sequences — they read the runtime's output, they
do not participate in the step loop, and they raise nothing the loop itself
would not already raise.

**Runtime Stability (λ).** Defined by the experiment spec as

.. math::

    \\lambda = \\lim_{t \\to \\infty} -\\frac{1}{t}
        \\ln \\frac{\\Vert \\Sigma(t) \\Vert}{\\Vert \\Sigma(0) \\Vert}

Two approximations make this measurable over a finite recorded run, both
documented here rather than hidden in the code:

1. The limit is replaced by the finite interval actually spanned by the
   supplied states — a finite-time estimate, not an asymptotic guarantee.
2. ``‖·‖`` is the Frobenius norm (``numpy.linalg.norm`` on a 2D array's
   default). Σ(t) is symmetric positive definite by construction
   (:func:`msr.linalg.invert_spd` of ``MeaningState.precision``), so the
   Frobenius norm is a legitimate, basis-independent size measure here; the
   spectral norm (largest eigenvalue) would be an equally defensible choice
   and is deliberately not what this implements.

**Drift Convergence.** Defined by the spec as
``lim_{t→∞} ‖γ(t) − γ_ref(t)‖ < ε_drift``. MSR has no notion of a reference
trajectory of its own (it is driven by whatever measurements arrive, not by
a predefined target), so :func:`drift_distance` takes the reference as an
explicit argument — the experiment defines what γ_ref means for its own
scenario, this module only supplies the pointwise-distance and
windowed-threshold check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from msr.abi import MeaningState, Vector
from msr.errors import DimensionMismatch, InsufficientHistory
from msr.linalg import invert_spd


def _covariance_norm(state: MeaningState) -> float:
    """‖Σ‖_F at one state, inverting the precision the runtime actually stored."""
    precision = np.asarray(state.precision, dtype=np.float64)
    covariance = invert_spd(precision, name="precision")
    return float(np.linalg.norm(covariance))


def runtime_stability_rate(states: Sequence[MeaningState]) -> float:
    """Finite-time Runtime Stability λ over ``states[0]`` .. ``states[-1]``.

    Positive λ means Σ shrank (the runtime grew more certain) over the
    interval; negative λ means it grew. Requires at least two states
    spanning a positive elapsed time, and a non-degenerate (non-zero-norm)
    covariance at both ends — anything else means there is no rate to
    measure, not that the rate is zero, so it is reported as
    :class:`~msr.errors.InsufficientHistory` rather than as ``0.0`` or
    ``nan``.
    """
    if len(states) < 2:
        raise InsufficientHistory("need at least 2 states to measure a rate")
    elapsed = states[-1].time_s - states[0].time_s
    if elapsed <= 0.0:
        raise InsufficientHistory("states must span a positive elapsed time")
    norm_0 = _covariance_norm(states[0])
    norm_t = _covariance_norm(states[-1])
    if norm_0 <= 0.0 or norm_t <= 0.0:  # pragma: no cover - needs float64 underflow
        # Guards a raw ZeroDivisionError from leaking out undressed: only
        # reachable if a precision's inverse underflows to the exact zero
        # matrix, which no precision this runtime actually produces (bounded
        # below by InformationAssimilator.precision_floor) can trigger.
        raise InsufficientHistory("covariance norm must be positive at both ends")
    return -math.log(norm_t / norm_0) / elapsed


def drift_distance(
    trajectory: Sequence[Vector], reference: Sequence[Vector]
) -> tuple[float, ...]:
    """Pointwise ``‖γ(t) - γ_ref(t)‖`` for two equal-length coordinate sequences."""
    if len(trajectory) != len(reference):
        raise DimensionMismatch(
            f"trajectory has {len(trajectory)} points, reference has {len(reference)}"
        )
    distances: list[float] = []
    for point, ref_point in zip(trajectory, reference, strict=True):
        array = np.asarray(point, dtype=np.float64)
        ref_array = np.asarray(ref_point, dtype=np.float64)
        if array.shape != ref_array.shape:
            raise DimensionMismatch(
                f"point has shape {array.shape}, reference point has {ref_array.shape}"
            )
        distances.append(float(np.linalg.norm(array - ref_array)))
    return tuple(distances)


def drift_converged(
    distances: Sequence[float], epsilon: float, *, window: int = 1
) -> bool:
    """True iff the last ``window`` distances are all below ``epsilon``.

    The empirically checkable stand-in for the spec's ``lim_{t→∞}``: a tail
    window standing in for "eventually and thereafter", not a proof that no
    later sample would exceed ``epsilon``.
    """
    if epsilon <= 0.0:
        raise DimensionMismatch("epsilon must be positive")
    if window < 1:
        raise DimensionMismatch("window must be >= 1")
    if len(distances) < window:
        raise InsufficientHistory(
            f"need at least {window} distances, got {len(distances)}"
        )
    return all(distance < epsilon for distance in distances[-window:])


__all__ = ["drift_converged", "drift_distance", "runtime_stability_rate"]
