"""EXP-Ubuntu004's stabilization invariant: the largest Lyapunov exponent.

The spec's Yield Rate criterion is "a normal trajectory that satisfies the
invariant condition (Lyapunov exponent λ_Lyapunov < 0)" — a *different*
quantity from :func:`msr.metrics.runtime_stability_rate`'s λ (a covariance
convergence rate). This module estimates the dynamical-systems quantity:
how fast two infinitesimally-separated trajectories starting near the same
point diverge (λ > 0) or converge (λ < 0) under the field's flow.

**Method: Benettin's algorithm**, applied to the deterministic drift only.
Alongside the already-realized trajectory θ₀, θ₁, ..., θₙ (from
``MeaningState`` history), a tangent (perturbation) vector δ is propagated by
the variational equation ``δθ̇ = J(θ) δθ``, where ``J(θ) = -mobility ·
∇²Φ(θ)`` is the Jacobian of the drift (:meth:`msr.field.FieldPrior.hessian`
is exactly ∇²Φ). δ is renormalized to unit length after every step; the
exponent is the time-averaged log growth factor:

.. math::

    \\lambda = \\frac{1}{T} \\sum_k \\ln \\Vert \\delta_k \\Vert
        \\quad\\text{(before each renormalization)}

Three approximations, documented here rather than hidden in the code:

1. **Deterministic backbone only.** The variational equation carries no
   stochastic term. At ``temperature=0`` (MSR's deterministic-replay
   default) this is exact; at ``temperature>0`` it reports the local
   expansion rate of the drift the noise is superimposed on, not a
   rigorous stochastic Lyapunov exponent.
2. **Single-step Euler per recorded interval.** The tangent map is
   integrated with one explicit-Euler step per *recorded* ``dt`` (the
   interval between consecutive ``MeaningState``s), not with the same
   adaptive substepping :meth:`msr.dynamics.LangevinFlow.step` privately
   applied to keep the *primal* trajectory stable. On a stiff field this
   makes the tangent-map estimate coarser than the trajectory it rides on.
3. **Fixed initial direction.** δ₀ is the normalized all-ones vector, not a
   random or seeded one. For the *largest* exponent this is a standard
   simplification — almost every generic direction aligns with the
   dominant expanding direction within a few steps — but it is not immune
   to the measure-zero case where δ₀ starts exactly orthogonal to that
   direction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from msr.abi import MeaningState
from msr.errors import DimensionMismatch, InsufficientHistory
from msr.field import FieldPrior
from msr.linalg import Array


@dataclass(frozen=True, slots=True)
class LyapunovEstimate:
    """A finite-time largest-Lyapunov-exponent estimate, with its own provenance.

    ``exponent < 0`` is the spec's stabilization invariant (nearby
    trajectories converge); ``exponent >= 0`` means they do not, by this
    estimate, over this window.
    """

    exponent: float
    steps: int
    window_seconds: float

    @property
    def is_contracting(self) -> bool:
        """The spec's stabilization invariant: λ_Lyapunov < 0."""
        return self.exponent < 0.0


def _initial_direction(dimension: int) -> Array:
    """A fixed, generic unit vector — deterministic, no seed required."""
    direction = np.ones(dimension, dtype=np.float64)
    return direction / float(np.linalg.norm(direction))


def largest_lyapunov_exponent(
    states: Sequence[MeaningState],
    prior: FieldPrior,
    mobility: float,
) -> LyapunovEstimate:
    """Estimate the largest Lyapunov exponent along an already-realized trajectory.

    ``prior`` and ``mobility`` **MUST** be the same field prior and
    :attr:`msr.dynamics.LangevinFlow.mobility` that produced ``states`` — this
    function does not re-derive them, it only differentiates the drift they
    imply.
    """
    if len(states) < 2:
        raise InsufficientHistory("need at least 2 states to estimate an exponent")
    window_seconds = states[-1].time_s - states[0].time_s
    if window_seconds <= 0.0:
        raise InsufficientHistory("states must span a positive elapsed time")
    if mobility <= 0.0:
        raise DimensionMismatch("mobility must be positive")

    dimension = states[0].dimension
    identity = np.eye(dimension, dtype=np.float64)
    delta = _initial_direction(dimension)
    log_sum = 0.0
    steps = 0

    for previous, current in pairwise(states):
        dt = current.time_s - previous.time_s
        if dt <= 0.0:
            raise DimensionMismatch("states must have strictly increasing time_s")
        theta = np.asarray(previous.theta, dtype=np.float64)
        jacobian = -mobility * prior.hessian(theta)
        delta = (identity + dt * jacobian) @ delta
        norm = float(np.linalg.norm(delta))
        if norm <= 0.0:  # pragma: no cover - a genuine fixed point of the tangent map
            raise InsufficientHistory(
                "perturbation collapsed to zero; cannot renormalize"
            )
        log_sum += math.log(norm)
        delta = delta / norm
        steps += 1

    return LyapunovEstimate(
        exponent=log_sum / window_seconds,
        steps=steps,
        window_seconds=window_seconds,
    )


__all__ = ["LyapunovEstimate", "largest_lyapunov_exponent"]
