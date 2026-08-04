"""The L2→L3 evolution: measurement assimilation, then Langevin flow on Φ.

Each MSR step is two physically distinct half-steps, in this order:

1. **Assimilation** (information geometry, L2). The measurement (θ_obs, Σ) is
   fused into the state in information form:
   ``P' = P + Σ⁻¹``, ``θ' = P'⁻¹ (P θ + Σ⁻¹ θ_obs)``.
   This is the natural update on the statistical manifold — a sharper
   measurement (larger Σ⁻¹) moves the state further, a vague one barely at all.

2. **Flow** (meaning physics, L3). The state then relaxes along the field for
   ``dt``: ``θ ← θ - γ ∇Φ(θ) dt + √(2Tdt) ξ``, and its precision decays toward
   a floor, because information about *where the state is* ages.

Separating them is what keeps "what was measured" distinguishable from "what
the accumulated knowledge did to it".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from msr.errors import CFLViolation, DimensionMismatch
from msr.field import FieldPrior
from msr.linalg import Array


@dataclass(frozen=True, slots=True)
class AssimilationResult:
    """State after fusing one measurement."""

    theta: Array
    precision: Array
    displacement: float


@dataclass(frozen=True, slots=True)
class InformationAssimilator:
    """Information-form fusion of a measurement into the runtime state.

    ``forgetting_tau`` is the timescale over which positional information
    decays toward ``precision_floor``: without it, an old state would become
    infinitely certain and new measurements could never move it again.
    """

    precision_floor: float = 1e-3
    forgetting_tau: float = 5.0

    def __post_init__(self) -> None:
        if self.precision_floor <= 0.0:
            raise DimensionMismatch("precision_floor must be positive")
        if self.forgetting_tau <= 0.0:
            raise DimensionMismatch("forgetting_tau must be positive")

    def assimilate(
        self,
        theta: Array,
        precision: Array,
        measured_theta: Array,
        measured_precision: Array,
    ) -> AssimilationResult:
        """Fuse ``(measured_theta, measured_precision)`` into ``(theta, precision)``."""
        posterior_precision = precision + measured_precision
        information = precision @ theta + measured_precision @ measured_theta
        posterior_theta: Array = np.linalg.solve(posterior_precision, information)
        displacement = float(np.linalg.norm(posterior_theta - theta))
        return AssimilationResult(
            theta=posterior_theta,
            precision=0.5 * (posterior_precision + posterior_precision.T),
            displacement=displacement,
        )

    def decay(self, precision: Array, dt: float) -> Array:
        """Relax precision toward the floor over ``dt`` (exponential forgetting)."""
        retained = float(np.exp(-dt / self.forgetting_tau))
        floor = self.precision_floor * np.eye(precision.shape[0], dtype=np.float64)
        decayed: Array = retained * precision + (1.0 - retained) * floor
        return 0.5 * (decayed + decayed.T)


@dataclass(frozen=True, slots=True)
class LangevinFlow:
    """Overdamped Langevin relaxation on Φ.

    ``temperature`` 0 (the default) makes the runtime fully deterministic —
    every experiment below is reproducible bit-for-bit. A positive temperature
    enables the exploratory regime; it is seeded, never drawn from global RNG
    state, so runs stay reproducible there too.

    **Stiffness guard.** MSR does not control the field it is given: HEKB can
    commit an arbitrarily sharp concept, and explicit Euler on a sharp well
    oscillates and diverges whenever ``γ·λ_max(∇²Φ)·dt`` exceeds ~2. The step is
    therefore subdivided so that each substep satisfies
    ``γ·stiffness·sub_dt ≤ stability_factor`` — a CFL-style condition on the
    field's curvature, not on the distance travelled — and additionally so that
    no substep displaces the state by more than ``max_displacement``, which
    covers the far-field case where the gradient is large but the curvature
    bound is loose. Without this, a legitimately-committed concept can blow up
    the runtime that committed it.

    **Fail closed.** ``max_substeps`` is a hard ceiling, not a cap: a field
    stiff enough to need more substeps than that raises :class:`CFLViolation`
    rather than integrating an under-resolved (and therefore no-longer-stable)
    step under the appearance of a stable one.
    """

    mobility: float = 1.0
    temperature: float = 0.0
    seed: int = 0
    stability_factor: float = 1.0
    max_displacement: float = 0.25
    max_substeps: int = 4096

    def __post_init__(self) -> None:
        if self.mobility <= 0.0:
            raise DimensionMismatch("mobility must be positive")
        if self.temperature < 0.0:
            raise DimensionMismatch("temperature must be non-negative")
        if not 0.0 < self.stability_factor < 2.0:
            raise DimensionMismatch("stability_factor must lie in (0, 2)")
        if self.max_displacement <= 0.0:
            raise DimensionMismatch("max_displacement must be positive")
        if self.max_substeps < 1:
            raise DimensionMismatch("max_substeps must be >= 1")

    def substep_count(self, theta: Array, prior: FieldPrior, dt: float) -> int:
        """How many substeps this ``dt`` needs at ``theta`` to stay stable.

        Raises :class:`CFLViolation` if that count exceeds ``max_substeps``
        instead of silently truncating to it — a truncated count no longer
        satisfies the stability condition it was computed for.
        """
        curvature = self.mobility * prior.stiffness * dt / self.stability_factor
        drift = float(np.linalg.norm(prior.gradient(theta))) * self.mobility * dt
        needed = max(curvature, drift / self.max_displacement)
        if needed <= 1.0:
            return 1
        substeps = int(np.ceil(needed))
        if substeps > self.max_substeps:
            raise CFLViolation(
                f"field curvature/drift requires {substeps} substeps for "
                f"dt={dt}, exceeding max_substeps={self.max_substeps}"
            )
        return substeps

    def make_rng(self) -> np.random.Generator:
        """A fresh generator for one runtime instance's whole lifetime."""
        return np.random.default_rng(self.seed)

    def step(
        self,
        theta: Array,
        prior: FieldPrior,
        dt: float,
        rng: np.random.Generator,
    ) -> Array:
        """Advance ``theta`` by one ``dt`` of relaxation on ``prior``'s field.

        Substepping keeps the total noise variance at ``2 T dt`` regardless of
        how many substeps are taken, so the stiffness guard does not change the
        temperature the trajectory actually experiences.
        """
        if dt <= 0.0:
            raise DimensionMismatch("dt must be positive")
        substeps = self.substep_count(theta, prior, dt)
        sub_dt = dt / substeps
        scale = (
            float(np.sqrt(2.0 * self.temperature * sub_dt))
            if self.temperature > 0.0
            else 0.0
        )
        moved: Array = theta
        for _ in range(substeps):
            moved = moved - self.mobility * prior.gradient(moved) * sub_dt
            if scale > 0.0:
                moved = moved + rng.standard_normal(moved.shape[0]) * scale
        return moved


__all__ = ["AssimilationResult", "InformationAssimilator", "LangevinFlow"]
