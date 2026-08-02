"""The meaning-space potential Φ and its basins.

Φ is *not* owned by MSR: it is the field prior HEKB supplies, expressed as a
sum of Gaussian wells, one per known concept. MSR only evaluates it. This is
what closes the slow loop — knowledge changes the shape of the space that
future observations flow through.

.. math::

    \\Phi(\\theta) = -\\sum_i d_i
        \\exp\\!\\big(-\\tfrac12 (\\theta - m_i)^T P_i (\\theta - m_i)\\big)

Empty prior (Bootstrap Mode) gives Φ ≡ 0 and ∇Φ ≡ 0: with no knowledge, the
space is flat and motion is driven purely by measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from msr.errors import DimensionMismatch, FrameMismatch
from msr.linalg import Array, as_spd, as_vector

#: Mahalanobis radius (in σ) within which a point is considered inside a basin.
DEFAULT_BASIN_RADIUS = 2.0


@dataclass(frozen=True, slots=True)
class GaussianWell:
    """One basin of Φ: a known concept's attractor in meaning space.

    ``precision`` is the well's inverse-covariance (sharpness) and ``depth``
    its energetic weight — how strongly established knowledge pulls an
    in-flight trajectory toward the concept it already knows.
    """

    well_id: str
    mean: Array
    precision: Array
    depth: float

    def __post_init__(self) -> None:
        if self.depth <= 0.0:
            raise DimensionMismatch("well depth must be positive")
        dimension = self.mean.shape[0]
        object.__setattr__(self, "mean", as_vector(self.mean, dimension, name="mean"))
        object.__setattr__(
            self, "precision", as_spd(self.precision, dimension, name="precision")
        )

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def mahalanobis(self, theta: Array) -> float:
        """Mahalanobis distance from ``theta`` to this well's mean."""
        delta = theta - self.mean
        quadratic = float(delta @ self.precision @ delta)
        return math.sqrt(max(quadratic, 0.0))

    @classmethod
    def isotropic(
        cls, well_id: str, mean: object, width: float, depth: float = 1.0
    ) -> GaussianWell:
        """Build a spherical well of standard deviation ``width``."""
        if width <= 0.0:
            raise DimensionMismatch("well width must be positive")
        center = np.asarray(mean, dtype=np.float64).reshape(-1)
        precision = np.eye(center.size, dtype=np.float64) / (width * width)
        return cls(well_id=well_id, mean=center, precision=precision, depth=depth)


@dataclass(frozen=True, slots=True)
class FieldPrior:
    """Φ over one frame, as a set of wells. Immutable; replaced, never mutated.

    Replacement rather than mutation is deliberate: a running trajectory must
    never observe a half-updated field, and every step is attributable to
    exactly one field version.
    """

    frame_id: str
    dimension: int
    wells: tuple[GaussianWell, ...] = ()
    basin_radius: float = DEFAULT_BASIN_RADIUS
    version: int = 0
    _stiffness: float = field(default=0.0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise DimensionMismatch("dimension must be positive")
        for well in self.wells:
            if well.dimension != self.dimension:
                raise DimensionMismatch(
                    f"well {well.well_id!r} has dimension {well.dimension}, "
                    f"field has {self.dimension}"
                )
        object.__setattr__(self, "_stiffness", self._compute_stiffness())

    def _compute_stiffness(self) -> float:
        """Upper bound on ‖∇²Φ‖ over the whole space.

        ``∇²Φ = Σ d_i e^{-½Q}(P_i - P_iΔΔᵀP_i)``, and the exponential and the
        rank-one term are both bounded by 1 and 0 respectively at worst, so
        ``Σ d_i λ_max(P_i)`` bounds the curvature everywhere. Integrators use it
        to pick a stable step; it is computed once per field, not per step,
        because fields are replaced rather than mutated.
        """
        total = 0.0
        for well in self.wells:
            total += well.depth * float(np.max(np.linalg.eigvalsh(well.precision)))
        return total

    @property
    def stiffness(self) -> float:
        """Curvature bound of Φ; 0 for an empty (flat) field."""
        return self._stiffness

    @property
    def is_empty(self) -> bool:
        """True in Bootstrap Mode: no knowledge yet, so the space is flat."""
        return len(self.wells) == 0

    def require_frame(self, frame_id: str) -> None:
        if frame_id != self.frame_id:
            raise FrameMismatch(
                f"field prior is in frame {self.frame_id!r}, got {frame_id!r}"
            )

    def _weights(self, theta: Array) -> Array:
        """Per-well Gaussian responses ``d_i * exp(-½ Δᵀ P Δ)`` at ``theta``."""
        responses = np.empty(len(self.wells), dtype=np.float64)
        for index, well in enumerate(self.wells):
            delta = theta - well.mean
            quadratic = float(delta @ well.precision @ delta)
            responses[index] = well.depth * math.exp(-0.5 * quadratic)
        return responses

    def potential(self, theta: Array) -> float:
        """Φ(θ). Zero for an empty field; negative inside any well."""
        if self.is_empty:
            return 0.0
        return float(-np.sum(self._weights(theta)))

    def gradient(self, theta: Array) -> Array:
        """∇Φ(θ). Descending it moves toward the nearest dominant well."""
        gradient = np.zeros(self.dimension, dtype=np.float64)
        if self.is_empty:
            return gradient
        responses = self._weights(theta)
        for response, well in zip(responses, self.wells, strict=True):
            gradient += response * (well.precision @ (theta - well.mean))
        return gradient

    def basin_of(self, theta: Array) -> str | None:
        """Which basin ``theta`` currently belongs to, or ``None`` if outside all.

        ``None`` is the physically meaningful answer in Bootstrap Mode and for
        genuinely novel meaning — it is what marks a stabilization as new
        knowledge rather than recognition of old knowledge.
        """
        best_id: str | None = None
        best_distance = self.basin_radius
        for well in self.wells:
            distance = well.mahalanobis(theta)
            if distance <= best_distance:
                best_distance = distance
                best_id = well.well_id
        return best_id

    def with_well(self, well: GaussianWell) -> FieldPrior:
        """Return a new field with ``well`` added (or replaced by id), version+1."""
        if well.dimension != self.dimension:
            raise DimensionMismatch(
                f"well {well.well_id!r} has dimension {well.dimension}, "
                f"field has {self.dimension}"
            )
        kept = tuple(w for w in self.wells if w.well_id != well.well_id)
        return FieldPrior(
            frame_id=self.frame_id,
            dimension=self.dimension,
            wells=(*kept, well),
            basin_radius=self.basin_radius,
            version=self.version + 1,
        )

    @classmethod
    def empty(
        cls, frame_id: str, dimension: int, basin_radius: float = DEFAULT_BASIN_RADIUS
    ) -> FieldPrior:
        """The Bootstrap-Mode field: no wells, flat space."""
        return cls(
            frame_id=frame_id,
            dimension=dimension,
            wells=(),
            basin_radius=basin_radius,
        )


__all__ = ["DEFAULT_BASIN_RADIUS", "FieldPrior", "GaussianWell"]
