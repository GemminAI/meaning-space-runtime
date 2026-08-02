"""Non-normative in-process stand-ins for CLE, HEKB and NVS-Kernel.

These exist so the developmental loop can be *run and measured* without the
real neighbours. They are deliberately minimal and are **not** an
implementation of CLE or HEKB: no category axioms, no naturality check, no
persistence, no control law. Production wiring binds the real components to the
same ports in :mod:`msr.ports`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from msr.abi import KernelView, StabilizedTrajectory
from msr.adapters.hekb import field_prior_from_concepts
from msr.field import DEFAULT_BASIN_RADIUS, FieldPrior

#: Smallest width a lifted concept may claim, in state units.
#:
#: A dwell that barely moved has a near-singular covariance, whose inverse would
#: be an almost-delta well — knowledge sharper than any measurement that
#: produced it, and stiff enough to destabilize the flow. Concepts are floored
#: at this width instead.
MIN_CONCEPT_WIDTH = 0.05


@dataclass(slots=True)
class ReferenceConcept:
    """The HEKB ``Concept`` shape MSR's adapter reads (id + geometry)."""

    id: str
    centroid: tuple[float, ...] | None = None
    hessian: tuple[tuple[float, ...], ...] | None = None
    invariants: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ReferenceHEKB:
    """An in-memory concept store that can emit a field prior."""

    frame_id: str
    dimension: int
    basin_radius: float = DEFAULT_BASIN_RADIUS
    concepts: dict[str, ReferenceConcept] = field(default_factory=dict)
    version: int = 0

    def commit(self, concept: ReferenceConcept) -> None:
        self.concepts[concept.id] = concept
        self.version += 1

    def field_prior(self, frame_id: str, dimension: int) -> FieldPrior | None:
        if frame_id != self.frame_id or dimension != self.dimension:
            return None
        return field_prior_from_concepts(
            tuple(self.concepts.values()),
            frame_id=self.frame_id,
            dimension=self.dimension,
            version=self.version,
            basin_radius=self.basin_radius,
        )


@dataclass(slots=True)
class ReferenceCLE:
    """Commits stabilized trajectories into HEKB as concepts.

    Novel trajectories seed a new concept (Bootstrap); trajectories that
    stabilized inside a known basin reinforce that concept — its well deepens,
    which is what makes recognition faster the second time.
    """

    hekb: ReferenceHEKB
    reinforcement: float = 0.5
    min_width: float = MIN_CONCEPT_WIDTH
    lifted: list[StabilizedTrajectory] = field(default_factory=list)
    committed: list[ReferenceConcept] = field(default_factory=list)
    _minted: int = field(default=0, init=False)

    def lift(self, trajectory: StabilizedTrajectory) -> None:
        """Satisfies :class:`msr.ports.LiftPort`; the concept lands in HEKB."""
        self.committed.append(self._lift(trajectory))

    def _lift(self, trajectory: StabilizedTrajectory) -> ReferenceConcept:
        self.lifted.append(trajectory)
        if trajectory.basin_id is not None:
            return self._reinforce(trajectory, trajectory.basin_id)
        self._minted += 1
        concept = ReferenceConcept(
            id=f"concept-{self._minted:03d}",
            centroid=trajectory.centroid,
            hessian=self._precision_of(trajectory, self.min_width),
            invariants={"depth": 1.0},
        )
        self.hekb.commit(concept)
        return concept

    def _reinforce(
        self, trajectory: StabilizedTrajectory, concept_id: str
    ) -> ReferenceConcept:
        existing = self.hekb.concepts.get(concept_id)
        if existing is None:  # pragma: no cover - basin ids come from the field
            existing = ReferenceConcept(id=concept_id, centroid=trajectory.centroid)
        depth = float(existing.invariants.get("depth", 1.0)) + self.reinforcement
        concept = ReferenceConcept(
            id=concept_id,
            centroid=existing.centroid,
            hessian=existing.hessian,
            invariants={**existing.invariants, "depth": depth},
        )
        self.hekb.commit(concept)
        return concept

    @staticmethod
    def _precision_of(
        trajectory: StabilizedTrajectory, min_width: float
    ) -> tuple[tuple[float, ...], ...] | None:
        """Invert the dwell covariance into a well precision, width-floored.

        The floor is load-bearing, not cosmetic: a stationary dwell has a
        singular covariance whose raw inverse is a delta-like well. If inversion
        still fails, the adapter's isotropic fallback takes over (``None``).
        """
        covariance = np.asarray(trajectory.covariance, dtype=np.float64)
        dimension = covariance.shape[0]
        regularized = covariance + (min_width * min_width) * np.eye(dimension)
        try:
            precision = np.linalg.inv(regularized)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this rare
            return None
        symmetric = 0.5 * (precision + precision.T)
        if not np.all(np.isfinite(symmetric)):  # pragma: no cover - defensive
            return None
        return tuple(tuple(float(v) for v in row) for row in symmetric)


@dataclass(slots=True)
class RecordingKernel:
    """Records every fast-loop view; stands in for NVS-Kernel."""

    views: list[KernelView] = field(default_factory=list)

    def observe(self, view: KernelView) -> None:
        self.views.append(view)


__all__ = [
    "MIN_CONCEPT_WIDTH",
    "RecordingKernel",
    "ReferenceCLE",
    "ReferenceConcept",
    "ReferenceHEKB",
]
