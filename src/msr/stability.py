"""Stabilization detection: when a trajectory has become knowledge-shaped.

MSR emits to CLE only when a trajectory *stops moving inside one basin* — the
``T`` in CLE's ``T → T/≈ → K``. The criterion is deliberately mechanical (speed
under a threshold, basin unchanged, for a dwell of N steps) so that "stabilized"
is a measurable predicate, not a judgment call.

The detector latches: one stabilization emits exactly one trajectory, and no
further emission happens until the state actually destabilizes again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from msr.abi import MeaningState, StabilizedTrajectory
from msr.errors import DimensionMismatch
from msr.linalg import to_matrix_tuple, to_tuple


@dataclass(frozen=True, slots=True)
class StabilizationCriteria:
    """Thresholds defining "stabilized"."""

    speed_threshold: float = 1e-2
    dwell_steps: int = 5
    max_segment: int = 4096

    def __post_init__(self) -> None:
        if self.speed_threshold <= 0.0:
            raise DimensionMismatch("speed_threshold must be positive")
        if self.dwell_steps < 1:
            raise DimensionMismatch("dwell_steps must be >= 1")
        if self.max_segment < self.dwell_steps:
            raise DimensionMismatch("max_segment must be >= dwell_steps")


@dataclass(slots=True)
class StabilizationDetector:
    """Accumulates quiescent states and emits one trajectory per stabilization."""

    criteria: StabilizationCriteria = field(default_factory=StabilizationCriteria)
    _segment: list[MeaningState] = field(default_factory=list, init=False)
    _basin_id: str | None = field(default=None, init=False)
    _latched: bool = field(default=False, init=False)
    _emitted: int = field(default=0, init=False)

    @property
    def is_stable(self) -> bool:
        """True once the current dwell has satisfied the criteria."""
        return self._latched

    @property
    def dwell_length(self) -> int:
        return len(self._segment)

    def reset(self) -> None:
        """Drop the current dwell (used when the field prior is replaced)."""
        self._segment.clear()
        self._basin_id = None
        self._latched = False

    def observe(self, state: MeaningState) -> StabilizedTrajectory | None:
        """Feed one state; return a trajectory iff this step stabilized it."""
        quiescent = state.speed <= self.criteria.speed_threshold
        continues = bool(self._segment) and state.basin_id == self._basin_id
        if not quiescent:
            self.reset()
            return None
        if not continues:
            self._segment = [state]
            self._basin_id = state.basin_id
            self._latched = False
            return None

        self._segment.append(state)
        if len(self._segment) > self.criteria.max_segment:
            del self._segment[0]
        if self._latched or len(self._segment) < self.criteria.dwell_steps:
            return None
        self._latched = True
        return self._build()

    def _build(self) -> StabilizedTrajectory:
        states = tuple(self._segment)
        points = np.array([state.theta for state in states], dtype=np.float64)
        centroid = points.mean(axis=0)
        if points.shape[0] > 1:
            covariance = np.cov(points, rowvar=False, ddof=0).reshape(
                points.shape[1], points.shape[1]
            )
        else:  # pragma: no cover - dwell_steps >= 1 makes this reachable only at 1
            covariance = np.zeros((points.shape[1], points.shape[1]), dtype=np.float64)
        provenance: list[str] = []
        for state in states:
            source = state.source_observation_id
            if source is not None and source not in provenance:
                provenance.append(source)
        self._emitted += 1
        return StabilizedTrajectory(
            trajectory_id=f"traj-{states[0].frame_id}-{states[0].step_index:06d}",
            frame_id=states[0].frame_id,
            basin_id=self._basin_id,
            states=states,
            centroid=to_tuple(centroid),
            covariance=to_matrix_tuple(covariance),
            dwell_steps=len(states),
            dwell_seconds=states[-1].time_s - states[0].time_s,
            provenance=tuple(provenance),
        )


__all__ = ["StabilizationCriteria", "StabilizationDetector"]
