"""The MSR ABI: the frozen value types crossing every MSR boundary.

Four types, four boundaries:

===========================  ==========================================
:class:`MeaningMeasurement`  Meaning Mapper -> MSR   (inbound, L1->L2)
:class:`MeaningState`        MSR internal snapshot   (observable)
:class:`KernelView`          MSR -> NVS-Kernel       (fast loop)
:class:`StabilizedTrajectory` MSR -> CLE             (slow loop)
===========================  ==========================================

All four are frozen and hold only plain Python scalars/tuples: once emitted, a
value can never be retroactively changed by later runtime evolution, and it can
be serialized without touching numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from msr.errors import DimensionMismatch
from msr.linalg import as_spd, as_vector, to_matrix_tuple, to_tuple

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class MeaningMeasurement:
    """A single semantic measurement: the sole inbound MSR event.

    This is Meaning Mapper's output read as an instrument reading, not as an
    annotation: a position ``theta`` in meaning space, in a declared frame,
    with a declared uncertainty ``sigma``. MSR never interprets ``provenance``;
    it only carries it through to the stabilized trajectory so that committed
    knowledge keeps a trace back to the observations that produced it.
    """

    observation_id: str
    frame_id: str
    theta: Vector
    sigma: Matrix
    timestamp_ns: int
    provenance: tuple[str, ...] = ()

    @property
    def dimension(self) -> int:
        return len(self.theta)

    def validate(self) -> None:
        """Check shapes and SPD-ness of Σ; raises on any defect."""
        dimension = self.dimension
        if dimension == 0:
            raise DimensionMismatch("theta must be non-empty")
        as_vector(self.theta, dimension, name="theta")
        as_spd(self.sigma, dimension, name="sigma")

    @classmethod
    def isotropic(
        cls,
        observation_id: str,
        frame_id: str,
        theta: object,
        variance: float,
        timestamp_ns: int,
        provenance: tuple[str, ...] = (),
    ) -> MeaningMeasurement:
        """Build a measurement with covariance ``variance * I``.

        Convenience for instruments that report a scalar uncertainty; the
        runtime still sees a full Σ, so no code path special-cases it.
        """
        values = np.asarray(theta, dtype=np.float64).reshape(-1)
        if variance <= 0.0:
            raise DimensionMismatch("variance must be positive")
        return cls(
            observation_id=observation_id,
            frame_id=frame_id,
            theta=to_tuple(values),
            sigma=to_matrix_tuple(variance * np.eye(values.size, dtype=np.float64)),
            timestamp_ns=timestamp_ns,
            provenance=provenance,
        )


@dataclass(frozen=True, slots=True)
class MeaningState:
    """An immutable snapshot of the runtime's position in meaning space."""

    frame_id: str
    step_index: int
    time_s: float
    theta: Vector
    precision: Matrix
    speed: float
    potential: float
    basin_id: str | None
    source_observation_id: str | None

    @property
    def dimension(self) -> int:
        return len(self.theta)


@dataclass(frozen=True, slots=True)
class KernelView:
    """Fast-loop payload handed to NVS-Kernel on every step.

    Deliberately flat and small: the kernel is a controller, so it receives the
    control-relevant reduction of the state (where, how fast, which direction
    the field pulls, how deep, how certain, which basin) rather than the
    runtime's internals. ``gradient`` is ∇Φ(θ) at the reported position — the
    same quantity the flow step just descended — so a kernel that needs the
    pull direction does not have to recompute it against its own copy of Φ.
    """

    frame_id: str
    step_index: int
    time_s: float
    theta: Vector
    speed: float
    potential: float
    gradient: Vector
    basin_id: str | None
    precision_trace: float
    stabilized: bool

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form, for transport to an out-of-process kernel."""
        return {
            "frame_id": self.frame_id,
            "step_index": self.step_index,
            "time_s": self.time_s,
            "theta": list(self.theta),
            "speed": self.speed,
            "potential": self.potential,
            "gradient": list(self.gradient),
            "basin_id": self.basin_id,
            "precision_trace": self.precision_trace,
            "stabilized": self.stabilized,
        }


@dataclass(frozen=True, slots=True)
class StabilizedTrajectory:
    """Slow-loop payload handed to CLE: a trajectory MSR considers stabilized.

    ``centroid``/``covariance`` summarize the dwell segment, and are exactly the
    geometric metadata HEKB echoes on a ``Concept`` once CLE lifts this
    trajectory — MSR computes them, CLE and HEKB only carry them.
    """

    trajectory_id: str
    frame_id: str
    basin_id: str | None
    states: tuple[MeaningState, ...]
    centroid: Vector
    covariance: Matrix
    dwell_steps: int
    dwell_seconds: float
    provenance: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dimension(self) -> int:
        return len(self.centroid)

    @property
    def is_novel(self) -> bool:
        """True when the trajectory stabilized outside every known basin.

        A novel stabilization is what seeds new knowledge in Bootstrap Mode: no
        field prior explained it, so CLE has something to lift that HEKB does
        not already contain.
        """
        return self.basin_id is None


__all__ = [
    "KernelView",
    "Matrix",
    "MeaningMeasurement",
    "MeaningState",
    "StabilizedTrajectory",
    "Vector",
]
