"""``MeaningSpaceRuntime`` — the L2/L3 runtime itself.

One instance owns one frame. It holds exactly one mutable thing (the current
position and precision in meaning space) and emits only frozen values. It does
not persist, does not name, does not lift, and does not decide: naming is CLE's,
persistence is HEKB's, control is NVS-Kernel's.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from msr.abi import KernelView, MeaningMeasurement, MeaningState, StabilizedTrajectory
from msr.dynamics import InformationAssimilator, LangevinFlow
from msr.errors import DimensionMismatch, FrameMismatch
from msr.field import FieldPrior
from msr.linalg import Array, as_spd, as_vector, invert_spd, to_matrix_tuple, to_tuple
from msr.stability import StabilizationDetector

NANOS_PER_SECOND = 1e9


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Timing bounds and history depth.

    ``min_dt``/``max_dt`` clamp the interval derived from measurement
    timestamps: a burst of same-microsecond measurements must not divide by
    ~zero, and a long silence must not integrate the field in one huge
    unstable jump.
    """

    default_dt: float = 0.05
    min_dt: float = 1e-4
    max_dt: float = 1.0
    history_capacity: int = 4096

    def __post_init__(self) -> None:
        if not 0.0 < self.min_dt <= self.max_dt:
            raise DimensionMismatch("require 0 < min_dt <= max_dt")
        if not self.min_dt <= self.default_dt <= self.max_dt:
            raise DimensionMismatch("default_dt must lie within [min_dt, max_dt]")
        if self.history_capacity < 1:
            raise DimensionMismatch("history_capacity must be >= 1")

    def clamp(self, dt: float) -> float:
        return min(max(dt, self.min_dt), self.max_dt)


@dataclass(frozen=True, slots=True)
class StepResult:
    """What one MSR step produced, for both loops at once."""

    state: MeaningState
    kernel_view: KernelView
    stabilized: StabilizedTrajectory | None = None

    @property
    def did_stabilize(self) -> bool:
        return self.stabilized is not None


@dataclass(slots=True)
class MeaningSpaceRuntime:
    """Observation → meaning-space trajectory, under a HEKB-supplied field."""

    frame_id: str
    dimension: int
    prior: FieldPrior | None = None
    flow: LangevinFlow = field(default_factory=LangevinFlow)
    assimilator: InformationAssimilator = field(default_factory=InformationAssimilator)
    detector: StabilizationDetector = field(default_factory=StabilizationDetector)
    config: RuntimeConfig = field(default_factory=RuntimeConfig)

    _theta: Array | None = field(default=None, init=False)
    _precision: Array | None = field(default=None, init=False)
    _step_index: int = field(default=-1, init=False)
    _time_s: float = field(default=0.0, init=False)
    _last_ns: int | None = field(default=None, init=False)
    _state: MeaningState | None = field(default=None, init=False)
    _history: deque[MeaningState] = field(default_factory=deque, init=False)
    _rng: np.random.Generator | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise DimensionMismatch("dimension must be positive")
        if self.prior is None:
            self.prior = FieldPrior.empty(self.frame_id, self.dimension)
        else:
            self._check_prior(self.prior)
        self._history = deque(maxlen=self.config.history_capacity)
        self._rng = self.flow.make_rng()

    # ---------------------------------------------------------------- state

    @property
    def field_prior(self) -> FieldPrior:
        assert self.prior is not None  # established in __post_init__
        return self.prior

    @property
    def state(self) -> MeaningState | None:
        """Latest emitted state, or ``None`` before the first measurement."""
        return self._state

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def history(self) -> tuple[MeaningState, ...]:
        return tuple(self._history)

    @property
    def is_bootstrap(self) -> bool:
        """True while no knowledge shapes the field (RFC-SensOS24 Bootstrap Mode)."""
        return self.field_prior.is_empty

    # ------------------------------------------------------------ slow loop

    def _check_prior(self, prior: FieldPrior) -> None:
        prior.require_frame(self.frame_id)
        if prior.dimension != self.dimension:
            raise DimensionMismatch(
                f"field prior has dimension {prior.dimension}, "
                f"runtime has {self.dimension}"
            )

    def set_field_prior(self, prior: FieldPrior) -> bool:
        """Install a new Φ (the HEKB → MSR arrow that closes the slow loop).

        Returns whether the current dwell was invalidated. The latch is dropped
        **only if the new field puts the current position in a different basin**
        — a field whose change does not reclassify the state leaves the dwell a
        valid statement.

        Resetting unconditionally would make the slow loop self-triggering: CLE
        reinforces a concept, HEKB's version bumps, the latch drops, the same
        state re-stabilizes, CLE reinforces again — an unbounded stream of
        duplicate trajectories from a runtime that never moved.
        """
        self._check_prior(prior)
        previous_basin = self._state.basin_id if self._state is not None else None
        self.prior = prior
        if self._theta is None:
            self.detector.reset()
            return True
        if prior.basin_of(self._theta) != previous_basin:
            self.detector.reset()
            return True
        return False

    # ------------------------------------------------------------ fast loop

    def ingest(self, measurement: MeaningMeasurement) -> StepResult:
        """Assimilate one measurement and advance the trajectory by one step."""
        measurement.validate()
        if measurement.frame_id != self.frame_id:
            raise FrameMismatch(
                f"runtime is in frame {self.frame_id!r}, "
                f"measurement is in {measurement.frame_id!r}"
            )
        if measurement.dimension != self.dimension:
            raise DimensionMismatch(
                f"runtime has dimension {self.dimension}, "
                f"measurement has {measurement.dimension}"
            )

        measured_theta = as_vector(measurement.theta, self.dimension, name="theta")
        sigma = as_spd(measurement.sigma, self.dimension, name="sigma")
        measured_precision = invert_spd(sigma, name="sigma")

        if self._theta is None or self._precision is None:
            return self._initialize(measurement, measured_theta, measured_precision)

        dt = self._elapsed(measurement.timestamp_ns)
        previous = self._theta
        fused = self.assimilator.assimilate(
            previous, self._precision, measured_theta, measured_precision
        )
        precision = self.assimilator.decay(fused.precision, dt)
        theta = self.flow.step(fused.theta, self.field_prior, dt, self._generator())
        speed = float(np.linalg.norm(theta - previous)) / dt
        return self._commit(
            theta=theta,
            precision=precision,
            dt=dt,
            speed=speed,
            timestamp_ns=measurement.timestamp_ns,
            source_observation_id=measurement.observation_id,
        )

    def advance(self, dt: float | None = None) -> StepResult:
        """Advance the field flow without a measurement (kernel-rate tick).

        The kernel may tick faster than the mapper measures; between
        measurements the trajectory still relaxes on Φ and still reports.
        """
        if self._theta is None or self._precision is None:
            raise DimensionMismatch("cannot advance before the first measurement")
        step_dt = self.config.clamp(self.config.default_dt if dt is None else dt)
        previous = self._theta
        precision = self.assimilator.decay(self._precision, step_dt)
        theta = self.flow.step(previous, self.field_prior, step_dt, self._generator())
        speed = float(np.linalg.norm(theta - previous)) / step_dt
        assert self._last_ns is not None
        timestamp_ns = self._last_ns + int(step_dt * NANOS_PER_SECOND)
        return self._commit(
            theta=theta,
            precision=precision,
            dt=step_dt,
            speed=speed,
            timestamp_ns=timestamp_ns,
            source_observation_id=None,
        )

    # ------------------------------------------------------------- internals

    def _generator(self) -> np.random.Generator:
        assert self._rng is not None  # established in __post_init__
        return self._rng

    def _elapsed(self, timestamp_ns: int) -> float:
        assert self._last_ns is not None
        return self.config.clamp((timestamp_ns - self._last_ns) / NANOS_PER_SECOND)

    def _initialize(
        self,
        measurement: MeaningMeasurement,
        measured_theta: Array,
        measured_precision: Array,
    ) -> StepResult:
        """First measurement: the state *is* the measurement, at rest.

        No flow is integrated, because there is no elapsed interval to
        integrate over and no prior position to have moved from.
        """
        return self._commit(
            theta=measured_theta,
            precision=measured_precision,
            dt=0.0,
            speed=0.0,
            timestamp_ns=measurement.timestamp_ns,
            source_observation_id=measurement.observation_id,
        )

    def _commit(
        self,
        *,
        theta: Array,
        precision: Array,
        dt: float,
        speed: float,
        timestamp_ns: int,
        source_observation_id: str | None,
    ) -> StepResult:
        self._theta = theta
        self._precision = precision
        self._step_index += 1
        # Runtime time advances by the *clamped* interval actually integrated,
        # not by wall-clock delta: speed, dwell_seconds and the trajectory all
        # have to refer to the same time the dynamics were evolved over.
        self._time_s += dt
        self._last_ns = timestamp_ns

        prior = self.field_prior
        state = MeaningState(
            frame_id=self.frame_id,
            step_index=self._step_index,
            time_s=self._time_s,
            theta=to_tuple(theta),
            precision=to_matrix_tuple(precision),
            speed=speed,
            potential=prior.potential(theta),
            basin_id=prior.basin_of(theta),
            source_observation_id=source_observation_id,
        )
        self._state = state
        self._history.append(state)

        stabilized = self.detector.observe(state)
        view = KernelView(
            frame_id=state.frame_id,
            step_index=state.step_index,
            time_s=state.time_s,
            theta=state.theta,
            speed=state.speed,
            potential=state.potential,
            basin_id=state.basin_id,
            precision_trace=float(np.trace(precision)),
            stabilized=self.detector.is_stable,
        )
        return StepResult(state=state, kernel_view=view, stabilized=stabilized)

    def snapshot(self) -> dict[str, Any]:
        """Diagnostic summary — for experiments and operators, not for the ABI."""
        return {
            "frame_id": self.frame_id,
            "dimension": self.dimension,
            "step_index": self._step_index,
            "time_s": self._time_s,
            "field_version": self.field_prior.version,
            "well_count": len(self.field_prior.wells),
            "bootstrap": self.is_bootstrap,
            "stable": self.detector.is_stable,
            "dwell_length": self.detector.dwell_length,
        }


__all__ = ["NANOS_PER_SECOND", "MeaningSpaceRuntime", "RuntimeConfig", "StepResult"]
