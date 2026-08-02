"""``MSRHost`` — the wiring that makes both loops run.

::

    Observation → MM → [ MSR ] ─── KernelView ──────────► NVS-Kernel   (fast)
                          │
                          └─ StabilizedTrajectory ─► CLE → HEKB
                          ▲                                    │
                          └────────── FieldPrior ──────────────┘        (slow)

The host owns no meaning-space state of its own; it only routes. Every port is
optional, so MSR runs headless (no kernel), knowledge-frozen (no HEKB) or
lift-free (no CLE) without special-casing inside the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from msr.abi import MeaningMeasurement
from msr.ports import FieldPriorSource, KernelPort, LiftPort
from msr.runtime import MeaningSpaceRuntime, StepResult


@dataclass(slots=True)
class LoopMetrics:
    """Counters for what actually crossed each boundary."""

    steps: int = 0
    kernel_views: int = 0
    lifts: int = 0
    prior_refreshes: int = 0


@dataclass(slots=True)
class MSRHost:
    """Routes MSR output to the kernel and CLE, and HEKB's field back into MSR."""

    runtime: MeaningSpaceRuntime
    kernel: KernelPort | None = None
    cle: LiftPort | None = None
    hekb: FieldPriorSource | None = None
    refresh_on_stabilize: bool = True
    metrics: LoopMetrics = field(default_factory=LoopMetrics)

    def ingest(self, measurement: MeaningMeasurement) -> StepResult:
        """One full developmental step for one measurement."""
        result = self.runtime.ingest(measurement)
        self._dispatch(result)
        return result

    def advance(self, dt: float | None = None) -> StepResult:
        """One field-only step (kernel ticks faster than the mapper measures)."""
        result = self.runtime.advance(dt)
        self._dispatch(result)
        return result

    def _dispatch(self, result: StepResult) -> None:
        self.metrics.steps += 1
        if self.kernel is not None:
            self.kernel.observe(result.kernel_view)
            self.metrics.kernel_views += 1
        if result.stabilized is None:
            return
        if self.cle is not None:
            self.cle.lift(result.stabilized)
            self.metrics.lifts += 1
        if self.refresh_on_stabilize:
            self.refresh_field_prior()

    def refresh_field_prior(self) -> bool:
        """Pull Φ from HEKB. Returns True iff a newer field was installed.

        Version comparison, not identity: re-pulling an unchanged field must be
        a no-op, otherwise every refresh would reset the stabilization latch and
        the runtime could never hold a dwell.
        """
        if self.hekb is None:
            return False
        prior = self.hekb.field_prior(self.runtime.frame_id, self.runtime.dimension)
        if prior is None or prior.version <= self.runtime.field_prior.version:
            return False
        self.runtime.set_field_prior(prior)
        self.metrics.prior_refreshes += 1
        return True


__all__ = ["LoopMetrics", "MSRHost"]
