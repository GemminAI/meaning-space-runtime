"""Meaning Space Runtime (MSR) — the L2/L3 runtime layer of SensOS.

MSR receives semantic measurements from Meaning Mapper, evolves them into
trajectories in meaning space under a HEKB-supplied potential Φ, reports state
to NVS-Kernel on every step (fast loop), and emits stabilized trajectories to
CLE (slow loop).

MSR does not measure, name, persist or control. It evolves.
"""

from __future__ import annotations

from msr.abi import (
    KernelView,
    MeaningMeasurement,
    MeaningState,
    StabilizedTrajectory,
)
from msr.dynamics import InformationAssimilator, LangevinFlow
from msr.errors import (
    DimensionMismatch,
    FrameMismatch,
    MSRError,
    NotPositiveDefinite,
)
from msr.field import FieldPrior, GaussianWell
from msr.host import LoopMetrics, MSRHost
from msr.ports import FieldPriorSource, KernelPort, LiftPort
from msr.runtime import MeaningSpaceRuntime, RuntimeConfig, StepResult
from msr.stability import StabilizationCriteria, StabilizationDetector
from msr.version import __version__

__all__ = [
    "DimensionMismatch",
    "FieldPrior",
    "FieldPriorSource",
    "FrameMismatch",
    "GaussianWell",
    "InformationAssimilator",
    "KernelPort",
    "KernelView",
    "LangevinFlow",
    "LiftPort",
    "LoopMetrics",
    "MSRError",
    "MSRHost",
    "MeaningMeasurement",
    "MeaningSpaceRuntime",
    "MeaningState",
    "NotPositiveDefinite",
    "RuntimeConfig",
    "StabilizationCriteria",
    "StabilizationDetector",
    "StabilizedTrajectory",
    "StepResult",
    "__version__",
]
