"""Trajectory geometry: measurement values over hidden-layer trajectories.

Independent of ``MeaningSpaceRuntime.ingest()``; imports no meaning_mapper,
HEKB or network code.
"""

from __future__ import annotations

from msr.geometry.analyze import (
    analyze_trajectory,
    codebook_sha256,
    compare_trajectories,
    to_json_bytes,
)
from msr.geometry.calibration import (
    Calibration,
    build_calibration,
    evaluate_shift,
    load_calibration,
    write_calibration,
)
from msr.geometry.errors import (
    CalibrationError,
    CrossSpaceComparison,
    InvalidTrajectory,
)
from msr.geometry.trajectory import Policy, TrajectoryInput

__all__ = [
    "Calibration",
    "CalibrationError",
    "CrossSpaceComparison",
    "InvalidTrajectory",
    "Policy",
    "TrajectoryInput",
    "analyze_trajectory",
    "build_calibration",
    "codebook_sha256",
    "compare_trajectories",
    "evaluate_shift",
    "load_calibration",
    "to_json_bytes",
    "write_calibration",
]
