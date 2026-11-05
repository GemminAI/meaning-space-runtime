"""Trajectory geometry: measurement values over hidden-layer trajectories.

Independent of ``MeaningSpaceRuntime.ingest()``; imports no meaning_mapper,
HEKB, sensos or network code. Measurement layer: :mod:`msr.geometry.analyze`;
IC-00/IC-02 envelopes: :mod:`msr.geometry.emit`, :mod:`msr.geometry.calibration`.
"""

from __future__ import annotations

from msr.geometry.analyze import analyze_trajectory, compare_trajectories
from msr.geometry.calibration import (
    Calibration,
    CalibrationArtifact,
    NullMember,
    build_calibration,
    calibration_from_envelope,
    evaluate_shift,
    evaluate_shift_quantized,
    load_calibration,
    write_calibration,
)
from msr.geometry.emit import SERIES_ORDER, emit_pair, emit_single
from msr.geometry.envelope import (
    ArtifactRef,
    Blob,
    Emission,
    EmitContext,
    PolicyRef,
    write_blob,
)
from msr.geometry.errors import (
    CalibrationError,
    CrossSpaceComparison,
    InvalidTrajectory,
)
from msr.geometry.trajectory import Policy, TrajectoryInput
from msr.geometry.wire import ContractError

__all__ = [
    "SERIES_ORDER",
    "ArtifactRef",
    "Blob",
    "Calibration",
    "CalibrationArtifact",
    "CalibrationError",
    "ContractError",
    "CrossSpaceComparison",
    "Emission",
    "EmitContext",
    "InvalidTrajectory",
    "NullMember",
    "Policy",
    "PolicyRef",
    "TrajectoryInput",
    "analyze_trajectory",
    "build_calibration",
    "calibration_from_envelope",
    "compare_trajectories",
    "emit_pair",
    "emit_single",
    "evaluate_shift",
    "evaluate_shift_quantized",
    "load_calibration",
    "write_blob",
    "write_calibration",
]
