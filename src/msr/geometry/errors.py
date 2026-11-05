"""Errors raised by the trajectory-geometry package."""

from __future__ import annotations

from msr.errors import MSRError


class InvalidTrajectory(MSRError):
    """A TrajectoryInput violated its contract (shape, source, policy, ...)."""


class CrossSpaceComparison(MSRError):
    """Two trajectories with different ``space_id`` were compared (D-69, H6)."""


class CalibrationError(MSRError):
    """A calibration file is malformed, mismatched or cannot be built."""


__all__ = ["CalibrationError", "CrossSpaceComparison", "InvalidTrajectory"]
