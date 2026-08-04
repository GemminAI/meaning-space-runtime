"""MSR error taxonomy.

Every error raised across an MSR boundary derives from :class:`MSRError`, so a
host runtime can distinguish "the measurement was rejected" from "the runtime
itself is broken" without catching bare exceptions.
"""

from __future__ import annotations


class MSRError(Exception):
    """Base class for every error raised by the Meaning Space Runtime."""


class FrameMismatch(MSRError):
    """A measurement or field prior declared a frame the runtime does not own.

    MSR never rebases coordinates. A measurement taken in frame ``F'`` is not
    comparable to state held in frame ``F``, so it is rejected rather than
    silently reinterpreted.
    """


class DimensionMismatch(MSRError):
    """A vector or matrix did not match the runtime's meaning-space dimension."""


class NotPositiveDefinite(MSRError):
    """A covariance/precision matrix was not symmetric positive definite.

    Measurement covariance Σ must be invertible for the information-form
    assimilation step; a singular or indefinite Σ is a measurement defect, not
    a runtime defect.
    """


class CFLViolation(MSRError):
    """A field's curvature demands more substeps than ``max_substeps`` allows.

    Raised instead of silently capping the substep count. A capped step no
    longer satisfies the CFL-style stability condition it was computed for,
    so continuing would integrate an unstable step under the appearance of a
    stable one — the guard must fail closed, not degrade quietly.
    """


__all__ = [
    "CFLViolation",
    "DimensionMismatch",
    "FrameMismatch",
    "MSRError",
    "NotPositiveDefinite",
]
