"""Meaning Mapper → MSR: parse a mapper payload into a measurement.

Accepts both the measurement-native field names (``theta``/``sigma``) and the
annotation-era names still emitted by deployed mappers (``coordinates``,
``covariance``, scalar ``variance``). Nothing is inferred: a payload without a
frame, a position, and an uncertainty is rejected, because a reading without a
declared frame and uncertainty is not a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from msr.abi import MeaningMeasurement
from msr.errors import DimensionMismatch
from msr.linalg import to_matrix_tuple, to_tuple

_THETA_KEYS = ("theta", "coordinates", "position")
_SIGMA_KEYS = ("sigma", "covariance")
_ID_KEYS = ("observation_id", "id", "measurement_id")
_FRAME_KEYS = ("frame_id", "frame")
_TIME_KEYS = ("timestamp_ns", "timestamp", "time_ns")


def _first(payload: Mapping[str, Any], keys: tuple[str, ...], what: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    raise DimensionMismatch(f"payload is missing {what} (any of {list(keys)})")


def measurement_from_payload(payload: Mapping[str, Any]) -> MeaningMeasurement:
    """Build a validated :class:`MeaningMeasurement` from a mapper payload."""
    theta = np.asarray(_first(payload, _THETA_KEYS, "theta"), dtype=np.float64).reshape(
        -1
    )
    if theta.size == 0:
        raise DimensionMismatch("theta must be non-empty")

    sigma_value: Any = None
    for key in _SIGMA_KEYS:
        if payload.get(key) is not None:
            sigma_value = payload[key]
            break
    if sigma_value is None:
        variance = payload.get("variance")
        if variance is None:
            raise DimensionMismatch(
                "payload is missing uncertainty (sigma, covariance or variance)"
            )
        scalar = float(variance)
        if scalar <= 0.0:
            raise DimensionMismatch("variance must be positive")
        sigma = scalar * np.eye(theta.size, dtype=np.float64)
    else:
        sigma = np.asarray(sigma_value, dtype=np.float64)
        if sigma.ndim == 1:
            sigma = np.diag(sigma)

    provenance_value = payload.get("provenance", ())
    provenance = tuple(str(item) for item in provenance_value)

    measurement = MeaningMeasurement(
        observation_id=str(_first(payload, _ID_KEYS, "observation id")),
        frame_id=str(_first(payload, _FRAME_KEYS, "frame id")),
        theta=to_tuple(theta),
        sigma=to_matrix_tuple(sigma),
        timestamp_ns=int(_first(payload, _TIME_KEYS, "timestamp")),
        provenance=provenance,
    )
    measurement.validate()
    return measurement


__all__ = ["measurement_from_payload"]
