"""Small SPD-matrix helpers shared by the assimilator and the field.

Kept in one place so that every symmetric-positive-definite check in MSR uses
the same definition (Cholesky succeeds after symmetrization) and the same
error type.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from msr.errors import DimensionMismatch, NotPositiveDefinite

Array = npt.NDArray[np.float64]

_SYMMETRY_TOLERANCE = 1e-9


def as_vector(values: object, dimension: int, *, name: str) -> Array:
    """Coerce ``values`` to a finite ``(dimension,)`` float64 array."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimension,):
        raise DimensionMismatch(
            f"{name} must have shape ({dimension},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise DimensionMismatch(f"{name} must be finite")
    return array


def as_spd(values: object, dimension: int, *, name: str) -> Array:
    """Coerce ``values`` to a symmetric positive definite ``(d, d)`` array.

    Symmetry is required within :data:`_SYMMETRY_TOLERANCE` and then enforced
    exactly by averaging with the transpose, so downstream Cholesky factors are
    reproducible bit-for-bit regardless of how the caller rounded its input.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimension, dimension):
        raise DimensionMismatch(
            f"{name} must have shape ({dimension}, {dimension}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise NotPositiveDefinite(f"{name} must be finite")
    if not np.allclose(array, array.T, atol=_SYMMETRY_TOLERANCE, rtol=0.0):
        raise NotPositiveDefinite(f"{name} must be symmetric")
    symmetric = 0.5 * (array + array.T)
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - message varies
        raise NotPositiveDefinite(f"{name} must be positive definite") from exc
    return symmetric


def invert_spd(matrix: Array, *, name: str) -> Array:
    """Invert an SPD matrix via its Cholesky factor, keeping the result symmetric."""
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - message varies
        raise NotPositiveDefinite(f"{name} must be positive definite") from exc
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    inverse: Array = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
    return 0.5 * (inverse + inverse.T)


def to_tuple(vector: Array) -> tuple[float, ...]:
    """Freeze a vector into a plain tuple for the immutable ABI types."""
    return tuple(float(value) for value in vector)


def to_matrix_tuple(matrix: Array) -> tuple[tuple[float, ...], ...]:
    """Freeze a matrix into nested tuples for the immutable ABI types."""
    return tuple(tuple(float(value) for value in row) for row in matrix)


__all__ = [
    "Array",
    "as_spd",
    "as_vector",
    "invert_spd",
    "to_matrix_tuple",
    "to_tuple",
]
