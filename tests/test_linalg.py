from __future__ import annotations

import numpy as np
import pytest

from msr.errors import DimensionMismatch, NotPositiveDefinite
from msr.linalg import as_spd, as_vector, invert_spd, to_matrix_tuple, to_tuple


def test_as_vector_accepts_correct_shape() -> None:
    assert as_vector([1.0, 2.0], 2, name="v").tolist() == [1.0, 2.0]


def test_as_vector_rejects_wrong_shape() -> None:
    with pytest.raises(DimensionMismatch):
        as_vector([1.0, 2.0, 3.0], 2, name="v")


def test_as_vector_rejects_non_finite() -> None:
    with pytest.raises(DimensionMismatch):
        as_vector([1.0, float("inf")], 2, name="v")


def test_as_spd_symmetrizes_exactly() -> None:
    nearly = [[2.0, 1.0], [1.0 + 1e-12, 2.0]]
    result = as_spd(nearly, 2, name="m")
    assert result[0][1] == result[1][0]


def test_as_spd_rejects_wrong_shape() -> None:
    with pytest.raises(DimensionMismatch):
        as_spd([[1.0, 0.0]], 2, name="m")


def test_as_spd_rejects_non_finite() -> None:
    with pytest.raises(NotPositiveDefinite):
        as_spd([[float("nan"), 0.0], [0.0, 1.0]], 2, name="m")


def test_as_spd_rejects_asymmetric() -> None:
    with pytest.raises(NotPositiveDefinite):
        as_spd([[1.0, 0.5], [-0.5, 1.0]], 2, name="m")


def test_as_spd_rejects_indefinite() -> None:
    with pytest.raises(NotPositiveDefinite):
        as_spd([[1.0, 2.0], [2.0, 1.0]], 2, name="m")


def test_invert_spd_round_trips() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    inverse = invert_spd(matrix, name="m")
    assert np.allclose(matrix @ inverse, np.eye(2))
    assert np.allclose(inverse, inverse.T)


def test_invert_spd_rejects_singular() -> None:
    with pytest.raises(NotPositiveDefinite):
        invert_spd(np.zeros((2, 2)), name="m")


def test_freezing_helpers() -> None:
    assert to_tuple(np.array([1.0, 2.0])) == (1.0, 2.0)
    assert to_matrix_tuple(np.eye(2)) == ((1.0, 0.0), (0.0, 1.0))
