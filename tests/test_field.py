from __future__ import annotations

import numpy as np
import pytest

from msr.errors import DimensionMismatch, FrameMismatch, NotPositiveDefinite
from msr.field import FieldPrior, GaussianWell


def test_empty_field_is_flat() -> None:
    prior = FieldPrior.empty("F", 3)
    theta = np.array([1.0, -2.0, 0.5])
    assert prior.is_empty is True
    assert prior.potential(theta) == 0.0
    assert np.allclose(prior.gradient(theta), np.zeros(3))
    assert prior.basin_of(theta) is None


def test_potential_is_minimal_at_the_well_mean() -> None:
    well = GaussianWell.isotropic("c1", [0.0, 0.0], width=1.0, depth=2.0)
    prior = FieldPrior("F", 2, (well,))
    assert prior.potential(np.zeros(2)) == pytest.approx(-2.0)
    assert prior.potential(np.array([1.0, 0.0])) > prior.potential(np.zeros(2))


def test_gradient_vanishes_at_the_mean_and_points_outward() -> None:
    well = GaussianWell.isotropic("c1", [0.0, 0.0], width=1.0)
    prior = FieldPrior("F", 2, (well,))
    assert np.allclose(prior.gradient(np.zeros(2)), np.zeros(2))
    gradient = prior.gradient(np.array([0.5, 0.0]))
    # -grad must point back toward the mean.
    assert gradient[0] > 0.0


def test_gradient_matches_finite_differences() -> None:
    wells = (
        GaussianWell.isotropic("c1", [1.0, 0.0], width=0.7, depth=1.5),
        GaussianWell.isotropic("c2", [-1.0, 0.5], width=1.3, depth=0.8),
    )
    prior = FieldPrior("F", 2, wells)
    theta = np.array([0.3, -0.2])
    epsilon = 1e-6
    numeric = np.array(
        [
            (
                prior.potential(theta + epsilon * unit)
                - prior.potential(theta - epsilon * unit)
            )
            / (2 * epsilon)
            for unit in np.eye(2)
        ]
    )
    assert np.allclose(prior.gradient(theta), numeric, atol=1e-6)


def test_basin_membership_uses_mahalanobis_radius() -> None:
    prior = FieldPrior(
        "F", 1, (GaussianWell.isotropic("c1", [0.0], width=1.0),), basin_radius=2.0
    )
    assert prior.basin_of(np.array([1.9])) == "c1"
    assert prior.basin_of(np.array([2.1])) is None


def test_basin_picks_the_nearest_well() -> None:
    prior = FieldPrior(
        "F",
        1,
        (
            GaussianWell.isotropic("far", [1.5], width=1.0),
            GaussianWell.isotropic("near", [0.1], width=1.0),
        ),
    )
    assert prior.basin_of(np.array([0.0])) == "near"


def test_with_well_replaces_by_id_and_bumps_version() -> None:
    prior = FieldPrior.empty("F", 2)
    first = prior.with_well(GaussianWell.isotropic("c1", [0.0, 0.0], width=1.0))
    second = first.with_well(GaussianWell.isotropic("c1", [1.0, 1.0], width=1.0))
    assert len(second.wells) == 1
    assert second.version == 2
    assert second.wells[0].mean.tolist() == [1.0, 1.0]
    # The original field is untouched: fields are replaced, never mutated.
    assert prior.is_empty and prior.version == 0


def test_with_well_rejects_dimension_mismatch() -> None:
    prior = FieldPrior.empty("F", 2)
    with pytest.raises(DimensionMismatch):
        prior.with_well(GaussianWell.isotropic("c1", [0.0], width=1.0))


def test_field_rejects_bad_construction() -> None:
    with pytest.raises(DimensionMismatch):
        FieldPrior("F", 0)
    with pytest.raises(DimensionMismatch):
        FieldPrior("F", 2, (GaussianWell.isotropic("c1", [0.0], width=1.0),))


def test_require_frame() -> None:
    prior = FieldPrior.empty("F", 1)
    prior.require_frame("F")
    with pytest.raises(FrameMismatch):
        prior.require_frame("G")


def test_well_validation() -> None:
    with pytest.raises(DimensionMismatch):
        GaussianWell.isotropic("c1", [0.0], width=0.0)
    with pytest.raises(DimensionMismatch):
        GaussianWell.isotropic("c1", [0.0], width=1.0, depth=0.0)
    with pytest.raises(NotPositiveDefinite):
        GaussianWell(
            well_id="c1",
            mean=np.zeros(2),
            precision=np.array([[1.0, 2.0], [2.0, 1.0]]),
            depth=1.0,
        )


def test_well_mahalanobis() -> None:
    well = GaussianWell.isotropic("c1", [0.0, 0.0], width=2.0)
    assert well.mahalanobis(np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert well.dimension == 2
