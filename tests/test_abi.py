from __future__ import annotations

import dataclasses

import pytest

from msr.abi import (
    KernelView,
    MeaningMeasurement,
    MeaningState,
    StabilizedTrajectory,
)
from msr.errors import DimensionMismatch, NotPositiveDefinite


def test_isotropic_measurement_is_valid() -> None:
    measurement = MeaningMeasurement.isotropic("o1", "F", [1.0, 2.0], 0.25, 10)
    measurement.validate()
    assert measurement.dimension == 2
    assert measurement.sigma == ((0.25, 0.0), (0.0, 0.25))


def test_isotropic_rejects_non_positive_variance() -> None:
    with pytest.raises(DimensionMismatch):
        MeaningMeasurement.isotropic("o1", "F", [1.0], 0.0, 0)


def test_validate_rejects_empty_theta() -> None:
    with pytest.raises(DimensionMismatch):
        MeaningMeasurement("o1", "F", (), (), 0).validate()


def test_validate_rejects_singular_sigma() -> None:
    measurement = MeaningMeasurement("o1", "F", (1.0, 2.0), ((0.0, 0.0), (0.0, 0.0)), 0)
    with pytest.raises(NotPositiveDefinite):
        measurement.validate()


def test_measurement_is_frozen() -> None:
    measurement = MeaningMeasurement.isotropic("o1", "F", [1.0], 1.0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        measurement.observation_id = "other"  # type: ignore[misc]


def _state() -> MeaningState:
    return MeaningState(
        frame_id="F",
        step_index=3,
        time_s=0.5,
        theta=(1.0, 0.0),
        precision=((1.0, 0.0), (0.0, 1.0)),
        speed=0.01,
        potential=-0.5,
        basin_id="c1",
        source_observation_id="o3",
    )


def test_state_dimension() -> None:
    assert _state().dimension == 2


def test_kernel_view_serializes() -> None:
    view = KernelView(
        frame_id="F",
        step_index=1,
        time_s=0.1,
        theta=(1.0, 0.0),
        speed=0.0,
        potential=-1.0,
        basin_id=None,
        precision_trace=2.0,
        stabilized=False,
    )
    payload = view.as_dict()
    assert payload["theta"] == [1.0, 0.0]
    assert payload["basin_id"] is None
    assert payload["stabilized"] is False


def test_trajectory_novelty_and_dimension() -> None:
    novel = StabilizedTrajectory(
        trajectory_id="t1",
        frame_id="F",
        basin_id=None,
        states=(_state(),),
        centroid=(1.0, 0.0),
        covariance=((0.0, 0.0), (0.0, 0.0)),
        dwell_steps=1,
        dwell_seconds=0.0,
    )
    assert novel.is_novel is True
    assert novel.dimension == 2
    known = dataclasses.replace(novel, basin_id="c1")
    assert known.is_novel is False
