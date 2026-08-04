from __future__ import annotations

import math

import pytest

from msr.abi import MeaningState
from msr.errors import DimensionMismatch, InsufficientHistory
from msr.field import FieldPrior, GaussianWell
from msr.lyapunov import largest_lyapunov_exponent

DT = 0.1


def _stationary_states(theta: float, count: int) -> list[MeaningState]:
    """States that never move — a synthetic fixture, not a realized flow.

    With a constant Jacobian (the trajectory never leaves the point the
    Hessian is evaluated at), the estimator's Euler recurrence collapses to
    a single constant growth factor per step, giving an exact closed-form
    prediction to test against.
    """
    return [
        MeaningState(
            frame_id="F",
            step_index=index,
            time_s=index * DT,
            theta=(theta,),
            precision=((1.0,),),
            speed=0.0,
            potential=0.0,
            basin_id=None,
            source_observation_id=None,
        )
        for index in range(count)
    ]


def test_exponent_is_negative_at_a_well_mean() -> None:
    # At Delta=0: hessian = depth * precision = 1.0; J = -mobility * 1.0 = -1.0.
    well = GaussianWell.isotropic("c1", [0.0], width=1.0, depth=1.0)
    prior = FieldPrior("F", 1, (well,))
    states = _stationary_states(0.0, count=5)
    estimate = largest_lyapunov_exponent(states, prior, mobility=1.0)
    expected = math.log(1.0 + DT * (-1.0)) / DT
    assert estimate.exponent == pytest.approx(expected)
    assert estimate.is_contracting is True
    assert estimate.steps == 4
    assert estimate.window_seconds == pytest.approx(0.4)


def test_exponent_is_positive_on_a_wells_repelling_shoulder() -> None:
    # 3 sigma out: hessian = response * (P - P^2 r^2) < 0 there, so J > 0.
    well = GaussianWell.isotropic("c1", [0.0], width=1.0, depth=1.0)
    prior = FieldPrior("F", 1, (well,))
    r = 3.0
    response = math.exp(-0.5 * r * r)
    hessian = response * (1.0 - r * r)
    assert hessian < 0.0  # sanity: this position is on the repelling shoulder
    states = _stationary_states(r, count=5)
    estimate = largest_lyapunov_exponent(states, prior, mobility=1.0)
    expected = math.log(1.0 + DT * (-1.0 * hessian)) / DT
    assert estimate.exponent == pytest.approx(expected)
    assert estimate.is_contracting is False


def test_exponent_is_zero_for_an_empty_flat_field() -> None:
    prior = FieldPrior.empty("F", 1)
    states = _stationary_states(0.0, count=3)
    estimate = largest_lyapunov_exponent(states, prior, mobility=1.0)
    assert estimate.exponent == pytest.approx(0.0, abs=1e-15)


def test_rejects_a_single_state() -> None:
    prior = FieldPrior.empty("F", 1)
    with pytest.raises(InsufficientHistory):
        largest_lyapunov_exponent(_stationary_states(0.0, count=1), prior, mobility=1.0)


def test_rejects_zero_elapsed_window() -> None:
    prior = FieldPrior.empty("F", 1)
    states = [
        MeaningState(
            frame_id="F",
            step_index=0,
            time_s=0.0,
            theta=(0.0,),
            precision=((1.0,),),
            speed=0.0,
            potential=0.0,
            basin_id=None,
            source_observation_id=None,
        )
        for _ in range(3)
    ]
    with pytest.raises(InsufficientHistory):
        largest_lyapunov_exponent(states, prior, mobility=1.0)


def test_rejects_non_positive_mobility() -> None:
    prior = FieldPrior.empty("F", 1)
    with pytest.raises(DimensionMismatch):
        largest_lyapunov_exponent(_stationary_states(0.0, count=3), prior, mobility=0.0)


def test_rejects_a_non_increasing_intermediate_timestamp() -> None:
    prior = FieldPrior.empty("F", 1)
    states = [
        MeaningState(
            frame_id="F",
            step_index=0,
            time_s=0.0,
            theta=(0.0,),
            precision=((1.0,),),
            speed=0.0,
            potential=0.0,
            basin_id=None,
            source_observation_id=None,
        ),
        MeaningState(
            frame_id="F",
            step_index=1,
            time_s=0.0,  # repeats the previous timestamp: zero intermediate dt
            theta=(0.0,),
            precision=((1.0,),),
            speed=0.0,
            potential=0.0,
            basin_id=None,
            source_observation_id=None,
        ),
        MeaningState(
            frame_id="F",
            step_index=2,
            time_s=1.0,
            theta=(0.0,),
            precision=((1.0,),),
            speed=0.0,
            potential=0.0,
            basin_id=None,
            source_observation_id=None,
        ),
    ]
    with pytest.raises(DimensionMismatch):
        largest_lyapunov_exponent(states, prior, mobility=1.0)
