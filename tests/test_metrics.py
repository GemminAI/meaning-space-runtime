from __future__ import annotations

import pytest

from msr.abi import MeaningState
from msr.errors import DimensionMismatch, InsufficientHistory
from msr.metrics import drift_converged, drift_distance, runtime_stability_rate


def _state(*, time_s: float, precision: tuple[tuple[float, ...], ...]) -> MeaningState:
    return MeaningState(
        frame_id="F",
        step_index=0,
        time_s=time_s,
        theta=(0.0, 0.0),
        precision=precision,
        speed=0.0,
        potential=0.0,
        basin_id=None,
        source_observation_id=None,
    )


IDENTITY = ((1.0, 0.0), (0.0, 1.0))
SHARPER = ((4.0, 0.0), (0.0, 4.0))


def test_runtime_stability_rate_is_positive_when_covariance_shrinks() -> None:
    states = [
        _state(time_s=0.0, precision=IDENTITY),
        _state(time_s=1.0, precision=SHARPER),
    ]
    assert runtime_stability_rate(states) > 0.0


def test_runtime_stability_rate_is_negative_when_covariance_grows() -> None:
    states = [
        _state(time_s=0.0, precision=SHARPER),
        _state(time_s=1.0, precision=IDENTITY),
    ]
    assert runtime_stability_rate(states) < 0.0


def test_runtime_stability_rate_uses_only_first_and_last_state() -> None:
    middle = _state(time_s=0.5, precision=((1000.0, 0.0), (0.0, 1000.0)))
    states = [
        _state(time_s=0.0, precision=IDENTITY),
        middle,
        _state(time_s=1.0, precision=SHARPER),
    ]
    endpoints_only = [states[0], states[-1]]
    assert runtime_stability_rate(states) == runtime_stability_rate(endpoints_only)


def test_runtime_stability_rate_rejects_a_single_state() -> None:
    with pytest.raises(InsufficientHistory):
        runtime_stability_rate([_state(time_s=0.0, precision=IDENTITY)])


def test_runtime_stability_rate_rejects_zero_elapsed_time() -> None:
    states = [
        _state(time_s=1.0, precision=IDENTITY),
        _state(time_s=1.0, precision=SHARPER),
    ]
    with pytest.raises(InsufficientHistory):
        runtime_stability_rate(states)


def test_drift_distance_matches_euclidean_norm() -> None:
    trajectory = [(0.0, 0.0), (3.0, 4.0)]
    reference = [(0.0, 0.0), (0.0, 0.0)]
    assert drift_distance(trajectory, reference) == (0.0, 5.0)


def test_drift_distance_rejects_length_mismatch() -> None:
    with pytest.raises(DimensionMismatch):
        drift_distance([(0.0,)], [(0.0,), (1.0,)])


def test_drift_distance_rejects_pointwise_shape_mismatch() -> None:
    with pytest.raises(DimensionMismatch):
        drift_distance([(1.0, 2.0), (3.0,)], [(1.0, 2.0), (3.0, 4.0)])


def test_drift_converged_checks_the_tail_window() -> None:
    distances = [10.0, 10.0, 0.01, 0.01, 0.01]
    assert drift_converged(distances, epsilon=0.1, window=3) is True
    assert drift_converged(distances, epsilon=0.1, window=5) is False


def test_drift_converged_default_window_is_the_last_sample() -> None:
    assert drift_converged([5.0, 0.01], epsilon=0.1) is True


def test_drift_converged_rejects_non_positive_epsilon() -> None:
    with pytest.raises(DimensionMismatch):
        drift_converged([0.1], epsilon=0.0)


def test_drift_converged_rejects_non_positive_window() -> None:
    with pytest.raises(DimensionMismatch):
        drift_converged([0.1], epsilon=0.1, window=0)


def test_drift_converged_rejects_insufficient_samples() -> None:
    with pytest.raises(InsufficientHistory):
        drift_converged([0.1], epsilon=0.1, window=2)
