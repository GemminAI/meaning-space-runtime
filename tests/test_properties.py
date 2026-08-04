"""Property-based tests (Hypothesis): determinism, replay, invariants, and
CFL fail-closed safety across the input space.

These complement the example-based suite. 100% line/branch coverage proves
every line has been *reached*, not that the numerically sensitive paths (SPD
inversion, information fusion, stiffness bounds, substep counts) hold across
inputs the hand-picked examples didn't happen to choose.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from msr.abi import MeaningMeasurement
from msr.dynamics import InformationAssimilator, LangevinFlow
from msr.errors import CFLViolation
from msr.field import FieldPrior, GaussianWell
from msr.linalg import as_spd, invert_spd
from msr.runtime import MeaningSpaceRuntime

_SETTINGS = settings(max_examples=50, deadline=None)
_FINITE = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)
_DIMENSION = st.integers(min_value=1, max_value=4)


def _draw_vector(data: st.DataObject, dimension: int) -> np.ndarray:
    values = data.draw(st.lists(_FINITE, min_size=dimension, max_size=dimension))
    return np.array(values, dtype=np.float64)


def _draw_spd(data: st.DataObject, dimension: int) -> np.ndarray:
    """Draw a random SPD matrix via A Aᵀ + 0.5 I (always positive definite)."""
    entries = st.floats(
        min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False
    )
    rows = data.draw(
        st.lists(
            st.lists(entries, min_size=dimension, max_size=dimension),
            min_size=dimension,
            max_size=dimension,
        )
    )
    a = np.array(rows, dtype=np.float64)
    return a @ a.T + 0.5 * np.eye(dimension)


# --------------------------------------------------------------- invariants


@given(data=st.data())
@_SETTINGS
def test_invert_spd_round_trips_for_arbitrary_spd(data: st.DataObject) -> None:
    dimension = data.draw(_DIMENSION)
    matrix = _draw_spd(data, dimension)
    inverse = invert_spd(matrix, name="m")
    assert np.allclose(inverse, inverse.T, atol=1e-8)
    assert np.allclose(matrix @ inverse, np.eye(dimension), atol=1e-6)


@given(data=st.data())
@_SETTINGS
def test_as_spd_symmetrizes_any_nearly_symmetric_input(data: st.DataObject) -> None:
    dimension = data.draw(_DIMENSION)
    matrix = _draw_spd(data, dimension)
    noise = data.draw(st.floats(min_value=-1e-10, max_value=1e-10, allow_nan=False))
    nearly_symmetric = matrix + noise * (np.triu(np.ones_like(matrix), k=1))
    result = as_spd(nearly_symmetric, dimension, name="m")
    assert np.array_equal(result, result.T)


@given(data=st.data())
@_SETTINGS
def test_assimilate_precision_is_exactly_additive(data: st.DataObject) -> None:
    dimension = data.draw(_DIMENSION)
    theta = _draw_vector(data, dimension)
    measured_theta = _draw_vector(data, dimension)
    precision = _draw_spd(data, dimension)
    measured_precision = _draw_spd(data, dimension)

    result = InformationAssimilator().assimilate(
        theta, precision, measured_theta, measured_precision
    )
    assert np.allclose(result.precision, precision + measured_precision, atol=1e-8)


@given(data=st.data())
@_SETTINGS
def test_assimilate_does_not_move_the_state_when_evidence_agrees(
    data: st.DataObject,
) -> None:
    """Fusing an observation identical to the current belief must not move it."""
    dimension = data.draw(_DIMENSION)
    theta = _draw_vector(data, dimension)
    precision = _draw_spd(data, dimension)
    measured_precision = _draw_spd(data, dimension)

    result = InformationAssimilator().assimilate(
        theta, precision, theta, measured_precision
    )
    assert result.displacement == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(result.theta, theta, atol=1e-6)


@given(data=st.data())
@_SETTINGS
def test_decay_trace_stays_between_current_and_floor(data: st.DataObject) -> None:
    dimension = data.draw(_DIMENSION)
    precision = _draw_spd(data, dimension)
    dt = data.draw(
        st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    assimilator = InformationAssimilator()

    decayed = assimilator.decay(precision, dt)
    assert np.allclose(decayed, decayed.T, atol=1e-8)

    original_trace = float(np.trace(precision))
    floor_trace = assimilator.precision_floor * dimension
    low, high = sorted((original_trace, floor_trace))
    assert low - 1e-6 <= float(np.trace(decayed)) <= high + 1e-6


# ------------------------------------------------------------- math review


@given(data=st.data())
@_SETTINGS
def test_field_gradient_matches_finite_differences(data: st.DataObject) -> None:
    """∇Φ must actually be the gradient of Φ, not merely share a name with it."""
    dimension = data.draw(st.integers(min_value=1, max_value=3))
    well_count = data.draw(st.integers(min_value=1, max_value=3))
    wells = tuple(
        GaussianWell.isotropic(
            f"w{i}",
            _draw_vector(data, dimension),
            width=data.draw(
                st.floats(
                    min_value=0.3, max_value=3.0, allow_nan=False, allow_infinity=False
                )
            ),
            depth=data.draw(
                st.floats(
                    min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False
                )
            ),
        )
        for i in range(well_count)
    )
    prior = FieldPrior("F", dimension, wells)
    theta = _draw_vector(data, dimension)

    analytic = prior.gradient(theta)
    eps = 1e-5
    numeric = np.empty(dimension, dtype=np.float64)
    for axis in range(dimension):
        bump = np.zeros(dimension, dtype=np.float64)
        bump[axis] = eps
        numeric[axis] = (
            prior.potential(theta + bump) - prior.potential(theta - bump)
        ) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-3, rtol=1e-3)


@given(data=st.data())
@_SETTINGS
def test_substep_count_satisfies_cfl_bound_or_fails_closed(data: st.DataObject) -> None:
    """Every returned substep count must actually satisfy the curvature bound
    it was computed for; every count that wouldn't must raise instead."""
    precision = data.draw(
        st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False)
    )
    dt = data.draw(
        st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    mobility = data.draw(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    max_substeps = data.draw(st.integers(min_value=1, max_value=64))

    prior = FieldPrior(
        "F",
        1,
        (
            GaussianWell(
                well_id="w",
                mean=np.zeros(1),
                precision=np.array([[precision]]),
                depth=1.0,
            ),
        ),
    )
    flow = LangevinFlow(mobility=mobility, max_substeps=max_substeps)
    theta = np.zeros(1)  # at the well mean: gradient is 0, only curvature drives this

    curvature_needed = mobility * prior.stiffness * dt / flow.stability_factor
    expected = 1 if curvature_needed <= 1.0 else int(np.ceil(curvature_needed))

    if expected > max_substeps:
        with pytest.raises(CFLViolation):
            flow.substep_count(theta, prior, dt)
    else:
        substeps = flow.substep_count(theta, prior, dt)
        assert substeps == expected
        sub_dt = dt / substeps
        assert mobility * prior.stiffness * sub_dt <= flow.stability_factor + 1e-9


# ----------------------------------------------------------- replay/determinism


@given(data=st.data())
@_SETTINGS
def test_zero_temperature_replay_is_bitwise_reproducible(data: st.DataObject) -> None:
    dimension = data.draw(st.integers(min_value=1, max_value=3))
    step_count = data.draw(st.integers(min_value=1, max_value=5))
    thetas = [_draw_vector(data, dimension) for _ in range(step_count)]

    def run() -> list[tuple[float, ...]]:
        engine = MeaningSpaceRuntime(frame_id="F", dimension=dimension)
        states = []
        for index, theta in enumerate(thetas):
            result = engine.ingest(
                MeaningMeasurement.isotropic(
                    f"o{index}", "F", theta, 0.5, index * 50_000_000
                )
            )
            states.append(result.state.theta)
        return states

    assert run() == run()


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    temperature=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@_SETTINGS
def test_seeded_positive_temperature_replay_is_reproducible(
    seed: int, temperature: float
) -> None:
    def run() -> list[tuple[float, ...]]:
        engine = MeaningSpaceRuntime(
            frame_id="F",
            dimension=1,
            flow=LangevinFlow(temperature=temperature, seed=seed),
        )
        engine.ingest(MeaningMeasurement.isotropic("o0", "F", [1.0], 0.5, 0))
        return [engine.advance(0.05).state.theta for _ in range(5)]

    assert run() == run()
