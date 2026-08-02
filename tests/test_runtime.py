from __future__ import annotations

import numpy as np
import pytest

from msr.abi import MeaningMeasurement
from msr.dynamics import InformationAssimilator, LangevinFlow
from msr.errors import DimensionMismatch, FrameMismatch
from msr.field import FieldPrior, GaussianWell
from msr.runtime import MeaningSpaceRuntime, RuntimeConfig
from msr.stability import StabilizationCriteria, StabilizationDetector

FRAME = "F"
STEP_NS = 50_000_000  # 50 ms


def measure(
    index: int, theta: list[float], variance: float = 0.1
) -> MeaningMeasurement:
    return MeaningMeasurement.isotropic(
        f"o{index}", FRAME, theta, variance, index * STEP_NS
    )


def runtime(**kwargs: object) -> MeaningSpaceRuntime:
    return MeaningSpaceRuntime(frame_id=FRAME, dimension=2, **kwargs)  # type: ignore[arg-type]


def test_first_measurement_initializes_at_rest() -> None:
    engine = runtime()
    result = engine.ingest(measure(0, [1.0, 2.0]))
    assert result.state.theta == (1.0, 2.0)
    assert result.state.speed == 0.0
    assert result.state.step_index == 0
    assert engine.step_index == 0
    assert result.state.time_s == 0.0
    assert result.state.source_observation_id == "o0"
    assert engine.state is result.state


def test_repeated_measurement_converges_and_stabilizes() -> None:
    engine = runtime(
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=3))
    )
    results = [engine.ingest(measure(i, [1.0, 0.0])) for i in range(6)]
    assert any(r.did_stabilize for r in results)
    trajectory = next(r.stabilized for r in results if r.stabilized is not None)
    assert trajectory.is_novel is True
    assert trajectory.frame_id == FRAME


def test_moving_measurements_drag_the_state_and_prevent_stabilization() -> None:
    engine = runtime()
    engine.ingest(measure(0, [0.0, 0.0]))
    result = engine.ingest(measure(1, [5.0, 0.0]))
    assert result.state.theta[0] > 0.0
    assert result.state.speed > 0.0
    assert result.stabilized is None


def test_field_pulls_the_state_into_a_basin() -> None:
    prior = FieldPrior(
        FRAME, 2, (GaussianWell.isotropic("c1", [0.0, 0.0], width=1.0, depth=5.0),)
    )
    engine = MeaningSpaceRuntime(
        frame_id=FRAME,
        dimension=2,
        prior=prior,
        assimilator=InformationAssimilator(forgetting_tau=0.2),
    )
    engine.ingest(measure(0, [0.6, 0.0], variance=10.0))
    for _ in range(39):
        engine.advance(0.05)
    assert engine.state is not None
    assert abs(engine.state.theta[0]) < 0.6
    assert engine.state.basin_id == "c1"
    assert engine.state.potential < 0.0


def test_advance_before_first_measurement_is_rejected() -> None:
    with pytest.raises(DimensionMismatch):
        runtime().advance()


def test_advance_uses_default_dt_and_advances_time() -> None:
    engine = runtime()
    engine.ingest(measure(0, [0.0, 0.0]))
    result = engine.advance()
    assert result.state.time_s == pytest.approx(RuntimeConfig().default_dt)
    assert result.state.source_observation_id is None


def test_frame_and_dimension_are_enforced() -> None:
    engine = runtime()
    with pytest.raises(FrameMismatch):
        engine.ingest(MeaningMeasurement.isotropic("o", "OTHER", [0.0, 0.0], 0.1, 0))
    with pytest.raises(DimensionMismatch):
        engine.ingest(MeaningMeasurement.isotropic("o", FRAME, [0.0], 0.1, 0))


def test_set_field_prior_validates_and_resets_the_latch() -> None:
    engine = runtime(
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=2))
    )
    for index in range(4):
        engine.ingest(measure(index, [1.0, 0.0]))
    assert engine.detector.is_stable is True

    prior = FieldPrior(
        FRAME, 2, (GaussianWell.isotropic("c1", [1.0, 0.0], width=1.0),), version=1
    )
    engine.set_field_prior(prior)
    assert engine.detector.is_stable is False
    assert engine.is_bootstrap is False

    with pytest.raises(FrameMismatch):
        engine.set_field_prior(FieldPrior.empty("OTHER", 2))
    with pytest.raises(DimensionMismatch):
        engine.set_field_prior(FieldPrior.empty(FRAME, 3))


def test_a_known_basin_is_reported_after_the_field_arrives() -> None:
    engine = runtime()
    engine.ingest(measure(0, [1.0, 0.0]))
    engine.set_field_prior(
        FieldPrior(
            FRAME, 2, (GaussianWell.isotropic("c1", [1.0, 0.0], width=1.0),), version=1
        )
    )
    result = engine.ingest(measure(1, [1.0, 0.0]))
    assert result.state.basin_id == "c1"
    assert result.kernel_view.basin_id == "c1"


def test_timestamp_gaps_are_clamped() -> None:
    engine = runtime(config=RuntimeConfig(min_dt=0.01, max_dt=0.2, default_dt=0.05))
    engine.ingest(measure(0, [0.0, 0.0]))
    huge = MeaningMeasurement.isotropic("o1", FRAME, [0.0, 0.0], 0.1, 10**12)
    result = engine.ingest(huge)
    assert result.state.time_s == pytest.approx(0.2)
    same_instant = MeaningMeasurement.isotropic("o2", FRAME, [0.0, 0.0], 0.1, 10**12)
    result = engine.ingest(same_instant)
    assert result.state.time_s == pytest.approx(0.21)


def test_history_is_capped_and_frozen() -> None:
    engine = runtime(config=RuntimeConfig(history_capacity=3))
    for index in range(6):
        engine.ingest(measure(index, [0.0, 0.0]))
    assert len(engine.history) == 3
    assert [s.step_index for s in engine.history] == [3, 4, 5]


def test_kernel_view_matches_state() -> None:
    engine = runtime()
    result = engine.ingest(measure(0, [1.0, 0.0]))
    view = result.kernel_view
    assert view.theta == result.state.theta
    assert view.speed == result.state.speed
    assert view.potential == result.state.potential
    assert view.precision_trace == pytest.approx(2.0 / 0.1)
    assert view.stabilized is False


def test_snapshot_reports_runtime_status() -> None:
    engine = runtime()
    engine.ingest(measure(0, [0.0, 0.0]))
    snapshot = engine.snapshot()
    assert snapshot["frame_id"] == FRAME
    assert snapshot["bootstrap"] is True
    assert snapshot["well_count"] == 0
    assert snapshot["step_index"] == 0


def test_construction_validates_dimension_and_prior() -> None:
    with pytest.raises(DimensionMismatch):
        MeaningSpaceRuntime(frame_id=FRAME, dimension=0)
    with pytest.raises(FrameMismatch):
        MeaningSpaceRuntime(
            frame_id=FRAME, dimension=2, prior=FieldPrior.empty("OTHER", 2)
        )
    with pytest.raises(DimensionMismatch):
        MeaningSpaceRuntime(
            frame_id=FRAME, dimension=2, prior=FieldPrior.empty(FRAME, 3)
        )


def test_config_validation() -> None:
    with pytest.raises(DimensionMismatch):
        RuntimeConfig(min_dt=0.0)
    with pytest.raises(DimensionMismatch):
        RuntimeConfig(min_dt=0.5, max_dt=0.1)
    with pytest.raises(DimensionMismatch):
        RuntimeConfig(default_dt=10.0)
    with pytest.raises(DimensionMismatch):
        RuntimeConfig(history_capacity=0)


def test_temperature_keeps_the_runtime_reproducible() -> None:
    def run() -> list[tuple[float, ...]]:
        engine = MeaningSpaceRuntime(
            frame_id=FRAME,
            dimension=2,
            prior=FieldPrior(
                FRAME, 2, (GaussianWell.isotropic("c1", [0.0, 0.0], width=1.0),)
            ),
            flow=LangevinFlow(temperature=0.05, seed=11),
        )
        engine.ingest(measure(0, [1.0, 1.0]))
        return [engine.advance(0.05).state.theta for _ in range(10)]

    first, second = run(), run()
    assert first == second
    assert not np.allclose(first[-1], first[0])
