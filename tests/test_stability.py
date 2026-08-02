from __future__ import annotations

import pytest

from msr.abi import MeaningState
from msr.errors import DimensionMismatch
from msr.stability import StabilizationCriteria, StabilizationDetector


def state(
    index: int,
    *,
    speed: float = 0.0,
    basin: str | None = None,
    theta: tuple[float, ...] = (0.0,),
    source: str | None = None,
) -> MeaningState:
    return MeaningState(
        frame_id="F",
        step_index=index,
        time_s=index * 0.1,
        theta=theta,
        precision=((1.0,),),
        speed=speed,
        potential=0.0,
        basin_id=basin,
        source_observation_id=source,
    )


def test_emits_once_after_the_dwell_is_met() -> None:
    detector = StabilizationDetector(StabilizationCriteria(dwell_steps=3))
    assert detector.observe(state(0)) is None
    assert detector.observe(state(1)) is None
    trajectory = detector.observe(state(2))
    assert trajectory is not None
    assert trajectory.dwell_steps == 3
    assert trajectory.trajectory_id == "traj-F-000000"
    # Latched: no second emission while the dwell continues.
    assert detector.observe(state(3)) is None
    assert detector.is_stable is True


def test_motion_breaks_the_dwell() -> None:
    detector = StabilizationDetector(StabilizationCriteria(dwell_steps=2))
    detector.observe(state(0))
    assert detector.observe(state(1, speed=1.0)) is None
    assert detector.is_stable is False
    assert detector.dwell_length == 0


def test_basin_change_restarts_the_dwell() -> None:
    detector = StabilizationDetector(StabilizationCriteria(dwell_steps=2))
    detector.observe(state(0, basin="a"))
    assert detector.observe(state(1, basin="b")) is None
    assert detector.dwell_length == 1
    assert detector.observe(state(2, basin="b")) is not None


def test_redemption_after_destabilization() -> None:
    detector = StabilizationDetector(StabilizationCriteria(dwell_steps=2))
    detector.observe(state(0))
    assert detector.observe(state(1)) is not None
    detector.observe(state(2, speed=5.0))
    detector.observe(state(3))
    assert detector.observe(state(4)) is not None


def test_segment_is_capped_and_summarized() -> None:
    detector = StabilizationDetector(
        StabilizationCriteria(dwell_steps=2, max_segment=3)
    )
    for index in range(6):
        detector.observe(state(index, theta=(float(index % 2),), source=f"o{index}"))
    assert detector.dwell_length == 3


def test_summary_statistics_and_provenance() -> None:
    detector = StabilizationDetector(StabilizationCriteria(dwell_steps=3))
    detector.observe(state(0, theta=(0.0,), source="o1"))
    detector.observe(state(1, theta=(2.0,), source="o1"))
    trajectory = detector.observe(state(2, theta=(4.0,), source="o2"))
    assert trajectory is not None
    assert trajectory.centroid == (2.0,)
    assert trajectory.covariance[0][0] == pytest.approx(8.0 / 3.0)
    assert trajectory.dwell_seconds == pytest.approx(0.2)
    assert trajectory.provenance == ("o1", "o2")
    assert trajectory.is_novel is True


def test_reset_clears_everything() -> None:
    detector = StabilizationDetector(StabilizationCriteria(dwell_steps=2))
    detector.observe(state(0))
    detector.observe(state(1))
    detector.reset()
    assert detector.is_stable is False
    assert detector.dwell_length == 0


def test_criteria_validation() -> None:
    with pytest.raises(DimensionMismatch):
        StabilizationCriteria(speed_threshold=0.0)
    with pytest.raises(DimensionMismatch):
        StabilizationCriteria(dwell_steps=0)
    with pytest.raises(DimensionMismatch):
        StabilizationCriteria(dwell_steps=5, max_segment=2)
