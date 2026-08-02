"""Regression tests for the two defects EXP-MSR-002 exposed.

1. A sharp well committed by HEKB made the explicit flow oscillate instead of
   settle (stiffness).
2. Installing any new field dropped the stabilization latch, so a reinforcing
   commit re-triggered stabilization forever (slow-loop runaway).
"""

from __future__ import annotations

import numpy as np
import pytest

from msr.abi import MeaningMeasurement
from msr.dynamics import LangevinFlow
from msr.errors import DimensionMismatch
from msr.field import FieldPrior, GaussianWell
from msr.host import MSRHost
from msr.reference import ReferenceCLE, ReferenceHEKB
from msr.runtime import MeaningSpaceRuntime
from msr.stability import StabilizationCriteria, StabilizationDetector

FRAME = "F"
STEP_NS = 50_000_000


def sharp_field(precision: float = 400.0) -> FieldPrior:
    return FieldPrior(
        FRAME,
        1,
        (
            GaussianWell(
                well_id="sharp",
                mean=np.zeros(1),
                precision=np.array([[precision]]),
                depth=1.0,
            ),
        ),
        version=1,
    )


def test_stiffness_bound_matches_the_wells() -> None:
    assert FieldPrior.empty(FRAME, 2).stiffness == 0.0
    assert sharp_field(400.0).stiffness == pytest.approx(400.0)
    # Depth-weighted and additive across wells.
    two = FieldPrior(
        FRAME,
        1,
        (
            GaussianWell.isotropic("a", [0.0], width=0.5, depth=2.0),
            GaussianWell.isotropic("b", [3.0], width=1.0, depth=1.0),
        ),
    )
    assert two.stiffness == pytest.approx(2.0 * 4.0 + 1.0)


def test_a_stiff_field_is_substepped_not_overshot() -> None:
    prior = sharp_field(400.0)
    flow = LangevinFlow()
    assert flow.substep_count(np.array([0.01]), prior, 0.05) == 20
    theta = np.array([0.02])
    for _ in range(20):
        theta = flow.step(theta, prior, 0.05, flow.make_rng())
    assert abs(float(theta[0])) < 0.02  # settles instead of oscillating


def test_without_the_guard_the_same_field_diverges() -> None:
    """Documents the failure mode: one Euler step at full dt overshoots ~19x."""
    prior = sharp_field(400.0)
    theta = np.array([0.001])
    overshot = theta - prior.gradient(theta) * 0.05
    assert abs(float(overshot[0])) > 10 * abs(float(theta[0]))


def test_far_field_uses_the_displacement_bound() -> None:
    prior = FieldPrior(
        FRAME, 1, (GaussianWell.isotropic("w", [0.0], width=1.0, depth=1.0),)
    )
    flow = LangevinFlow(max_displacement=0.001, stability_factor=1.9)
    assert flow.substep_count(np.array([1.0]), prior, 0.05) > 1


def test_substeps_are_capped() -> None:
    flow = LangevinFlow(max_substeps=3)
    assert flow.substep_count(np.array([0.01]), sharp_field(1e6), 0.05) == 3


def test_flow_validates_the_new_parameters() -> None:
    with pytest.raises(DimensionMismatch):
        LangevinFlow(stability_factor=0.0)
    with pytest.raises(DimensionMismatch):
        LangevinFlow(stability_factor=2.0)
    with pytest.raises(DimensionMismatch):
        LangevinFlow(max_displacement=0.0)
    with pytest.raises(DimensionMismatch):
        LangevinFlow(max_substeps=0)


def test_substepping_preserves_noise_variance() -> None:
    """Total noise must not depend on how many substeps the guard chose."""
    prior = FieldPrior.empty(FRAME, 1)
    flow = LangevinFlow(temperature=1.0, seed=3)
    rng = flow.make_rng()
    samples = np.array(
        [float(flow.step(np.zeros(1), prior, 0.2, rng)[0]) for _ in range(20000)]
    )
    assert float(np.var(samples)) == pytest.approx(2.0 * 1.0 * 0.2, rel=0.05)


# --------------------------------------------------------------- latch policy


def test_latch_survives_a_field_change_that_keeps_the_basin() -> None:
    engine = MeaningSpaceRuntime(
        frame_id=FRAME,
        dimension=1,
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=2)),
    )
    for index in range(3):
        engine.ingest(
            MeaningMeasurement.isotropic(
                f"o{index}", FRAME, [0.0], 0.1, index * STEP_NS
            )
        )
    assert engine.detector.is_stable is True
    field = FieldPrior(
        FRAME, 1, (GaussianWell.isotropic("c1", [0.0], width=1.0),), version=1
    )
    assert engine.set_field_prior(field) is True  # None -> "c1": basin changed
    for index in range(3, 6):
        engine.ingest(
            MeaningMeasurement.isotropic(
                f"o{index}", FRAME, [0.0], 0.1, index * STEP_NS
            )
        )
    assert engine.detector.is_stable is True
    deeper = FieldPrior(
        FRAME,
        1,
        (GaussianWell.isotropic("c1", [0.0], width=1.0, depth=2.0),),
        version=2,
    )
    assert engine.set_field_prior(deeper) is False  # same basin: dwell still valid
    assert engine.detector.is_stable is True


def test_latch_resets_before_the_first_measurement() -> None:
    engine = MeaningSpaceRuntime(frame_id=FRAME, dimension=1)
    assert engine.set_field_prior(FieldPrior.empty(FRAME, 1)) is True


def test_the_slow_loop_reaches_a_fixed_point() -> None:
    """Seed once, reinforce once, then stop — not an unbounded lift stream."""
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=1)
    cle = ReferenceCLE(hekb=hekb)
    host = MSRHost(
        runtime=MeaningSpaceRuntime(
            frame_id=FRAME,
            dimension=1,
            detector=StabilizationDetector(StabilizationCriteria(dwell_steps=5)),
        ),
        cle=cle,
        hekb=hekb,
    )
    lift_steps = []
    for index in range(40):
        result = host.ingest(
            MeaningMeasurement.isotropic(
                f"o{index}", FRAME, [1.0], 0.1, index * STEP_NS
            )
        )
        if result.did_stabilize:
            lift_steps.append(index)
    assert lift_steps == [4, 9]
    assert sorted(hekb.concepts) == ["concept-001"]
    assert host.runtime.state is not None
    assert host.runtime.state.speed == 0.0
