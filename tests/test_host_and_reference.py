from __future__ import annotations

import numpy as np
import pytest

from msr.abi import MeaningMeasurement
from msr.field import FieldPrior, GaussianWell
from msr.host import MSRHost
from msr.ports import FieldPriorSource, KernelPort, LiftPort
from msr.reference import (
    RecordingKernel,
    ReferenceCLE,
    ReferenceConcept,
    ReferenceHEKB,
)
from msr.runtime import MeaningSpaceRuntime
from msr.stability import StabilizationCriteria, StabilizationDetector

FRAME = "F"
STEP_NS = 50_000_000


def measure(index: int, theta: list[float]) -> MeaningMeasurement:
    return MeaningMeasurement.isotropic(f"o{index}", FRAME, theta, 0.1, index * STEP_NS)


def build_host(**kwargs: object) -> tuple[MSRHost, RecordingKernel, ReferenceCLE]:
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=2)
    cle = ReferenceCLE(hekb=hekb)
    kernel = RecordingKernel()
    runtime = MeaningSpaceRuntime(
        frame_id=FRAME,
        dimension=2,
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=3)),
    )
    host = MSRHost(runtime=runtime, kernel=kernel, cle=cle, hekb=hekb, **kwargs)  # type: ignore[arg-type]
    return host, kernel, cle


def test_ports_are_satisfied_structurally() -> None:
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=2)
    assert isinstance(RecordingKernel(), KernelPort)
    assert isinstance(ReferenceCLE(hekb=hekb), LiftPort)
    assert isinstance(hekb, FieldPriorSource)


def test_every_step_reaches_the_kernel() -> None:
    host, kernel, _ = build_host()
    for index in range(5):
        host.ingest(measure(index, [1.0, 0.0]))
    assert len(kernel.views) == 5
    assert host.metrics.steps == 5
    assert host.metrics.kernel_views == 5


def test_stabilization_reaches_cle_and_closes_the_slow_loop() -> None:
    host, _, cle = build_host()
    for index in range(5):
        host.ingest(measure(index, [1.0, 0.0]))
    assert len(cle.lifted) == 1
    assert host.metrics.lifts == 1
    assert host.metrics.prior_refreshes == 1
    # The lifted trajectory is now a well in the runtime's field.
    assert host.runtime.is_bootstrap is False
    assert [w.well_id for w in host.runtime.field_prior.wells] == ["concept-001"]


def test_advance_also_routes_to_the_kernel() -> None:
    host, kernel, _ = build_host()
    host.ingest(measure(0, [1.0, 0.0]))
    host.advance(0.05)
    assert len(kernel.views) == 2


def test_refresh_is_a_no_op_without_hekb_or_new_version() -> None:
    runtime = MeaningSpaceRuntime(frame_id=FRAME, dimension=2)
    assert MSRHost(runtime=runtime).refresh_field_prior() is False

    hekb = ReferenceHEKB(frame_id=FRAME, dimension=2)
    host = MSRHost(runtime=runtime, hekb=hekb)
    assert host.refresh_field_prior() is False  # version 0, not newer
    hekb.commit(ReferenceConcept(id="c1", centroid=(0.0, 0.0)))
    assert host.refresh_field_prior() is True
    assert host.refresh_field_prior() is False  # unchanged version


def test_hekb_returns_nothing_for_a_foreign_frame() -> None:
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=2)
    assert hekb.field_prior("OTHER", 2) is None
    assert hekb.field_prior(FRAME, 3) is None


def test_refresh_on_stabilize_can_be_disabled() -> None:
    host, _, cle = build_host(refresh_on_stabilize=False)
    for index in range(5):
        host.ingest(measure(index, [1.0, 0.0]))
    assert len(cle.lifted) == 1
    assert host.metrics.prior_refreshes == 0
    assert host.runtime.is_bootstrap is True


def test_host_runs_headless_without_ports() -> None:
    runtime = MeaningSpaceRuntime(
        frame_id=FRAME,
        dimension=2,
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=2)),
    )
    host = MSRHost(runtime=runtime)
    results = [host.ingest(measure(index, [1.0, 0.0])) for index in range(4)]
    assert any(r.did_stabilize for r in results)
    assert host.metrics.kernel_views == 0
    assert host.metrics.lifts == 0


def test_reference_cle_reinforces_a_known_basin() -> None:
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=2)
    cle = ReferenceCLE(hekb=hekb, reinforcement=0.5)
    runtime = MeaningSpaceRuntime(
        frame_id=FRAME,
        dimension=2,
        prior=FieldPrior(
            FRAME, 2, (GaussianWell.isotropic("c1", [1.0, 0.0], width=1.0),), version=1
        ),
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=3)),
    )
    hekb.commit(
        ReferenceConcept(id="c1", centroid=(1.0, 0.0), invariants={"depth": 1.0})
    )
    host = MSRHost(runtime=runtime, cle=cle, hekb=hekb, refresh_on_stabilize=False)
    for index in range(5):
        host.ingest(measure(index, [1.0, 0.0]))
    assert cle.lifted[0].basin_id == "c1"
    assert hekb.concepts["c1"].invariants["depth"] == pytest.approx(1.5)
    assert len(hekb.concepts) == 1  # reinforced, not duplicated


def test_reference_cle_inverts_the_dwell_covariance() -> None:
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=2)
    cle = ReferenceCLE(hekb=hekb)
    runtime = MeaningSpaceRuntime(
        frame_id=FRAME,
        dimension=2,
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=3)),
    )
    host = MSRHost(runtime=runtime, cle=cle, hekb=hekb, refresh_on_stabilize=False)
    for index in range(5):
        host.ingest(measure(index, [1.0, 0.0]))
    concept = hekb.concepts["concept-001"]
    assert concept.hessian is not None
    # A stationary dwell has ~zero covariance, so the ridge dominates and the
    # precision is large but finite.
    assert np.all(np.isfinite(np.asarray(concept.hessian)))
    assert concept.centroid == pytest.approx((1.0, 0.0))
