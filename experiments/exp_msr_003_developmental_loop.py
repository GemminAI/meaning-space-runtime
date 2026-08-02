"""EXP-MSR-003 — Slow loop: does prior exposure change how MSR handles noise?

This is the RFC-SensOS24 developmental claim, reduced to something measurable:
knowledge committed to HEKB should make a *later, noisier* observation of the
same thing resolvable, because the concept's well absorbs measurement jitter
that a flat space cannot.

Protocol (identical phase-2/3 measurement streams in both arms, seeded)
    Phase 1  clean stationary exposure at A   — experienced arm only
    Phase 2  clean stationary exposure at B   — both arms (symmetry control)
    Phase 3  noisy exposure around A          — both arms, byte-identical stream

Hypothesis
    The experienced arm stabilizes during phase 3; the naive arm does not, or
    does so strictly later.

Falsifiable failure
    The naive arm stabilizes as fast as the experienced arm — in which case the
    field prior contributes nothing and the slow loop is decorative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from msr.abi import MeaningMeasurement
from msr.dynamics import InformationAssimilator
from msr.host import MSRHost
from msr.reference import ReferenceCLE, ReferenceHEKB
from msr.runtime import MeaningSpaceRuntime
from msr.stability import StabilizationCriteria, StabilizationDetector

FRAME = "exp-msr-003"
DIMENSION = 2
STEP_NS = 50_000_000
A = np.array([1.0, -0.5])
B = np.array([-1.5, 1.0])
CLEAN_STEPS = 8
NOISY_STEPS = 80
NOISE_STD = 0.05
VARIANCE = 0.1


def _stream(
    prefix: str, centre: np.ndarray, count: int, start: int, noise_std: float, seed: int
) -> list[MeaningMeasurement]:
    rng = np.random.default_rng(seed)
    stream: list[MeaningMeasurement] = []
    for offset in range(count):
        theta = centre + (
            rng.standard_normal(DIMENSION) * noise_std if noise_std > 0.0 else 0.0
        )
        stream.append(
            MeaningMeasurement.isotropic(
                f"{prefix}-{offset}",
                FRAME,
                theta.tolist(),
                VARIANCE,
                (start + offset) * STEP_NS,
            )
        )
    return stream


def _arm(experienced: bool, phase3: list[MeaningMeasurement]) -> dict[str, Any]:
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=DIMENSION)
    cle = ReferenceCLE(hekb=hekb)
    host = MSRHost(
        runtime=MeaningSpaceRuntime(
            frame_id=FRAME,
            dimension=DIMENSION,
            detector=StabilizationDetector(StabilizationCriteria(dwell_steps=5)),
            # A 20 Hz measurement stream needs a short forgetting time, or the
            # runtime tracks the mapper far too sluggishly to be controlled.
            assimilator=InformationAssimilator(forgetting_tau=0.5),
        ),
        cle=cle,
        hekb=hekb,
    )

    if experienced:
        for measurement in _stream("p1", A, CLEAN_STEPS, 0, 0.0, seed=1):
            host.ingest(measurement)
    for measurement in _stream("p2", B, CLEAN_STEPS, CLEAN_STEPS, 0.0, seed=2):
        host.ingest(measurement)

    # Both arms end phase 2 holding exactly one concept, but not the same one:
    # the experienced arm learned A in phase 1, the naive arm only ever saw B.
    concepts_before = {
        name: hekb.concepts[name].centroid for name in sorted(hekb.concepts)
    }
    stabilized_at: int | None = None
    speeds: list[float] = []
    for index, measurement in enumerate(phase3):
        result = host.ingest(measurement)
        speeds.append(result.state.speed)
        if result.did_stabilize and stabilized_at is None:
            stabilized_at = index
    return {
        "experienced": experienced,
        "concepts_before_phase3": concepts_before,
        "stabilized_at_phase3_step": stabilized_at,
        "mean_phase3_speed": float(np.mean(speeds)),
        "mean_settled_speed": float(np.mean(speeds[-20:])),
        "final_phase3_speed": speeds[-1],
        "final_basin": host.runtime.state.basin_id if host.runtime.state else None,
    }


def run() -> dict[str, Any]:
    # One stream object list, reused verbatim by both arms.
    phase3 = _stream("p3", A, NOISY_STEPS, 2 * CLEAN_STEPS, NOISE_STD, seed=1234)
    experienced = _arm(True, phase3)
    naive = _arm(False, phase3)
    naive_step = naive["stabilized_at_phase3_step"]
    exp_step = experienced["stabilized_at_phase3_step"]
    return {
        "experiment": "EXP-MSR-003",
        "noise_std": NOISE_STD,
        "experienced_arm": experienced,
        "naive_arm": naive,
        "pass": bool(
            exp_step is not None
            and (naive_step is None or naive_step > exp_step)
            and experienced["mean_settled_speed"] < naive["mean_settled_speed"]
        ),
    }


def main() -> None:
    result = run()
    output = Path(__file__).with_name("results") / "exp_msr_003.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
