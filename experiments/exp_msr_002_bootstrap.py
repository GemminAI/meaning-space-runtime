"""EXP-MSR-002 — Bootstrap: does an empty HEKB still produce knowledge?

Hypothesis
    With no field at all (Φ ≡ 0, RFC-SensOS24 Bootstrap Mode), a repeated
    measurement still stabilizes, and the resulting trajectory is marked novel
    (``basin_id is None``) — which is exactly what gives CLE something to lift.

Falsifiable failure (the control arm)
    A drifting measurement stream must NOT stabilize. If it does, the detector
    is measuring nothing and every stabilization in the ecosystem is noise.

Deterministic: temperature = 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from msr.abi import MeaningMeasurement
from msr.host import MSRHost
from msr.reference import ReferenceCLE, ReferenceHEKB
from msr.runtime import MeaningSpaceRuntime
from msr.stability import StabilizationCriteria, StabilizationDetector

FRAME = "exp-msr-002"
DIMENSION = 2
STEP_NS = 50_000_000
STEPS = 30


def _host() -> tuple[MSRHost, ReferenceCLE, ReferenceHEKB]:
    hekb = ReferenceHEKB(frame_id=FRAME, dimension=DIMENSION)
    cle = ReferenceCLE(hekb=hekb)
    runtime = MeaningSpaceRuntime(
        frame_id=FRAME,
        dimension=DIMENSION,
        detector=StabilizationDetector(StabilizationCriteria(dwell_steps=5)),
    )
    return MSRHost(runtime=runtime, cle=cle, hekb=hekb), cle, hekb


def _arm(theta_at: object) -> dict[str, Any]:
    host, cle, hekb = _host()
    lift_steps: list[int] = []
    for index in range(STEPS):
        theta = theta_at(index)  # type: ignore[operator]
        result = host.ingest(
            MeaningMeasurement.isotropic(
                f"obs-{index}", FRAME, theta, 0.1, index * STEP_NS
            )
        )
        if result.did_stabilize:
            lift_steps.append(index)
    quiet_from = STEPS - 10
    return {
        "stabilized_at_step": lift_steps[0] if lift_steps else None,
        "lifts": len(cle.lifted),
        "lift_steps": lift_steps,
        "novel": [t.is_novel for t in cle.lifted],
        "concepts": sorted(hekb.concepts),
        # The loop must reach a fixed point: no further lifts once the concept
        # has been seeded and reinforced.
        "lifts_in_final_10_steps": sum(1 for s in lift_steps if s >= quiet_from),
        "final_speed": host.runtime.state.speed if host.runtime.state else None,
    }


def run() -> dict[str, Any]:
    stationary = _arm(lambda index: [1.0, -0.5])
    drifting = _arm(lambda index: [0.1 * index, 0.0])
    return {
        "experiment": "EXP-MSR-002",
        "stationary_arm": stationary,
        "drifting_control_arm": drifting,
        "pass": bool(
            stationary["stabilized_at_step"] is not None
            and stationary["novel"][0] is True
            and stationary["concepts"] == ["concept-001"]
            and stationary["lifts_in_final_10_steps"] == 0
            and stationary["final_speed"] == 0.0
            and drifting["stabilized_at_step"] is None
            and drifting["lifts"] == 0
        ),
    }


def main() -> None:
    result = run()
    output = Path(__file__).with_name("results") / "exp_msr_002.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
