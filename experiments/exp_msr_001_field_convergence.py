"""EXP-MSR-001 — Fast loop: does a known field actually attract a trajectory?

Hypothesis
    With a HEKB-supplied well at m, a trajectory starting off-centre relaxes
    toward m, its potential decreases monotonically, and it ends up reported to
    the kernel as inside that basin.

Falsifiable failure
    Potential increases, or the state never enters the basin, or the reported
    ``basin_id`` disagrees with the geometric distance.

Deterministic: temperature = 0.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from msr.abi import MeaningMeasurement
from msr.field import FieldPrior, GaussianWell
from msr.host import MSRHost
from msr.reference import RecordingKernel
from msr.runtime import MeaningSpaceRuntime

FRAME = "exp-msr-001"
DIMENSION = 2
START = [1.4, 1.4]
WELL_MEAN = [0.0, 0.0]


def run() -> dict[str, Any]:
    prior = FieldPrior(
        frame_id=FRAME,
        dimension=DIMENSION,
        wells=(GaussianWell.isotropic("c-known", WELL_MEAN, width=1.0, depth=4.0),),
        version=1,
    )
    kernel = RecordingKernel()
    host = MSRHost(
        runtime=MeaningSpaceRuntime(frame_id=FRAME, dimension=DIMENSION, prior=prior),
        kernel=kernel,
    )

    # One vague measurement to position the runtime, then pure field relaxation.
    host.ingest(MeaningMeasurement.isotropic("obs-0", FRAME, START, 4.0, 0))
    for _ in range(200):
        host.advance(0.05)

    potentials = [view.potential for view in kernel.views]
    distances = [
        float(np.linalg.norm(np.array(view.theta) - np.array(WELL_MEAN)))
        for view in kernel.views
    ]
    monotone = all(b <= a + 1e-12 for a, b in pairwise(potentials))
    first_in_basin = next(
        (i for i, view in enumerate(kernel.views) if view.basin_id == "c-known"), None
    )
    return {
        "experiment": "EXP-MSR-001",
        "steps": len(kernel.views),
        "start_distance": distances[0],
        "final_distance": distances[-1],
        "start_potential": potentials[0],
        "final_potential": potentials[-1],
        "potential_monotonically_decreasing": monotone,
        "first_step_inside_basin": first_in_basin,
        "final_basin_id": kernel.views[-1].basin_id,
        "pass": bool(
            monotone
            and distances[-1] < distances[0]
            and kernel.views[-1].basin_id == "c-known"
        ),
    }


def main() -> None:
    result = run()
    output = Path(__file__).with_name("results") / "exp_msr_001.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
