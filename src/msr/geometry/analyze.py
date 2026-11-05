"""Trajectory -> Interpretation-candidate dict (producer: meaning-space-runtime).

Emits measurement values, series and a calibration verdict only. It never names
a meaning (loop, anomaly, hallucination), never assigns P/L/U, and never writes
to HEKB — nvs-runtime mediates that (D-68, D-89, H7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from msr.geometry.calibration import (
    Calibration,
    ShiftVerdict,
    evaluate_shift,
    load_calibration,
)
from msr.geometry.errors import CalibrationError
from msr.geometry.metrics import (
    displacement_metrics,
    diversity_metrics,
    mean_step_distance,
    persistence_metrics,
    recurrence_metrics,
)
from msr.geometry.trajectory import TrajectoryInput, require_same_space
from msr.version import __version__

PRODUCER = "meaning-space-runtime"


def codebook_sha256(codebook: NDArray[np.float64]) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(codebook, dtype=np.float64).tobytes()
    ).hexdigest()


def analyze_trajectory(
    traj: TrajectoryInput,
    *,
    data_dir: Path | None = None,
    codebook: NDArray[np.float64] | None = None,
    epsilon: float | None = None,
    delta: float | None = None,
    w_min: int | None = None,
) -> dict[str, Any]:
    """Compute metrics + verdict. Explicit thresholds override calibration."""
    loaded = load_calibration(data_dir, traj.model_build_id) if data_dir else None
    calib: Calibration | None = loaded.calibration if loaded else None
    if calib is not None:
        if calib.space_id != traj.space_id:
            raise CalibrationError(
                f"calibration is for space_id {calib.space_id!r}, "
                f"trajectory is {traj.space_id!r}"
            )
        ref = calib.codebook_ref or {}
        expected = ref.get("sha256")
        if codebook is not None and expected and codebook_sha256(codebook) != expected:
            raise CalibrationError(
                "codebook sha256 does not match calibration codebook_ref"
            )
    eps = epsilon if epsilon is not None else (calib.epsilon if calib else None)
    dlt = delta if delta is not None else (calib.delta if calib else None)
    wm = w_min if w_min is not None else (calib.w_min if calib else None)

    points = traj.points
    disp, s1 = displacement_metrics(points, dlt)
    rec, s2 = recurrence_metrics(points, eps, wm)
    div, s3 = diversity_metrics(points, codebook)
    per, s4 = persistence_metrics(points, codebook, eps)
    series = {**s1, **s2, **s3, **s4}

    verdict: ShiftVerdict | None = None  # calibrated but fewer than 2 points
    if calib is None:
        verdict = "UNCALIBRATED"
    elif disp["mean"] is not None:
        verdict = evaluate_shift(float(disp["mean"]), calib)  # type: ignore[arg-type]
    return {
        "producer": {"name": PRODUCER, "version": __version__},
        "space_id": traj.space_id,
        "source": traj.source,
        "layer": traj.layer,
        "input_ref": {
            "sha256": traj.sha256(),
            "model_build_id": traj.model_build_id,
            "policy": {"name": traj.policy.name, "sha256": traj.policy.sha256},
            "length": traj.length,
            "dimension": traj.dimension,
        },
        "calibration": {
            "model_build_id": calib.model_build_id if calib else None,
            "sha256": loaded.sha256 if loaded else None,
        },
        "thresholds": {"epsilon": eps, "delta": dlt, "w_min": wm},
        "metrics": {
            "displacement": disp,
            "recurrence": rec,
            "diversity": div,
            "persistence": per,
        },
        "series": series,
        "verdict": {"state": verdict, "statistic": "displacement.mean"},
    }


def compare_trajectories(
    left: TrajectoryInput,
    right: TrajectoryInput,
    *,
    calib: Calibration | None = None,
    allow_cross_space: bool = False,
) -> dict[str, Any]:
    """Mean per-step distance between two trajectories of one space + verdict."""
    require_same_space(left, right, allow_cross_space=allow_cross_space)
    distance = mean_step_distance(left.points, right.points)
    return {
        "space_id": left.space_id,
        "distance": distance,
        "verdict": evaluate_shift(distance, calib),
    }


def to_json_bytes(result: Mapping[str, Any]) -> bytes:
    """Canonical bytes: the same input always yields the same bytes."""
    return json.dumps(
        result, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode()


__all__ = [
    "PRODUCER",
    "analyze_trajectory",
    "codebook_sha256",
    "compare_trajectories",
    "to_json_bytes",
]
