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
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from msr.geometry.calibration import (
    Calibration,
    LoadedCalibration,
    evaluate_shift,
    load_calibration,
)
from msr.geometry.errors import CalibrationError, InvalidTrajectory
from msr.geometry.metrics import (
    DEFAULT_BLOCK_SIZE,
    displacement_metrics,
    diversity_metrics,
    mean_step_distance,
    persistence_metrics,
    recurrence_metrics,
)
from msr.geometry.trajectory import TrajectoryInput, require_same_space
from msr.version import __version__

PRODUCER = "meaning-space-runtime"
DISTANCE_DEFINITION = "mean_step_l2"  # = mlba-core trajectory_distance
VERDICT_REASON = "shift verdict applies to trajectory pairs (SPEC-MSR-001 v2 §3.2)"
Alignment = Literal["equal_only", "common_prefix"]


def codebook_sha256(codebook: NDArray[np.float64]) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(codebook, dtype=np.float64).tobytes()
    ).hexdigest()


def _input_ref(traj: TrajectoryInput) -> dict[str, Any]:
    return {
        "sha256": traj.sha256(),
        "model_build_id": traj.model_build_id,
        "policy": {"name": traj.policy.name, "sha256": traj.policy.sha256},
        "length": traj.length,
        "dimension": traj.dimension,
    }


def _calibration_ref(loaded: LoadedCalibration | None) -> dict[str, Any]:
    return {
        "model_build_id": loaded.calibration.model_build_id if loaded else None,
        "sha256": loaded.sha256 if loaded else None,
    }


def _require_calibration_space(calib: Calibration, *spaces: str) -> None:
    for space in spaces:
        if calib.space_id != space:
            raise CalibrationError(
                f"calibration is for space_id {calib.space_id!r}, "
                f"trajectory is {space!r}"
            )


def _resolve(
    explicit: float | None, from_calib: float | None
) -> tuple[Any, str | None]:
    if explicit is not None:
        return explicit, "explicit"
    if from_calib is not None:
        return from_calib, "calibration"
    return None, None


def analyze_trajectory(
    traj: TrajectoryInput,
    *,
    data_dir: Path | None = None,
    codebook: NDArray[np.float64] | None = None,
    epsilon: float | None = None,
    delta: float | None = None,
    w_min: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    """Compute metrics for one trajectory. Explicit thresholds override calibration.

    No shift verdict is produced here: it is defined for trajectory pairs
    (SPEC-MSR-001 v2 §3.2) and lives in :func:`compare_trajectories`.
    """
    loaded = load_calibration(data_dir, traj.model_build_id) if data_dir else None
    calib: Calibration | None = loaded.calibration if loaded else None
    if calib is not None:
        _require_calibration_space(calib, traj.space_id)
        ref = calib.codebook_ref or {}
        expected = ref.get("sha256")
        if codebook is not None and expected and codebook_sha256(codebook) != expected:
            raise CalibrationError(
                "codebook sha256 does not match calibration codebook_ref"
            )
    eps, eps_src = _resolve(epsilon, calib.epsilon if calib else None)
    dlt, dlt_src = _resolve(delta, calib.delta if calib else None)
    wm, wm_src = _resolve(w_min, calib.w_min if calib else None)

    points = traj.points
    disp, s1 = displacement_metrics(points, dlt)
    rec, s2 = recurrence_metrics(points, eps, wm, block_size=block_size)
    div, s3 = diversity_metrics(points, codebook)
    per, s4 = persistence_metrics(points, codebook, eps)
    series = {**s1, **s2, **s3, **s4}
    return {
        "producer": {"name": PRODUCER, "version": __version__},
        "space_id": traj.space_id,
        "source": traj.source,
        "layer": traj.layer,
        "input_ref": _input_ref(traj),
        "calibration": _calibration_ref(loaded),
        "thresholds": {
            "epsilon": eps,
            "delta": dlt,
            "w_min": wm,
            "source": {"epsilon": eps_src, "delta": dlt_src, "w_min": wm_src},
        },
        "metrics": {
            "displacement": disp,
            "recurrence": rec,
            "diversity": div,
            "persistence": per,
        },
        "series": series,
        "verdict": {"state": None, "reason": VERDICT_REASON},
    }


def compare_trajectories(
    left: TrajectoryInput,
    right: TrajectoryInput,
    *,
    calibration: LoadedCalibration | None = None,
    alignment: Alignment = "equal_only",
    allow_cross_space: bool = False,
) -> dict[str, Any]:
    """Mean per-step L2 between two trajectories + the SPEC §3.2 verdict."""
    require_same_space(left, right, allow_cross_space=allow_cross_space)
    if alignment not in ("equal_only", "common_prefix"):
        raise InvalidTrajectory(f"unknown alignment {alignment!r}")
    if alignment == "equal_only" and left.length != right.length:
        raise InvalidTrajectory(
            f"lengths differ ({left.length} vs {right.length}); "
            "pass alignment='common_prefix' to compare the common prefix"
        )
    calib = calibration.calibration if calibration else None
    if calib is not None:
        _require_calibration_space(calib, left.space_id, right.space_id)
    used = min(left.length, right.length)
    distance = mean_step_distance(left.points, right.points)
    return {
        "producer": {"name": PRODUCER, "version": __version__},
        "space_id": left.space_id,
        "inputs": {"left": _input_ref(left), "right": _input_ref(right)},
        "calibration": _calibration_ref(calibration),
        "alignment": {"mode": alignment, "points_used": used},
        "distance": {"definition": DISTANCE_DEFINITION, "value": distance},
        "verdict": {"state": evaluate_shift(distance, calib)},
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
