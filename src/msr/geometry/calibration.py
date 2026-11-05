"""Calibration files and the four-way shift verdict (SPEC-MSR-001 v2 §3).

No threshold lives in code. ε, δ, w_min and the noise-floor interval come from
``<data_dir>/calibration/<model_build_id>.json``; without that file every
threshold judgment is ``UNCALIBRATED``.

``evaluate_shift`` follows the same rule as
``sensos_core.manifold_shift.evaluate_manifold_shift`` but does not import it
(MSR imports no neighbour code); the tests check the two agree.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from msr.geometry.errors import CalibrationError
from msr.geometry.metrics import (
    mean_step_distance,
    pairwise_distances,
    step_displacements,
)
from msr.geometry.trajectory import TrajectoryInput

ShiftVerdict = Literal[
    "UNCALIBRATED", "STAGNANT", "NOISE_FLOOR_INTERNAL", "STRUCTURAL_SHIFT_DETECTED"
]
METHOD_VERSION = "msr-geometry-calibration/1"


@dataclass(frozen=True, slots=True)
class Calibration:
    """SPEC §3.1 file contents plus space_id, ε, δ, w_min, codebook reference."""

    model_build_id: str
    space_id: str
    measured_at: str
    noise_floor_lower: float
    noise_floor_upper: float
    background_mean: float
    sample_config: dict[str, Any] = field(default_factory=dict)
    epsilon: float | None = None
    delta: float | None = None
    w_min: int | None = None
    codebook_ref: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model_build_id": self.model_build_id,
            "space_id": self.space_id,
            "measured_at": self.measured_at,
            "noise_floor_lower": self.noise_floor_lower,
            "noise_floor_upper": self.noise_floor_upper,
            "background_mean": self.background_mean,
            "sample_config": self.sample_config,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "w_min": self.w_min,
            "codebook_ref": self.codebook_ref,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class LoadedCalibration:
    calibration: Calibration
    sha256: str


def calibration_path(data_dir: Path, model_build_id: str) -> Path:
    return Path(data_dir) / "calibration" / f"{model_build_id}.json"


def load_calibration(data_dir: Path, model_build_id: str) -> LoadedCalibration | None:
    """``None`` (never a fallback to another build) when the file is absent."""
    path = calibration_path(data_dir, model_build_id)
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
        calib = Calibration(
            model_build_id=data["model_build_id"],
            space_id=data["space_id"],
            measured_at=data["measured_at"],
            noise_floor_lower=float(data["noise_floor_lower"]),
            noise_floor_upper=float(data["noise_floor_upper"]),
            background_mean=float(data["background_mean"]),
            sample_config=data.get("sample_config", {}),
            epsilon=data.get("epsilon"),
            delta=data.get("delta"),
            w_min=data.get("w_min"),
            codebook_ref=data.get("codebook_ref"),
            provenance=data.get("provenance", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"malformed calibration file {path}: {exc!r}") from exc
    if calib.model_build_id != model_build_id:
        raise CalibrationError(
            f"{path} declares model_build_id {calib.model_build_id!r}, "
            f"expected {model_build_id!r}"
        )
    return LoadedCalibration(calib, hashlib.sha256(raw).hexdigest())


def write_calibration(data_dir: Path, calib: Calibration) -> Path:
    path = calibration_path(data_dir, calib.model_build_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(calib.to_json_dict(), sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def evaluate_shift(d: float, calib: Calibration | None) -> ShiftVerdict:
    """SPEC §3.2: UNCALIBRATED / STAGNANT / NOISE_FLOOR_INTERNAL / STRUCTURAL."""
    if calib is None:
        return "UNCALIBRATED"
    if d < calib.noise_floor_lower:
        return "STAGNANT"
    if d <= calib.noise_floor_upper:
        return "NOISE_FLOOR_INTERNAL"
    return "STRUCTURAL_SHIFT_DETECTED"


def build_calibration(
    null_trajectories: Sequence[TrajectoryInput],
    *,
    w_min: int = 1,
    epsilon_quantile: float = 0.05,
    delta_quantile: float = 0.10,
    floor_quantiles: tuple[float, float] = (0.05, 0.95),
    codebook_ref: dict[str, Any] | None = None,
    measured_at: str | None = None,
) -> Calibration:
    """Derive a calibration from a set of null (noise-only) trajectories.

    ε = ``epsilon_quantile`` of within-trajectory point distances
    (``|i-j| > w_min``); δ = ``delta_quantile`` of step displacements; the
    noise-floor interval = ``floor_quantiles`` of the pairwise mean per-step
    distance between null trajectories (common prefix). Records when, from
    which inputs (sha256) and by which method version it was made.
    """
    if len(null_trajectories) < 2:
        raise CalibrationError("need at least 2 null trajectories")
    first = null_trajectories[0]
    for traj in null_trajectories:
        if (
            traj.space_id != first.space_id
            or traj.model_build_id != first.model_build_id
        ):
            raise CalibrationError(
                "null trajectories must share space_id and model_build_id"
            )
        if traj.dimension != first.dimension:
            raise CalibrationError("null trajectories must share dimension")
    if w_min < 0:
        raise CalibrationError("w_min must be >= 0")
    pooled: list[float] = []
    steps: list[float] = []
    for traj in null_trajectories:
        dist = pairwise_distances(traj.points)
        rows, cols = np.triu_indices(traj.length, k=w_min + 1)
        pooled.extend(float(x) for x in dist[rows, cols])
        steps.extend(float(x) for x in step_displacements(traj.points))
    if not pooled or not steps:
        raise CalibrationError("null trajectories too short for the requested w_min")
    between = [
        mean_step_distance(a.points, b.points)
        for a, b in itertools.combinations(null_trajectories, 2)
    ]
    lower, upper = (float(np.quantile(between, q)) for q in floor_quantiles)
    return Calibration(
        model_build_id=first.model_build_id,
        space_id=first.space_id,
        measured_at=measured_at or datetime.now(UTC).isoformat(timespec="seconds"),
        noise_floor_lower=lower,
        noise_floor_upper=upper,
        background_mean=float(np.mean(between)),
        sample_config={
            "trajectories": len(null_trajectories),
            "points": int(min(t.length for t in null_trajectories)),
            "dims": first.dimension,
        },
        epsilon=float(np.quantile(pooled, epsilon_quantile)),
        delta=float(np.quantile(steps, delta_quantile)),
        w_min=w_min,
        codebook_ref=codebook_ref,
        provenance={
            "method_version": METHOD_VERSION,
            "input_sha256": [t.sha256() for t in null_trajectories],
            "quantiles": {
                "epsilon": epsilon_quantile,
                "delta": delta_quantile,
                "floor": list(floor_quantiles),
            },
        },
    )


__all__ = [
    "METHOD_VERSION",
    "Calibration",
    "LoadedCalibration",
    "ShiftVerdict",
    "build_calibration",
    "calibration_path",
    "evaluate_shift",
    "load_calibration",
    "write_calibration",
]
