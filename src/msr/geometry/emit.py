"""IC-02 emission: trajectories -> IC-00 envelopes (design v0.6.2 §3.3, App. A).

``emit_single`` and ``emit_pair`` run the measurement layer
(:mod:`msr.geometry.analyze`), quantize every real number onto the 2**-30 grid
(ties-to-even; discrete integers stay as they are), move the per-prefix series
into an int64 Blob, and wrap the payload in an envelope. Thresholds are
quantized *before* they are used, so the payload's thresholds are exactly the
ones the numbers were computed with, and the pair verdict is decided on the
payload's integers.

``series`` is an object keyed by series name; every value is a 1-D int64 Blob
(``shape [T]``, little-endian, no header). Real series are quantized at 2**-30;
the three count series are stored as they are (``quantization: null``). "No
value" is ``-2**63`` and the Blob then declares ``undefined: "int64_min"``; the
positions are those of ``msr/README.md`` (contract 0.7.0, §3.3.2, §7.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from msr.geometry.analyze import (
    VERDICT_REASON,
    Alignment,
    analyze_trajectory,
    compare_trajectories,
    input_ref,
)
from msr.geometry.calibration import (
    EXPONENT,
    CalibrationArtifact,
    evaluate_shift_quantized,
    payload_alignment,
)
from msr.geometry.envelope import (
    BLOB_MEDIA_TYPE,
    E_CONTEXT,
    ArtifactRef,
    Blob,
    Emission,
    EmitContext,
    alignment_extensions,
    build_envelope,
    int64_blob,
)
from msr.geometry.errors import CalibrationError, InvalidTrajectory
from msr.geometry.metrics import DEFAULT_BLOCK_SIZE
from msr.geometry.trajectory import TrajectoryInput
from msr.geometry.wire import (
    BLOB_RANGE,
    E_QUANT_RANGE,
    INT64_RANGE,
    MAX_SAFE,
    ContractError,
    dequantize,
    quantize,
)

PAYLOAD_SCHEMA = ("msr.geometry-measurement", "0.7.0")
UNDEFINED = INT64_RANGE[
    0
]  # the Blob's "no value" mark; -2**63 is never a measured value
# series name -> (kind, unit), the contract's table (msr/README.md)
SERIES: dict[str, tuple[str, str]] = {
    "displacement_mean": ("real", "distance"),
    "displacement_median": ("real", "distance"),
    "displacement_longest_sub_delta_run": ("count", "steps"),
    "recurrence_pair_fraction": ("real", "fraction"),
    "recurrence_return_count": ("count", "returns"),
    "recurrence_return_rate": ("real", "returns_per_point"),
    "diversity_cell_entropy": ("real", "nats"),
    "diversity_participation_ratio": ("real", "dimensionless"),
    "persistence_longest_stay": ("count", "points"),
    "persistence_dominant_cell_share": ("real", "fraction"),
}


def _q(value: float | None) -> int | None:
    return None if value is None else quantize(value, EXPONENT)


def _count(value: object) -> int | None:
    return cast("int | None", value)


def _real(value: object) -> float | None:
    return cast("float | None", value)


def _threshold_real(
    explicit: float | None, calibration: int | None
) -> tuple[int | None, str | None]:
    if explicit is not None:
        return quantize(explicit, EXPONENT), "explicit"
    if calibration is not None:
        return calibration, "calibration"
    return None, None


def _threshold_w_min(
    explicit: int | None, calibration: int | None
) -> tuple[int | None, str | None]:
    if explicit is not None:
        if isinstance(explicit, bool) or not 0 <= explicit <= MAX_SAFE:
            raise ContractError(E_CONTEXT, "w_min must be an int in [0, 2^53-1]")
        return explicit, "explicit"
    if calibration is not None:
        return calibration, "calibration"
    return None, None


def _series_blobs(
    series: dict[str, list[Any]], length: int
) -> tuple[dict[str, dict[str, Any]], tuple[Blob, ...]]:
    refs: dict[str, dict[str, Any]] = {}
    blobs: list[Blob] = []
    for name, (kind, unit) in SERIES.items():
        values: list[int] = []
        for v in series[name]:
            if v is None:
                values.append(UNDEFINED)
            elif kind == "real":
                values.append(quantize(v, EXPONENT, int_range=BLOB_RANGE))
            else:
                values.append(_count_value(v))
        blob = int64_blob(values)
        ref: dict[str, Any] = {
            "uri": blob.uri,
            "sha256": blob.sha256,
            "media_type": BLOB_MEDIA_TYPE,
            "dtype": "int64",
            "shape": [length],
            "byte_order": "little",
            "order": "C",
            "quantization": (
                {
                    "quantum": {"base": 2, "exponent": EXPONENT},
                    "rounding": "ties-to-even",
                    "tolerance_quanta": 1,
                }
                if kind == "real"
                else None
            ),
            "unit": unit,
        }
        if UNDEFINED in values:
            ref["undefined"] = "int64_min"
        refs[name] = ref
        blobs.append(blob)
    return refs, tuple(blobs)


def _count_value(value: int) -> int:
    if not 0 <= value <= BLOB_RANGE[1]:
        raise ContractError(E_QUANT_RANGE, f"count {value} outside [0, 2^63-1]")
    return int(value)


def emit_single(
    traj: TrajectoryInput,
    ctx: EmitContext,
    *,
    trajectory_ref: ArtifactRef,
    calibration: CalibrationArtifact | None = None,
    codebook: NDArray[np.float64] | None = None,
    codebook_ref: ArtifactRef | None = None,
    epsilon: float | None = None,
    delta: float | None = None,
    w_min: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Emission:
    """One trajectory's measurement as an envelope (payload ``mode: "single"``)."""
    if (codebook is None) != (codebook_ref is None):
        raise ContractError(E_CONTEXT, "codebook and codebook_ref come together")
    calibrated = calibration.calibration if calibration else None
    if (
        calibrated is not None
        and codebook_ref is not None
        and calibrated.codebook_ref is not None
        and calibrated.codebook_ref != codebook_ref
    ):
        raise CalibrationError(
            "codebook_ref differs from the calibration's codebook_ref"
        )
    values = calibration.values if calibration else {}
    eps_q, eps_src = _threshold_real(epsilon, values.get("epsilon"))
    dlt_q, dlt_src = _threshold_real(delta, values.get("delta"))
    wm, wm_src = _threshold_w_min(w_min, values.get("w_min"))
    measured = analyze_trajectory(
        traj,
        calibration=calibration,
        codebook=codebook,
        epsilon=None if eps_q is None else dequantize(eps_q, EXPONENT),
        delta=None if dlt_q is None else dequantize(dlt_q, EXPONENT),
        w_min=wm,
        block_size=block_size,
    )
    metrics = measured["metrics"]
    disp, rec = metrics["displacement"], metrics["recurrence"]
    div, per = metrics["diversity"], metrics["persistence"]
    series_refs, blobs = _series_blobs(measured["series"], traj.length)
    payload = {
        "mode": "single",
        "space_id": traj.space_id,
        "source": traj.source,
        "layer": traj.layer,
        "input_ref": input_ref(traj),
        "calibration": None if calibration is None else calibration.ref.to_dict(),
        "thresholds": {
            "epsilon": eps_q,
            "delta": dlt_q,
            "w_min": wm,
            "source": {"epsilon": eps_src, "delta": dlt_src, "w_min": wm_src},
        },
        "metrics": {
            "displacement": {
                "count": disp["count"],
                "mean": _q(_real(disp["mean"])),
                "median": _q(_real(disp["median"])),
                "longest_sub_delta_run": _count(disp["longest_sub_delta_run"]),
            },
            "recurrence": {
                "pair_fraction": _q(_real(rec["pair_fraction"])),
                "return_count": _count(rec["return_count"]),
                "return_rate": _q(_real(rec["return_rate"])),
            },
            "diversity": {
                "cell_entropy": _q(_real(div["cell_entropy"])),
                "participation_ratio": _q(_real(div["participation_ratio"])),
            },
            "persistence": {
                "longest_stay": _count(per["longest_stay"]),
                "dominant_cell_share": _q(_real(per["dominant_cell_share"])),
            },
        },
        "series": series_refs,
        "verdict": {"state": None, "reason": VERDICT_REASON},
    }
    parents: list[tuple[str, Sequence[ArtifactRef]]] = [
        ("trajectory", [trajectory_ref])
    ]
    if calibration is not None:
        parents.append(("calibration", [calibration.ref]))
    if codebook_ref is not None:
        parents.append(("codebook", [codebook_ref]))
    envelope = build_envelope(
        ctx,
        payload_schema=PAYLOAD_SCHEMA,
        payload=payload,
        space_id=traj.space_id,
        parents=parents,
        calibration_ref=None if calibration is None else calibration.ref,
        extensions={},
    )
    return Emission(envelope, blobs)


def emit_pair(
    left: TrajectoryInput,
    right: TrajectoryInput,
    ctx: EmitContext,
    *,
    left_ref: ArtifactRef,
    right_ref: ArtifactRef,
    calibration: CalibrationArtifact | None = None,
    alignment: Alignment = "equal_only",
) -> Emission:
    """Two trajectories compared, as an envelope (payload ``mode: "pair"``).

    ``common_prefix`` is written as ``x-common_prefix`` and needs
    ``ctx.alignment_approval`` (S17). A calibration made for another alignment
    is refused (S13). Trajectories of different spaces are always refused.
    """
    if left.model_build_id != right.model_build_id:
        raise InvalidTrajectory("left and right must share model_build_id (S18)")
    mode = payload_alignment(alignment)
    extensions = alignment_extensions(mode, ctx)
    measured = compare_trajectories(
        left, right, calibration=calibration, alignment=alignment
    )
    distance_q = quantize(measured["distance"]["value"], EXPONENT)
    payload = {
        "mode": "pair",
        "space_id": left.space_id,
        "inputs": {"left": input_ref(left), "right": input_ref(right)},
        "calibration": None if calibration is None else calibration.ref.to_dict(),
        "alignment": {
            "mode": mode,
            "points_used": measured["alignment"]["points_used"],
        },
        "distance": {"definition": "mean_step_l2", "value": distance_q},
        "verdict": {
            "state": evaluate_shift_quantized(
                distance_q, calibration, distance_exponent=EXPONENT
            )
        },
    }
    parents: list[tuple[str, Sequence[ArtifactRef]]] = [
        ("left", [left_ref]),
        ("right", [right_ref]),
    ]
    if calibration is not None:
        parents.append(("calibration", [calibration.ref]))
    envelope = build_envelope(
        ctx,
        payload_schema=PAYLOAD_SCHEMA,
        payload=payload,
        space_id=left.space_id,
        parents=parents,
        calibration_ref=None if calibration is None else calibration.ref,
        extensions=extensions,
    )
    return Emission(envelope)


__all__ = [
    "PAYLOAD_SCHEMA",
    "SERIES",
    "UNDEFINED",
    "emit_pair",
    "emit_single",
]
