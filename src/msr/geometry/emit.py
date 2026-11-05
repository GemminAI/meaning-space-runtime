"""IC-02 emission: trajectories -> IC-00 envelopes (design v0.6.2 §3.3, App. A).

``emit_single`` and ``emit_pair`` run the measurement layer
(:mod:`msr.geometry.analyze`), quantize every real number onto the 2**-30 grid
(ties-to-even; discrete integers stay as they are), move the per-prefix series
into an int64 Blob, and wrap the payload in an envelope. Thresholds are
quantized *before* they are used, so the payload's thresholds are exactly the
ones the numbers were computed with, and the pair verdict is decided on the
payload's integers.

Series Blob layout (``unit`` = ``"msr-geometry-series/1"``): shape ``[10, T]``,
row ``k`` = :data:`SERIES_ORDER` ``[k]``, column ``t`` = the value from points
``0..t`` as ``round_ties_to_even(value * 2**30)`` (counts too), little-endian
int64, C order, no header; ``-2**63`` marks "undefined".
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
    INT64_RANGE,
    MAX_SAFE,
    ContractError,
    dequantize,
    quantize,
)

PAYLOAD_SCHEMA = ("msr.geometry-measurement", "0.6.2")
SERIES_UNIT = "msr-geometry-series/1"
UNDEFINED = INT64_RANGE[0]  # the Blob's "no value" marker
SERIES_ORDER: tuple[str, ...] = (
    "displacement_mean",
    "displacement_median",
    "displacement_longest_sub_delta_run",
    "recurrence_pair_fraction",
    "recurrence_return_count",
    "recurrence_return_rate",
    "diversity_cell_entropy",
    "diversity_participation_ratio",
    "persistence_longest_stay",
    "persistence_dominant_cell_share",
)


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


def _series_blob_ref(
    series: dict[str, list[Any]], length: int
) -> tuple[Any, dict[str, Any]]:
    rows = [
        [
            UNDEFINED if v is None else quantize(v, EXPONENT, int_range=INT64_RANGE)
            for v in series[name]
        ]
        for name in SERIES_ORDER
    ]
    blob = int64_blob(rows)
    ref = {
        "uri": blob.uri,
        "sha256": blob.sha256,
        "media_type": BLOB_MEDIA_TYPE,
        "dtype": "int64",
        "shape": [len(SERIES_ORDER), length],
        "byte_order": "little",
        "order": "C",
        "quantization": {
            "quantum": {"base": 2, "exponent": EXPONENT},
            "rounding": "ties-to-even",
            "tolerance_quanta": 1,
        },
        "unit": SERIES_UNIT,
    }
    return blob, ref


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
    blob, series_ref = _series_blob_ref(measured["series"], traj.length)
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
        "series": series_ref,
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
    return Emission(envelope, (blob,))


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
        "verdict": {"state": evaluate_shift_quantized(distance_q, calibration)},
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
    "SERIES_ORDER",
    "SERIES_UNIT",
    "UNDEFINED",
    "emit_pair",
    "emit_single",
]
