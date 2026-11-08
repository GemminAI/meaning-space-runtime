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
value" is ``-2**63``; every series always declares ``undefined: "int64_min"``
(contract 0.7.1, whether or not a value is missing). The positions are those of
``msr/README.md`` (§3.3.2, §7.1).
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
from msr.geometry.errors import CalibrationError, InvalidTrajectory, InvalidWindow
from msr.geometry.metrics import DEFAULT_BLOCK_SIZE
from msr.geometry.trajectory import TrajectoryInput
from msr.geometry.window import TrajectoryWindow, check_window_spaces
from msr.geometry.wire import (
    BLOB_RANGE,
    E_QUANT_RANGE,
    INT64_RANGE,
    MAX_SAFE,
    ContractError,
    dequantize,
    quantize,
)

PAYLOAD_SCHEMA = ("msr.geometry-measurement", "0.9.0")
PAYLOAD_SCHEMA_V0100 = (
    "msr.geometry-measurement",
    "0.10.0",
)  # IC-02 0.10.0 (D-132): the window's rule/start/length


def _payload_schema(
    schema_version: str | None, *windows: TrajectoryWindow | None
) -> tuple[str, str]:
    """0.9.0 unless a 0.10.0 window is measured or 0.10.0 is asked for.

    Otherwise the output stays byte-identical to before 0.10.0.

    A 0.10.0 measurement may have a 0.9.0 window as its parent (D-133 (5)); a 0.9.0
    measurement can only declare the 0.9.0 window (prefix_rows, start 0, length 128).
    """
    held = [w for w in windows if w is not None]
    if schema_version is None:
        schema_version = (
            "0.10.0" if any(w.schema_version == "0.10.0" for w in held) else "0.9.0"
        )
    if schema_version == "0.10.0":
        return PAYLOAD_SCHEMA_V0100
    if schema_version != "0.9.0":
        raise ContractError(E_CONTEXT, f"unknown schema_version {schema_version!r}")
    if any((w.rule, w.start, w.length) != ("prefix_rows", 0, 128) for w in held):
        raise InvalidWindow(
            "a 0.9.0 measurement can declare only the 0.9.0 window "
            "(prefix_rows, 0, 128)"
        )
    return PAYLOAD_SCHEMA


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
            # 0.7.1: every series can hold "no value" by contract, so it always says so
            "undefined": "int64_min",
        }
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
    window: TrajectoryWindow | None = None,
    calibration: CalibrationArtifact | None = None,
    codebook: NDArray[np.float64] | None = None,
    codebook_ref: ArtifactRef | None = None,
    epsilon: float | None = None,
    delta: float | None = None,
    w_min: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    schema_version: str | None = None,
) -> Emission:
    """One trajectory's measurement as an envelope (payload ``mode: "single"``).

    ``window=None``: the whole trajectory is measured; ``input_scope`` is
    ``{"kind": "full"}`` and ``trajectory_ref`` is the (non-window) trajectory
    artifact. ``window=w`` (IC-02 0.9.0, D-129): the first-128-rows window artifact
    is measured; pass ``traj=w.trajectory`` and ``trajectory_ref=w.ref`` (anything
    else is refused), and ``input_scope`` declares the window. A window artifact
    passed as a ``full`` input cannot be told apart here (MSR does not hold the
    parent); the Gate refuses it.

    ``schema_version=None`` (default): ``0.10.0`` when the window was read under 0.10.0
    (IC-02 0.10.0, D-132: ``input_scope`` declares rule/start/length), otherwise
    ``0.9.0`` -- a full input and a 0.9.0 window are emitted exactly as before.
    ``"0.10.0"`` may also
    be asked for with a 0.9.0 window as the parent (D-133 (5)).
    """
    schema = _payload_schema(schema_version, window)
    if window is not None and (
        traj is not window.trajectory or trajectory_ref != window.ref
    ):
        raise InvalidTrajectory(
            "with window=, traj must be window.trajectory and trajectory_ref window.ref"
        )
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
    if (
        calibration is not None
        and calibration.exponent != EXPONENT
        and "calibration" in (eps_src, dlt_src)
    ):
        raise CalibrationError(
            f"quantum differs: thresholds 2**{EXPONENT}, "
            f"calibration 2**{calibration.exponent} (no conversion)"
        )
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
        "input_scope": {"kind": "full"} if window is None else dict(window.scope),
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
        payload_schema=schema,
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
    left_window: TrajectoryWindow | None = None,
    right_window: TrajectoryWindow | None = None,
    schema_version: str | None = None,
) -> Emission:
    """Two trajectories compared, as an envelope (payload ``mode: "pair"``).

    ``common_prefix`` is written as ``x-common_prefix`` and needs
    ``ctx.alignment_approval`` (S17). A calibration made for another alignment
    is refused (S13). Trajectories of different spaces are always refused.

    Windows (IC-02 0.10.0): ``left_window=`` / ``right_window=`` say the side is a
    window artifact (``left`` must be its ``trajectory`` and ``left_ref`` its ``ref``).
    Two windows of different lengths are refused (``pair_window_length``; the Gate
    refuses them whatever the alignment, ``common_prefix`` is not adopted, D-120).
    Windows sharing a ``space_id`` must share ``(start, length)``
    (``window_space_position``). Comparing windows at different positions is NOT
    DECIDED (D-133 (6)):
    this function does not decide it; what it does today is recorded in the tests.
    """
    schema = _payload_schema(schema_version, left_window, right_window)
    for side, w, t, r in (
        ("left", left_window, left, left_ref),
        ("right", right_window, right, right_ref),
    ):
        if w is not None and (t is not w.trajectory or r != w.ref):
            raise InvalidTrajectory(
                f"with {side}_window=, {side} must be its trajectory "
                f"and {side}_ref its ref"
            )
    if left_window is not None and right_window is not None:
        check_window_spaces([left_window, right_window])
        if left_window.length != right_window.length:
            raise InvalidWindow(
                f"pair_window_length: window lengths {left_window.length} != "
                f"{right_window.length}; windows of different lengths cannot be "
                "compared (equal_only; common_prefix is not adopted, D-120)"
            )
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
        payload_schema=schema,
        payload=payload,
        space_id=left.space_id,
        parents=parents,
        calibration_ref=None if calibration is None else calibration.ref,
        extensions=extensions,
    )
    return Emission(envelope)


__all__ = [
    "PAYLOAD_SCHEMA",
    "PAYLOAD_SCHEMA_V0100",
    "SERIES",
    "UNDEFINED",
    "emit_pair",
    "emit_single",
]
