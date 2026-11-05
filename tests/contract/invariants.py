"""Semantic invariants (design v0.6.2 §8) that MSR can decide on its own.

S3, S13, S17, S18, S19, S20, S21, S23 and S24, written against the envelope and
the stored artifacts it points at (``stored`` maps artifact_id -> a dict with the
fields the check needs: ``envelope`` for a calibration, ``space_id`` for an
input trajectory). Independent of the emitting code on purpose: it reads only
what the contract puts in the envelope.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from typing import Any

from contract_support import validator

ALIGN_EXT = "x-meaning-space-runtime-alignment"
INT64_MIN = -(2**63)
# series name -> (metrics group, metrics key); the contract's table (msr/README.md)
SERIES = {
    "displacement_mean": ("displacement", "mean"),
    "displacement_median": ("displacement", "median"),
    "displacement_longest_sub_delta_run": ("displacement", "longest_sub_delta_run"),
    "recurrence_pair_fraction": ("recurrence", "pair_fraction"),
    "recurrence_return_count": ("recurrence", "return_count"),
    "recurrence_return_rate": ("recurrence", "return_rate"),
    "diversity_cell_entropy": ("diversity", "cell_entropy"),
    "diversity_participation_ratio": ("diversity", "participation_ratio"),
    "persistence_longest_stay": ("persistence", "longest_stay"),
    "persistence_dominant_cell_share": ("persistence", "dominant_cell_share"),
}
ITEMSIZE = {"int64": 8, "float32": 4, "float64": 8}


def ints(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 8}q", data))


def blob_violations(ref: Mapping[str, Any], data: bytes | None) -> set[str]:
    """Blob reference checks of design §7.1 (S34, S24/S25, S36 declaration, counts)."""
    if list(validator("common/blob").iter_errors(ref)):
        return {"schema"}
    out: set[str] = set()
    if ref["uri"] != "blob:sha256:" + ref["sha256"]:
        out.add("S34")
    if data is None:
        return out
    code = "S24" if ref["dtype"] == "int64" else "S25"
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        out.add(code)
    if len(data) != math.prod(ref["shape"]) * ITEMSIZE[ref["dtype"]]:
        out.add(code)
        return out
    if ref["dtype"] == "int64":
        values = ints(data)
        declared = ref.get("undefined") == "int64_min"
        if INT64_MIN in values and not declared:
            out.add("S36")
        if ref["quantization"] is None and any(
            v < 0 for v in values if v != INT64_MIN or not declared
        ):
            out.add("count_negative")
    return out


def expected_undefined(
    name: str, length: int, th: Mapping[str, Any], has_codebook: bool
) -> set[int]:
    """Positions with no value, by msr/README.md."""
    everything = set(range(length))
    eps, delta, w_min = th["epsilon"], th["delta"], th["w_min"]
    if name in (
        "displacement_mean",
        "displacement_median",
        "diversity_participation_ratio",
    ):
        return {0}
    if name == "displacement_longest_sub_delta_run":
        return everything if delta is None else {0}
    if name == "recurrence_pair_fraction":
        if eps is None or w_min is None:
            return everything
        return set(range(min(length, w_min + 1)))
    if name in ("recurrence_return_count", "recurrence_return_rate"):
        return everything if eps is None or w_min is None else set()
    if name == "diversity_cell_entropy":
        return set() if has_codebook else everything
    return set() if has_codebook or eps is not None else everything


def verdict(d: int, calibration_payload: Mapping[str, Any] | None) -> str:
    if calibration_payload is None:
        return "UNCALIBRATED"
    lo = calibration_payload["values"]["noise_floor_lower"]
    hi = calibration_payload["values"]["noise_floor_upper"]
    if d < lo:
        return "STAGNANT"
    return "NOISE_FLOOR_INTERNAL" if d <= hi else "STRUCTURAL_SHIFT_DETECTED"


def _role_ids(envelope: Mapping[str, Any], role: str) -> list[str] | None:
    for parent in envelope["parents"]:
        if parent["role"] == role:
            ids: list[str] = parent["artifact_ids"]
            return ids
    return None


def _calibration(
    envelope: Mapping[str, Any], stored: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    ref = envelope["calibration_ref"]
    if ref is None:
        return None
    found: Mapping[str, Any] = stored[ref["artifact_id"]]["envelope"]
    return found


def _approved(envelope: Mapping[str, Any]) -> bool:
    ext = envelope["extensions"].get(ALIGN_EXT)
    return (
        isinstance(ext, dict)
        and bool(ext.get("reason"))
        and bool(ext.get("approved_by"))
    )


def violations(
    envelope: Mapping[str, Any],
    blobs: Mapping[str, bytes],
    stored: Mapping[str, Any],
) -> set[str]:
    out: set[str] = set()
    payload = envelope["payload"]
    if envelope["payload_schema"]["id"] == "msr.calibration":
        _calibration_payload(envelope, payload, out)
        return out
    single = payload["mode"] == "single"
    calibration = _calibration(envelope, stored)
    cal_payload = None if calibration is None else calibration["payload"]

    if single and payload["verdict"]["state"] is not None:
        out.add("S3")
    if cal_payload is not None:
        if cal_payload["space_id"] != payload["space_id"]:
            out.add("S13")
        if (
            not single
            and cal_payload["between_alignment"] != payload["alignment"]["mode"]
        ):
            out.add("S13")
    mode = None if single else payload["alignment"]["mode"]
    cal_mode = None if cal_payload is None else cal_payload["between_alignment"]
    if "x-common_prefix" in (mode, cal_mode) and not _approved(envelope):
        out.add("S17")
    if not single:
        _s18(envelope, payload, stored, out)
    _s20(envelope, payload, out)
    if cal_payload is not None and single:
        for key in ("epsilon", "delta", "w_min"):
            if (
                payload["thresholds"]["source"][key] == "calibration"
                and payload["thresholds"][key] != cal_payload["values"][key]
            ):
                out.add("S23")
    if single:
        _series(envelope, payload, blobs, out)
    else:
        want = verdict(payload["distance"]["value"], cal_payload)
        if payload["verdict"]["state"] != want:
            out.add("S38")
    return out


def _s18(
    envelope: Mapping[str, Any],
    payload: Mapping[str, Any],
    stored: Mapping[str, Any],
    out: set[str],
) -> None:
    left, right = payload["inputs"]["left"], payload["inputs"]["right"]
    for role in ("left", "right"):
        ids = _role_ids(envelope, role)
        if ids is None or len(ids) != 1:
            out.add("S18")
            continue
        if stored[ids[0]]["space_id"] != payload["space_id"]:
            out.add("S18")
    if (
        left["model_build_id"] != right["model_build_id"]
        or left["dimension"] != right["dimension"]
    ):
        out.add("S18")
    used = payload["alignment"]["points_used"]
    if payload["alignment"]["mode"] == "equal_only":
        if left["length"] != right["length"] or used != left["length"]:
            out.add("S18")
    elif used != min(left["length"], right["length"]):
        out.add("S18")


def _s20(
    envelope: Mapping[str, Any], payload: Mapping[str, Any], out: set[str]
) -> None:
    ref = envelope["calibration_ref"]
    in_role = _role_ids(envelope, "calibration")
    if payload["calibration"] != ref:
        out.add("S20")
    if ref is None:
        if in_role is not None:
            out.add("S20")
    elif in_role != [ref["artifact_id"]]:
        out.add("S20")


def _series(
    envelope: Mapping[str, Any],
    payload: Mapping[str, Any],
    blobs: Mapping[str, bytes],
    out: set[str],
) -> None:
    length = payload["input_ref"]["length"]
    has_codebook = any(p["role"] == "codebook" for p in envelope["parents"])
    if set(payload["series"]) != set(SERIES):
        out.add("S35")
        return
    for name, (group, key) in SERIES.items():
        ref = payload["series"][name]
        data = blobs.get(ref["uri"])
        out |= blob_violations(ref, data)
        if ref.get("undefined") != "int64_min":
            out.add("S36")  # 0.7.1: every series declares it, always
        if ref["shape"] != [length]:
            out.add("S35")
        if data is None or len(data) != 8 * length:
            continue
        values = ints(data)
        undefined = {t for t, v in enumerate(values) if v == INT64_MIN}
        if undefined != expected_undefined(
            name, length, payload["thresholds"], has_codebook
        ):
            out.add("S36")
        metric = payload["metrics"][group][key]
        if (INT64_MIN if metric is None else metric) != values[-1]:
            out.add("S37")


def _calibration_payload(
    envelope: Mapping[str, Any], payload: Mapping[str, Any], out: set[str]
) -> None:
    values, quantiles = payload["values"], payload["method"]["quantiles"]
    if values["noise_floor_lower"] > values["noise_floor_upper"]:
        out.add("S19")
    if not quantiles["floor"][0] < quantiles["floor"][1]:
        out.add("S19")
    role = _role_ids(envelope, "null_set")
    null_set = payload["null_set"]
    if (
        role is None
        or set(null_set) != set(role)
        or len(null_set) != len(set(null_set))
        or len(null_set) != payload["sample_config"]["trajectories"]
        or null_set != sorted(null_set, key=lambda a: a.encode())
    ):
        out.add("S19")
    codebook = _role_ids(envelope, "codebook")
    cb_ref = values["codebook_ref"]
    if cb_ref is None:
        if codebook is not None:
            out.add("S21")
    elif codebook != [cb_ref["artifact_id"]]:
        out.add("S21")
    if envelope["calibration_ref"] is not None:
        out.add("calibration_self_reference")
    if payload["between_alignment"] == "x-common_prefix" and not _approved(envelope):
        out.add("S17")


def series_payload_violations(
    payload: Mapping[str, Any],
    parents: list[dict[str, Any]],
    blobs: Mapping[str, bytes],
) -> set[str]:
    """Schema (msr/geometry-measurement) then the series checks, as the frozen
    ``series.json`` vectors state them: ``schema`` alone, else S34-S37."""
    if list(validator("msr/geometry-measurement").iter_errors(payload)):
        return {"schema"}
    out: set[str] = set()
    _series({"parents": parents}, payload, blobs, out)
    return out
