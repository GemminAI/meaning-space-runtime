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
from collections.abc import Mapping
from typing import Any

ALIGN_EXT = "x-meaning-space-runtime-alignment"


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
        _s24(payload["series"], blobs, out)
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


def _s24(series: Mapping[str, Any], blobs: Mapping[str, bytes], out: set[str]) -> None:
    if series["dtype"] != "int64":
        return
    data = blobs.get(series["uri"])
    if data is None:
        out.add("S24")
        return
    if hashlib.sha256(data).hexdigest() != series["sha256"]:
        out.add("S24")
    if len(data) != math.prod(series["shape"]) * 8:
        out.add("S24")


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
