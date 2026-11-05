"""An independent, test-side implementation of the frozen IC-00 reference checks
(ids, ``check_envelope``) written from the contract text, so the frozen vectors can
be run against MSR without importing anything from sensos.

MSR itself numbers nothing, so none of this is in ``src``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, NamedTuple

from contract_support import validator

from msr.geometry import ids
from msr.geometry.wire import MAX_SAFE, ContractError, jcs


class Violation(NamedTuple):
    code: str
    message: str


def _sha(obj: Any) -> str:
    return hashlib.sha256(jcs(obj)).hexdigest()


def _need_int(value: Any, lo: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lo <= value <= MAX_SAFE
    ):
        raise ContractError("E_ID_INPUT", "bad integer")
    return value


def _need_str(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("E_ID_INPUT", "bad string")
    return value


def session_id_experiment(
    experiment_spec_hash: str,
    item_id: str,
    condition: str,
    replicate: int,
    rng_seed: int,
) -> str:
    return "ses:" + _sha(
        {
            "experiment_spec_hash": _need_str(experiment_spec_hash),
            "item_id": _need_str(item_id),
            "condition": _need_str(condition),
            "replicate": _need_int(replicate, 0),
            "rng_seed": _need_int(rng_seed, -MAX_SAFE),
        }
    )


def observation_id(session_id: str, ordinal: int) -> str:
    if not isinstance(session_id, str) or not re.fullmatch(
        r"ses:[0-9a-f]{64}", session_id
    ):
        raise ContractError("E_ID_INPUT", "bad session_id")
    return "obs:" + _sha({"session_id": session_id, "ordinal": _need_int(ordinal, 0)})


KINDS: dict[str, frozenset[str]] = {
    "nvs-kernel": frozenset({"observation"}),
    "nvs-runtime": frozenset({"observation", "interpretation", "outcome"}),
    "hekb-vnext": frozenset({"observation"}),
    "hext": frozenset({"evidence"}),
    "meaning-mapper": frozenset({"interpretation"}),
    "meaning-space-runtime": frozenset({"interpretation"}),
    "categorical-lift-engine": frozenset({"interpretation"}),
    "sensos-ux": frozenset({"selection"}),
}
PREFIX_KINDS = {
    "adapter:": frozenset({"interpretation"}),
    "capability:": frozenset({"interpretation"}),
}


def _allowed(name: str) -> frozenset[str]:
    if name in KINDS:
        return KINDS[name]
    return next((k for p, k in PREFIX_KINDS.items() if name.startswith(p)), frozenset())


def _b(text: str) -> bytes:
    return text.encode("utf-8")


def _s16(env: Mapping[str, Any]) -> tuple[list[Violation], bool]:
    out: list[Violation] = []
    duplicate = False
    for name, fields in (
        ("policy_refs", ("name", "sha256")),
        ("rule_refs", ("name", "version", "sha256")),
    ):
        items = list(env[name])
        keys = [tuple(_b(i[f]) for f in fields) for i in items]
        if len(set(keys)) != len(keys):
            duplicate = True
            out.append(Violation("S16", f"duplicate in {name}"))
        elif keys != sorted(keys):
            out.append(Violation("S16", f"{name} not sorted"))
    roles = [p["role"] for p in env["parents"]]
    if len(set(roles)) != len(roles):
        duplicate = True
        out.append(Violation("S16", "duplicate role"))
    if [_b(r) for r in roles] != sorted(_b(r) for r in roles):
        out.append(Violation("S16", "roles not in name order"))
    for parent in env["parents"]:
        if parent["ordered"]:
            continue
        found = list(parent["artifact_ids"])
        if len(set(found)) != len(found):
            duplicate = True
            out.append(Violation("S16", "duplicate ids"))
        elif [_b(i) for i in found] != sorted(_b(i) for i in found):
            out.append(Violation("S16", "ids not sorted"))
    return out, duplicate


def _s8(env: Mapping[str, Any], parents: list[Mapping[str, Any]]) -> list[Violation]:
    own = env["observation_id"]
    if not env["parents"]:
        if env["kind"] == "observation" and own is None:
            return [Violation("S8", "root observation needs an observation_id")]
        return []
    seen = {p.get("observation_id") for p in parents} - {None}
    if len(seen) == 1:
        return [] if own in seen else [Violation("S8", "must inherit")]
    return [] if own is None else [Violation("S8", "must be null")]


def check_envelope(
    env: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]]
) -> list[Violation]:
    errors = list(validator("common/artifact").iter_errors(env))
    if errors:
        return [Violation("schema", errors[0].message)]
    out: list[Violation] = []
    if env["kind"] not in _allowed(env["producer"]["name"]):
        out.append(Violation("S4", "producer may not produce this kind"))
    s16, duplicate = _s16(env)
    out.extend(s16)
    parent_ids = [a for p in env["parents"] for a in p["artifact_ids"]]
    unknown = [a for a in parent_ids if a not in lookup]
    out.extend(Violation("parent_unknown", a) for a in unknown)
    cal = env["calibration_ref"]
    if cal is not None:
        stored = lookup.get(cal["artifact_id"])
        if stored is None:
            out.append(Violation("calibration_unknown", cal["artifact_id"]))
        elif stored.get("content_id") != cal["content_id"]:
            out.append(Violation("calibration_mismatch", cal["artifact_id"]))
    if not unknown and not duplicate:
        contents = {a: lookup[a]["content_id"] for a in parent_ids}
        if ids.content_id(env, contents) != env["content_id"]:
            out.append(Violation("content_id_mismatch", ""))
        if ids.artifact_id(env, contents) != env["artifact_id"]:
            out.append(Violation("artifact_id_mismatch", ""))
    known = [lookup[a] for a in parent_ids if a in lookup]
    out.extend(_s8(env, known))
    refs = list(known) + (
        [lookup[cal["artifact_id"]]]
        if cal is not None and cal["artifact_id"] in lookup
        else []
    )
    out.extend(
        Violation("S5", "parent seq not before") for r in refs if r["seq"] >= env["seq"]
    )
    return out
