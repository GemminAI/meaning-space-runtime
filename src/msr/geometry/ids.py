"""``content_id`` and ``artifact_id`` (design v0.6.2 §2.1).

Preimage key names are the ones fixed in the sensos contracts README (the design
lists the fields but not their JSON names)::

    content_id  = "cnt:" + sha256(JCS({contract, kind, payload_schema, payload,
                    producer{name, version}, policy_refs, rule_refs,
                    calibration_content_id, space_id, event_span,
                    parent_contents[{role, ordered, content_ids}]}))
    artifact_id = "art:" + sha256(JCS({content_id, session_id, observation_id,
                    producer_build, parents[{role, ordered, artifact_ids}]}))

``producer.build`` enters ``artifact_id`` only. ``seq``, ``produced_at``,
``observed_at`` and ``extensions`` enter neither. MSR numbers nothing: the
``session_id`` / ``observation_id`` / ``seq`` values are the caller's.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from msr.geometry.wire import E_LOOKUP, E_PARENTS, ContractError, jcs


def _bytes(text: str) -> bytes:
    return text.encode("utf-8")


def _sha(obj: Any) -> str:
    return hashlib.sha256(jcs(obj)).hexdigest()


def sorted_set(
    items: Sequence[Mapping[str, Any]], fields: tuple[str, ...], what: str
) -> list[dict[str, Any]]:
    """Sort a set-like array by the UTF-8 bytes of ``fields``; refuse duplicates."""
    ordered = sorted(items, key=lambda item: tuple(_bytes(item[f]) for f in fields))
    keys = [tuple(item[f] for f in fields) for item in ordered]
    if len(set(keys)) != len(keys):
        raise ContractError(E_PARENTS, f"duplicate entry in {what}")
    return [dict(item) for item in ordered]


def normalized_parents(parents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Roles by name bytes; ``ordered: false`` ids by id bytes (no duplicates)."""
    names = [p["role"] for p in parents]
    if len(set(names)) != len(names):
        raise ContractError(E_PARENTS, "duplicate role in parents")
    out: list[dict[str, Any]] = []
    for parent in sorted(parents, key=lambda p: _bytes(p["role"])):
        ids = list(parent["artifact_ids"])
        if not parent["ordered"]:
            ids.sort(key=_bytes)
            if len(set(ids)) != len(ids):
                raise ContractError(
                    E_PARENTS, f"duplicate artifact id in role {parent['role']!r}"
                )
        out.append(
            {"role": parent["role"], "ordered": parent["ordered"], "artifact_ids": ids}
        )
    return out


def content_id(envelope: Mapping[str, Any], parent_contents: Mapping[str, str]) -> str:
    """``parent_contents`` maps each parent ``artifact_id`` to its ``content_id``."""
    contents: list[dict[str, Any]] = []
    for role in normalized_parents(envelope["parents"]):
        cids: list[str] = []
        for artifact in role["artifact_ids"]:
            if artifact not in parent_contents:
                raise ContractError(E_LOOKUP, f"parent {artifact} has no content_id")
            cids.append(parent_contents[artifact])
        if not role["ordered"]:
            cids.sort(key=_bytes)
        contents.append(
            {"role": role["role"], "ordered": role["ordered"], "content_ids": cids}
        )
    calibration = envelope["calibration_ref"]
    preimage = {
        "contract": envelope["contract"],
        "kind": envelope["kind"],
        "payload_schema": envelope["payload_schema"],
        "payload": envelope["payload"],
        "producer": {
            "name": envelope["producer"]["name"],
            "version": envelope["producer"]["version"],
        },
        "policy_refs": sorted_set(
            envelope["policy_refs"], ("name", "sha256"), "policy_refs"
        ),
        "rule_refs": sorted_set(
            envelope["rule_refs"], ("name", "version", "sha256"), "rule_refs"
        ),
        "calibration_content_id": None
        if calibration is None
        else calibration["content_id"],
        "space_id": envelope["space_id"],
        "event_span": envelope["event_span"],
        "parent_contents": contents,
    }
    return "cnt:" + _sha(preimage)


def artifact_id(envelope: Mapping[str, Any], parent_contents: Mapping[str, str]) -> str:
    preimage = {
        "content_id": content_id(envelope, parent_contents),
        "session_id": envelope["session_id"],
        "observation_id": envelope["observation_id"],
        "producer_build": envelope["producer"]["build"],
        "parents": normalized_parents(envelope["parents"]),
    }
    return "art:" + _sha(preimage)


__all__ = ["artifact_id", "content_id", "normalized_parents", "sorted_set"]
