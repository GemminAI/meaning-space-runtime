"""MSR's own wire implementation against the vendored language-neutral vectors."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import pytest
from contract_support import VECTORS
from ic00_gate import check_envelope, observation_id, session_id_experiment
from invariants import blob_violations

from msr.geometry import ids
from msr.geometry.wire import (
    ENVELOPE_RANGE,
    INT64_RANGE,
    ContractError,
    dequantize,
    jcs,
    quantize,
    strict_load,
)


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((VECTORS / name).read_text("utf-8"))
    return data


CANON = _load("canonicalize.json")["cases"]
QUANT = _load("quantize.json")["cases"]
ENV = _load("envelope-ids.json")
ENV_BY_ID = {c["id"]: c for c in ENV["cases"]}
PARENT_CONTENTS = {k: v["content_id"] for k, v in ENV["lookup"].items()}


def _input_bytes(case: dict[str, Any]) -> bytes:
    if "input_hex" in case:
        return bytes.fromhex(case["input_hex"])
    return str(case["input_utf8"]).encode("utf-8")


@pytest.mark.parametrize("case", CANON, ids=[c["id"] for c in CANON])
def test_canonicalize_vectors(case: dict[str, Any]) -> None:
    expect = case["expect"]
    if expect["result"] == "reject":
        with pytest.raises(ContractError) as info:
            jcs(strict_load(_input_bytes(case)))
        assert info.value.code == expect["code"]
        return
    out = jcs(strict_load(_input_bytes(case)))
    assert out == expect["jcs_utf8"].encode("utf-8")
    assert hashlib.sha256(out).hexdigest() == expect["jcs_sha256"]


@pytest.mark.parametrize("case", QUANT, ids=[c["id"] for c in QUANT])
def test_quantize_vectors(case: dict[str, Any]) -> None:
    value = struct.unpack(">d", bytes.fromhex(case["value_bits"]))[0]
    rng = INT64_RANGE if case["range"] == "int64" else ENVELOPE_RANGE
    expect = case["expect"]
    if expect["result"] == "reject":
        with pytest.raises(ContractError) as info:
            quantize(value, case["exponent"], int_range=rng)
        assert info.value.code == expect["code"]
        return
    q = quantize(value, case["exponent"], int_range=rng)
    assert q == expect["q"]
    assert abs(dequantize(q, case["exponent"], int_range=rng) - value) <= 2.0 ** (
        case["exponent"] - 1
    )


@pytest.mark.parametrize("case", ENV["cases"], ids=[c["id"] for c in ENV["cases"]])
def test_envelope_id_vectors(case: dict[str, Any]) -> None:
    env = case["envelope"]
    assert ids.content_id(env, PARENT_CONTENTS) == case["expect"]["content_id"]
    assert ids.artifact_id(env, PARENT_CONTENTS) == case["expect"]["artifact_id"]
    assert hashlib.sha256(jcs(env)).hexdigest() == case["expect"]["envelope_jcs_sha256"]


@pytest.mark.parametrize(
    "rel", ENV["relations"], ids=[r["b"] for r in ENV["relations"]]
)
def test_envelope_relations(rel: dict[str, Any]) -> None:
    a, b = ENV_BY_ID[rel["a"]]["expect"], ENV_BY_ID[rel["b"]]["expect"]
    assert (a["content_id"] == b["content_id"]) == (rel["content_id"] == "same")
    assert (a["artifact_id"] == b["artifact_id"]) == (rel["artifact_id"] == "same")


def test_vector_counts() -> None:
    assert (len(CANON), len(QUANT), len(ENV["cases"]), len(ENV["relations"])) == (
        46,
        29,
        17,
        16,
    )


IDS = _load("ids.json")["cases"]


@pytest.mark.parametrize("case", IDS, ids=[c["id"] for c in IDS])
def test_id_vectors(case: dict[str, Any]) -> None:
    def run(i: dict[str, Any]) -> str:
        if case["op"] == "session_experiment":
            return session_id_experiment(**i)
        return observation_id(i["session_id"], i["ordinal"])

    expect = case["expect"]
    if expect["result"] == "reject":
        with pytest.raises(ContractError) as info:
            run(case["input"])
        assert info.value.code == expect["code"]
        return
    assert run(case["input"]) == next(v for k, v in expect.items() if k != "result")


CHECK = _load("check-envelope.json")["cases"]


@pytest.mark.parametrize("case", CHECK, ids=[c["id"] for c in CHECK])
def test_check_envelope_vectors(case: dict[str, Any]) -> None:
    codes = sorted({v.code for v in check_envelope(case["envelope"], case["lookup"])})
    assert codes == case["expect"]["codes"]
    assert (not codes) == (case["expect"]["result"] == "accept")


def test_frozen_vector_counts() -> None:
    def accepts(cases: list[dict[str, Any]]) -> int:
        return sum(1 for c in cases if c["expect"]["result"] == "accept")

    assert (len(IDS), accepts(IDS)) == (10, 5)
    assert (len(CHECK), accepts(CHECK)) == (37, 7)
    assert (len(CANON), accepts(CANON)) == (46, 16)
    assert (len(QUANT), accepts(QUANT)) == (29, 21)


BLOB = _load("blob/blob-refs.json")["cases"]


@pytest.mark.parametrize("case", BLOB, ids=[c["id"] for c in BLOB])
def test_blob_vectors(case: dict[str, Any]) -> None:
    data = bytes.fromhex(case["data_hex"]) if "data_hex" in case else None
    assert sorted(blob_violations(case["ref"], data)) == case["expect"]["codes"]


def test_blob_vector_counts() -> None:
    accepts = sum(1 for c in BLOB if c["expect"]["result"] == "accept")
    assert (len(BLOB), accepts) == (32, 8)
