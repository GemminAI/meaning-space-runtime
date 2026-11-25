"""IC-00 version is an argument. The product default is 1.1.0.

The 1.1.0 envelope schema is a byte copy of sensos-wt-u6t
``docs/architecture/contracts/IC-00/1.1.0/artifact.schema.json``
(sha256 d1ba7c8ff80700eb6111a81e0b06e3ba0b9367887be8588a7187fdee9bb9e39e).
It lives outside ``vendor/`` because that directory is locked by
``contract.lock`` and ``test_no_unlocked_file_in_vendor``. Provenance stays
the vendored 1.0.0 schema, which this artifact schema ``$ref``s.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from contract_support import VENDOR, ctx, ref, traj
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import msr.geometry as g
from msr.geometry.envelope import build_envelope
from msr.geometry.wire import jcs

# emit_single(traj(9), ctx(), trajectory_ref=ref(50)) when the default was 1.0.0.
BEFORE_JCS_SHA256 = "f671394435908f69e521333750852c13c099a22f814b685c328d95a2d3b83013"
BEFORE_CONTENT_ID = "cnt:5cf24452383a3f9590cccb4df6e5385cc654b6f20d76941da31ed1ec98b76cb6"
SCHEMA_SHA256 = "d1ba7c8ff80700eb6111a81e0b06e3ba0b9367887be8588a7187fdee9bb9e39e"
SCHEMA_PATH = (
    Path(__file__).parent / "vendor_ic00_1_1_0" / "IC-00" / "1.1.0" / "artifact.schema.json"
)


def _schema_errors(envelope: dict) -> list[str]:
    raw = SCHEMA_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SCHEMA_SHA256
    doc = json.loads(raw)
    provenance = json.loads((VENDOR / "common/provenance.schema.json").read_text("utf-8"))
    registry = (
        Registry()
        .with_resource(doc["$id"], Resource.from_contents(doc))
        .with_resource(provenance["$id"], Resource.from_contents(provenance))
    )
    return [e.message for e in Draft202012Validator(doc, registry=registry).iter_errors(envelope)]


def _bare(**kw: object) -> g.EmitContext:
    """A context that does not name a version, so the product default applies."""
    return g.EmitContext(
        "ses:" + "1" * 64,
        "obs:" + "2" * 64,
        1,
        producer_build="test-build",
        produced_at="2026-11-06T00:00:00Z",
        **kw,
    )


def _emit(**kw):
    return g.emit_single(traj(9), _bare(), trajectory_ref=ref(50), **kw)


def test_explicit_1_0_0_matches_the_output_from_before_the_default_change() -> None:
    explicit = _emit(contract_version="1.0.0")
    raw = jcs(explicit.envelope)
    assert hashlib.sha256(raw).hexdigest() == BEFORE_JCS_SHA256
    assert explicit.envelope["content_id"] == BEFORE_CONTENT_ID
    assert explicit.envelope["contract"] == {"id": "IC-00", "version": "1.0.0"}
    via_context = g.emit_single(
        traj(9), _bare(contract_version="1.0.0"), trajectory_ref=ref(50)
    )
    assert jcs(via_context.envelope) == raw


def test_the_omitted_version_is_1_1_0_and_differs_from_1_0_0() -> None:
    omitted = _emit()
    none = _emit(contract_version=None)
    pinned = _emit(contract_version="1.0.0")
    assert omitted.envelope["contract"] == {"id": "IC-00", "version": "1.1.0"}
    assert jcs(none.envelope) == jcs(omitted.envelope)
    assert omitted.envelope["content_id"] != pinned.envelope["content_id"]
    assert omitted.envelope["payload"] == pinned.envelope["payload"]
    assert omitted.envelope["payload_schema"] == pinned.envelope["payload_schema"]
    assert _schema_errors(omitted.envelope) == []


def test_build_envelope_takes_the_same_argument() -> None:
    context = _bare()
    default = build_envelope(
        context,
        payload_schema=("msr.geometry-measurement", "0.9.0"),
        payload={"n": 1},
        space_id="S1",
        parents=[],
        calibration_ref=None,
        extensions={},
    )
    explicit = build_envelope(
        context,
        payload_schema=("msr.geometry-measurement", "0.9.0"),
        payload={"n": 1},
        space_id="S1",
        parents=[],
        calibration_ref=None,
        extensions={},
        contract_version="1.0.0",
    )
    assert default["contract"]["version"] == "1.1.0"
    assert explicit["contract"]["version"] == "1.0.0"
    assert default["content_id"] != explicit["content_id"]


@pytest.mark.parametrize("version", ["1.2.0", "2.0.0", ""])
def test_an_unsupported_version_is_refused(version: str) -> None:
    with pytest.raises(g.ContractError, match="unsupported contract version"):
        _emit(contract_version=version)
    with pytest.raises(g.ContractError, match="unsupported contract version"):
        ctx(contract_version=version)


@pytest.mark.parametrize("version", [1, 1.0, True, ["1.1.0"]])
def test_a_non_string_version_is_refused(version: object) -> None:
    with pytest.raises(g.ContractError, match="must be a string"):
        _emit(contract_version=version)
    with pytest.raises(g.ContractError, match="must be a string"):
        ctx(contract_version=version)


def test_window_and_full_both_accept_either_version() -> None:
    points = traj(1, n=3, d=2)
    origin = "ab" * 32
    window = g.TrajectoryWindow(
        points,
        ref(50),
        {"trajectory_id": origin, "position": "final_norm", "length": 3},
        {
            "kind": "window",
            "rule": "rows",
            "start": 0,
            "length": 3,
            "source_trajectory_id": origin,
            "content_sha256": "cd" * 32,
        },
        ref(7).artifact_id,
        "0.10.0",
    )
    for version in ("1.0.0", "1.1.0"):
        measured = g.emit_single(
            window.trajectory,
            ctx(),
            trajectory_ref=window.ref,
            window=window,
            contract_version=version,
        )
        assert measured.envelope["contract"]["version"] == version
        assert measured.envelope["payload"]["input_scope"]["kind"] == "window"
        full = g.emit_single(
            points, ctx(), trajectory_ref=ref(51), contract_version=version
        )
        assert full.envelope["contract"]["version"] == version
        assert full.envelope["payload"]["input_scope"] == {"kind": "full"}
    omitted = g.emit_single(
        window.trajectory, _bare(), trajectory_ref=window.ref, window=window
    )
    assert omitted.envelope["contract"]["version"] == "1.1.0"
