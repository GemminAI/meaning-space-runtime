"""A calibration built with the product default (IC-00 1.1.0) must read back.

The reader (``_check_contract``) accepts exactly the contract versions in
``SUPPORTED_CONTRACT_VERSIONS`` with id ``IC-00``; the Gate (not the reader)
does the JSON-Schema check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from contract_support import ctx, nulls, parent_contents
from test_contract_version_arg import _schema_errors

import msr.geometry as g
from msr.geometry.calibration import calibration_path
from msr.geometry.wire import jcs

# build_calibration(nulls(), ctx(2)); ctx() pins contract_version="1.0.0".
PINNED_1_0_0_ARTIFACT_ID = (
    "art:7b7e586ec8ee358af984dada06bbceb57d1392ff1a12e894649242a4d7246e85"
)
PINNED_1_0_0_CONTENT_ID = (
    "cnt:0443fb4e285ca917d2608bca40b3f8a3042dee1bd8303c5804ed854a84beda58"
)


def _bare(seq: int = 2, **kw: Any) -> g.EmitContext:
    """A context that names no contract version, so the default applies."""
    return g.EmitContext(
        "ses:" + "1" * 64,
        "obs:" + "2" * 64,
        seq,
        producer_build="test-build",
        produced_at="2026-11-06T00:00:00Z",
        **kw,
    )


def _roundtrip(art: g.CalibrationArtifact, tmp_path: Path) -> g.CalibrationArtifact:
    g.write_calibration(tmp_path, art)
    return g.load_calibration(
        tmp_path, art.ref.artifact_id, parent_contents=parent_contents()
    )


def test_default_context_roundtrips_as_1_1_0(tmp_path: Path) -> None:
    art = g.build_calibration(nulls(), _bare())
    assert art.envelope["contract"] == {"id": "IC-00", "version": "1.1.0"}
    loaded = _roundtrip(art, tmp_path)
    assert loaded.envelope == art.envelope and loaded.ref == art.ref


def test_explicit_1_0_0_roundtrips(tmp_path: Path) -> None:
    art = g.build_calibration(nulls(), _bare(contract_version="1.0.0"))
    assert art.envelope["contract"] == {"id": "IC-00", "version": "1.0.0"}
    assert _roundtrip(art, tmp_path).envelope == art.envelope


def test_a_pinned_1_0_0_calibration_reads_back(tmp_path: Path) -> None:
    art = g.build_calibration(nulls(), ctx(2))
    assert art.ref.artifact_id == PINNED_1_0_0_ARTIFACT_ID
    assert art.ref.content_id == PINNED_1_0_0_CONTENT_ID
    loaded = _roundtrip(art, tmp_path)
    assert loaded.ref.artifact_id == PINNED_1_0_0_ARTIFACT_ID
    assert loaded.envelope["contract"]["version"] == "1.0.0"


def _store_tampered(tmp_path: Path, contract: Any) -> str:
    art = g.build_calibration(nulls(), _bare())
    env = json.loads(jcs(art.envelope))
    env["contract"] = contract
    aid = art.ref.artifact_id
    path = calibration_path(tmp_path, aid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jcs(env))
    return aid


@pytest.mark.parametrize(
    "contract",
    [
        {"id": "IC-00", "version": "1.2.0"},  # unsupported version
        {"id": "IC-00", "version": ""},
        {"id": "IC-00", "version": 1},  # not a str
        {"id": "IC-00", "version": ["1.1.0"]},
        {"id": "IC-02", "version": "1.1.0"},  # another id
        {"id": "IC-00", "version": "1.1.0", "extra": 1},  # extra key
        {"id": "IC-00"},
        "IC-00",
        None,
    ],
)
def test_unsupported_contract_is_refused(tmp_path: Path, contract: Any) -> None:
    aid = _store_tampered(tmp_path, contract)
    with pytest.raises(g.CalibrationError):
        g.load_calibration(tmp_path, aid, parent_contents=parent_contents())


def test_default_envelope_passes_the_vendored_ic00_1_1_0_schema() -> None:
    art = g.build_calibration(nulls(), _bare())
    assert _schema_errors(art.envelope) == []
