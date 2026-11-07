"""The vendored contract files are exactly what contract.lock records."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from contract_support import HERE, VENDOR, schema_errors  # noqa: F401

LOCK: dict[str, Any] = json.loads((HERE / "contract.lock").read_text("utf-8"))


def test_lock_shape_and_commit() -> None:
    assert LOCK["contract_id"] == "IC-00" and LOCK["version"] == "1.0.0"
    assert LOCK["source"]["commit"] == "bbc085b87d1ed54869ae325b8c2f05dc711cd108"
    assert LOCK["source"]["branch"] == "feature/contracts-ic02-v080"
    assert LOCK["source"]["tag"] is None
    assert "contracts/ic00-v1.0.0" in LOCK["source"]["tag_note"]
    assert LOCK["contracts"] == {
        "IC-00": "1.0.0",
        "common/blob": "0.7.0",
        "msr/geometry-measurement": "0.7.1",
        "msr/calibration": "0.8.0",
    }
    paths = {f["path"] for f in LOCK["files"]}
    for needed in (
        "IC-00/1.0.0/FREEZE.md",
        "common/artifact.schema.json",
        "common/blob.schema.json",
        "msr/geometry-measurement.schema.json",
        "msr/calibration.schema.json",
        "msr/README.md",
        "msr/vectors/series.json",
        "msr/vectors/calibration-parents.json",
        "msr/vectors/calibration-exponent.json",
        "common/vectors/canonicalize.json",
        "common/vectors/quantize.json",
        "common/vectors/ids.json",
        "common/vectors/envelope-ids.json",
        "common/vectors/check-envelope.json",
        "common/vectors/blob/blob-refs.json",
    ):
        assert needed in paths


def test_frozen_files_match_the_freeze_record() -> None:
    """FREEZE.md (vendored from the frozen tag) lists sha256 for the IC-00 files."""
    text = (VENDOR / "IC-00/1.0.0/FREEZE.md").read_text("utf-8")
    rows = re.findall(
        r"^\| `docs/architecture/contracts/([^`]+)` \| `([0-9a-f]{64})` \|$", text, re.M
    )
    assert (
        len(rows) == 8
    )  # 3 schemas + 5 vectors (the reference-implementation rows live in tools/)
    for path, digest in rows:
        assert hashlib.sha256((VENDOR / path).read_bytes()).hexdigest() == digest, path
    assert "f4a865ccf58027c64ed4b9197bc9bd0ebb76fa8f63bd69e1bf7c7a7dd137b911" in text


@pytest.mark.parametrize("entry", LOCK["files"], ids=lambda e: str(e["path"]))
def test_vendored_file_matches_lock(entry: dict[str, str]) -> None:
    data = (VENDOR / entry["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_no_unlocked_file_in_vendor() -> None:
    on_disk = {str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*") if p.is_file()}
    assert on_disk == {f["path"] for f in LOCK["files"]}


def test_schema_ids_carry_the_locked_versions() -> None:
    ids = {
        f["path"]: json.loads((VENDOR / f["path"]).read_text("utf-8"))["$id"]
        for f in LOCK["files"]
        if f["path"].endswith(".schema.json")
    }
    assert ids == {
        "common/artifact.schema.json": "sensos:contracts/common/artifact/1.0.0",
        "common/provenance.schema.json": "sensos:contracts/common/provenance/1.0.0",
        "common/uncertainty.schema.json": "sensos:contracts/common/uncertainty/1.0.0",
        "common/blob.schema.json": "sensos:contracts/common/blob/0.7.0",
        "msr/geometry-measurement.schema.json": "sensos:contracts/msr/geometry-measurement/0.7.1",
        "msr/calibration.schema.json": "sensos:contracts/msr/calibration/0.8.0",
    }


def test_output_declares_the_vendored_versions() -> None:
    from contract_support import calibration, ctx, ref, traj

    import msr.geometry as g

    art = calibration()
    em = g.emit_single(traj(9), ctx(), trajectory_ref=ref(50))
    assert art.envelope["contract"] == {"id": "IC-00", "version": "1.0.0"}
    assert art.envelope["payload_schema"] == {
        "id": "msr.calibration",
        "version": "0.8.0",
    }
    assert em.envelope["payload_schema"] == {
        "id": "msr.geometry-measurement",
        "version": "0.7.1",
    }


def test_vendor_equals_sensos_commit_when_available() -> None:
    sensos = Path.home() / "GemminAI" / "sensos"
    if not (sensos / ".git").exists():
        pytest.skip("sensos checkout not present")
    for entry in LOCK["files"]:
        spec = f"{LOCK['source']['commit']}:{LOCK['source']['root']}{entry['path']}"
        shown = subprocess.run(
            ["git", "-C", str(sensos), "show", spec], capture_output=True, check=False
        )
        if shown.returncode != 0:
            pytest.skip("locked sensos commit not in the local checkout")
        assert hashlib.sha256(shown.stdout).hexdigest() == entry["sha256"]


PINNED = {
    "msr/README.md": "d16f04c6ddb904c2a6b7d75372d55b10ca4c3040f78c1bb04d7ed488f5f7c974",
    "msr/vectors/calibration-parents.json": "6c176b2673632281ff2676cf095462ead8b4b3f1eb77f9d9e128219a3e9ee0a1",
    "msr/vectors/calibration-exponent.json": "123ce297480be9b7b47cc46a7aa1d6466ba9301ce0f0094778e19ab6b70aa1a3",
    "msr/geometry-measurement.schema.json": "d9aa7ea59237c4c6f19d33bbac6506ce0983b10ce46a3ee3a94b076b9bd05f6e",
    "msr/vectors/series.json": "de25c3ddab5c8632a63a8006bf3b0ea8b2ad7a958a2639a5a0dfde964a4f01e9",
    "msr/calibration.schema.json": "794852390b3ec765fdeeb8c71e9fad75b9e5f3c81f4f26daa769dff6b4955d4f",
}


@pytest.mark.parametrize(("path", "digest"), sorted(PINNED.items()))
def test_msr_contract_files_are_the_ic02_0_8_0_ones(path: str, digest: str) -> None:
    recorded = {f["path"]: f["sha256"] for f in LOCK["files"]}
    assert recorded[path] == digest
    assert hashlib.sha256((VENDOR / path).read_bytes()).hexdigest() == digest


def test_archive_is_not_vendored() -> None:
    assert not (VENDOR / "msr" / "archive").exists()
