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
    assert LOCK["contract_id"] == "IC-00" and LOCK["version"] == "0.6.2"
    assert re.fullmatch(r"[0-9a-f]{40}", LOCK["source"]["commit"])
    assert LOCK["source"]["commit"].startswith("6a34b99")
    paths = {f["path"] for f in LOCK["files"]}
    for needed in (
        "common/artifact.schema.json",
        "msr/geometry-measurement.schema.json",
        "msr/calibration.schema.json",
        "common/vectors/canonicalize.json",
        "common/vectors/quantize.json",
        "common/vectors/envelope-ids.json",
    ):
        assert needed in paths


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
    assert ids["common/artifact.schema.json"].endswith("/0.6.2")
    assert ids["msr/geometry-measurement.schema.json"].endswith("/0.6.2")
    assert ids["msr/calibration.schema.json"].endswith("/0.6.1")


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
