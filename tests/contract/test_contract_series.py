"""Series table (unit, kind, undefined) against the vendored schema and README,
and the vendored series vectors against MSR's own T=1 / T=2 output."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest
from contract_support import POLICY, VENDOR, ctx, ref
from invariants import series_payload_violations, violations

import msr.geometry as g
from msr.geometry.emit import SERIES

CASES: list[dict[str, Any]] = json.loads(
    (VENDOR / "msr/vectors/series.json").read_text("utf-8")
)["cases"]


def _schema_series() -> dict[str, Any]:
    schema = json.loads(
        (VENDOR / "msr/geometry-measurement.schema.json").read_text("utf-8")
    )
    series: dict[str, Any] = schema["$defs"]["single"]["properties"]["series"]
    return series


def _readme_rows() -> dict[str, tuple[str, str, str]]:
    rows: dict[str, tuple[str, str, str]] = {}
    for line in (VENDOR / "msr/README.md").read_text("utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].strip("`") in SERIES:
            found = re.fullmatch(r"(実数|個数)・`([^`]+)`", cells[2])
            assert found is not None
            rows[cells[0].strip("`")] = (found.group(1), found.group(2), cells[5])
    return rows


def test_series_table_matches_the_vendored_schema() -> None:
    schema = _schema_series()
    assert set(schema["properties"]) == set(SERIES) == set(schema["required"])
    for name, (kind, unit) in SERIES.items():
        extra = schema["properties"][name]["allOf"][1]
        assert extra["properties"]["unit"] == {"const": unit}, name
        assert (extra["properties"]["quantization"] == {"type": "null"}) == (
            kind == "count"
        ), name
        assert extra["required"] == ["undefined"], name  # always declared


def test_series_table_matches_the_vendored_readme() -> None:
    rows = _readme_rows()
    assert set(rows) == set(SERIES)
    for name, (kind, unit) in SERIES.items():
        readme_kind, readme_unit, undefined = rows[name]
        assert (readme_kind == "個数") == (kind == "count"), name
        assert readme_unit == unit, name
        assert undefined == "常に", name


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_series_vectors(case: dict[str, Any]) -> None:
    blobs = {uri: bytes.fromhex(h) for uri, h in case["blobs"].items()}
    found = series_payload_violations(case["payload"], case["parents"], blobs)
    assert sorted(found) == case["expect"]["codes"]


def test_series_vector_counts() -> None:
    accepts = [c for c in CASES if c["expect"]["result"] == "accept"]
    assert (len(accepts), len(CASES) - len(accepts)) == (4, 5)


def _trajectory(length: int) -> g.TrajectoryInput:
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])[:length]
    return g.TrajectoryInput(
        "S1", "T-hid", points, tuple(range(length)), POLICY, "m1", layer=12
    )


@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("delta", [None, 0.5])
def test_msr_output_has_the_shape_of_the_accepted_vectors(
    length: int, delta: float | None
) -> None:
    em = g.emit_single(_trajectory(length), ctx(), trajectory_ref=ref(50), delta=delta)
    payload = em.envelope["payload"]
    twin = next(
        c
        for c in CASES
        if c["id"] == f"accept-T{length}-{'delta' if delta else 'no-delta'}"
    )
    # same metrics, same series refs, same Blob bytes as the contract's vector
    assert payload["metrics"] == twin["payload"]["metrics"]
    assert payload["series"] == twin["payload"]["series"]
    assert {b.uri: b.data.hex() for b in em.blobs} == twin["blobs"]
    disp = payload["metrics"]["displacement"]
    if length == 1:  # no displacement interval: count 0, the rest has no value
        assert disp == {
            "count": 0,
            "mean": None,
            "median": None,
            "longest_sub_delta_run": None,
        }
    # S35-S37 hold on the real output
    blobs = {b.uri: b.data for b in em.blobs}
    assert violations(em.envelope, blobs, {}) == set()
