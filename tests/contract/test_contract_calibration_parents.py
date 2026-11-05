"""The reader's parent-reference checks against the contract's five points
(vendored ``msr/README.md`` and ``msr/vectors/calibration-parents.json``)."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from contract_support import VENDOR

import msr.geometry as g
import msr.geometry.calibration as calibration_module

CASES: list[dict[str, Any]] = json.loads(
    (VENDOR / "msr/vectors/calibration-parents.json").read_text("utf-8")
)["cases"]


def _is_valid(case: dict[str, Any]) -> bool:
    try:
        calibration_module._check_contract(case["envelope"], case["parent_contents"])
    except g.CalibrationError:
        return False
    return True


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reader_agrees_with_the_vendored_parent_vectors(case: dict[str, Any]) -> None:
    assert _is_valid(case) == (case["expect"]["result"] == "accept")


def test_vector_set_has_the_two_accepts_and_a_reject_for_each_point() -> None:
    accepts = [c for c in CASES if c["expect"]["result"] == "accept"]
    assert (len(accepts), len(CASES) - len(accepts)) == (2, 10)
    for point in ("point1", "point2", "point3", "point4", "point5"):
        assert any(c["id"].startswith(point) for c in CASES), point


def test_readme_states_the_five_points_and_the_division_of_labour() -> None:
    text = (VENDOR / "msr/README.md").read_text("utf-8")
    section = text.split("## 親の参照整合", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| (\d) \|", section, re.M)
    assert rows == ["1", "2", "3", "4", "5"]
    for needle in ("検証失敗", "nvs-runtime", "Gate", "MSR は HEKB を呼ばない"):
        assert needle in section
    assert "nvs-runtime fetched" in (g.load_calibration.__doc__ or "")


def test_every_parent_contents_entry_must_have_a_cnt_content_id() -> None:
    """Contract point 1: not only the parents the envelope names."""
    case = next(c for c in CASES if c["id"] == "accept-null-set-only")
    for unrelated in ("not-a-content-id", "cnt:short", "", 5):
        contents = {**case["parent_contents"], "art:" + "e" * 64: unrelated}
        with pytest.raises(g.CalibrationError):
            calibration_module._check_contract(case["envelope"], contents)
    ok = {**case["parent_contents"], "art:" + "e" * 64: "cnt:" + "e" * 64}
    calibration_module._check_contract(
        case["envelope"], ok
    )  # unrelated but well-formed
