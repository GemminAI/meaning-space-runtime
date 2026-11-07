"""IC-02 0.8.0: a calibration declares its quantization exponent; readers follow it.

Provisional decision (Tomo 11-07 15:29/15:30): ``exponent`` is a required integer in the
payload, not fixed at -30; the contract sets no range; a 0.7.x payload (no field) is never
read as 0.8.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from contract_support import (
    VENDOR,
    ctx,
    deepcopy,
    envelope_errors,
    nulls,
    parent_contents,
    ref,
    schema_errors,
    traj,
)

import msr.geometry as g
import msr.geometry.calibration as calibration_module

VALUE_KEYS = (
    "noise_floor_lower",
    "noise_floor_upper",
    "background_mean",
    "epsilon",
    "delta",
)


def _build(exponent: int | None) -> g.CalibrationArtifact:
    kw: dict[str, Any] = {} if exponent is None else {"exponent": exponent}
    return g.build_calibration(nulls(), ctx(2), **kw)


def test_default_declares_minus_30() -> None:
    art = _build(None)
    assert art.payload["exponent"] == -30 and art.exponent == -30
    assert art.envelope["payload_schema"]["version"] == "0.8.0"


def test_declared_minus_20_is_read_at_its_true_value() -> None:
    fine, coarse = _build(-30), _build(-20)
    assert coarse.payload["exponent"] == -20 and coarse.exponent == -20
    assert envelope_errors(coarse.envelope) == []
    # the integers are ~1024 times smaller, the real values are the same (to one quantum)
    assert coarse.values["epsilon"] * 1024 == pytest.approx(
        fine.values["epsilon"], abs=1024
    )
    assert coarse.calibration.epsilon == coarse.values["epsilon"] / 2**20
    for key in VALUE_KEYS:
        assert (
            abs(getattr(coarse.calibration, key) - getattr(fine.calibration, key))
            <= 2.0**-20
        )
    assert coarse.calibration.epsilon != pytest.approx(
        fine.calibration.epsilon / 1024
    )  # not 1024x off


def test_store_and_load_keep_the_declared_exponent(tmp_path: Path) -> None:
    for exponent in (-30, -20):
        art = _build(exponent)
        g.write_calibration(tmp_path, art)
        loaded = g.load_calibration(
            tmp_path, art.ref.artifact_id, parent_contents=parent_contents()
        )
        assert loaded.exponent == exponent and loaded.calibration == art.calibration
        assert loaded.ref == art.ref  # ids recompute


def test_the_exponent_is_part_of_the_identity() -> None:
    assert _build(-30).ref != _build(-20).ref


def _mutate(art: g.CalibrationArtifact, fn: Any) -> dict[str, Any]:
    env = deepcopy(art.envelope)
    fn(env["payload"])
    return env  # ids not recomputed: the checks below look at the payload only


@pytest.mark.parametrize(
    "fn",
    [
        lambda p: p.pop("exponent"),
        lambda p: p.update(exponent="-30"),
        lambda p: p.update(exponent=-20.5),
        lambda p: p.update(exponent=-20.0),
        lambda p: p.update(exponent=True),
        lambda p: p.update(exponent=None),
    ],
    ids=["missing", "string", "fraction", "integral-float", "bool", "null"],
)
def test_reader_refuses_a_missing_or_malformed_exponent(fn: Any) -> None:
    art = _build(-20)
    env = _mutate(art, fn)
    with pytest.raises(g.CalibrationError):
        calibration_module._check_contract(env, parent_contents())
    with pytest.raises(g.CalibrationError):
        g.calibration_from_envelope(env)
    assert (
        schema_errors("msr/calibration", env["payload"]) != []
        or env["payload"].get("exponent") == -20.0
    )


def test_a_0_7_x_payload_is_not_read_as_0_8_0() -> None:
    art = _build(-30)
    env = _mutate(art, lambda p: p.pop("exponent"))
    env["payload_schema"] = {"id": "msr.calibration", "version": "0.7.0"}
    with pytest.raises(g.CalibrationError, match=r"not an msr\.calibration payload"):
        g.calibration_from_envelope(env)
    with pytest.raises(g.CalibrationError):
        calibration_module._check_contract(env, parent_contents())


def test_an_absurd_exponent_is_a_refusal_not_a_crash() -> None:
    env = _mutate(_build(-30), lambda p: p.update(exponent=10**30))
    with pytest.raises(g.CalibrationError):
        g.calibration_from_envelope(env)


def test_no_range_is_applied_to_the_exponent_on_read() -> None:
    env = _mutate(_build(-30), lambda p: p.update(exponent=-31))
    assert g.calibration_from_envelope(env).exponent == -31


def test_quantiles_stay_on_the_probability_grid() -> None:
    q = _build(-20).payload["method"]["quantiles"]
    assert q["epsilon"] == round(0.05 * 2**30) and q["floor"] == [
        round(0.05 * 2**30),
        round(0.95 * 2**30),
    ]


def test_a_different_quantum_is_not_converted_into_a_measurement() -> None:
    coarse = _build(-20)
    with pytest.raises(g.CalibrationError, match="no conversion"):
        g.emit_pair(
            traj(8),
            traj(9),
            ctx(),
            left_ref=ref(50),
            right_ref=ref(51),
            calibration=coarse,
        )
    with pytest.raises(g.CalibrationError, match="no conversion"):
        g.emit_single(traj(8), ctx(), trajectory_ref=ref(50), calibration=coarse)
    fine = _build(-30)
    g.emit_single(
        traj(8), ctx(), trajectory_ref=ref(50), calibration=fine
    )  # unchanged for -30


def test_vendored_exponent_vectors_agree_with_the_vendored_schema() -> None:
    cases = json.loads(
        (VENDOR / "msr/vectors/calibration-exponent.json").read_text("utf-8")
    )["cases"]
    assert len(cases) == 11
    for case in cases:
        if (
            case["payload_schema"]["version"] != "0.8.0"
            or case["id"] == "reject-exponent-integral-float"
        ):
            continue  # 0.7.0 is not vendored; a float like -20.0 is the Gate's explicit check
        assert (schema_errors("msr/calibration", case["payload"]) == []) == (
            case["expect"]["result"] == "accept"
        ), case["id"]
