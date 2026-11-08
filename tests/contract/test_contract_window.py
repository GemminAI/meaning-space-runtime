"""IC-02 0.9.0 (D-129): MSR reads a first-128-rows window artifact and measures it.

The vendored window vectors (``vendor_window/``, pinned by ``window.lock``) are the contract's own
cases. MSR's reader must agree with every one it can decide on its own; the cases only the Gate can decide
(the stored original says something else, ``window_origin*``) are accepted by the reader and are listed
here, so a silent widening of what the reader accepts shows up as a failure.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest
from contract_support import (
    HERE,
    VENDOR_WINDOW,
    ctx,
    envelope_errors,
    geometry_v090_validator,
    old_geometry_validator,
    ref,
    traj,
    window_validator,
)

import msr.geometry as g

LOCK: dict[str, Any] = json.loads((HERE / "window.lock").read_text("utf-8"))
VEC: dict[str, Any] = json.loads(
    (VENDOR_WINDOW / "msr/archive/0.9.0/vectors/window.json").read_text("utf-8")
)
BLOBS = {u: bytes.fromhex(h) for u, h in VEC["blobs"].items()}
CASES = {c["id"]: c for c in VEC["cases"]}
WINDOWS = [c for c in VEC["cases"] if c["target"] == "window"]
MEASUREMENTS = [c for c in VEC["cases"] if c["target"] == "measurement"]
# Codes the reader cannot reach: it does not hold the stored original (the Gate does).
GATE_ONLY = {"window_origin", "window_origin_unreadable"}


def _parent_contents(case: dict[str, Any]) -> dict[str, str]:
    return {a: e["content_id"] for a, e in case["lookup"].items()}


def _read(case: dict[str, Any]) -> g.TrajectoryWindow:
    return g.window_from_envelope(
        case["envelope"], BLOBS, parent_contents=_parent_contents(case)
    )


def _good() -> g.TrajectoryWindow:
    return _read(CASES["accept-window-first-128"])


# ---------------------------------------------------------------- the vendored window files


def test_window_lock_matches_the_vendored_files_and_the_commit_of_the_main_lock() -> (
    None
):
    main = json.loads((HERE / "contract.lock").read_text("utf-8"))
    assert LOCK["contract_id"] == "IC-02" and LOCK["version"] == "0.10.0"
    assert (
        LOCK["source"]["commit"] == main["source"]["commit"]
        and LOCK["source"]["tag"] is None
    )
    on_disk = {
        str(p.relative_to(VENDOR_WINDOW))
        for p in VENDOR_WINDOW.rglob("*")
        if p.is_file()
    }
    assert on_disk == {f["path"] for f in LOCK["files"]}
    for entry in LOCK["files"]:
        assert (
            hashlib.sha256((VENDOR_WINDOW / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        ), entry["path"]
    assert (
        json.loads(
            (
                VENDOR_WINDOW / "msr/archive/0.9.0/trajectory-window.schema.json"
            ).read_text("utf-8")
        )["$id"]
        == "sensos:contracts/msr/trajectory-window/0.9.0"
    )


def test_vendored_old_schema_is_the_ic02_0_8_0_one() -> None:
    old = json.loads(
        (
            VENDOR_WINDOW / "msr/archive/0.8.0/geometry-measurement.schema.json"
        ).read_text("utf-8")
    )
    assert old["$id"] == "sensos:contracts/msr/geometry-measurement/0.7.1"
    assert "input_scope" not in json.dumps(old)


def test_vendored_vectors_have_the_expected_size() -> None:
    accepts = [c for c in VEC["cases"] if c["expect"]["result"] == "accept"]
    assert (len(VEC["cases"]), len(accepts), len(WINDOWS), len(MEASUREMENTS)) == (
        47,
        6,
        30,
        17,
    )


# ---------------------------------------------------------------- the reader against the window vectors


@pytest.mark.parametrize("case", WINDOWS, ids=[c["id"] for c in WINDOWS])
def test_reader_agrees_with_the_window_vectors(case: dict[str, Any]) -> None:
    expected = case["expect"]
    refused_by_reader = bool(expected["ic00"]) or bool(
        set(expected["payload"]) - GATE_ONLY
    )
    if refused_by_reader:
        with pytest.raises(g.InvalidWindow):
            _read(case)
    else:
        window = _read(case)
        assert window.trajectory.length == 128 and window.scope["length"] == 128


def test_cases_only_the_gate_can_refuse_are_exactly_these() -> None:
    only_gate = sorted(
        c["id"]
        for c in WINDOWS
        if c["expect"]["result"] == "reject"
        and not c["expect"]["ic00"]
        and set(c["expect"]["payload"]) <= GATE_ONLY
    )
    assert only_gate == [
        "reject-window-origin-length-differs-from-stored",
        "reject-window-origin-position-differs-from-stored",
        "reject-window-origin-trajectory-id-differs-from-stored",
        "reject-window-stored-parent-unreadable",
    ]


def test_a_window_without_its_blobs_or_with_foreign_bytes_is_refused() -> None:
    case = CASES["accept-window-first-128"]
    pc = _parent_contents(case)
    uri = case["envelope"]["payload"]["points"]["uri"]
    with pytest.raises(g.InvalidWindow):
        g.window_from_envelope(
            case["envelope"],
            {u: d for u, d in BLOBS.items() if u != uri},
            parent_contents=pc,
        )
    with pytest.raises(g.InvalidWindow):
        g.window_from_envelope(
            case["envelope"], {**BLOBS, uri: bytes(len(BLOBS[uri]))}, parent_contents=pc
        )
    with pytest.raises(g.InvalidWindow):  # the source trajectory was not fetched
        g.window_from_envelope(case["envelope"], BLOBS, parent_contents={})
    with pytest.raises(
        g.InvalidWindow
    ):  # a tampered payload no longer recomputes its ids
        env = json.loads(json.dumps(case["envelope"]))
        env["payload"]["origin"]["position"] = "p9"
        g.window_from_envelope(env, BLOBS, parent_contents=pc)


def test_a_measurement_envelope_is_not_a_window() -> None:
    case = CASES["accept-single-window-scope"]
    with pytest.raises(g.InvalidWindow):
        g.window_from_envelope(
            case["envelope"], BLOBS, parent_contents=_parent_contents(case)
        )


def test_the_window_reads_as_the_first_128_rows_it_holds() -> None:
    window = _good()
    payload = CASES["accept-window-first-128"]["envelope"]["payload"]
    rows = np.frombuffer(BLOBS[payload["points"]["uri"]], "<f4").reshape(
        128, payload["dimension"]
    )
    assert np.array_equal(window.trajectory.points, rows.astype(np.float64))
    assert window.trajectory.token_index == tuple(range(128))
    assert (
        window.trajectory.space_id
        == CASES["accept-window-first-128"]["envelope"]["space_id"]
    )
    assert (
        window.ref.artifact_id
        == CASES["accept-window-first-128"]["envelope"]["artifact_id"]
    )
    assert (
        window.scope
        == CASES["accept-single-window-scope"]["envelope"]["payload"]["input_scope"]
    )
    assert g.prefix_matches(window, rows.tobytes() + bytes(40))
    assert not g.prefix_matches(window, bytes(rows.nbytes))
    assert not g.prefix_matches(window, rows.tobytes()[:-4])


# ---------------------------------------------------------------- emit_single with a window / without one


def test_single_of_a_window_declares_the_window_and_validates_against_the_vendored_schema() -> (
    None
):
    window = _good()
    em = g.emit_single(
        window.trajectory, ctx(), trajectory_ref=window.ref, window=window
    )
    payload = em.envelope["payload"]
    assert em.envelope["payload_schema"] == {
        "id": "msr.geometry-measurement",
        "version": "0.9.0",
    }
    assert payload["input_scope"] == dict(window.scope)
    assert payload["input_ref"]["length"] == 128
    assert [p for p in em.envelope["parents"] if p["role"] == "trajectory"] == [
        {
            "role": "trajectory",
            "ordered": False,
            "artifact_ids": [window.ref.artifact_id],
        }
    ]
    assert envelope_errors(em.envelope) == []
    # the window's own series are those of a plain measurement of the same rows (the scope changes nothing numerically)
    plain = g.emit_single(window.trajectory, ctx(), trajectory_ref=window.ref)
    assert {
        k: v for k, v in plain.envelope["payload"].items() if k != "input_scope"
    } == {k: v for k, v in payload.items() if k != "input_scope"}


def test_single_without_a_window_is_a_full_input() -> None:
    full = g.emit_single(traj(7, n=200), ctx(), trajectory_ref=ref(50))
    assert full.envelope["payload"]["input_scope"] == {"kind": "full"}
    assert (
        full.envelope["payload"]["input_ref"]["length"] == 200
    )  # the whole trajectory, never cut to 128
    assert envelope_errors(full.envelope) == []


def test_emit_refuses_a_window_with_another_trajectory_or_reference() -> None:
    window = _good()
    with pytest.raises(g.InvalidTrajectory):
        g.emit_single(
            traj(3, n=128, d=window.trajectory.dimension),
            ctx(),
            trajectory_ref=window.ref,
            window=window,
        )
    with pytest.raises(g.InvalidTrajectory):
        g.emit_single(window.trajectory, ctx(), trajectory_ref=ref(51), window=window)


def test_no_emission_is_the_old_rule() -> None:
    em = g.emit_single(traj(1), ctx(), trajectory_ref=ref(50))
    assert em.envelope["payload_schema"]["version"] == "0.9.0"
    pair = g.emit_pair(traj(1), traj(2), ctx(), left_ref=ref(50), right_ref=ref(51))
    assert (
        pair.envelope["payload_schema"]["version"] == "0.9.0"
        and "input_scope" not in pair.envelope["payload"]
    )
    assert envelope_errors(pair.envelope) == []


def test_an_old_0_7_1_payload_is_still_read_by_the_old_rule_only() -> None:
    old_case = CASES["accept-single-0.7.1-old-rule-no-scope"]
    assert (
        list(old_geometry_validator().iter_errors(old_case["envelope"]["payload"]))
        == []
    )
    assert list(
        geometry_v090_validator().iter_errors(old_case["envelope"]["payload"])
    )  # not read as 0.9.0


# ---------------------------------------------------------------- the vendored schemas against the vectors


@pytest.mark.parametrize("case", WINDOWS, ids=[c["id"] for c in WINDOWS])
def test_window_schema_agrees_with_the_vectors(case: dict[str, Any]) -> None:
    errors = list(window_validator().iter_errors(case["envelope"]["payload"]))
    assert bool(errors) == ("schema" in case["expect"]["payload"])


@pytest.mark.parametrize("case", MEASUREMENTS, ids=[c["id"] for c in MEASUREMENTS])
def test_measurement_schema_agrees_with_the_vectors(case: dict[str, Any]) -> None:
    env = case["envelope"]
    schema = (
        old_geometry_validator()
        if env["payload_schema"]["version"] == "0.7.1"
        else geometry_v090_validator()
    )
    assert bool(list(schema.iter_errors(env["payload"]))) == (
        "schema" in case["expect"]["payload"]
    )


def test_a_malformed_envelope_is_refused_as_an_invalid_window() -> None:
    with pytest.raises(g.InvalidWindow):
        g.window_from_envelope({}, BLOBS, parent_contents={})
    with pytest.raises(g.InvalidWindow):
        g.window_from_envelope({"payload_schema": 1}, BLOBS, parent_contents={})
