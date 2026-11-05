"""MSR's output envelopes pass the vendored schemas and the wire rules."""

from __future__ import annotations

import json

import numpy as np
import pytest
from contract_support import (
    ALIGN_EXT,
    APPROVAL,
    ctx,
    decode_blob,
    envelope_errors,
    ref,
    traj,
)
from contract_support import calibration as make_calibration

import msr.geometry as g
from msr.geometry import ids
from msr.geometry.analyze import VERDICT_REASON
from msr.geometry.emit import SERIES_ORDER, UNDEFINED
from msr.geometry.wire import ContractError, jcs, strict_load


def _no_float(token: str) -> object:
    raise AssertionError(f"float token {token!r} in output")


def assert_strict_integer_json(emission: g.Emission) -> None:
    data = emission.to_bytes()
    assert strict_load(data) == emission.envelope  # own strict reader accepts it
    json.loads(data, parse_float=_no_float, parse_constant=_no_float)
    assert jcs(strict_load(data)) == data  # already canonical


def assert_ids_recompute(emission: g.Emission, parent_contents: dict[str, str]) -> None:
    env = emission.envelope
    assert env["content_id"] == ids.content_id(env, parent_contents)
    assert env["artifact_id"] == ids.artifact_id(env, parent_contents)


def parent_map(*refs: g.ArtifactRef) -> dict[str, str]:
    return {r.artifact_id: r.content_id for r in refs}


def test_single_with_calibration_and_codebook() -> None:
    cal = make_calibration()
    codebook = np.random.default_rng(3).normal(size=(5, 4))
    em = g.emit_single(
        traj(9),
        ctx(11),
        trajectory_ref=ref(50),
        calibration=cal,
        codebook=codebook,
        codebook_ref=ref(51),
    )
    assert envelope_errors(em.envelope) == []
    assert_strict_integer_json(em)
    assert_ids_recompute(em, parent_map(ref(50), ref(51), cal.ref))
    payload = em.envelope["payload"]
    assert payload["mode"] == "single"
    assert payload["calibration"] == cal.ref.to_dict() == em.envelope["calibration_ref"]
    assert payload["verdict"] == {"state": None, "reason": VERDICT_REASON}
    assert payload["thresholds"]["source"] == {
        "epsilon": "calibration",
        "delta": "calibration",
        "w_min": "calibration",
    }
    assert [p["role"] for p in em.envelope["parents"]] == [
        "calibration",
        "codebook",
        "trajectory",
    ]
    assert payload["metrics"]["diversity"]["cell_entropy"] is not None


def test_single_without_calibration_uses_explicit_thresholds_quantized() -> None:
    t = traj(9)
    em = g.emit_single(
        t, ctx(), trajectory_ref=ref(50), epsilon=1.3, delta=0.2, w_min=0
    )
    assert envelope_errors(em.envelope) == []
    th = em.envelope["payload"]["thresholds"]
    assert th["epsilon"] == round(1.3 * 2**30) and th["delta"] == round(0.2 * 2**30)
    assert th["source"] == {
        "epsilon": "explicit",
        "delta": "explicit",
        "w_min": "explicit",
    }
    assert em.envelope["calibration_ref"] is None
    # the numbers were computed with the quantized thresholds
    same = g.emit_single(
        t,
        ctx(),
        trajectory_ref=ref(50),
        epsilon=th["epsilon"] / 2**30,
        delta=th["delta"] / 2**30,
        w_min=0,
    )
    assert same.envelope["content_id"] == em.envelope["content_id"]
    bare = g.emit_single(t, ctx(), trajectory_ref=ref(50))
    assert envelope_errors(bare.envelope) == []
    m = bare.envelope["payload"]["metrics"]
    assert m["recurrence"] == {
        "pair_fraction": None,
        "return_count": None,
        "return_rate": None,
    }
    assert bare.envelope["payload"]["thresholds"]["w_min"] is None


def test_series_blob_is_the_canonical_int64_array() -> None:
    t = traj(9, n=30)
    em = g.emit_single(
        t, ctx(), trajectory_ref=ref(50), epsilon=2.0, delta=0.5, w_min=1
    )
    blob = em.blobs[0]
    s = em.envelope["payload"]["series"]
    assert (s["media_type"], s["dtype"], s["byte_order"], s["order"]) == (
        "application/vnd.sensos.ndarray-raw",
        "int64",
        "little",
        "C",
    )
    assert s["shape"] == [len(SERIES_ORDER), 30] == list(blob.shape)
    assert s["sha256"] == blob.sha256 and s["uri"] == blob.uri
    assert len(blob.data) == len(SERIES_ORDER) * 30 * 8  # no header
    rows = decode_blob(blob.data, (len(SERIES_ORDER), 30))
    ref_series = g.analyze_trajectory(t, epsilon=2.0, delta=0.5, w_min=1)["series"]
    for k, name in enumerate(SERIES_ORDER):
        expect = [
            UNDEFINED if v is None else round(v * 2**30) for v in ref_series[name]
        ]
        assert rows[k].tolist() == expect, name
    assert rows[0, 0] == UNDEFINED  # displacement_mean has no value at t=0


def test_pair_with_and_without_calibration() -> None:
    cal = make_calibration()
    left, right = traj(8), traj(9)
    refs = parent_map(ref(60), ref(61), cal.ref)
    em = g.emit_pair(
        left, right, ctx(13), left_ref=ref(60), right_ref=ref(61), calibration=cal
    )
    assert envelope_errors(em.envelope) == []
    assert_strict_integer_json(em)
    assert_ids_recompute(em, refs)
    p = em.envelope["payload"]
    assert p["mode"] == "pair" and p["alignment"] == {
        "mode": "equal_only",
        "points_used": 40,
    }
    assert p["distance"]["definition"] == "mean_step_l2"
    assert [r["role"] for r in em.envelope["parents"]] == [
        "calibration",
        "left",
        "right",
    ]
    assert em.blobs == ()
    bare = g.emit_pair(left, right, ctx(), left_ref=ref(60), right_ref=ref(61))
    assert envelope_errors(bare.envelope) == []
    assert bare.envelope["payload"]["verdict"] == {"state": "UNCALIBRATED"}
    assert bare.envelope["extensions"] == {}


def test_pair_verdict_is_decided_on_the_integers() -> None:
    cal = make_calibration()
    upper = cal.values["noise_floor_upper"]
    lower = cal.values["noise_floor_lower"]
    for q, state in (
        (lower - 1, "STAGNANT"),
        (lower, "NOISE_FLOOR_INTERNAL"),
        (upper, "NOISE_FLOOR_INTERNAL"),
        (upper + 1, "STRUCTURAL_SHIFT_DETECTED"),
    ):
        assert g.evaluate_shift_quantized(q, cal) == state
    assert g.evaluate_shift_quantized(5, None) == "UNCALIBRATED"
    far = g.emit_pair(
        traj(8),
        g.TrajectoryInput(
            "S1",
            "T-hid",
            traj(8).points + 1e3,
            tuple(range(40)),
            traj(8).policy,
            "build-1",
            layer=3,
        ),
        ctx(),
        left_ref=ref(60),
        right_ref=ref(61),
        calibration=cal,
    )
    assert far.envelope["payload"]["verdict"]["state"] == "STRUCTURAL_SHIFT_DETECTED"


def test_common_prefix_needs_an_approval_and_is_spelled_x_common_prefix() -> None:
    left, right = traj(8, n=40), traj(9, n=25)
    with pytest.raises(g.InvalidTrajectory):  # equal_only refuses unequal lengths
        g.emit_pair(left, right, ctx(), left_ref=ref(60), right_ref=ref(61))
    with pytest.raises(ContractError) as info:
        g.emit_pair(
            left,
            right,
            ctx(),
            left_ref=ref(60),
            right_ref=ref(61),
            alignment="common_prefix",
        )
    assert info.value.code == "S17"
    em = g.emit_pair(
        left,
        right,
        ctx(alignment_approval=APPROVAL),
        left_ref=ref(60),
        right_ref=ref(61),
        alignment="common_prefix",
    )
    assert envelope_errors(em.envelope) == []
    assert em.envelope["payload"]["alignment"] == {
        "mode": "x-common_prefix",
        "points_used": 25,
    }
    assert em.envelope["extensions"] == {ALIGN_EXT: APPROVAL}
    for bad in (
        {"reason": "", "approved_by": "tomo"},
        {"reason": "x", "approved_by": ""},
        {"reason": "x"},
        {"approved_by": "tomo"},
    ):
        with pytest.raises(ContractError):
            g.emit_pair(
                left,
                right,
                ctx(alignment_approval=bad),
                left_ref=ref(60),
                right_ref=ref(61),
                alignment="common_prefix",
            )


def test_pair_refusals() -> None:
    a = traj(8)
    with pytest.raises(g.CrossSpaceComparison):
        g.emit_pair(a, traj(9, space="S2"), ctx(), left_ref=ref(60), right_ref=ref(61))
    other_build = g.TrajectoryInput(
        "S1", "T-hid", a.points, a.token_index, a.policy, "build-2", layer=3
    )
    with pytest.raises(g.InvalidTrajectory):
        g.emit_pair(a, other_build, ctx(), left_ref=ref(60), right_ref=ref(61))
    cal = make_calibration()
    with pytest.raises(g.CalibrationError):  # S13: calibration for another space
        g.emit_pair(
            traj(8, space="S2"),
            traj(9, space="S2"),
            ctx(),
            left_ref=ref(60),
            right_ref=ref(61),
            calibration=cal,
        )
    approved = ctx(alignment_approval=APPROVAL)
    with pytest.raises(g.CalibrationError):  # S13: alignment differs
        g.emit_pair(
            traj(8, n=40),
            traj(9, n=25),
            approved,
            left_ref=ref(60),
            right_ref=ref(61),
            calibration=cal,
            alignment="common_prefix",
        )


def test_single_refusals() -> None:
    cb = np.zeros((3, 4))
    with pytest.raises(ContractError):  # codebook and its ref come together
        g.emit_single(traj(9), ctx(), trajectory_ref=ref(50), codebook=cb)
    with pytest.raises(ContractError):
        g.emit_single(traj(9), ctx(), trajectory_ref=ref(50), codebook_ref=ref(51))
    with pytest.raises(ContractError):
        g.emit_single(traj(9), ctx(), trajectory_ref=ref(50), w_min=-1)
    with pytest.raises(ContractError):
        g.emit_single(traj(9), ctx(), trajectory_ref=ref(50), w_min=True)
    cal = g.build_calibration(
        [
            g.NullMember(t.trajectory, t.ref)
            for t in __import__("contract_support").nulls()
        ],
        ctx(),
        codebook_ref=ref(70),
    )
    with pytest.raises(g.CalibrationError):  # a different codebook than calibrated
        g.emit_single(
            traj(9),
            ctx(),
            trajectory_ref=ref(50),
            calibration=cal,
            codebook=cb,
            codebook_ref=ref(71),
        )
    ok = g.emit_single(
        traj(9),
        ctx(),
        trajectory_ref=ref(50),
        calibration=cal,
        codebook=np.random.default_rng(1).normal(size=(3, 4)),
        codebook_ref=ref(70),
    )
    assert envelope_errors(ok.envelope) == []
    with pytest.raises(g.CalibrationError):  # trajectory of another space
        g.emit_single(
            traj(9, space="S2"), ctx(), trajectory_ref=ref(50), calibration=cal
        )


def test_no_meaning_words_in_the_output() -> None:
    cal = make_calibration()
    em = g.emit_single(traj(9), ctx(), trajectory_ref=ref(50), calibration=cal)
    text = em.to_bytes().decode().lower()
    for word in ("loop", "anomaly", "hallucination", '"p"', '"l"', '"u"'):
        assert word not in text
    assert em.envelope["kind"] == "interpretation"
    assert em.envelope["producer"]["name"] == "meaning-space-runtime"
    assert em.envelope["observed_at"] is None  # MSR creates no observation


def test_envelope_carries_exactly_what_the_caller_supplied() -> None:
    c = ctx(
        42,
        observed_at={"value": "2026-11-06", "precision": "day", "tz": "unknown"},
        event_span=(3, 9),
        policy_refs=(
            g.PolicyRef("z-policy", "b" * 64),
            g.PolicyRef("a-policy", "c" * 64),
        ),
    )
    em = g.emit_pair(traj(8), traj(9), c, left_ref=ref(60), right_ref=ref(61))
    env = em.envelope
    assert envelope_errors(env) == []
    assert (env["session_id"], env["observation_id"], env["seq"]) == (
        c.session_id,
        c.observation_id,
        42,
    )
    assert env["event_span"] == {"token_start": 3, "token_end": 9}
    assert [p["name"] for p in env["policy_refs"]] == ["a-policy", "z-policy"]
    assert env["observed_at"]["precision"] == "day"
    assert env["producer"]["build"] == "test-build"
    default = g.EmitContext("ses:" + "1" * 64, None, 0)
    assert default.producer_build
    stamped = g.emit_pair(
        traj(8), traj(9), default, left_ref=ref(60), right_ref=ref(61)
    )
    assert envelope_errors(stamped.envelope) == []
    assert stamped.envelope["observation_id"] is None
