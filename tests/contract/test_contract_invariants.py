"""Semantic invariants S3, S13, S17-S21, S23, S24: they hold on MSR's output,
and each one is actually detected when the output is tampered with."""

from __future__ import annotations

from typing import Any

import pytest
from contract_support import APPROVAL, ctx, deepcopy, muts, nulls, ref, traj
from invariants import violations

import msr.geometry as g


@pytest.fixture(scope="module")
def world() -> dict[str, Any]:
    cal = g.build_calibration(nulls(), ctx(2))
    cb_cal = g.build_calibration(nulls(), ctx(2), codebook_ref=ref(70))
    single = g.emit_single(
        traj(9), ctx(3), trajectory_ref=ref(50), calibration=cal, epsilon=1.5
    )
    pair = g.emit_pair(
        traj(8), traj(9), ctx(3), left_ref=ref(60), right_ref=ref(61), calibration=cal
    )
    approved = ctx(3, alignment_approval=APPROVAL)
    cp_cal = g.build_calibration(
        [g.NullMember(m.trajectory, m.ref) for m in nulls()],
        ctx(2, alignment_approval=APPROVAL),
        alignment="common_prefix",
    )
    cp_pair = g.emit_pair(
        traj(8, n=40),
        traj(9, n=25),
        approved,
        left_ref=ref(60),
        right_ref=ref(61),
        calibration=cp_cal,
        alignment="common_prefix",
    )
    stored = {
        cal.ref.artifact_id: {"envelope": cal.envelope},
        cp_cal.ref.artifact_id: {"envelope": cp_cal.envelope},
        ref(60).artifact_id: {"space_id": "S1"},
        ref(61).artifact_id: {"space_id": "S1"},
    }
    return {
        "cal": cal,
        "cb_cal": cb_cal,
        "single": single,
        "pair": pair,
        "cp_cal": cp_cal,
        "cp_pair": cp_pair,
        "stored": stored,
        "blobs": {b.uri: b.data for b in single.blobs},
    }


def check(env: dict[str, Any], w: dict[str, Any]) -> set[str]:
    return violations(env, w["blobs"], w["stored"])


@pytest.mark.parametrize("key", ["single", "pair", "cp_pair"])
def test_emissions_satisfy_the_invariants(world: dict[str, Any], key: str) -> None:
    assert check(world[key].envelope, world) == set()


@pytest.mark.parametrize("key", ["cal", "cb_cal", "cp_cal"])
def test_calibrations_satisfy_the_invariants(world: dict[str, Any], key: str) -> None:
    assert check(world[key].envelope, world) == set()


def test_s3_single_has_no_four_valued_verdict(world: dict[str, Any]) -> None:
    env = deepcopy(world["single"].envelope)
    env["payload"]["verdict"]["state"] = "STAGNANT"
    assert check(env, world) == {"S3"}


def test_s13_space_and_alignment(world: dict[str, Any]) -> None:
    env = deepcopy(world["single"].envelope)
    env["payload"]["space_id"] = "S2"
    assert "S13" in check(env, world)
    pair = deepcopy(world["pair"].envelope)
    stored = deepcopy(world["stored"])
    cal_id = pair["calibration_ref"]["artifact_id"]
    stored[cal_id]["envelope"]["payload"]["between_alignment"] = "x-common_prefix"
    assert "S13" in violations(pair, world["blobs"], stored)


def test_s17_extension_required_for_common_prefix(world: dict[str, Any]) -> None:
    env = deepcopy(world["cp_pair"].envelope)
    del env["extensions"]["x-meaning-space-runtime-alignment"]
    assert "S17" in check(env, world)
    env["extensions"]["x-meaning-space-runtime-alignment"] = {
        "reason": "",
        "approved_by": "t",
    }
    assert "S17" in check(env, world)
    cal = deepcopy(world["cp_cal"].envelope)
    cal["extensions"] = {}
    assert "S17" in check(cal, world)


def test_s18_pair_inputs(world: dict[str, Any]) -> None:
    base = world["pair"].envelope
    for mutate in muts(
        lambda e: e["payload"]["inputs"]["left"].__setitem__("model_build_id", "x"),
        lambda e: e["payload"]["inputs"]["right"].__setitem__("dimension", 9),
        lambda e: e["payload"]["inputs"]["right"].__setitem__("length", 41),
        lambda e: e["payload"]["alignment"].__setitem__("points_used", 39),
        lambda e: e["parents"][1].__setitem__("artifact_ids", []),
    ):
        env = deepcopy(base)
        mutate(env)
        assert "S18" in check(env, world)
    env = deepcopy(world["cp_pair"].envelope)
    env["payload"]["alignment"]["points_used"] = 40
    assert "S18" in check(env, world)
    stored = deepcopy(world["stored"])
    stored[ref(61).artifact_id]["space_id"] = "S9"
    assert "S18" in violations(base, world["blobs"], stored)


def test_s19_calibration(world: dict[str, Any]) -> None:
    base = world["cal"].envelope
    for mutate in muts(
        lambda e: e["payload"]["values"].__setitem__("noise_floor_lower", 2**50),
        lambda e: e["payload"]["method"]["quantiles"].__setitem__("floor", [9, 9]),
        lambda e: e["payload"]["null_set"].pop(),
        lambda e: e["payload"].__setitem__("null_set", e["payload"]["null_set"] * 2),
        lambda e: e["payload"]["sample_config"].__setitem__("trajectories", 9),
        lambda e: e["payload"]["null_set"].reverse(),
        lambda e: e["parents"].clear(),
    ):
        env = deepcopy(base)
        mutate(env)
        assert "S19" in check(env, world)


def test_s20_calibration_reference_agrees(world: dict[str, Any]) -> None:
    base = world["single"].envelope
    for mutate in muts(
        lambda e: e["payload"]["calibration"].__setitem__(
            "content_id", "cnt:" + "0" * 64
        ),
        lambda e: e.__setitem__("calibration_ref", None),
        lambda e: e["parents"].pop(0),
    ):
        env = deepcopy(base)
        mutate(env)
        assert "S20" in check(env, world)
    bare = g.emit_single(traj(9), ctx(), trajectory_ref=ref(50))
    blobs = {b.uri: b.data for b in bare.blobs}
    assert violations(bare.envelope, blobs, world["stored"]) == set()
    odd = deepcopy(bare.envelope)
    odd["parents"].append(
        {"role": "calibration", "ordered": False, "artifact_ids": [ref(1).artifact_id]}
    )
    assert "S20" in violations(odd, blobs, world["stored"])


def test_s21_codebook_reference_agrees(world: dict[str, Any]) -> None:
    base = world["cb_cal"].envelope
    env = deepcopy(base)
    env["payload"]["values"]["codebook_ref"] = ref(71).to_dict()
    assert "S21" in check(env, world)
    env = deepcopy(base)
    env["parents"] = [p for p in env["parents"] if p["role"] != "codebook"]
    assert "S21" in check(env, world)
    env = deepcopy(world["cal"].envelope)
    env["parents"].append(
        {"role": "codebook", "ordered": False, "artifact_ids": [ref(70).artifact_id]}
    )
    assert "S21" in check(env, world)
    env = deepcopy(world["cal"].envelope)
    env["calibration_ref"] = ref(5).to_dict()
    assert "calibration_self_reference" in check(env, world)


def test_s23_thresholds_follow_the_calibration(world: dict[str, Any]) -> None:
    env = world["single"].envelope
    src = env["payload"]["thresholds"]["source"]
    assert src["epsilon"] == "explicit" and src["delta"] == "calibration"
    assert check(env, world) == set()
    mutated = deepcopy(env)
    mutated["payload"]["thresholds"]["delta"] += 1
    assert "S23" in check(mutated, world)
    mutated = deepcopy(env)
    mutated["payload"]["thresholds"]["w_min"] += 1
    assert "S23" in check(mutated, world)
    mutated = deepcopy(env)
    mutated["payload"]["thresholds"]["epsilon"] += 1  # explicit: free to differ
    assert check(mutated, world) == set()


def test_s24_blob_hash_and_length(world: dict[str, Any]) -> None:
    env, uri = world["single"].envelope, world["single"].blobs[0].uri
    assert check(env, world) == set()
    for data in (
        world["blobs"][uri][:-8],
        world["blobs"][uri] + b"\0" * 8,
        b"x" * len(world["blobs"][uri]),
    ):
        assert "S24" in violations(env, {uri: data}, world["stored"])
    assert "S24" in violations(env, {}, world["stored"])
    fl = deepcopy(env)
    fl["payload"]["series"]["dtype"] = "float64"  # S24 applies to int Blobs only
    assert "S24" not in violations(fl, {}, world["stored"])
