"""Calibration as an independent, immutable artifact."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from contract_support import (
    APPROVAL,
    ctx,
    deepcopy,
    envelope_errors,
    muts,
    nulls,
    parent_contents,
    ref,
    traj,
)
from contract_support import calibration as make_calibration

import msr.geometry as g
import msr.geometry.calibration as calibration_module
from msr.geometry.calibration import calibration_from_envelope, payload_alignment
from msr.geometry.envelope import write_blob
from msr.geometry.wire import ContractError, jcs


def test_calibration_envelope_validates_and_names_its_parents() -> None:
    art = make_calibration()
    assert envelope_errors(art.envelope) == []
    env = art.envelope
    assert env["kind"] == "interpretation" and env["calibration_ref"] is None
    assert env["payload_schema"] == {"id": "msr.calibration", "version": "0.7.0"}
    assert [p["role"] for p in env["parents"]] == ["null_set"]
    assert env["parents"][0]["artifact_ids"] == env["payload"]["null_set"]
    assert env["parents"][0]["ordered"] is False
    assert "measured_at" not in env["payload"]
    assert art.ref.artifact_id == env["artifact_id"]
    with_cb = make_calibration(codebook_ref=ref(70))
    assert envelope_errors(with_cb.envelope) == []
    assert [p["role"] for p in with_cb.envelope["parents"]] == ["codebook", "null_set"]
    assert with_cb.envelope["payload"]["values"]["codebook_ref"] == ref(70).to_dict()


def test_values_are_on_the_quantum_grid_and_levels_are_recorded() -> None:
    art = make_calibration()
    v, q = art.values, art.payload["method"]["quantiles"]
    c = art.calibration
    assert c.epsilon == v["epsilon"] / 2**30 and c.w_min == v["w_min"] == 1
    assert q["epsilon"] == round(0.05 * 2**30) and q["floor"] == [
        round(0.05 * 2**30),
        round(0.95 * 2**30),
    ]
    assert art.payload["between_alignment"] == "equal_only"
    assert art.payload["sample_config"] == {"trajectories": 4, "points": 40, "dims": 4}


def test_null_set_order_does_not_change_the_artifact() -> None:
    members = nulls()
    a = g.build_calibration(members, ctx())
    b = g.build_calibration(list(reversed(members)), ctx())
    assert a.envelope["content_id"] == b.envelope["content_id"]
    assert a.envelope["artifact_id"] == b.envelope["artifact_id"]


def test_equal_only_refuses_unequal_lengths() -> None:
    mixed = [g.NullMember(traj(i, n=30 + 5 * (i % 2)), ref(i)) for i in range(4)]
    with pytest.raises(g.CalibrationError, match="differ in length"):
        g.build_calibration(mixed, ctx())
    with pytest.raises(ContractError) as info:  # the extension needs an approval
        g.build_calibration(mixed, ctx(), alignment="common_prefix")
    assert info.value.code == "S17"
    art = g.build_calibration(
        mixed, ctx(alignment_approval=APPROVAL), alignment="common_prefix"
    )
    assert envelope_errors(art.envelope) == []
    assert art.payload["between_alignment"] == "x-common_prefix"
    assert art.payload["sample_config"]["points"] == 30
    assert art.envelope["extensions"]["x-meaning-space-runtime-alignment"] == APPROVAL


def test_build_input_errors() -> None:
    members = nulls()
    with pytest.raises(g.CalibrationError, match="distinct"):
        g.build_calibration([members[0], g.NullMember(traj(7), members[0].ref)], ctx())
    for kw in (
        {"epsilon_quantile": 0.0},
        {"delta_quantile": 1.0},
        {"floor_quantiles": (0.9, 0.1)},
        {"floor_quantiles": (0.0, 0.5)},
    ):
        with pytest.raises(g.CalibrationError):
            g.build_calibration(members, ctx(), **kw)
    with pytest.raises(g.CalibrationError, match="unknown alignment"):
        g.build_calibration(members, ctx(), alignment="bogus")  # type: ignore[arg-type]
    with pytest.raises(g.CalibrationError, match="max_pooled_pairs"):
        g.build_calibration(members, ctx(), max_pooled_pairs=10)
    tiny = [g.NullMember(traj(i, n=1), ref(i)) for i in range(2)]
    with pytest.raises(g.CalibrationError, match="too short"):
        g.build_calibration(tiny, ctx())
    flat = [
        g.NullMember(
            g.TrajectoryInput(
                "S1",
                "T-hid",
                np.zeros((6, 2)),
                tuple(range(6)),
                traj(0).policy,
                "build-1",
                layer=3,
            ),
            ref(i),
        )
        for i in range(2)
    ]
    with pytest.raises(g.CalibrationError, match="smaller than one quantum"):
        g.build_calibration(flat, ctx())


def test_block_size_does_not_change_the_artifact() -> None:
    members = nulls(3, n=37)
    base = g.build_calibration(members, ctx(), block_size=1000)
    for block in (1, 5, 36, 37):
        again = g.build_calibration(members, ctx(), block_size=block)
        assert again.envelope["artifact_id"] == base.envelope["artifact_id"]


def test_store_load_roundtrip_by_artifact_id(tmp_path: Path) -> None:
    art = make_calibration()
    path = g.write_calibration(tmp_path, art)
    assert path == tmp_path / "calibration" / f"{art.ref.artifact_id}.json"
    assert path.read_bytes() == jcs(art.envelope)
    loaded = g.load_calibration(
        tmp_path, art.ref.artifact_id, parent_contents=parent_contents()
    )
    assert loaded.envelope == art.envelope and loaded.calibration == art.calibration
    assert loaded.file_sha256 and loaded.file_sha256 != art.file_sha256
    assert loaded.ref == art.ref
    assert envelope_errors(loaded.envelope) == []  # the vendored schema agrees


def test_a_stored_calibration_is_never_overwritten(tmp_path: Path) -> None:
    art = make_calibration()
    path = g.write_calibration(tmp_path, art)
    before = path.read_bytes()
    with pytest.raises(g.CalibrationError, match="immutable"):
        g.write_calibration(tmp_path, art)
    # rebuilt (seq / produced_at are not hashed): the very same artifact
    again = make_calibration(seq=9)
    assert again.ref == art.ref
    with pytest.raises(g.CalibrationError, match="immutable"):
        g.write_calibration(tmp_path, again)
    assert path.read_bytes() == before
    # another build of the same content: same content_id, new artifact, own file
    other_build = g.EmitContext(
        ctx().session_id, ctx().observation_id, 9, producer_build="build-2"
    )
    rebuilt = g.build_calibration(nulls(), other_build)
    assert rebuilt.envelope["content_id"] == art.envelope["content_id"]
    assert rebuilt.ref.artifact_id != art.ref.artifact_id
    assert g.write_calibration(tmp_path, rebuilt) != path
    assert path.read_bytes() == before


def test_load_authority_is_parse_shape_and_recomputed_ids(tmp_path: Path) -> None:
    art = make_calibration()
    aid = art.ref.artifact_id
    contents = parent_contents()
    with pytest.raises(g.CalibrationError, match="no calibration"):
        g.load_calibration(tmp_path, aid, parent_contents=contents)
    path = g.write_calibration(tmp_path, art)
    sidecar = path.with_name(path.name + ".sha256")
    good = path.read_bytes()
    # no .sha256 file: still accepted (it is a help, not the authority)
    sidecar.unlink()
    assert g.load_calibration(tmp_path, aid, parent_contents=contents).ref == art.ref
    # a .sha256 file that disagrees: refused
    sidecar.write_text("0" * 64 + "\n")
    with pytest.raises(g.CalibrationError, match=r"\.sha256"):
        g.load_calibration(tmp_path, aid, parent_contents=contents)
    sidecar.unlink()
    # tampered content with no .sha256 to notice: the ids no longer recompute
    path.write_bytes(good.replace(b'"build-1"', b'"build-9"', 1))
    with pytest.raises(g.CalibrationError, match="do not recompute"):
        g.load_calibration(tmp_path, aid, parent_contents=contents)
    path.write_bytes(good)
    # parents' content_ids are needed; a wrong one does not recompute either
    with pytest.raises(g.CalibrationError, match="does not match the contract"):
        g.load_calibration(tmp_path, aid, parent_contents={})  # parents unknown
    wrong = {k: ref(999).content_id for k in contents}
    with pytest.raises(g.CalibrationError, match="do not recompute"):
        g.load_calibration(tmp_path, aid, parent_contents=wrong)
    # not strict JSON / not the named artifact / not the contract's shape
    other = {**art.envelope, "artifact_id": "art:" + "0" * 64}
    extra = {**art.envelope, "unexpected": 1}
    broken = {**art.envelope, "producer": None}  # a reader must not crash on it
    for body in (b'{"a":1,"a":2}', b"[1]", jcs(other), jcs(extra), jcs(broken)):
        path.write_bytes(body)
        with pytest.raises(g.CalibrationError):
            g.load_calibration(tmp_path, aid, parent_contents=contents)


def test_lookup_is_by_artifact_id_only(tmp_path: Path) -> None:
    for name in ("build-1", "../x", "art:abc", ""):
        with pytest.raises(g.CalibrationError, match="bad artifact_id"):
            g.load_calibration(tmp_path, name, parent_contents={})
        with pytest.raises(g.CalibrationError, match="bad artifact_id"):
            calibration_module.calibration_path(tmp_path, name)
    public = [
        obj
        for name, obj in vars(calibration_module).items()
        if callable(obj)
        and not name.startswith("_")
        and getattr(obj, "__module__", "") == calibration_module.__name__
    ]
    assert public
    for obj in public:
        if inspect.isfunction(obj):
            assert "model_build_id" not in inspect.signature(obj).parameters, obj
    assert not any("model_build" in n for n in dir(g) if callable(getattr(g, n)))
    with pytest.raises(TypeError):
        g.load_calibration(tmp_path, model_build_id="build-1", parent_contents={})  # type: ignore[call-arg]


def test_calibration_from_envelope_rejects_malformed() -> None:
    art = make_calibration()
    for mutate in muts(
        lambda e: e["payload"].pop("values"),
        lambda e: e["payload"]["values"].__setitem__("w_min", "x"),
        lambda e: e.__setitem__("artifact_id", "nope"),
        lambda e: e["payload"]["values"].__setitem__(
            "codebook_ref", {"artifact_id": 1}
        ),
        lambda e: e["payload_schema"].__setitem__("id", "msr.geometry-measurement"),
        lambda e: e["payload"]["values"].__setitem__("epsilon", 2**60),
    ):
        env = {**art.envelope, "payload": {**art.payload, "values": dict(art.values)}}
        env["payload_schema"] = dict(art.envelope["payload_schema"])
        mutate(env)
        with pytest.raises(g.CalibrationError):
            calibration_from_envelope(env)
    assert (
        calibration_from_envelope(art.envelope, file_sha256="f" * 64).file_sha256
        == "f" * 64
    )
    assert payload_alignment("common_prefix") == "x-common_prefix"


def test_write_blob_is_content_addressed_and_never_altered(tmp_path: Path) -> None:
    em = g.emit_single(traj(9), ctx(), trajectory_ref=ref(50))
    blob = em.blobs[0]
    path = write_blob(tmp_path / "blobs", blob)
    assert path.name == f"{blob.sha256}.bin" and path.read_bytes() == blob.data
    assert write_blob(tmp_path / "blobs", blob) == path  # idempotent
    path.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="different bytes"):
        write_blob(tmp_path / "blobs", blob)


def test_emit_context_and_refs_validate() -> None:
    ok = ("ses:" + "1" * 64, "obs:" + "2" * 64)
    bad_contexts: list[tuple[Any, Any, Any, dict[str, Any]]] = [
        ("nope", ok[1], 1, {}),
        (ok[0], "obs:short", 1, {}),
        (ok[0], ok[1], -1, {}),
        (ok[0], ok[1], True, {}),
        (ok[0], ok[1], 1, {"producer_build": ""}),
        (ok[0], ok[1], 1, {"produced_at": "yesterday"}),
        (ok[0], ok[1], 1, {"event_span": (5, 2)}),
    ]
    for session, observation, seq, kw in bad_contexts:
        with pytest.raises(ContractError):
            g.EmitContext(session, observation, seq, **kw)
    with pytest.raises(ContractError):
        g.ArtifactRef("art:1", "cnt:" + "0" * 64)
    with pytest.raises(ContractError):
        g.ArtifactRef("art:" + "0" * 64, "cnt:1")
    with pytest.raises(ContractError):
        g.PolicyRef("", "0" * 64)
    # one artifact id given with two content ids is refused
    clash = g.ArtifactRef(ref(50).artifact_id, ref(51).content_id)
    cal = make_calibration()
    with pytest.raises(ContractError, match="two different content_ids"):
        g.emit_pair(
            traj(8),
            traj(9),
            ctx(),
            left_ref=ref(50),
            right_ref=clash,
            calibration=cal,
        )


def test_a_different_quantum_is_refused_not_converted() -> None:
    art = make_calibration()
    coarse = g.CalibrationArtifact(art.envelope, art.calibration, exponent=-20)
    with pytest.raises(g.CalibrationError, match="no conversion"):
        g.evaluate_shift_quantized(5, coarse)
    assert g.evaluate_shift_quantized(5, art, distance_exponent=-30) in (
        "STAGNANT",
        "NOISE_FLOOR_INTERNAL",
        "STRUCTURAL_SHIFT_DETECTED",
    )
    with pytest.raises(g.CalibrationError, match="no conversion"):
        g.evaluate_shift_quantized(5, art, distance_exponent=-31)


def _mut(path: tuple[Any, ...], value: Any) -> Any:
    def apply(e: dict[str, Any]) -> object:
        node = e
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return None

    return apply


def _drop(path: tuple[Any, ...]) -> Any:
    def apply(e: dict[str, Any]) -> object:
        node = e
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
        return None

    return apply


A = "art:" + "7" * 64
CONTRACT_BREAKS = {
    "contract-version": _mut(("contract", "version"), "0.6.2"),
    "kind": _mut(("kind",), "evidence"),
    "payload-schema-0.7.1": _mut(("payload_schema", "version"), "0.7.1"),
    "payload-schema-id": _mut(("payload_schema", "id"), "msr.geometry-measurement"),
    "producer": _mut(("producer", "name"), "meaning-mapper"),
    "calibration-ref": _mut(("calibration_ref",), {"artifact_id": A}),
    "artifact-id": _mut(("artifact_id",), "nope"),
    "content-id": _mut(("content_id",), "cnt:x"),
    "extra-envelope-key": _mut(("unexpected",), 1),
    "extra-payload-key": _mut(("payload", "file_sha256"), "0" * 64),
    "empty-model-build": _mut(("payload", "model_build_id"), ""),
    "space-mismatch": _mut(("space_id",), "S9"),
    "floor-above-ceiling": _mut(("payload", "values", "noise_floor_lower"), 2**50),
    "negative-lower": _mut(("payload", "values", "noise_floor_lower"), -1),
    "epsilon-zero": _mut(("payload", "values", "epsilon"), 0),
    "epsilon-float": _mut(("payload", "values", "epsilon"), 1.5),
    "delta-negative": _mut(("payload", "values", "delta"), -1),
    "w-min-bool": _mut(("payload", "values", "w_min"), True),
    "background-too-big": _mut(("payload", "values", "background_mean"), 2**53),
    "alignment-unknown": _mut(("payload", "between_alignment"), "other"),
    "method-name": _mut(("payload", "method", "name"), "other"),
    "method-extra": _mut(("payload", "method", "extra"), 1),
    "quantile-zero": _mut(("payload", "method", "quantiles", "epsilon"), 0),
    "quantile-one": _mut(("payload", "method", "quantiles", "delta"), 2**30),
    "floor-equal": _mut(("payload", "method", "quantiles", "floor"), [5, 5]),
    "floor-length": _mut(("payload", "method", "quantiles", "floor"), [5]),
    "floor-not-list": _mut(("payload", "method", "quantiles", "floor"), 5),
    "null-set-short": _mut(("payload", "null_set"), [A]),
    "null-set-not-id": _mut(("payload", "null_set"), ["x", "y"]),
    "null-set-duplicate": _mut(("payload", "null_set"), [A, A]),
    "null-set-unsorted": _mut(("payload", "null_set"), ["art:" + "9" * 64, A]),
    "trajectory-count": _mut(("payload", "sample_config", "trajectories"), 9),
    "points-zero": _mut(("payload", "sample_config", "points"), 0),
    "sample-config-key": _drop(("payload", "sample_config", "dims")),
    "parent-ordered": _mut(("parents", 0, "ordered"), True),
    "parent-role-extra": _mut(("parents", 0, "role"), "history"),
    "null-set-role-differs": _mut(("parents", 0, "artifact_ids"), [A]),
    "codebook-role-without-ref": lambda e: e["parents"].append(
        {"role": "codebook", "ordered": False, "artifact_ids": [A]}
    ),
    "role-twice": lambda e: e["parents"].append(deepcopy(e["parents"][0])),
    "common-prefix-without-approval": _mut(
        ("payload", "between_alignment"), "x-common_prefix"
    ),
}


@pytest.mark.parametrize("name", list(CONTRACT_BREAKS))
def test_reader_refuses_a_calibration_that_breaks_the_contract(name: str) -> None:
    art = make_calibration()
    env = deepcopy(art.envelope)
    CONTRACT_BREAKS[name](env)
    with pytest.raises((g.CalibrationError, KeyError, TypeError)):
        try:
            calibration_module._check_contract(env, parent_contents())
        except (KeyError, TypeError, AttributeError) as exc:
            raise g.CalibrationError(str(exc)) from exc


def test_reader_checks_the_codebook_parent_and_the_approval() -> None:
    cb = make_calibration(codebook_ref=ref(70))
    contents = parent_contents(ref(70))
    calibration_module._check_contract(cb.envelope, contents)  # the good case
    for bad in (
        {**contents, ref(70).artifact_id: ref(71).content_id},  # content_id differs
        parent_contents(),  # codebook not among the parents
    ):
        with pytest.raises(g.CalibrationError):
            calibration_module._check_contract(cb.envelope, bad)
    env = deepcopy(cb.envelope)
    env["payload"]["values"]["codebook_ref"] = None  # role present, ref gone
    with pytest.raises(g.CalibrationError):
        calibration_module._check_contract(env, contents)
    env = deepcopy(cb.envelope)
    env["parents"] = [p for p in env["parents"] if p["role"] != "codebook"]
    with pytest.raises(g.CalibrationError):  # ref present, role gone
        calibration_module._check_contract(env, contents)
    mixed = [g.NullMember(traj(i, n=30 + 5 * (i % 2)), ref(i)) for i in range(4)]
    cp = g.build_calibration(
        mixed, ctx(alignment_approval=APPROVAL), alignment="common_prefix"
    )
    contents = {m.ref.artifact_id: m.ref.content_id for m in mixed}
    calibration_module._check_contract(cp.envelope, contents)
    for approval in (
        {},
        {"reason": "", "approved_by": "t"},
        {"reason": "r", "approved_by": ""},
    ):
        env = deepcopy(cp.envelope)
        env["extensions"] = {"x-meaning-space-runtime-alignment": approval}
        with pytest.raises(g.CalibrationError):
            calibration_module._check_contract(env, contents)
    env = deepcopy(cp.envelope)
    env["extensions"] = {}
    with pytest.raises(g.CalibrationError):
        calibration_module._check_contract(env, contents)
    env = deepcopy(cp.envelope)
    env["extensions"] = {"x-meaning-space-runtime-alignment": "approved"}
    with pytest.raises(g.CalibrationError):
        calibration_module._check_contract(env, contents)
