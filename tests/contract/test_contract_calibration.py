"""Calibration as an independent, immutable artifact."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from contract_support import APPROVAL, ctx, envelope_errors, muts, nulls, ref, traj
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
    assert env["payload_schema"] == {"id": "msr.calibration", "version": "0.6.1"}
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
    loaded = g.load_calibration(tmp_path, art.ref.artifact_id)
    assert loaded.envelope == art.envelope and loaded.calibration == art.calibration
    assert loaded.file_sha256 and loaded.file_sha256 != art.file_sha256
    assert loaded.ref == art.ref


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


def test_load_checks_integrity_and_identity(tmp_path: Path) -> None:
    art = make_calibration()
    aid = art.ref.artifact_id
    with pytest.raises(g.CalibrationError, match="no calibration"):
        g.load_calibration(tmp_path, aid)
    path = g.write_calibration(tmp_path, art)
    sidecar = path.with_name(path.name + ".sha256")
    good = path.read_bytes()
    path.write_bytes(good.replace(b'"S1"', b'"S2"', 1))
    with pytest.raises(g.CalibrationError, match="integrity"):
        g.load_calibration(tmp_path, aid)
    path.write_bytes(good)
    sidecar.unlink()
    with pytest.raises(g.CalibrationError, match="integrity"):
        g.load_calibration(tmp_path, aid)
    # a file that is intact but not strict JSON / not the named artifact
    for body in (b'{"a":1,"a":2}', b"[1]", b'{"artifact_id":"art:' + b"0" * 64 + b'"}'):
        path.write_bytes(body)
        sidecar.write_text(hashlib.sha256(body).hexdigest())
        with pytest.raises(g.CalibrationError):
            g.load_calibration(tmp_path, aid)


def test_lookup_is_by_artifact_id_only(tmp_path: Path) -> None:
    for name in ("build-1", "../x", "art:abc", ""):
        with pytest.raises(g.CalibrationError, match="bad artifact_id"):
            g.load_calibration(tmp_path, name)
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
        g.load_calibration(tmp_path, model_build_id="build-1")  # type: ignore[call-arg]


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
