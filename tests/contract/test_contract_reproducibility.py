"""S15a: the same input in another process gives the same content_id,
artifact_id and Blob sha256 (the build is the same: this interpreter)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from contract_support import ctx, nulls, ref, traj

import msr.geometry as g

SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {here!r})
    import msr.geometry as g
    from contract_support import ctx, nulls, ref, traj

    cal = g.build_calibration(nulls(), ctx(2))
    single = g.emit_single(traj(9), ctx(3), trajectory_ref=ref(50), calibration=cal)
    pair = g.emit_pair(traj(8), traj(9), ctx(3), left_ref=ref(60),
                       right_ref=ref(61), calibration=cal)
    print(json.dumps({{
        "cal": [cal.envelope["content_id"], cal.envelope["artifact_id"]],
        "single": [single.envelope["content_id"], single.envelope["artifact_id"],
                   single.blobs[0].sha256],
        "pair": [pair.envelope["content_id"], pair.envelope["artifact_id"]],
        "bytes": single.to_bytes().decode(),
    }}))
    """
)


def run_once(hash_seed: str) -> dict[str, object]:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(here=str(Path(__file__).parent))],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    result: dict[str, object] = json.loads(out.stdout)
    return result


def test_two_processes_agree_on_ids_and_blob_hash() -> None:
    first, second = run_once("1"), run_once("2")  # different str-hash seeds too
    assert first == second
    cal = g.build_calibration(nulls(), ctx(2))
    single = g.emit_single(traj(9), ctx(3), trajectory_ref=ref(50), calibration=cal)
    assert first["cal"] == [cal.envelope["content_id"], cal.envelope["artifact_id"]]
    assert first["single"] == [
        single.envelope["content_id"],
        single.envelope["artifact_id"],
        single.blobs[0].sha256,
    ]
    assert first["bytes"] == single.to_bytes().decode()


def test_seq_and_produced_at_enter_no_id() -> None:
    a = g.emit_single(traj(9), ctx(3), trajectory_ref=ref(50))
    b = g.emit_single(
        traj(9),
        g.EmitContext(
            ctx().session_id,
            ctx().observation_id,
            999,
            producer_build="test-build",
            produced_at="2030-01-01T00:00:00Z",
        ),
        trajectory_ref=ref(50),
    )
    assert a.envelope["artifact_id"] == b.envelope["artifact_id"]
    assert a.envelope["seq"] != b.envelope["seq"]
    assert a.blobs[0].sha256 == b.blobs[0].sha256


def test_build_enters_artifact_id_but_not_content_id() -> None:
    other = g.EmitContext(
        ctx().session_id, ctx().observation_id, 3, producer_build="another-build"
    )
    a = g.emit_single(traj(9), ctx(3), trajectory_ref=ref(50))
    b = g.emit_single(traj(9), other, trajectory_ref=ref(50))
    assert a.envelope["content_id"] == b.envelope["content_id"]
    assert a.envelope["artifact_id"] != b.envelope["artifact_id"]
    moved = g.emit_single(traj(10), ctx(3), trajectory_ref=ref(50))
    assert moved.envelope["content_id"] != a.envelope["content_id"]
