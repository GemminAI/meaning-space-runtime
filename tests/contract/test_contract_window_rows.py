"""IC-02 0.10.0 (D-132, D-133): MSR reads a window with a position and a length and measures it.

``window.lock`` pins sensos main (PR #8). The determined vectors (``window-rows.json``, 61 cases) are replayed
against what MSR can decide itself (the reader, the schemas, the emitters); the cases only the Gate can decide
(it holds the stored original, and the claims of a measurement against its parent) are listed here, so a silent
widening of what MSR accepts shows up as a failure. The undetermined vectors (D-133 (6) and one more) are NOT
asserted as expectations: the tests record what MSR does today.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest
from contract_support import (
    VENDOR_WINDOW,
    ctx,
    envelope_errors,
    geometry_v090_validator,
    traj,
    validator,
    window_v0100_validator,
    window_validator,
)

import msr.geometry as g
from msr.geometry.emit import PAYLOAD_SCHEMA, PAYLOAD_SCHEMA_V0100

DET: dict[str, Any] = json.loads(
    (VENDOR_WINDOW / "msr/vectors/window-rows.json").read_text("utf-8")
)
UND: dict[str, Any] = json.loads(
    (VENDOR_WINDOW / "msr/vectors/window-rows-undetermined.json").read_text("utf-8")
)
OLD: dict[str, Any] = json.loads(
    (VENDOR_WINDOW / "msr/archive/0.9.0/vectors/window.json").read_text("utf-8")
)
BLOBS = {
    u: bytes.fromhex(h)
    for u, h in {**DET["blobs"], **UND["blobs"], **OLD["blobs"]}.items()
}
CASES = {c["id"]: c for c in DET["cases"]}
UCASES = {c["id"]: c for c in UND["cases"]}
WINDOWS = [c for c in DET["cases"] if c["target"] == "window"]
SINGLES = [
    c
    for c in DET["cases"]
    if c["target"] == "measurement" and c["envelope"]["payload"]["mode"] == "single"
]
PAIRS = [
    c
    for c in DET["cases"]
    if c["target"] == "measurement" and c["envelope"]["payload"]["mode"] == "pair"
]
WINDOW_ID = "msr.trajectory-window"
# Codes MSR cannot reach: the Gate holds the stored original (window_origin*).
GATE_ONLY_WINDOW = {"window_origin", "window_origin_unreadable"}
# Measurement cases only the Gate can refuse: MSR derives scope and input_ref.length from the window it
# holds, so it cannot write them; and a window used as ``full`` is invisible without the parent (Gate).
GATE_ONLY_SINGLE = {
    "reject-single-scope-full-but-parent-is-window",
}


def _contents(*sources: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for src in sources:
        out.update({a: e["content_id"] for a, e in src.items()})
    return out


def _lookup_contents(case: dict[str, Any]) -> dict[str, str]:
    return {a: e["content_id"] for a, e in case["lookup"].items()}


def _read_env(env: dict[str, Any], case: dict[str, Any]) -> g.TrajectoryWindow:
    stored = {a: s["envelope"] for a, s in case["stored"].items()}
    return g.window_from_envelope(
        env, BLOBS, parent_contents={**_contents(stored), **_lookup_contents(case)}
    )


def _msr_window_verdict(case: dict[str, Any]) -> str:
    """accept / reject by what MSR itself holds: the reader, then the space/position rule over the stored windows."""
    try:
        window = g.window_from_envelope(
            case["envelope"], BLOBS, parent_contents=_lookup_contents(case)
        )
        others = [
            _read_env(s["envelope"], case)
            for a, s in case["stored"].items()
            if s["envelope"]["payload_schema"]["id"] == WINDOW_ID
            and a != case["envelope"]["artifact_id"]
        ]
        g.check_window_spaces([*others, window])
    except g.InvalidWindow:
        return "reject"
    return "accept"


# ---------------------------------------------------------------- the vendored files


def test_window_lock_is_sensos_main_with_pr8_and_files_match() -> None:
    lock = json.loads((VENDOR_WINDOW.parent / "window.lock").read_text("utf-8"))
    main = json.loads((VENDOR_WINDOW.parent / "contract.lock").read_text("utf-8"))
    assert lock["version"] == "0.10.0" and lock["source"]["tag"] is None
    assert lock["source"]["commit"] == main["source"]["commit"]
    assert lock["source"]["commit"] == "56171e064059b8a27ad895e24482afa53a5c70c1"
    assert {f["path"] for f in lock["files"]} == {
        str(p.relative_to(VENDOR_WINDOW))
        for p in VENDOR_WINDOW.rglob("*")
        if p.is_file()
    }
    for f in lock["files"]:
        digest = hashlib.sha256((VENDOR_WINDOW / f["path"]).read_bytes()).hexdigest()
        assert digest == f["sha256"], f["path"]


def test_vendored_vectors_have_the_expected_size() -> None:
    assert (
        len(DET["cases"]),
        sum(c["expect"]["result"] == "accept" for c in DET["cases"]),
    ) == (61, 22)
    assert (len(WINDOWS), len(SINGLES), len(PAIRS)) == (43, 16, 2)
    assert sorted(UCASES) == [
        "undetermined-pair-of-equal-length-windows-at-different-positions",
        "undetermined-token-index-offset-from-row-numbers",
    ]
    assert all(c["tag"] == "未規定" for c in UND["cases"])


def test_the_0_10_0_schemas_are_the_ones_the_vectors_were_made_for() -> None:
    assert (
        json.loads(
            (VENDOR_WINDOW / "msr/trajectory-window.schema.json").read_text("utf-8")
        )["$id"]
        == "sensos:contracts/msr/trajectory-window/0.10.0"
    )
    assert validator("msr/geometry-measurement").schema["$id"].endswith("/0.10.0")


# ---------------------------------------------------------------- the 61 determined cases


@pytest.mark.parametrize("case", WINDOWS, ids=[c["id"] for c in WINDOWS])
def test_reader_agrees_with_the_window_vectors(case: dict[str, Any]) -> None:
    expected = case["expect"]
    version = case["envelope"]["payload_schema"]["version"]
    gate_only = (
        bool(expected["payload"]) and set(expected["payload"]) <= GATE_ONLY_WINDOW
    )
    expect_reject = bool(expected["ic00"]) or bool(
        set(expected["payload"]) - GATE_ONLY_WINDOW
    )
    assert _msr_window_verdict(case) == ("reject" if expect_reject else "accept"), case[
        "id"
    ]
    assert (
        gate_only is False or not expect_reject
    )  # nothing here the reader would have to refuse
    schema = window_validator() if version == "0.9.0" else window_v0100_validator()
    assert bool(list(schema.iter_errors(case["envelope"]["payload"]))) == (
        "schema" in expected["payload"]
    )


def test_cases_only_the_gate_can_refuse_are_exactly_these() -> None:
    only_gate = sorted(
        c["id"]
        for c in WINDOWS
        if c["expect"]["result"] == "reject"
        and not c["expect"]["ic00"]
        and set(c["expect"]["payload"]) <= GATE_ONLY_WINDOW
    )
    assert only_gate == [
        "reject-rows-origin-length-differs-from-stored",
        "reject-rows-origin-trajectory-id-differs-from-stored",
    ]


@pytest.mark.parametrize("case", SINGLES, ids=[c["id"] for c in SINGLES])
def test_single_vectors(case: dict[str, Any]) -> None:
    env, expected = case["envelope"], case["expect"]
    payload = env["payload"]
    schema_errors = envelope_errors(env)
    assert bool(schema_errors) == ("schema" in expected["payload"]), schema_errors
    if "schema" in expected["payload"]:
        return
    parent = next(p for p in env["parents"] if p["role"] == "trajectory")[
        "artifact_ids"
    ][0]
    window = _read_env(case["stored"][parent]["envelope"], case)
    em = g.emit_single(
        window.trajectory,
        ctx(),
        trajectory_ref=window.ref,
        window=window,
        schema_version=env["payload_schema"]["version"],
    )
    emitted = em.envelope["payload"]
    if expected["result"] == "accept":
        # what MSR writes for this window is exactly what the contract accepts
        assert emitted["input_scope"] == payload["input_scope"]
        assert emitted["input_ref"]["length"] == payload["input_ref"]["length"]
        assert em.envelope["payload_schema"] == env["payload_schema"]
        assert envelope_errors(em.envelope) == []
    elif case["id"] in GATE_ONLY_SINGLE:
        assert payload["input_scope"] == {"kind": "full"}
    else:  # window_mismatch / window_length: MSR cannot write a scope or a length that differs from its window
        assert (
            emitted["input_scope"] != payload["input_scope"]
            or emitted["input_ref"]["length"] != payload["input_ref"]["length"]
        )


@pytest.mark.parametrize("case", PAIRS, ids=[c["id"] for c in PAIRS])
def test_pair_vectors_windows_of_different_lengths_are_refused(
    case: dict[str, Any],
) -> None:
    assert case["expect"]["payload"] == ["pair_window_length"]
    env = case["envelope"]
    wl = _read_env(case["stored"][_role(env, "left")]["envelope"], case)
    wr = _read_env(case["stored"][_role(env, "right")]["envelope"], case)
    assert wl.length != wr.length
    for alignment in (
        "equal_only",
        "common_prefix",
    ):  # common_prefix is not adopted (D-120)
        with pytest.raises(g.InvalidWindow, match="pair_window_length"):
            g.emit_pair(
                wl.trajectory,
                wr.trajectory,
                ctx(),
                left_ref=wl.ref,
                right_ref=wr.ref,
                left_window=wl,
                right_window=wr,
                alignment=alignment,
            )


def _role(env: dict[str, Any], role: str) -> str:
    ids: list[str] = next(p for p in env["parents"] if p["role"] == role)[
        "artifact_ids"
    ]
    return ids[0]


def test_the_count_of_determined_cases_msr_agrees_with() -> None:
    """61 determined cases: 43 windows + 16 singles + 2 pairs, each checked above against its expectation."""
    agree = 0
    for case in WINDOWS:
        expect_reject = bool(case["expect"]["ic00"]) or bool(
            set(case["expect"]["payload"]) - GATE_ONLY_WINDOW
        )
        agree += _msr_window_verdict(case) == ("reject" if expect_reject else "accept")
    gate_only_windows = 2
    assert agree == len(WINDOWS)
    assert agree + len(SINGLES) + len(PAIRS) == len(DET["cases"]) == 61
    assert (
        gate_only_windows + len(GATE_ONLY_SINGLE) == 3
    )  # cases MSR accepts and only the Gate refuses


# ---------------------------------------------------------------- explicit: head / middle / tail, boundary, bad values


def test_rows_head_middle_tail_read_as_the_rows_they_declare() -> None:
    for cid, start, length in (
        ("accept-rows-head", 0, 128),
        ("accept-rows-middle", 86, 128),
        ("accept-rows-tail", 1372, 128),
    ):
        c = CASES[cid]
        w = _read_env(c["envelope"], c)
        assert (w.rule, w.start, w.length) == ("rows", start, length)
        assert w.trajectory.length == length and w.schema_version == "0.10.0"
        assert w.scope["start"] == start
        raw = np.ascontiguousarray(w.trajectory.points, dtype="<f4").tobytes()
        assert g.rows_match(
            w, bytes(start * w.trajectory.dimension * 4) + raw + bytes(8)
        )
        # the rows are at [start, ...): the window alone is not the original's first rows unless start is 0
        assert g.rows_match(w, raw) == (start == 0)
        assert not g.rows_match(
            w, bytes(start * w.trajectory.dimension * 4) + raw[:-4]
        )  # too short
        assert not g.rows_match(
            w, bytes(len(raw) + start * w.trajectory.dimension * 4)
        )  # other rows


def test_boundary_end_equals_origin_length_and_one_over() -> None:
    ok = CASES["accept-rows-boundary-end-equals-origin-length"]
    w = _read_env(ok["envelope"], ok)
    assert w.start + w.length == ok["envelope"]["payload"]["origin"]["length"]
    over = CASES["reject-rows-end-exceeds-origin-by-1"]
    with pytest.raises(g.InvalidWindow, match="window_range"):
        g.window_from_envelope(
            over["envelope"], BLOBS, parent_contents=_lookup_contents(over)
        )


@pytest.mark.parametrize(
    "cid",
    [
        "reject-rows-start-negative",
        "reject-rows-length-zero",
        "reject-rows-length-negative",
        "reject-rows-start-string",
        "reject-rows-length-string",
        "reject-rows-start-boolean",
        "reject-rows-start-null",
        "reject-window-rule-unknown-suffix-rows",
        "reject-prefix-rows-start-nonzero",
        "reject-rows-length-missing",
    ],
)
def test_bad_start_or_length_is_refused(cid: str) -> None:
    c = CASES[cid]
    with pytest.raises(g.InvalidWindow):
        g.window_from_envelope(
            c["envelope"], BLOBS, parent_contents=_lookup_contents(c)
        )


def test_token_index_outside_origin_is_refused_and_inside_is_not() -> None:
    for cid in (
        "reject-token-index-beyond-origin-length",
        "reject-token-index-negative",
    ):
        c = CASES[cid]
        with pytest.raises(g.InvalidWindow, match="window_token_range"):
            g.window_from_envelope(
                c["envelope"], BLOBS, parent_contents=_lookup_contents(c)
            )
    c = CASES["reject-rows-longer-than-origin"]
    with pytest.raises(g.InvalidWindow, match="window_range"):
        g.window_from_envelope(
            c["envelope"], BLOBS, parent_contents=_lookup_contents(c)
        )


def test_prefix_rows_of_any_length_and_free_policy_name_are_read() -> None:
    c = CASES["accept-prefix-rows-length-64"]
    w = _read_env(c["envelope"], c)
    assert (w.rule, w.start, w.length) == ("prefix_rows", 0, 64)
    c = CASES["accept-policy-name-free-form"]
    assert _read_env(c["envelope"], c).trajectory.policy.name == "my-middle-window"
    raw = np.ascontiguousarray(w.trajectory.points, dtype="<f4").tobytes()
    assert g.prefix_matches(w, raw + bytes(12))
    assert not g.prefix_matches(w, raw[:-4])


# ---------------------------------------------------------------- the space is separated by position (D-133 (4))


def test_two_positions_in_one_space_are_refused() -> None:
    c = CASES["reject-space-shared-by-two-positions"]
    assert _msr_window_verdict(c) == "reject"
    other = next(
        s["envelope"]
        for s in c["stored"].values()
        if s["envelope"]["payload_schema"]["id"] == WINDOW_ID
    )
    a = _read_env(other, c)
    b = g.window_from_envelope(
        c["envelope"], BLOBS, parent_contents=_lookup_contents(c)
    )
    assert a.space_id == b.space_id and (a.start, a.length) != (b.start, b.length)
    with pytest.raises(g.InvalidWindow, match="window_space_position"):
        g.check_window_spaces([a, b])
    g.check_window_spaces([a, a])  # the same position in the same space is fine
    g.check_window_spaces([])


def test_windows_of_one_original_in_separate_spaces_are_accepted_together() -> None:
    c = CASES["accept-windows-of-one-original-in-separate-spaces"]
    assert _msr_window_verdict(c) == "accept"


# ---------------------------------------------------------------- a 0.9.0 window, and the output of 0.9.0 stays byte-identical


def _good_090() -> g.TrajectoryWindow:
    case = next(c for c in OLD["cases"] if c["id"] == "accept-window-first-128")
    return g.window_from_envelope(
        case["envelope"], BLOBS, parent_contents=_lookup_contents(case)
    )


def test_a_0_9_0_window_is_read_under_its_own_version() -> None:
    w = _good_090()
    assert w.schema_version == "0.9.0"
    assert (w.rule, w.start, w.length) == ("prefix_rows", 0, 128)
    em = g.emit_single(w.trajectory, ctx(), trajectory_ref=w.ref, window=w)
    assert em.envelope["payload_schema"] == {
        "id": PAYLOAD_SCHEMA[0],
        "version": "0.9.0",
    }
    assert envelope_errors(em.envelope) == []


def test_0_9_0_window_output_is_byte_identical_to_before_0_10_0() -> None:
    """The digest was computed on main 4a573f3 (before this change) over every accepted window of the 0.9.0 vectors."""
    h = hashlib.sha256()
    n = 0
    for c in OLD["cases"]:
        if c["target"] != "window" or c["expect"]["result"] != "accept":
            continue
        w = g.window_from_envelope(
            c["envelope"], BLOBS, parent_contents=_lookup_contents(c)
        )
        em = g.emit_single(w.trajectory, ctx(), trajectory_ref=w.ref, window=w)
        h.update(
            json.dumps(em.envelope, sort_keys=True, separators=(",", ":")).encode()
        )
        h.update(json.dumps(dict(w.scope), sort_keys=True).encode())
        n += 1
    assert n == 2
    assert (
        h.hexdigest()
        == "1e81877cb48e0f5b85fe144b125a22e489e61ae6343abe38b9c2e67f705368b8"
    )


def test_a_0_10_0_window_is_emitted_under_0_10_0_and_validates() -> None:
    c = CASES["accept-rows-middle"]
    w = _read_env(c["envelope"], c)
    em = g.emit_single(w.trajectory, ctx(), trajectory_ref=w.ref, window=w)
    assert em.envelope["payload_schema"] == {
        "id": PAYLOAD_SCHEMA_V0100[0],
        "version": "0.10.0",
    }
    assert em.envelope["payload"]["input_scope"] == dict(w.scope)
    assert em.envelope["payload"]["input_ref"]["length"] == w.length
    assert envelope_errors(em.envelope) == []
    with pytest.raises(
        g.InvalidWindow
    ):  # a 0.9.0 measurement declares only the 0.9.0 window
        g.emit_single(
            w.trajectory, ctx(), trajectory_ref=w.ref, window=w, schema_version="0.9.0"
        )
    with pytest.raises(g.ContractError):
        g.emit_single(
            w.trajectory, ctx(), trajectory_ref=w.ref, window=w, schema_version="0.11.0"
        )


def test_a_0_10_0_measurement_may_have_a_0_9_0_window_as_parent() -> None:
    """D-133 (5)."""
    w = _good_090()
    em = g.emit_single(
        w.trajectory, ctx(), trajectory_ref=w.ref, window=w, schema_version="0.10.0"
    )
    assert em.envelope["payload_schema"]["version"] == "0.10.0"
    assert em.envelope["payload"]["input_scope"]["rule"] == "prefix_rows"
    assert envelope_errors(em.envelope) == []
    # the full input stays under 0.9.0 unless 0.10.0 is asked for; both are read by the vendored schemas
    full = g.emit_single(traj(7, n=200), ctx(), trajectory_ref=_ref(50))
    assert full.envelope["payload_schema"]["version"] == "0.9.0"
    full10 = g.emit_single(
        traj(7, n=200), ctx(), trajectory_ref=_ref(50), schema_version="0.10.0"
    )
    assert full10.envelope["payload"] == full.envelope["payload"]
    assert list(geometry_v090_validator().iter_errors(full.envelope["payload"])) == []
    assert envelope_errors(full10.envelope) == []


def _ref(i: int) -> g.ArtifactRef:
    return g.ArtifactRef(f"art:{i:064x}", f"cnt:{i + 1000:064x}")


# ---------------------------------------------------------------- pairs of windows


def _window_pair() -> tuple[g.TrajectoryWindow, g.TrajectoryWindow]:
    a = CASES["accept-rows-head"]
    b = CASES["accept-rows-middle"]
    return _read_env(a["envelope"], a), _read_env(b["envelope"], b)


def test_emit_pair_checks_window_ties_and_lengths() -> None:
    head, mid = _window_pair()
    with pytest.raises(g.InvalidTrajectory):  # left must be left_window.trajectory
        g.emit_pair(
            mid.trajectory,
            mid.trajectory,
            ctx(),
            left_ref=head.ref,
            right_ref=mid.ref,
            left_window=head,
            right_window=mid,
        )
    with pytest.raises(g.InvalidTrajectory):
        g.emit_pair(
            head.trajectory,
            head.trajectory,
            ctx(),
            left_ref=head.ref,
            right_ref=mid.ref,
            left_window=head,
            right_window=mid,
        )
    # same position, same space: a pair of equal-length windows is emitted under 0.10.0 when a window is 0.10.0
    em = g.emit_pair(
        head.trajectory,
        head.trajectory,
        ctx(),
        left_ref=head.ref,
        right_ref=head.ref,
        left_window=head,
        right_window=head,
    )
    assert em.envelope["payload_schema"]["version"] == "0.10.0"
    assert envelope_errors(em.envelope) == []
    old = _good_090()
    em9 = g.emit_pair(
        old.trajectory,
        old.trajectory,
        ctx(),
        left_ref=old.ref,
        right_ref=old.ref,
        left_window=old,
        right_window=old,
    )
    assert em9.envelope["payload_schema"]["version"] == "0.9.0"


def test_emit_pair_refuses_windows_sharing_a_space_at_two_positions() -> None:
    _, mid = _window_pair()
    head_a = g.window_from_envelope(
        CASES["accept-rows-head"]["envelope"],
        BLOBS,
        parent_contents=_lookup_contents(CASES["accept-rows-head"]),
    )
    shared = g.TrajectoryWindow(
        mid.trajectory,
        mid.ref,
        mid.origin,
        {**mid.scope},
        mid.source_artifact_id,
        mid.schema_version,
    )
    assert (
        shared.space_id != head_a.space_id
    )  # sensos gives each position its own space
    object.__setattr__(
        shared,
        "trajectory",
        g.TrajectoryInput(
            head_a.space_id,
            mid.trajectory.source,
            mid.trajectory.points,
            mid.trajectory.token_index,
            mid.trajectory.policy,
            mid.trajectory.model_build_id,
            layer=mid.trajectory.layer,
        ),
    )
    with pytest.raises(g.InvalidWindow, match="window_space_position"):
        g.emit_pair(
            head_a.trajectory,
            shared.trajectory,
            ctx(),
            left_ref=head_a.ref,
            right_ref=shared.ref,
            left_window=head_a,
            right_window=shared,
        )


# ---------------------------------------------------------------- undetermined: recorded, not asserted as expectations


def test_record_token_number_not_equal_to_row_number() -> None:
    """RECORD (未規定): D-133 (1) fixes only the range; the reader takes token numbers inside [0, origin.length)."""
    c = UCASES["undetermined-token-index-offset-from-row-numbers"]
    w = _read_env(c["envelope"], c)
    assert (w.start, w.trajectory.token_index[0]) == (86, 7)


def test_record_pair_of_windows_at_different_positions() -> None:
    """RECORD (未規定, D-133 (6)): equal lengths, different positions. What MSR does today is noted, not required."""
    c = UCASES["undetermined-pair-of-equal-length-windows-at-different-positions"]
    left = _read_env(c["stored"][_role(c["envelope"], "left")]["envelope"], c)
    right = _read_env(c["stored"][_role(c["envelope"], "right")]["envelope"], c)
    assert left.length == right.length and (left.start, left.length) != (
        right.start,
        right.length,
    )
    try:
        g.emit_pair(
            left.trajectory,
            right.trajectory,
            ctx(),
            left_ref=left.ref,
            right_ref=right.ref,
            left_window=left,
            right_window=right,
        )
        observed = "emitted"
    except g.CrossSpaceComparison:
        observed = "refused: the two positions have different space_ids (D-133 (4)) and MSR refuses cross-space"
    # not an expectation of the contract: sensos leaves it open. Update this note if the behaviour changes.
    assert observed.startswith("refused")
