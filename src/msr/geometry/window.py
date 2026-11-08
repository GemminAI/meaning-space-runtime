"""A window as an input of ``emit_single``.

IC-02 0.9.0: the first 128 rows (D-129). IC-02 0.10.0: a position and a length
(D-132, D-133).

A window is its own artifact (``msr.trajectory-window`` 0.9.0 or 0.10.0, made by
nvs-runtime from a stored trajectory; MSR never makes one). This module *reads* it:
:func:`window_from_envelope` verifies what a reader must verify (design section 5:
never trust a copy without checking) and returns a :class:`TrajectoryWindow` -- the
:class:`~msr.geometry.trajectory.TrajectoryInput` to measure, the
:class:`~msr.geometry.envelope.ArtifactRef` to put under the parents' role
``trajectory``, and the ``input_scope`` the measurement declares.

Two versions are read, each under its own rule (never one as the other):

* 0.9.0: rule ``prefix_rows``, start 0, length 128, policy ``first-128-rows/1``
  (fixed; the output is unchanged from before 0.10.0, byte for byte).
* 0.10.0 (D-132, D-133): rule ``prefix_rows`` (start 0) or ``rows`` (start >= 0,
  length >= 1, integers); ``start + length <= origin.length``; every ``token_index``
  entry in ``[0, origin.length)``; ``policy.name`` any non-empty string (its sha256
  is checked). Positions are not narrowed.

The space of a window is separated by its position: :func:`check_window_spaces`
refuses two windows that share a ``space_id`` but not ``(start, length)`` (D-133
(4)). The Gate does the same over what it stores.

What this reader checks: payload schema id/version, kind ``observation``, producer
``nvs-runtime``, ``space_id`` present, parents (exactly one, role
``source_trajectory``, ``ordered: false``, present in ``parent_contents``), every
payload field's set / type / constant, ``policy.sha256``, the Blobs' sha256 / uri /
byte length / finiteness (float32) / strictly increasing ``token_index``,
``content_sha256``, and that ``content_id`` / ``artifact_id`` recompute. What it
cannot check: that the stored original says the same ``trajectory_id`` /
``length`` / ``position`` (it does not hold the original; the Gate does), and that
the rows are really the RAW rows ``[start, start + length)`` (see :func:`rows_match`,
:func:`prefix_matches`).
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from msr.geometry import ids
from msr.geometry.envelope import ArtifactRef
from msr.geometry.errors import InvalidWindow
from msr.geometry.trajectory import Policy, TrajectoryInput
from msr.geometry.wire import ContractError

WINDOW_SCHEMA = ("msr.trajectory-window", "0.9.0")
WINDOW_SCHEMA_V0100 = ("msr.trajectory-window", "0.10.0")
WINDOW_RULE = "prefix_rows"
ROWS_RULE = "rows"
WINDOW_START = 0
WINDOW_LENGTH = 128
WINDOW_POLICY = "first-128-rows/1"
SOURCE_ROLE = "source_trajectory"
WINDOW_PRODUCER = "nvs-runtime"
WINDOW_KIND = "observation"
MAX_SAFE = 2**53 - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MEDIA = "application/vnd.sensos.ndarray-raw"


@dataclass(frozen=True, slots=True, eq=False)
class TrajectoryWindow:
    """A verified window: what ``emit_single`` needs, and what it declares."""

    trajectory: TrajectoryInput
    ref: ArtifactRef
    origin: Mapping[str, Any]  # {trajectory_id, position, length}
    scope: Mapping[str, Any]  # the measurement's ``input_scope``
    source_artifact_id: str
    schema_version: str = (
        "0.9.0"  # the payload_schema version the artifact was read under
    )

    @property
    def rule(self) -> str:
        return str(self.scope["rule"])

    @property
    def start(self) -> int:
        return int(self.scope["start"])

    @property
    def length(self) -> int:
        return int(self.scope["length"])

    @property
    def space_id(self) -> str:
        return self.trajectory.space_id


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise InvalidWindow(what)


def _is_int(v: object, lo: int = 0, hi: int = MAX_SAFE) -> bool:
    return type(v) is int and lo <= v <= hi


def _obj(value: object, keys: set[str], what: str) -> dict[str, Any]:
    _need(isinstance(value, dict), f"{what}: not an object")
    assert isinstance(value, dict)
    _need(set(value) == keys, f"{what}: keys {sorted(value)} != {sorted(keys)}")
    return value


def _sha_str(v: object) -> bool:
    return isinstance(v, str) and _SHA256.fullmatch(v) is not None


def _blob_common(ref: dict[str, Any], what: str, dtype: str) -> None:
    _need(
        ref["media_type"] == _MEDIA and ref["dtype"] == dtype,
        f"{what}: media_type/dtype",
    )
    _need(
        ref["byte_order"] == "little" and ref["order"] == "C",
        f"{what}: byte_order/order",
    )
    _need(isinstance(ref["unit"], str) and bool(ref["unit"]), f"{what}: unit")
    _need(
        _sha_str(ref["sha256"]) and ref["uri"] == "blob:sha256:" + ref["sha256"],
        f"{what}: sha256 / uri",
    )


def window_from_envelope(
    envelope: Mapping[str, Any],
    blobs: Mapping[str, bytes],
    *,
    parent_contents: Mapping[str, str],
) -> TrajectoryWindow:
    """Verify a stored window artifact and return it as a measurable input.

    ``parent_contents`` (artifact_id -> content_id of the source trajectory) must be
    what nvs-runtime fetched from HEKB and confirmed; MSR does not call HEKB.
    ``blobs`` maps ``uri`` -> bytes.
    """
    try:
        return _read(envelope, blobs, parent_contents)
    except (KeyError, TypeError, AttributeError, ContractError) as exc:
        raise InvalidWindow(f"not a valid window artifact: {exc!r}") from exc


def _read(
    envelope: Mapping[str, Any],
    blobs: Mapping[str, bytes],
    parent_contents: Mapping[str, str],
) -> TrajectoryWindow:
    declared = envelope["payload_schema"]
    _need(
        declared
        in (
            {"id": WINDOW_SCHEMA[0], "version": WINDOW_SCHEMA[1]},
            {"id": WINDOW_SCHEMA_V0100[0], "version": WINDOW_SCHEMA_V0100[1]},
        ),
        "payload_schema is not msr.trajectory-window 0.9.0 or 0.10.0",
    )
    version = declared["version"]
    fixed = version == WINDOW_SCHEMA[1]
    _need(envelope["kind"] == WINDOW_KIND, f"kind must be {WINDOW_KIND!r}")
    _need(
        envelope["producer"]["name"] == WINDOW_PRODUCER,
        f"producer must be {WINDOW_PRODUCER!r}",
    )
    space_id = envelope["space_id"]
    _need(
        isinstance(space_id, str) and bool(space_id),
        "space_id must be a non-empty string",
    )
    parents = envelope["parents"]
    _need(
        isinstance(parents, list)
        and len(parents) == 1
        and parents[0]["role"] == SOURCE_ROLE
        and parents[0]["ordered"] is False
        and len(parents[0]["artifact_ids"]) == 1,
        f"parents must be exactly one role {SOURCE_ROLE!r}, one artifact, "
        "ordered false",
    )
    source = parents[0]["artifact_ids"][0]
    _need(
        source in parent_contents,
        f"source trajectory {source} is not among the fetched parents",
    )

    p = _obj(
        envelope["payload"],
        {
            "window",
            "origin",
            "source",
            "layer",
            "dimension",
            "model_build_id",
            "policy",
            "points",
            "token_index",
        },
        "payload",
    )
    w = _obj(p["window"], {"rule", "start", "length", "content_sha256"}, "window")
    if fixed:
        _need(
            w["rule"] == WINDOW_RULE
            and type(w["start"]) is int
            and w["start"] == WINDOW_START,
            "window: rule/start are fixed (prefix_rows, 0)",
        )
        _need(
            type(w["length"]) is int and w["length"] == WINDOW_LENGTH,
            f"window.length must be {WINDOW_LENGTH}",
        )
    else:
        _need(
            w["rule"] in (WINDOW_RULE, ROWS_RULE),
            "window.rule must be prefix_rows or rows",
        )
        _need(_is_int(w["start"]), "window.start must be an integer >= 0")
        _need(_is_int(w["length"], 1), "window.length must be an integer >= 1")
        _need(
            w["rule"] != WINDOW_RULE or w["start"] == WINDOW_START,
            "window: prefix_rows is the case start = 0",
        )
    start, length = w["start"], w["length"]
    _need(_sha_str(w["content_sha256"]), "window.content_sha256")
    o = _obj(p["origin"], {"trajectory_id", "position", "length"}, "origin")
    _need(
        _sha_str(o["trajectory_id"])
        and isinstance(o["position"], str)
        and bool(o["position"]),
        "origin.trajectory_id / position",
    )
    if fixed:
        _need(
            _is_int(o["length"], WINDOW_LENGTH),
            "origin.length must be an integer >= 128",
        )
    else:
        _need(_is_int(o["length"], 1), "origin.length must be an integer >= 1")
        _need(
            start + length <= o["length"],
            f"window_range: start {start} + length {length} > "
            f"origin.length {o['length']}",
        )
    _need(p["source"] in ("T-tok", "T-hid"), "source must be T-tok or T-hid")
    layer = p["layer"]
    _need(
        layer is None or _is_int(layer) or (isinstance(layer, str) and bool(layer)),
        "layer",
    )
    _need(
        (layer is None) == (p["source"] == "T-tok"),
        "layer is null exactly when source is T-tok",
    )
    dim = p["dimension"]
    _need(_is_int(dim, 1), "dimension")
    _need(
        isinstance(p["model_build_id"], str) and bool(p["model_build_id"]),
        "model_build_id",
    )
    pol = _obj(p["policy"], {"name", "sha256"}, "policy")
    if fixed:
        _need(pol["name"] == WINDOW_POLICY, f"policy.name must be {WINDOW_POLICY!r}")
    else:
        _need(
            isinstance(pol["name"], str) and bool(pol["name"]),
            "policy.name must be a non-empty string",
        )
    _need(
        pol["sha256"] == hashlib.sha256(pol["name"].encode("utf-8")).hexdigest(),
        "policy.sha256 is not the sha256 of policy.name",
    )

    pts = _obj(
        p["points"],
        {
            "uri",
            "sha256",
            "media_type",
            "dtype",
            "shape",
            "byte_order",
            "order",
            "unit",
        },
        "points",
    )
    _blob_common(pts, "points", "float32")
    _need(
        pts["shape"] == [length, dim],
        f"points.shape must be [{length}, {dim}]",
    )
    tix = _obj(
        p["token_index"],
        {
            "uri",
            "sha256",
            "media_type",
            "dtype",
            "shape",
            "byte_order",
            "order",
            "quantization",
            "unit",
        },
        "token_index",
    )
    _blob_common(tix, "token_index", "int64")
    _need(
        tix["shape"] == [length] and tix["quantization"] is None,
        "token_index.shape / quantization",
    )
    _need(
        w["content_sha256"] == pts["sha256"],
        "window.content_sha256 differs from points.sha256",
    )

    data, raw = blobs.get(pts["uri"]), blobs.get(tix["uri"])
    _need(data is not None and raw is not None, "the window's Blobs are not provided")
    assert data is not None and raw is not None
    _need(
        hashlib.sha256(data).hexdigest() == pts["sha256"],
        "points bytes differ from points.sha256",
    )
    _need(
        len(data) == length * dim * 4,
        f"points must hold {length} x {dim} float32",
    )
    _need(
        hashlib.sha256(raw).hexdigest() == tix["sha256"],
        "token_index bytes differ from token_index.sha256",
    )
    _need(len(raw) == 8 * length, f"token_index must hold {length} int64")
    values = struct.unpack(f"<{length * dim}f", data)
    _need(all(math.isfinite(v) for v in values), "points hold a NaN or an infinity")
    index = struct.unpack(f"<{length}q", raw)
    _need(
        all(b > a for a, b in pairwise(index)),
        "token_index is not strictly increasing",
    )
    if not fixed:
        _need(
            index[0] >= 0 and index[-1] < o["length"],
            f"window_token_range: token_index [{index[0]}, {index[-1]}] is outside "
            f"[0, {o['length']})",
        )

    recomputed = (
        ids.content_id(envelope, parent_contents),
        ids.artifact_id(envelope, parent_contents),
    )
    _need(
        recomputed == (envelope["content_id"], envelope["artifact_id"]),
        "content_id / artifact_id do not recompute",
    )

    traj = TrajectoryInput(
        space_id,
        p["source"],
        np.frombuffer(data, dtype="<f4").reshape(length, dim).astype(np.float64),
        tuple(int(v) for v in index),
        Policy(pol["name"], pol["sha256"]),
        p["model_build_id"],
        layer=layer,
    )
    scope = {
        "kind": "window",
        "rule": w["rule"],
        "start": start,
        "length": length,
        "source_trajectory_id": o["trajectory_id"],
        "content_sha256": w["content_sha256"],
    }
    return TrajectoryWindow(
        traj,
        ArtifactRef(envelope["artifact_id"], envelope["content_id"]),
        dict(o),
        scope,
        source,
        version,
    )


def prefix_matches(window: TrajectoryWindow, source_rows: bytes) -> bool:
    """For a caller holding the original RAW rows (float32 LE): the first rows."""
    n = window.trajectory.length * window.trajectory.dimension * 4
    mine = np.ascontiguousarray(window.trajectory.points, dtype="<f4").tobytes()
    return len(source_rows) >= n and mine == source_rows[:n]


def rows_match(window: TrajectoryWindow, source_rows: bytes) -> bool:
    """Are these the original RAW rows ``[start, start + length)`` (float32 LE)?"""
    row = window.trajectory.dimension * 4
    a, n = window.start * row, window.length * row
    mine = np.ascontiguousarray(window.trajectory.points, dtype="<f4").tobytes()
    return len(source_rows) >= a + n and mine == source_rows[a : a + n]


def check_window_spaces(windows: Iterable[TrajectoryWindow]) -> None:
    """D-133 (4): windows in one ``space_id`` must share ``(start, length)``."""
    seen: dict[str, tuple[int, int]] = {}
    for w in windows:
        position = (w.start, w.length)
        if seen.setdefault(w.space_id, position) != position:
            raise InvalidWindow(
                f"window_space_position: space_id {w.space_id!r} is used by windows "
                f"at {seen[w.space_id]} and {position}"
            )


__all__ = [
    "ROWS_RULE",
    "WINDOW_LENGTH",
    "WINDOW_SCHEMA",
    "WINDOW_SCHEMA_V0100",
    "TrajectoryWindow",
    "check_window_spaces",
    "prefix_matches",
    "rows_match",
    "window_from_envelope",
]
