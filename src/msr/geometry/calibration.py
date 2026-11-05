"""Calibration as an independent artifact (design v0.6.2 §3.3.1) and the verdict.

A calibration is an IC-00 envelope whose payload follows
``msr/calibration.schema.json``. Other artifacts point at it by
``{artifact_id, content_id}`` (never by ``model_build_id`` and never by the
file's sha256). It is immutable (H1): a stored calibration is never overwritten,
and a re-calibration is a new artifact. The file's sha256 is used only to check
that the file read back is intact.

No threshold lives in code: ε, δ, w_min and the noise-floor interval come from a
calibration; without one every shift verdict is ``UNCALIBRATED``.
``evaluate_shift`` follows ``sensos_core.manifold_shift.evaluate_manifold_shift``
without importing it (MSR imports no neighbour code); the tests check they agree.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from msr.geometry import ids
from msr.geometry.envelope import (
    ALIGNMENT_EXTENSION,
    CONTRACT,
    PRODUCER_NAME,
    ArtifactRef,
    EmitContext,
    alignment_extensions,
    build_envelope,
)
from msr.geometry.errors import CalibrationError
from msr.geometry.metrics import (
    DEFAULT_BLOCK_SIZE,
    mean_step_distance,
    pooled_pair_count,
    pooled_pair_distances,
    step_displacements,
)
from msr.geometry.trajectory import TrajectoryInput
from msr.geometry.wire import (
    MAX_SAFE,
    ContractError,
    dequantize,
    jcs,
    quantize,
    strict_load,
)

ShiftVerdict = Literal[
    "UNCALIBRATED", "STAGNANT", "NOISE_FLOOR_INTERNAL", "STRUCTURAL_SHIFT_DETECTED"
]
Alignment = Literal["equal_only", "common_prefix"]
METHOD_NAME = "msr-geometry-calibration"
METHOD_VERSION = "msr-geometry-calibration/1"
PAYLOAD_SCHEMA = ("msr.calibration", "0.7.0")
EXPONENT = -30  # every real value in the contract is a multiple of 2**-30
DEFAULT_MAX_POOLED_PAIRS = 50_000_000  # ~400 MB of float64

_ARTIFACT = re.compile(r"^art:[0-9a-f]{64}$")
_ONE = 2**30  # probability 1.0 on the 2**-30 grid


def payload_alignment(mode: Alignment) -> str:
    """Contract spelling: ``common_prefix`` is written ``x-common_prefix``."""
    if mode == "equal_only":
        return "equal_only"
    if mode == "common_prefix":
        return "x-common_prefix"
    raise CalibrationError(f"unknown alignment {mode!r}")


@dataclass(frozen=True, slots=True)
class Calibration:
    """Float view of a calibration's values (each exactly a multiple of 2**-30)."""

    model_build_id: str
    space_id: str
    noise_floor_lower: float
    noise_floor_upper: float
    background_mean: float
    epsilon: float | None = None
    delta: float | None = None
    w_min: int | None = None
    between_alignment: str = "equal_only"
    codebook_ref: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """A calibration envelope together with its decoded values.

    ``file_sha256`` is set only when the artifact was read back from a file.
    """

    envelope: dict[str, Any]
    calibration: Calibration
    file_sha256: str | None = None
    exponent: int = EXPONENT  # the declared quantum of ``values`` (contract 0.7.0)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(self.envelope["artifact_id"], self.envelope["content_id"])

    @property
    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = self.envelope["payload"]
        return payload

    @property
    def values(self) -> dict[str, Any]:
        """The payload's integer values (``noise_floor_lower`` ... ``w_min``)."""
        values: dict[str, Any] = self.payload["values"]
        return values


def calibration_from_envelope(
    envelope: Mapping[str, Any], *, file_sha256: str | None = None
) -> CalibrationArtifact:
    """Decode a calibration envelope (shape errors become ``CalibrationError``)."""
    try:
        if envelope["payload_schema"] != {
            "id": PAYLOAD_SCHEMA[0],
            "version": PAYLOAD_SCHEMA[1],
        }:
            raise CalibrationError("not an msr.calibration payload")
        payload = envelope["payload"]
        values = payload["values"]
        raw_ref = values["codebook_ref"]
        calibration = Calibration(
            model_build_id=payload["model_build_id"],
            space_id=payload["space_id"],
            noise_floor_lower=dequantize(values["noise_floor_lower"], EXPONENT),
            noise_floor_upper=dequantize(values["noise_floor_upper"], EXPONENT),
            background_mean=dequantize(values["background_mean"], EXPONENT),
            epsilon=dequantize(values["epsilon"], EXPONENT),
            delta=dequantize(values["delta"], EXPONENT),
            w_min=int(values["w_min"]),
            between_alignment=payload["between_alignment"],
            codebook_ref=None if raw_ref is None else ArtifactRef(**raw_ref),
        )
        artifact = CalibrationArtifact(dict(envelope), calibration, file_sha256)
        artifact.ref  # noqa: B018 - validates artifact_id / content_id
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        raise CalibrationError(f"malformed calibration envelope: {exc!r}") from exc
    return artifact


def _path(data_dir: Path, artifact_id: str, suffix: str) -> Path:
    if not _ARTIFACT.match(artifact_id):
        raise CalibrationError(f"bad artifact_id {artifact_id!r}")
    return Path(data_dir) / "calibration" / f"{artifact_id}{suffix}"


def calibration_path(data_dir: Path, artifact_id: str) -> Path:
    """``<data_dir>/calibration/<artifact_id>.json``."""
    return _path(data_dir, artifact_id, ".json")


def write_calibration(data_dir: Path, artifact: CalibrationArtifact) -> Path:
    """Store the envelope's canonical bytes; refuse to overwrite (H1).

    A ``<artifact_id>.json.sha256`` file records the file's sha256 for the
    integrity check on load; it is not part of the artifact.
    """
    artifact_id = artifact.ref.artifact_id
    path = calibration_path(data_dir, artifact_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = jcs(artifact.envelope)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise CalibrationError(
            f"{path} already exists; calibrations are immutable (write a new artifact)"
        ) from exc
    _path(data_dir, artifact_id, ".json.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n", encoding="ascii"
    )
    return path


_ENVELOPE_KEYS = frozenset(
    [
        "contract",
        "content_id",
        "artifact_id",
        "kind",
        "payload_schema",
        "payload",
        "producer",
        "session_id",
        "observation_id",
        "event_span",
        "space_id",
        "parents",
        "policy_refs",
        "calibration_ref",
        "rule_refs",
        "observed_at",
        "seq",
        "produced_at",
        "extensions",
    ]
)
_PAYLOAD_KEYS = frozenset(
    [
        "space_id",
        "model_build_id",
        "method",
        "null_set",
        "between_alignment",
        "values",
        "sample_config",
    ]
)
_VALUE_KEYS = frozenset(
    [
        "noise_floor_lower",
        "noise_floor_upper",
        "background_mean",
        "epsilon",
        "delta",
        "w_min",
        "codebook_ref",
    ]
)


_ART = re.compile(r"^art:[0-9a-f]{64}$")
_CNT = re.compile(r"^cnt:[0-9a-f]{64}$")
_CAL_ROLES = frozenset({"null_set", "codebook"})


def _is_int(value: object, lo: int, hi: int = MAX_SAFE) -> bool:
    return type(value) is int and lo <= value <= hi


def _is_id(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.match(value) is not None


def _check_contract(
    envelope: Mapping[str, Any], parent_contents: Mapping[str, str]
) -> None:
    """What a *reader* verifies about a calibration without trusting the Gate:
    the contract version, the fields it uses (type and range), the invariants
    S17/S19/S21, and that every parent the envelope names has a content_id in
    ``parent_contents`` (the full JSON-Schema check is the Gate's job)."""
    payload = envelope.get("payload")
    values = payload.get("values") if isinstance(payload, dict) else None
    ok = (
        set(envelope) == _ENVELOPE_KEYS
        and envelope["contract"] == CONTRACT
        and envelope["kind"] == "interpretation"
        and envelope["payload_schema"]
        == {"id": PAYLOAD_SCHEMA[0], "version": PAYLOAD_SCHEMA[1]}
        and envelope["producer"]["name"] == PRODUCER_NAME
        and envelope["calibration_ref"] is None
        and _is_id(_ART, envelope["artifact_id"])
        and _is_id(_CNT, envelope["content_id"])
        and isinstance(payload, dict)
        and set(payload) == _PAYLOAD_KEYS
        and isinstance(values, dict)
        and set(values) == _VALUE_KEYS
        and all(isinstance(payload[k], str) and payload[k] for k in _TEXT_KEYS)
        and envelope["space_id"] == payload["space_id"]
        and _is_int(values["noise_floor_lower"], 0)
        and _is_int(values["noise_floor_upper"], values["noise_floor_lower"])
        and _is_int(values["background_mean"], 0)
        and _is_int(values["epsilon"], 1)
        and _is_int(values["delta"], 0)
        and _is_int(values["w_min"], 0)
        and payload["between_alignment"] in ("equal_only", "x-common_prefix")
        and _quantiles_ok(payload["method"])
        and _null_set_ok(payload)
        and _parents_ok(envelope, payload, parent_contents)
        and (
            payload["between_alignment"] == "equal_only"
            or _approved(envelope["extensions"])
        )
    )
    if not ok:
        raise CalibrationError("calibration envelope does not match the contract")


_TEXT_KEYS = ("space_id", "model_build_id")


def _approved(extensions: Mapping[str, Any]) -> bool:
    ext = extensions.get(ALIGNMENT_EXTENSION)
    return (
        isinstance(ext, dict)
        and isinstance(ext.get("reason"), str)
        and bool(ext["reason"])
        and isinstance(ext.get("approved_by"), str)
        and bool(ext["approved_by"])
    )


def _quantiles_ok(method: Any) -> bool:
    q = method["quantiles"]
    floor = q["floor"]
    return (
        set(method) == {"name", "version", "quantiles"}
        and method["name"] == METHOD_NAME
        and isinstance(method["version"], str)
        and set(q) == {"epsilon", "delta", "floor"}
        and _is_int(q["epsilon"], 1, _ONE - 1)
        and _is_int(q["delta"], 1, _ONE - 1)
        and isinstance(floor, list)
        and len(floor) == 2
        and _is_int(floor[0], 1, _ONE - 1)
        and _is_int(floor[1], floor[0] + 1, _ONE - 1)
    )


def _null_set_ok(payload: Mapping[str, Any]) -> bool:
    null_set = payload["null_set"]
    config = payload["sample_config"]
    return (
        isinstance(null_set, list)
        and len(null_set) >= 2
        and all(_is_id(_ART, a) for a in null_set)
        and null_set == sorted(set(null_set), key=lambda a: a.encode())
        and isinstance(config, dict)
        and set(config) == {"trajectories", "points", "dims"}
        and config["trajectories"] == len(null_set)
        and _is_int(config["points"], 1)
        and _is_int(config["dims"], 1)
    )


def _parents_ok(
    envelope: Mapping[str, Any],
    payload: Mapping[str, Any],
    parent_contents: Mapping[str, str],
) -> bool:
    roles = {p["role"]: p for p in envelope["parents"]}
    codebook = payload["values"]["codebook_ref"]
    cb_ids = [] if "codebook" not in roles else roles["codebook"]["artifact_ids"]
    named = [a for p in envelope["parents"] for a in p["artifact_ids"]]
    return (
        len(roles) == len(envelope["parents"])
        and set(roles) <= _CAL_ROLES
        and "null_set" in roles
        and all(p["ordered"] is False for p in roles.values())
        and roles["null_set"]["artifact_ids"] == payload["null_set"]
        and (
            cb_ids == []
            if codebook is None
            else cb_ids == [codebook["artifact_id"]]
            and parent_contents.get(codebook["artifact_id"]) == codebook["content_id"]
        )
        and all(a in parent_contents for a in named)
        and all(_is_id(_CNT, v) for v in parent_contents.values())  # every entry
    )


def load_calibration(
    data_dir: Path, artifact_id: str, *, parent_contents: Mapping[str, str]
) -> CalibrationArtifact:
    """Read a stored calibration by ``artifact_id`` and verify it (design §3.3.1).

    ``parent_contents`` (artifact_id -> content_id of every parent: the
    ``null_set`` members and the codebook) must be what **nvs-runtime fetched
    from HEKB and verified**; MSR does not call HEKB and does not trust a caller's
    arbitrary parent information beyond this contract's own cross-checks (every
    named parent must be present, the codebook's content_id must equal
    ``codebook_ref``). A calibration that was not stored through the Gate is not
    trusted: strict parse, contract version, field types and ranges, parent
    consistency (S17/S19/S21), then ``content_id`` and ``artifact_id`` are
    recomputed from those parents. The file alone is never self-sufficient.
    A ``.json.sha256`` file is optional; if present and different the file is
    refused.
    """
    path = calibration_path(data_dir, artifact_id)
    if not path.exists():
        raise CalibrationError(f"no calibration {artifact_id} under {data_dir}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = _path(data_dir, artifact_id, ".json.sha256")
    if sidecar.exists() and sidecar.read_text(encoding="ascii").strip() != digest:
        raise CalibrationError(f"{path} does not match its .sha256 file")
    try:
        envelope = strict_load(raw)
    except ContractError as exc:
        raise CalibrationError(f"{path} is not strict JSON: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("artifact_id") != artifact_id:
        raise CalibrationError(f"{path} does not hold artifact {artifact_id}")
    try:
        _check_contract(envelope, parent_contents)
        recomputed = (
            ids.content_id(envelope, parent_contents),
            ids.artifact_id(envelope, parent_contents),
        )
    except (KeyError, TypeError, AttributeError, ContractError) as exc:
        raise CalibrationError(f"{path} is not a valid calibration: {exc!r}") from exc
    if recomputed != (envelope["content_id"], artifact_id):
        raise CalibrationError(f"{path}: content_id / artifact_id do not recompute")
    return calibration_from_envelope(envelope, file_sha256=digest)


def evaluate_shift(d: float, calib: Calibration | None) -> ShiftVerdict:
    """SPEC §3.2 on floats. **Non-normative**: never used for a payload's verdict.

    The payload's ``verdict.state`` is :func:`evaluate_shift_quantized` (design
    §3.3.3); this float form only serves the in-memory measurement layer.
    """
    if calib is None:
        return "UNCALIBRATED"
    return _verdict(d, calib.noise_floor_lower, calib.noise_floor_upper)


def evaluate_shift_quantized(
    distance_q: int,
    artifact: CalibrationArtifact | None,
    *,
    distance_exponent: int = EXPONENT,
) -> ShiftVerdict:
    """The pair verdict (design §3.3.3), decided on the payload's integers.

    ``d < lo`` STAGNANT; ``lo <= d <= hi`` NOISE_FLOOR_INTERNAL (the boundaries
    belong to the noise band); ``d > hi`` STRUCTURAL_SHIFT_DETECTED; no
    calibration UNCALIBRATED. A distance and a calibration declared with
    different quanta are refused, never converted.
    """
    if artifact is None:
        return "UNCALIBRATED"
    if artifact.exponent != distance_exponent:
        raise CalibrationError(
            f"quantum differs: distance 2**{distance_exponent}, "
            f"calibration 2**{artifact.exponent} (no conversion)"
        )
    values = artifact.values
    return _verdict(
        distance_q, values["noise_floor_lower"], values["noise_floor_upper"]
    )


def _verdict(d: float, lower: float, upper: float) -> ShiftVerdict:
    if d < lower:
        return "STAGNANT"
    if d <= upper:
        return "NOISE_FLOOR_INTERNAL"
    return "STRUCTURAL_SHIFT_DETECTED"


@dataclass(frozen=True, slots=True)
class NullMember:
    """A null (noise-only) trajectory and the artifact that stores it."""

    trajectory: TrajectoryInput
    ref: ArtifactRef


def _level(value: float, name: str) -> int:
    q = quantize(value, EXPONENT)
    if not 0 < q < _ONE:
        raise CalibrationError(
            f"{name} quantile {value} must be strictly inside (0, 1)"
        )
    return q


def build_calibration(
    null_set: Sequence[NullMember],
    ctx: EmitContext,
    *,
    w_min: int = 1,
    epsilon_quantile: float = 0.05,
    delta_quantile: float = 0.10,
    floor_quantiles: tuple[float, float] = (0.05, 0.95),
    codebook_ref: ArtifactRef | None = None,
    alignment: Alignment = "equal_only",
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_pooled_pairs: int = DEFAULT_MAX_POOLED_PAIRS,
) -> CalibrationArtifact:
    """Derive a calibration artifact from a set of null trajectories.

    ε = ``epsilon_quantile`` of within-trajectory point distances
    (``j - i > w_min``); δ = ``delta_quantile`` of step displacements; the
    noise-floor interval = ``floor_quantiles`` of the pairwise mean per-step
    distance between null trajectories. Quantile levels are snapped to the
    2**-30 grid first and the snapped level is what is used and recorded.
    ``alignment`` ("equal_only" by default) governs the between-trajectory
    distances: unequal lengths are refused unless ``common_prefix`` is chosen,
    which needs ``ctx.alignment_approval`` (S17).

    Raises ``CalibrationError`` when the pooled within-trajectory pairs exceed
    ``max_pooled_pairs`` (the exact quantile needs them in memory).
    """
    if len(null_set) < 2:
        raise CalibrationError("need at least 2 null trajectories")
    refs = [m.ref for m in null_set]
    if len({r.artifact_id for r in refs}) != len(refs):
        raise CalibrationError("null_set refs must be distinct artifacts")
    trajs = [m.trajectory for m in null_set]
    first = trajs[0]
    for traj in trajs:
        if (
            traj.space_id != first.space_id
            or traj.model_build_id != first.model_build_id
        ):
            raise CalibrationError(
                "null trajectories must share space_id and model_build_id"
            )
        if traj.dimension != first.dimension:
            raise CalibrationError("null trajectories must share dimension")
    if w_min < 0:
        raise CalibrationError("w_min must be >= 0")
    mode = payload_alignment(alignment)
    if alignment == "equal_only" and len({t.length for t in trajs}) != 1:
        raise CalibrationError(
            "null trajectories differ in length; equal_only refuses "
            "(common_prefix needs an approval)"
        )
    extensions = alignment_extensions(mode, ctx)
    q_eps, q_delta = (
        _level(epsilon_quantile, "epsilon"),
        _level(delta_quantile, "delta"),
    )
    q_lo, q_hi = (
        _level(floor_quantiles[0], "floor"),
        _level(floor_quantiles[1], "floor"),
    )
    if not q_lo < q_hi:
        raise CalibrationError("floor quantiles must satisfy lower < upper")
    pairs = sum(pooled_pair_count(t.length, w_min) for t in trajs)
    if pairs == 0:
        raise CalibrationError("null trajectories too short for the requested w_min")
    if pairs > max_pooled_pairs:
        raise CalibrationError(
            f"{pairs} pooled pairs exceed max_pooled_pairs={max_pooled_pairs}"
        )
    pooled = np.concatenate(
        [pooled_pair_distances(t.points, w_min, block_size=block_size) for t in trajs]
    )
    steps = np.concatenate([step_displacements(t.points) for t in trajs])
    between = [
        mean_step_distance(a.points, b.points)
        for a, b in itertools.combinations(trajs, 2)
    ]
    lower, upper = (
        float(np.quantile(between, dequantize(q, EXPONENT))) for q in (q_lo, q_hi)
    )
    epsilon = float(np.quantile(pooled, dequantize(q_eps, EXPONENT)))
    delta = float(np.quantile(steps, dequantize(q_delta, EXPONENT)))
    q_epsilon = quantize(epsilon, EXPONENT)
    if q_epsilon < 1:
        raise CalibrationError("epsilon is smaller than one quantum (2**-30)")
    payload = {
        "space_id": first.space_id,
        "model_build_id": first.model_build_id,
        "method": {
            "name": METHOD_NAME,
            "version": METHOD_VERSION,
            "quantiles": {"epsilon": q_eps, "delta": q_delta, "floor": [q_lo, q_hi]},
        },
        "null_set": sorted((r.artifact_id for r in refs), key=lambda a: a.encode()),
        "between_alignment": mode,
        "values": {
            "noise_floor_lower": quantize(lower, EXPONENT),
            "noise_floor_upper": quantize(upper, EXPONENT),
            "background_mean": quantize(float(np.mean(between)), EXPONENT),
            "epsilon": q_epsilon,
            "delta": quantize(delta, EXPONENT),
            "w_min": w_min,
            "codebook_ref": None if codebook_ref is None else codebook_ref.to_dict(),
        },
        "sample_config": {
            "trajectories": len(trajs),
            "points": int(min(t.length for t in trajs)),
            "dims": first.dimension,
        },
    }
    parents: list[tuple[str, Sequence[ArtifactRef]]] = [("null_set", refs)]
    if codebook_ref is not None:
        parents.append(("codebook", [codebook_ref]))
    envelope = build_envelope(
        ctx,
        payload_schema=PAYLOAD_SCHEMA,
        payload=payload,
        space_id=first.space_id,
        parents=parents,
        calibration_ref=None,
        extensions=extensions,
    )
    return calibration_from_envelope(envelope)


__all__ = [
    "DEFAULT_MAX_POOLED_PAIRS",
    "EXPONENT",
    "METHOD_VERSION",
    "Alignment",
    "Calibration",
    "CalibrationArtifact",
    "NullMember",
    "ShiftVerdict",
    "build_calibration",
    "calibration_from_envelope",
    "calibration_path",
    "evaluate_shift",
    "evaluate_shift_quantized",
    "load_calibration",
    "payload_alignment",
    "write_calibration",
]
