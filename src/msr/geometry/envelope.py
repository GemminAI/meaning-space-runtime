"""IC-00 envelope assembly for MSR artifacts.

MSR numbers nothing: ``session_id``, ``observation_id``, ``seq`` and the
``artifact_id`` of every input come from the caller (nvs-runtime) through
:class:`EmitContext` and :class:`ArtifactRef`. This module only assembles the
envelope, computes ``content_id`` / ``artifact_id`` (:mod:`msr.geometry.ids`)
and writes Blobs.
"""

from __future__ import annotations

import hashlib
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from msr.geometry import ids
from msr.geometry.wire import E_INVARIANT, MAX_SAFE, ContractError, jcs
from msr.version import __version__

PRODUCER_NAME = "meaning-space-runtime"
CONTRACT = {"id": "IC-00", "version": "0.6.2"}
ALIGNMENT_EXTENSION = "x-meaning-space-runtime-alignment"
BLOB_MEDIA_TYPE = "application/vnd.sensos.ndarray-raw"

E_CONTEXT = "E_CONTEXT"

_ARTIFACT = re.compile(r"^art:[0-9a-f]{64}$")
_CONTENT = re.compile(r"^cnt:[0-9a-f]{64}$")
_SESSION = re.compile(r"^ses:[0-9a-f]{64}$")
_OBSERVATION = re.compile(r"^obs:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?(Z|[+-]\d{2}:\d{2})$"
)


def default_build() -> str:
    """Identify the numeric environment (BLAS/CPU differences are not assumed away)."""
    return (
        f"python-{platform.python_version()}/numpy-{np.__version__}/"
        f"{platform.system().lower()}-{platform.machine()}"
    )


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """An already-stored artifact, referenced by both of its ids (§2.3)."""

    artifact_id: str
    content_id: str

    def __post_init__(self) -> None:
        if not _ARTIFACT.match(self.artifact_id):
            raise ContractError(E_CONTEXT, f"bad artifact_id {self.artifact_id!r}")
        if not _CONTENT.match(self.content_id):
            raise ContractError(E_CONTEXT, f"bad content_id {self.content_id!r}")

    def to_dict(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "content_id": self.content_id}


@dataclass(frozen=True, slots=True)
class PolicyRef:
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not _SHA256.match(self.sha256):
            raise ContractError(E_CONTEXT, "policy ref needs a name and a sha256")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class EmitContext:
    """Identity and bookkeeping supplied by the caller, never invented by MSR.

    ``produced_at`` defaults to the current UTC time; it enters no id.
    ``alignment_approval`` ({reason, approved_by}) is required to emit anything
    aligned by ``common_prefix`` (S17).
    """

    session_id: str
    observation_id: str | None
    seq: int
    producer_build: str = field(default_factory=default_build)
    produced_at: str | None = None
    observed_at: Mapping[str, Any] | None = None
    event_span: tuple[int, int] | None = None
    policy_refs: tuple[PolicyRef, ...] = ()
    alignment_approval: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not _SESSION.match(self.session_id):
            raise ContractError(E_CONTEXT, f"bad session_id {self.session_id!r}")
        if self.observation_id is not None and not _OBSERVATION.match(
            self.observation_id
        ):
            raise ContractError(
                E_CONTEXT, f"bad observation_id {self.observation_id!r}"
            )
        if isinstance(self.seq, bool) or not 0 <= self.seq <= MAX_SAFE:
            raise ContractError(E_CONTEXT, "seq must be an int in [0, 2^53-1]")
        if not self.producer_build:
            raise ContractError(E_CONTEXT, "producer_build must be non-empty")
        if self.produced_at is not None and not _RFC3339.match(self.produced_at):
            raise ContractError(E_CONTEXT, "produced_at must be RFC 3339")
        if self.event_span is not None:
            start, end = self.event_span
            if not 0 <= start <= end <= MAX_SAFE:
                raise ContractError(E_CONTEXT, "event_span must be 0 <= start <= end")


@dataclass(frozen=True, slots=True)
class Blob:
    """Canonical bytes of an array (int64, little-endian, C order, no header)."""

    data: bytes
    shape: tuple[int, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def uri(self) -> str:
        return f"blob:sha256:{self.sha256}"


@dataclass(frozen=True, slots=True)
class Emission:
    """An envelope plus the Blobs its payload references."""

    envelope: dict[str, Any]
    blobs: tuple[Blob, ...] = ()

    def to_bytes(self) -> bytes:
        """The envelope's canonical (JCS) bytes."""
        return jcs(self.envelope)


def int64_blob(values: Sequence[Sequence[int]]) -> Blob:
    """A rows x cols int64 array as little-endian C-order bytes."""
    array = np.asarray(values, dtype="<i8")
    return Blob(array.tobytes(order="C"), (int(array.shape[0]), int(array.shape[1])))


def write_blob(directory: Path, blob: Blob) -> Path:
    """Store ``blob`` as ``<directory>/<sha256>.bin`` (idempotent, never altered)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{blob.sha256}.bin"
    if path.exists():
        if path.read_bytes() != blob.data:
            raise ContractError(E_INVARIANT, f"{path} exists with different bytes")
        return path
    path.write_bytes(blob.data)
    return path


def alignment_extensions(mode: str, ctx: EmitContext) -> dict[str, Any]:
    """S17: ``x-common_prefix`` needs ``{reason, approved_by}`` in the envelope."""
    if mode != "x-common_prefix":
        return {}
    approval = ctx.alignment_approval
    if (
        approval is None
        or not isinstance(approval.get("reason"), str)
        or not approval["reason"]
        or not isinstance(approval.get("approved_by"), str)
        or not approval["approved_by"]
    ):
        raise ContractError(
            "S17",
            "common_prefix alignment needs alignment_approval {reason, approved_by}",
        )
    return {
        ALIGNMENT_EXTENSION: {
            "reason": approval["reason"],
            "approved_by": approval["approved_by"],
        }
    }


def build_envelope(
    ctx: EmitContext,
    *,
    payload_schema: tuple[str, str],
    payload: Mapping[str, Any],
    space_id: str,
    parents: Sequence[tuple[str, Sequence[ArtifactRef]]],
    calibration_ref: ArtifactRef | None,
    extensions: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble an IC-00 envelope and fill ``content_id`` / ``artifact_id``.

    ``parents`` is ``(role, refs)`` per role; every role MSR emits is
    ``ordered: false`` (ids are stored sorted, duplicates refused).
    """
    parent_contents: dict[str, str] = {}
    roles: list[dict[str, Any]] = []
    for role, refs in parents:
        for ref in refs:
            known = parent_contents.setdefault(ref.artifact_id, ref.content_id)
            if known != ref.content_id:
                raise ContractError(
                    E_CONTEXT, f"{ref.artifact_id} given with two different content_ids"
                )
        roles.append(
            {
                "role": role,
                "ordered": False,
                "artifact_ids": [ref.artifact_id for ref in refs],
            }
        )
    envelope: dict[str, Any] = {
        "contract": dict(CONTRACT),
        "content_id": "",
        "artifact_id": "",
        "kind": "interpretation",
        "payload_schema": {"id": payload_schema[0], "version": payload_schema[1]},
        "payload": dict(payload),
        "producer": {
            "name": PRODUCER_NAME,
            "version": __version__,
            "build": ctx.producer_build,
        },
        "session_id": ctx.session_id,
        "observation_id": ctx.observation_id,
        "event_span": (
            None
            if ctx.event_span is None
            else {"token_start": ctx.event_span[0], "token_end": ctx.event_span[1]}
        ),
        "space_id": space_id,
        "parents": ids.normalized_parents(roles),
        "policy_refs": ids.sorted_set(
            [p.to_dict() for p in ctx.policy_refs], ("name", "sha256"), "policy_refs"
        ),
        "calibration_ref": None
        if calibration_ref is None
        else calibration_ref.to_dict(),
        "rule_refs": [],
        "observed_at": None if ctx.observed_at is None else dict(ctx.observed_at),
        "seq": ctx.seq,
        "produced_at": ctx.produced_at or _now(),
        "extensions": dict(extensions),
    }
    envelope["content_id"] = ids.content_id(envelope, parent_contents)
    envelope["artifact_id"] = ids.artifact_id(envelope, parent_contents)
    return envelope


__all__ = [
    "ALIGNMENT_EXTENSION",
    "BLOB_MEDIA_TYPE",
    "CONTRACT",
    "E_CONTEXT",
    "PRODUCER_NAME",
    "ArtifactRef",
    "Blob",
    "Emission",
    "EmitContext",
    "PolicyRef",
    "alignment_extensions",
    "build_envelope",
    "default_build",
    "int64_blob",
    "write_blob",
]
