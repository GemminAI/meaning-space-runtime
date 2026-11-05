"""Shared helpers for the IC-00 / IC-02 contract tests (not a test module)."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import msr.geometry as g

HERE = Path(__file__).parent
VENDOR = HERE / "vendor"
VECTORS = VENDOR / "common" / "vectors"
SCHEMAS = (
    "common/artifact",
    "common/provenance",
    "common/blob",
    "msr/geometry-measurement",
    "msr/calibration",
)
POLICY = g.Policy("last-token", "a" * 64)
ALIGN_EXT = "x-meaning-space-runtime-alignment"
APPROVAL = {"reason": "short run for a smoke test", "approved_by": "tomo"}


def _registry() -> Registry:
    registry: Registry = Registry()
    for name in SCHEMAS:
        doc = json.loads((VENDOR / f"{name}.schema.json").read_text("utf-8"))
        registry = registry.with_resource(doc["$id"], Resource.from_contents(doc))
    return registry


def validator(name: str) -> Draft202012Validator:
    doc = json.loads((VENDOR / f"{name}.schema.json").read_text("utf-8"))
    return Draft202012Validator(doc, registry=_registry())


def schema_errors(name: str, instance: object) -> list[str]:
    return [e.message for e in validator(name).iter_errors(instance)]


def envelope_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Schema errors of the envelope and of its payload (by ``payload_schema``)."""
    errors = schema_errors("common/artifact", envelope)
    payload_schema = envelope["payload_schema"]["id"]
    name = {
        "msr.geometry-measurement": "msr/geometry-measurement",
        "msr.calibration": "msr/calibration",
    }[payload_schema]
    return errors + schema_errors(name, envelope["payload"])


def ref(i: int) -> g.ArtifactRef:
    return g.ArtifactRef(f"art:{i:064x}", f"cnt:{i + 1000:064x}")


def ctx(seq: int = 1, **kw: Any) -> g.EmitContext:
    return g.EmitContext(
        "ses:" + "1" * 64,
        "obs:" + "2" * 64,
        seq,
        producer_build="test-build",
        produced_at="2026-11-06T00:00:00Z",
        **kw,
    )


def traj(seed: int, n: int = 40, d: int = 4, space: str = "S1") -> g.TrajectoryInput:
    points = np.cumsum(np.random.default_rng(seed).normal(size=(n, d)), axis=0)
    return g.TrajectoryInput(
        space, "T-hid", points, tuple(range(n)), POLICY, "build-1", layer=3
    )


def nulls(count: int = 4, **kw: Any) -> list[g.NullMember]:
    return [g.NullMember(traj(100 + i, **kw), ref(10 + i)) for i in range(count)]


def calibration(seq: int = 2, **kw: Any) -> g.CalibrationArtifact:
    return g.build_calibration(nulls(), ctx(seq), **kw)


def decode_blob(data: bytes, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    return np.frombuffer(data, dtype="<i8").reshape(shape)


def deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


Mutation = Callable[[dict[str, Any]], object]


def muts(*fns: Mutation) -> tuple[Mutation, ...]:
    """Typed container so lambda mutations get their signature from context."""
    return fns


def parent_contents(*extra: g.ArtifactRef) -> dict[str, str]:
    """artifact_id -> content_id of the default null set (and ``extra``)."""
    refs = [m.ref for m in nulls()] + list(extra)
    return {r.artifact_id: r.content_id for r in refs}
