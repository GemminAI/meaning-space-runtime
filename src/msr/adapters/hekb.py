"""HEKB → MSR: turn committed concepts into the field prior Φ.

A HEKB ``Concept`` already carries exactly the geometry this needs — the
``centroid``/``hessian``/``invariants`` metadata it echoes from upstream (see
exp7100's ``Concept``). One concept becomes one Gaussian well: knowledge is
literally the shape of the space that future observations flow through.

Concepts without usable geometry are skipped, not guessed at: a concept with no
centroid has no position in this frame, and inventing one would put a well
somewhere no observation ever was.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from msr.errors import MSRError
from msr.field import DEFAULT_BASIN_RADIUS, FieldPrior, GaussianWell
from msr.linalg import as_spd


@runtime_checkable
class ConceptLike(Protocol):
    """The HEKB concept shape MSR reads. Structural — HEKB is never imported."""

    @property
    def id(self) -> str: ...

    @property
    def centroid(self) -> tuple[float, ...] | None: ...


def _hessian_of(concept: ConceptLike) -> object | None:
    return getattr(concept, "hessian", None)


def _invariants_of(concept: ConceptLike) -> dict[str, float]:
    invariants = getattr(concept, "invariants", None)
    return dict(invariants) if isinstance(invariants, dict) else {}


def well_from_concept(
    concept: ConceptLike,
    dimension: int,
    *,
    default_width: float = 1.0,
    default_depth: float = 1.0,
) -> GaussianWell | None:
    """Build one well from one concept, or ``None`` if it has no usable geometry.

    The concept's Hessian is used as the well's precision when it is a valid SPD
    matrix of the right size; otherwise the well falls back to isotropic
    ``default_width``. A non-SPD Hessian is a saddle or a flat direction — real
    information, but not a basin, so it is degraded rather than trusted.
    """
    centroid = concept.centroid
    if centroid is None:
        return None
    mean = np.asarray(centroid, dtype=np.float64).reshape(-1)
    if mean.size != dimension or not np.all(np.isfinite(mean)):
        return None

    invariants = _invariants_of(concept)
    depth = float(invariants.get("depth", default_depth))
    if depth <= 0.0:
        depth = default_depth

    hessian = _hessian_of(concept)
    if hessian is not None:
        try:
            precision = as_spd(hessian, dimension, name="hessian")
        except MSRError:
            precision = None
        if precision is not None:
            return GaussianWell(
                well_id=concept.id, mean=mean, precision=precision, depth=depth
            )

    width = float(invariants.get("width", default_width))
    if width <= 0.0:
        width = default_width
    return GaussianWell.isotropic(concept.id, mean, width=width, depth=depth)


def field_prior_from_concepts(
    concepts: object,
    frame_id: str,
    dimension: int,
    *,
    version: int = 1,
    basin_radius: float = DEFAULT_BASIN_RADIUS,
    default_width: float = 1.0,
    default_depth: float = 1.0,
) -> FieldPrior:
    """Build the whole field prior from an iterable of HEKB concepts."""
    wells: list[GaussianWell] = []
    for concept in concepts:  # type: ignore[attr-defined]
        well = well_from_concept(
            concept,
            dimension,
            default_width=default_width,
            default_depth=default_depth,
        )
        if well is not None:
            wells.append(well)
    return FieldPrior(
        frame_id=frame_id,
        dimension=dimension,
        wells=tuple(wells),
        basin_radius=basin_radius,
        version=version,
    )


__all__ = ["ConceptLike", "field_prior_from_concepts", "well_from_concept"]
