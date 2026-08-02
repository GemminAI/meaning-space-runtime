from __future__ import annotations

import numpy as np
import pytest

from msr.adapters.hekb import field_prior_from_concepts, well_from_concept
from msr.adapters.mapper import measurement_from_payload
from msr.errors import DimensionMismatch
from msr.reference import ReferenceConcept

# --------------------------------------------------------------- HEKB adapter


def test_concept_with_hessian_becomes_an_anisotropic_well() -> None:
    concept = ReferenceConcept(
        id="c1",
        centroid=(1.0, 0.0),
        hessian=((4.0, 0.0), (0.0, 1.0)),
        invariants={"depth": 2.5},
    )
    well = well_from_concept(concept, 2)
    assert well is not None
    assert well.well_id == "c1"
    assert well.depth == 2.5
    assert well.precision[0][0] == 4.0


def test_concept_without_hessian_falls_back_to_isotropic() -> None:
    concept = ReferenceConcept(id="c1", centroid=(0.0, 0.0), invariants={"width": 2.0})
    well = well_from_concept(concept, 2)
    assert well is not None
    assert well.precision[0][0] == pytest.approx(0.25)


def test_non_spd_hessian_is_degraded_not_trusted() -> None:
    concept = ReferenceConcept(
        id="c1", centroid=(0.0, 0.0), hessian=((1.0, 2.0), (2.0, 1.0))
    )
    well = well_from_concept(concept, 2)
    assert well is not None
    assert np.allclose(well.precision, np.eye(2))


def test_invalid_geometry_is_skipped() -> None:
    assert well_from_concept(ReferenceConcept(id="c1"), 2) is None
    assert well_from_concept(ReferenceConcept(id="c2", centroid=(0.0,)), 2) is None
    assert (
        well_from_concept(ReferenceConcept(id="c3", centroid=(float("nan"), 0.0)), 2)
        is None
    )


def test_non_positive_invariants_fall_back_to_defaults() -> None:
    concept = ReferenceConcept(
        id="c1", centroid=(0.0,), invariants={"depth": -1.0, "width": 0.0}
    )
    well = well_from_concept(concept, 1, default_width=3.0, default_depth=7.0)
    assert well is not None
    assert well.depth == 7.0
    assert well.precision[0][0] == pytest.approx(1.0 / 9.0)


def test_field_prior_from_concepts_skips_unusable_and_versions() -> None:
    concepts = (
        ReferenceConcept(id="good", centroid=(0.0, 0.0)),
        ReferenceConcept(id="bad", centroid=None),
    )
    prior = field_prior_from_concepts(concepts, "F", 2, version=4)
    assert [w.well_id for w in prior.wells] == ["good"]
    assert prior.version == 4
    assert prior.frame_id == "F"


def test_concept_with_non_dict_invariants_is_tolerated() -> None:
    class Odd:
        id = "c1"
        centroid = (0.0,)
        invariants = "not-a-dict"

    well = well_from_concept(Odd(), 1)
    assert well is not None


# ------------------------------------------------------------- mapper adapter


def test_payload_with_native_names() -> None:
    measurement = measurement_from_payload(
        {
            "observation_id": "o1",
            "frame_id": "F",
            "theta": [1.0, 2.0],
            "sigma": [[0.5, 0.0], [0.0, 0.5]],
            "timestamp_ns": 42,
            "provenance": ["ev-1"],
        }
    )
    assert measurement.theta == (1.0, 2.0)
    assert measurement.provenance == ("ev-1",)
    assert measurement.timestamp_ns == 42


def test_payload_with_annotation_era_names() -> None:
    measurement = measurement_from_payload(
        {
            "id": "o1",
            "frame": "F",
            "coordinates": [1.0],
            "covariance": [[0.25]],
            "timestamp": 7,
        }
    )
    assert measurement.frame_id == "F"
    assert measurement.sigma == ((0.25,),)


def test_payload_with_diagonal_or_scalar_uncertainty() -> None:
    diagonal = measurement_from_payload(
        {
            "id": "o",
            "frame": "F",
            "theta": [1.0, 2.0],
            "sigma": [1.0, 4.0],
            "timestamp": 0,
        }
    )
    assert diagonal.sigma == ((1.0, 0.0), (0.0, 4.0))
    scalar = measurement_from_payload(
        {"id": "o", "frame": "F", "theta": [1.0, 2.0], "variance": 0.5, "timestamp": 0}
    )
    assert scalar.sigma == ((0.5, 0.0), (0.0, 0.5))


def test_payload_rejects_missing_or_invalid_fields() -> None:
    base = {"id": "o", "frame": "F", "theta": [1.0], "variance": 1.0, "timestamp": 0}
    for missing in ("id", "frame", "theta", "timestamp"):
        payload = dict(base)
        del payload[missing]
        with pytest.raises(DimensionMismatch):
            measurement_from_payload(payload)
    with pytest.raises(DimensionMismatch):
        measurement_from_payload({**base, "theta": []})
    with pytest.raises(DimensionMismatch):
        measurement_from_payload({**base, "variance": None, "sigma": None})
    with pytest.raises(DimensionMismatch):
        measurement_from_payload({**base, "variance": 0.0})
