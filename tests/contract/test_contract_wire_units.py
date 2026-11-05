"""Edge cases of the wire layer that the shared vectors do not reach."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from contract_support import ref

from msr.geometry import ids
from msr.geometry.errors import InvalidTrajectory
from msr.geometry.metrics import pooled_pair_count, pooled_pair_distances
from msr.geometry.wire import (
    ContractError,
    canonicalize,
    dequantize,
    jcs,
    quantize,
    strict_load,
)


def code_of(fn: Any, *args: Any, **kw: Any) -> str:
    with pytest.raises(ContractError) as info:
        fn(*args, **kw)
    return str(info.value.code)


def test_quantize_rejects_non_numbers_and_ties_to_even() -> None:
    assert code_of(quantize, "1", -30) == "E_QUANT_NONFINITE"
    assert code_of(quantize, True, -30) == "E_QUANT_NONFINITE"
    assert code_of(quantize, math.nan, -30) == "E_QUANT_NONFINITE"
    assert code_of(quantize, 1e300, -30) == "E_QUANT_RANGE"
    assert quantize(3, 0) == 3  # ints are fine
    # built-in round(): ties go to even (numpy.rint would agree; no NumPy needed)
    assert [quantize(x, 0) for x in (0.5, 1.5, 2.5, 3.5, -0.5, -1.5)] == [
        0,
        2,
        2,
        4,
        0,
        -2,
    ]
    assert quantize(0.5, -1) == 1  # 0.5 * 2 = 1


def test_dequantize_rejects_bad_input() -> None:
    assert code_of(dequantize, 1.5, -30) == "E_QUANT_RANGE"
    assert code_of(dequantize, True, -30) == "E_QUANT_RANGE"
    assert code_of(dequantize, 2**53, -30) == "E_QUANT_RANGE"
    assert dequantize(2**30, -30) == 1.0


def test_canonicalize_rejects_what_it_cannot_write() -> None:
    assert code_of(canonicalize, 1.0) == "E_FLOAT_TYPE"
    assert code_of(canonicalize, {1: 2}) == "E_NOT_CANONICALIZABLE"
    assert code_of(canonicalize, {1, 2}) == "E_NOT_CANONICALIZABLE"
    assert code_of(canonicalize, 2**53) == "E_INT_RANGE"
    assert code_of(canonicalize, "a\ud800") == "E_SURROGATE"
    assert jcs({"b": [1, None, True, False], "a": "x\n\u0001"}) == (
        b'{"a":"x\\n\\u0001","b":[1,null,true,false]}'
    )
    assert code_of(strict_load, b"[1.0]") == "E_FLOAT_TOKEN"


def envelope(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "contract": {"id": "IC-00", "version": "0.6.2"},
        "kind": "interpretation",
        "payload_schema": {"id": "x", "version": "1"},
        "payload": {},
        "producer": {"name": "p", "version": "1", "build": "b"},
        "session_id": "ses:" + "1" * 64,
        "observation_id": None,
        "event_span": None,
        "space_id": None,
        "parents": [],
        "policy_refs": [],
        "calibration_ref": None,
        "rule_refs": [],
    }
    base.update(kw)
    return base


def test_ids_refuse_duplicates_and_unknown_parents() -> None:
    a = ref(1)
    role = {"role": "r", "ordered": False, "artifact_ids": [a.artifact_id]}
    contents = {a.artifact_id: a.content_id}
    assert ids.content_id(envelope(parents=[role]), contents).startswith("cnt:")
    dup_role = [role, {**role, "artifact_ids": [ref(2).artifact_id]}]
    assert code_of(ids.normalized_parents, dup_role) == "E_PARENTS"
    dup_ids = [{**role, "artifact_ids": [a.artifact_id, a.artifact_id]}]
    assert code_of(ids.normalized_parents, dup_ids) == "E_PARENTS"
    assert code_of(ids.content_id, envelope(parents=[role]), {}) == "E_LOOKUP"
    twice = [{"name": "n", "sha256": "0" * 64}] * 2
    assert code_of(ids.sorted_set, twice, ("name", "sha256"), "policy_refs") == (
        "E_PARENTS"
    )
    # ordered roles keep their order; unordered ones are sorted
    two = [ref(9).artifact_id, ref(3).artifact_id]
    kept = ids.normalized_parents([{"role": "r", "ordered": True, "artifact_ids": two}])
    assert kept[0]["artifact_ids"] == two
    sorted_ = ids.normalized_parents(
        [{"role": "r", "ordered": False, "artifact_ids": two}]
    )
    assert sorted_[0]["artifact_ids"] == sorted(two)


def test_pooled_pairs_match_the_full_matrix() -> None:
    pts = np.random.default_rng(4).normal(size=(23, 3))
    full = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    for w_min in (0, 1, 5, 21, 30):
        want = [full[i, j] for i in range(23) for j in range(i + w_min + 1, 23)]
        assert pooled_pair_count(23, w_min) == len(want)
        for block in (1, 4, 100):
            got = pooled_pair_distances(pts, w_min, block_size=block)
            assert got.tolist() == want
    with pytest.raises(InvalidTrajectory):
        pooled_pair_distances(pts, -1)
    with pytest.raises(InvalidTrajectory):
        pooled_pair_distances(pts, 1, block_size=0)
