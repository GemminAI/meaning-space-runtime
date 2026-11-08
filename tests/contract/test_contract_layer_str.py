"""``TrajectoryInput.layer`` may be an int or a string (IC-02 allows both); a digits-only string is
refused so the integer 12 and the string "12" cannot name the same layer (hidden-trajectory H6)."""

from __future__ import annotations

import numpy as np
import pytest
from contract_support import POLICY, ctx, envelope_errors, ref

import msr.geometry as g
from msr.geometry.errors import InvalidTrajectory


def _traj(layer: object, *, source: str = "T-hid") -> g.TrajectoryInput:
    points = np.cumsum(np.random.default_rng(7).normal(size=(30, 4)), axis=0)
    return g.TrajectoryInput(
        "S1",
        source,
        points,
        tuple(range(30)),
        POLICY,
        "build-1",
        layer=layer,  # type: ignore[arg-type]
    )


def test_string_layer_flows_through_analyze_and_emit() -> None:
    t = _traj("final_norm")
    assert t.layer == "final_norm"
    assert g.analyze_trajectory(t, epsilon=1.0, w_min=2)["layer"] == "final_norm"
    em = g.emit_single(t, ctx(3), trajectory_ref=ref(50), epsilon=1.0, w_min=2)
    assert em.envelope["payload"]["layer"] == "final_norm"
    assert envelope_errors(em.envelope) == []


def test_int_and_string_layers_are_different_inputs() -> None:
    assert _traj(12).sha256() != _traj("L12").sha256()
    assert _traj("a").sha256() != _traj("b").sha256()


@pytest.mark.parametrize("bad", ["12", "0", "007", "", True, 1.5, b"x", ["x"]])
def test_bad_layer_is_refused(bad: object) -> None:
    with pytest.raises(InvalidTrajectory):
        _traj(bad)


def test_t_tok_still_has_no_layer() -> None:
    with pytest.raises(InvalidTrajectory):
        _traj("final_norm", source="T-tok")
