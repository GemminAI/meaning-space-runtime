from __future__ import annotations

import numpy as np
import pytest

from msr.dynamics import InformationAssimilator, LangevinFlow
from msr.errors import DimensionMismatch
from msr.field import FieldPrior, GaussianWell


def test_assimilation_is_precision_weighted() -> None:
    assimilator = InformationAssimilator()
    theta = np.array([0.0])
    precision = np.array([[1.0]])
    result = assimilator.assimilate(
        theta, precision, np.array([1.0]), np.array([[3.0]])
    )
    # Posterior mean = (1*0 + 3*1) / 4
    assert result.theta[0] == pytest.approx(0.75)
    assert result.precision[0][0] == pytest.approx(4.0)
    assert result.displacement == pytest.approx(0.75)


def test_a_vague_measurement_barely_moves_a_confident_state() -> None:
    assimilator = InformationAssimilator()
    confident = assimilator.assimilate(
        np.array([0.0]), np.array([[100.0]]), np.array([1.0]), np.array([[0.01]])
    )
    assert confident.theta[0] < 0.001


def test_precision_decays_toward_the_floor() -> None:
    assimilator = InformationAssimilator(precision_floor=0.1, forgetting_tau=1.0)
    precision = np.array([[10.0]])
    for _ in range(200):
        precision = assimilator.decay(precision, 0.5)
    assert precision[0][0] == pytest.approx(0.1, abs=1e-6)


def test_assimilator_validates_configuration() -> None:
    with pytest.raises(DimensionMismatch):
        InformationAssimilator(precision_floor=0.0)
    with pytest.raises(DimensionMismatch):
        InformationAssimilator(forgetting_tau=0.0)


def test_flow_descends_the_potential() -> None:
    prior = FieldPrior("F", 1, (GaussianWell.isotropic("c1", [0.0], width=1.0),))
    flow = LangevinFlow(mobility=1.0)
    rng = flow.make_rng()
    theta = np.array([0.8])
    before = prior.potential(theta)
    for _ in range(50):
        theta = flow.step(theta, prior, 0.05, rng)
    assert abs(theta[0]) < 0.8
    assert prior.potential(theta) < before


def test_flow_on_an_empty_field_is_motionless() -> None:
    flow = LangevinFlow()
    rng = flow.make_rng()
    theta = np.array([1.0, -1.0])
    assert np.allclose(flow.step(theta, FieldPrior.empty("F", 2), 0.1, rng), theta)


def test_zero_temperature_is_deterministic() -> None:
    prior = FieldPrior("F", 1, (GaussianWell.isotropic("c1", [0.0], width=1.0),))
    flow = LangevinFlow(temperature=0.0)
    first = flow.step(np.array([0.5]), prior, 0.1, flow.make_rng())
    second = flow.step(np.array([0.5]), prior, 0.1, flow.make_rng())
    assert first.tolist() == second.tolist()


def test_positive_temperature_is_stochastic_but_seeded() -> None:
    prior = FieldPrior.empty("F", 1)
    flow = LangevinFlow(temperature=0.5, seed=7)
    noisy = flow.step(np.array([0.0]), prior, 0.1, flow.make_rng())
    assert noisy[0] != 0.0
    assert (
        noisy.tolist()
        == flow.step(np.array([0.0]), prior, 0.1, flow.make_rng()).tolist()
    )


def test_flow_validates_configuration_and_dt() -> None:
    with pytest.raises(DimensionMismatch):
        LangevinFlow(mobility=0.0)
    with pytest.raises(DimensionMismatch):
        LangevinFlow(temperature=-1.0)
    flow = LangevinFlow()
    with pytest.raises(DimensionMismatch):
        flow.step(np.array([0.0]), FieldPrior.empty("F", 1), 0.0, flow.make_rng())
