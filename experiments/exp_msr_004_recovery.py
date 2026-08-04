"""EXP-Ubuntu004 — OSS Runtime Validation: NVS-Kernel basin recovery.

The first experiment in this repository to connect a *real* neighbour
(RFC-MSR01 Section 8: "No real CLE, HEKB, or NVS-Kernel is connected" — this
closes that gap for NVS-Kernel specifically). NVS-Kernel is used strictly as
a read-only decision oracle: this experiment calls its field/geometry math
to get a correction direction, and owns every state transition, RNG use,
and recorded state itself (``experiments/_nvs_bridge.py``). NVS-Kernel's own
architecture separates deciding from executing (an ``ApprovalInterlock``
gates its tiered ``ControlEngine``); this experiment does not touch that
layer or that separation at all — it uses the lower-level, unconditional
field math (``PotentialField.gradient``, an exact analytic computation, not
a safety decision) as the source of "which way is back to the basin,"
because the full risk-tiered ``ControlEngine`` requires a safe-set/anchor
model this synthetic scenario has no honest way to construct.

Protocol
    1. Settle a real ``MeaningSpaceRuntime`` into one Gaussian well via a
       clean, deterministic measurement stream (temperature=0) — a
       genuinely-produced stabilized state, not a hand-fabricated one.
    2. **Parity check.** Convert the field to NVS-Kernel's ``WellCache``
       (``_nvs_bridge.field_to_well_cache``) and verify NVS-Kernel's
       independently-implemented ``PotentialField.gradient`` agrees with
       MSR's own ``FieldPrior.gradient`` at the settled point, to near
       machine precision. This is a correctness check on the bridge itself,
       not on either repo.
    3. For each of several fixed (not random) perturbation offsets, all
       large enough to exit the basin: displace the settled state, then run
       a bounded recovery loop where NVS-Kernel's gradient supplies the
       direction and this experiment applies
       ``theta <- theta - mobility * gradient * dt`` — the same functional
       form as ``msr.dynamics.LangevinFlow``'s deterministic term, so
       ``msr.lyapunov.largest_lyapunov_exponent`` can be run over the
       recovered segment using MSR's *own* field prior faithfully. Two
       distinct moments are tracked per trial: first basin re-entry (Basin
       Recovery Rate's τ_recovery — a border-crossing, transient claim) and
       MSR's own ``StabilizationDetector`` confirming a genuine quiescent
       dwell (what Drift Convergence / Runtime Stability / the Lyapunov
       invariant are evaluated over — asymptotic, settled-state claims).
       Conflating the two was tried first and produced a spurious failure
       (recorded in ``docs/RFC_ALIGNMENT.md``) before this split was made.
    4. Every cross-repo call is wrapped in
       ``_nvs_bridge.quarantine_safe``: a raised ``MSRError``/
       ``KernelError``/``ValueError`` anywhere never escapes this
       experiment uncaught.

Hypothesis
    A perturbation that leaves the basin re-enters it and then reaches a
    genuine stabilized dwell within a bounded step budget, and that dwell is
    Lyapunov-contracting (the spec's stabilization invariant) — for offsets
    that do not send the state past a second, farther attractor (none exists
    here: single-well field).

Falsifiable failure
    Any trial fails to re-enter the basin or fails to reach a confirmed
    dwell within the step budget, any cross-repo call is quarantined, or
    the field-adapter parity check exceeds tolerance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from _nvs_bridge import HarnessQuarantine, field_to_well_cache, quarantine_safe
from nvs_kernel.field.potential import PotentialField

from msr.abi import MeaningMeasurement, MeaningState, StabilizedTrajectory
from msr.dynamics import InformationAssimilator
from msr.field import DEFAULT_BASIN_RADIUS, FieldPrior, GaussianWell
from msr.host import MSRHost
from msr.lyapunov import largest_lyapunov_exponent
from msr.metrics import drift_converged, drift_distance, runtime_stability_rate
from msr.reference import RecordingKernel
from msr.runtime import MeaningSpaceRuntime
from msr.stability import StabilizationDetector

FRAME = "exp-msr-004"
DIMENSION = 2
STEP_NS = 50_000_000
WELL_MEAN = np.array([1.0, -0.5])
WELL_WIDTH = 0.5
WELL_DEPTH = 1.5
SETTLE_STEPS = 10
SETTLE_VARIANCE = 0.05

RECOVERY_DT = 0.1
RECOVERY_MOBILITY = 1.0
MAX_RECOVERY_STEPS = 200
DRIFT_EPSILON = 0.05
DRIFT_WINDOW = 20

# Four fixed perturbation offsets, each a Euclidean distance of 1.5 from the
# well mean -- a Mahalanobis distance of 3.0 (basin_radius defaults to 2.0),
# clearly outside the basin. Not random: enumerated, so every run is the
# same experiment.
PERTURBATIONS: tuple[np.ndarray, ...] = (
    np.array([1.5, 0.0]),
    np.array([-1.5, 0.0]),
    np.array([0.0, 1.5]),
    np.array([1.5 / math.sqrt(2), -1.5 / math.sqrt(2)]),
)

PARITY_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class RecoveryTrial:
    basin_reentry_step: int | None
    stabilized: StabilizedTrajectory | None
    states: tuple[MeaningState, ...]
    quarantine: HarnessQuarantine | None

    @property
    def recovered_within_budget(self) -> bool:
        """Basin Recovery Rate's criterion: back in normal territory within budget."""
        return self.basin_reentry_step is not None


def _settle(prior: FieldPrior) -> MeaningState:
    """A genuinely-produced stabilized state: clean measurements at the well mean."""
    host = MSRHost(
        runtime=MeaningSpaceRuntime(frame_id=FRAME, dimension=DIMENSION, prior=prior),
        kernel=RecordingKernel(),
    )
    result = None
    for index in range(SETTLE_STEPS):
        measurement = MeaningMeasurement.isotropic(
            f"settle-{index}",
            FRAME,
            WELL_MEAN.tolist(),
            SETTLE_VARIANCE,
            index * STEP_NS,
        )
        result = host.ingest(measurement)
    assert result is not None
    assert result.state is not None
    return result.state


def _run_recovery_trial(
    prior: FieldPrior,
    origin_state: MeaningState,
    offset: np.ndarray,
    assimilator: InformationAssimilator,
) -> RecoveryTrial:
    """Run NVS-Kernel-guided gradient recovery until MSR's own detector confirms
    re-stabilization (a genuine quiescent dwell), or the step budget is spent.

    Two distinct moments matter, and are reported separately: the step the
    trajectory first re-enters the basin (Basin Recovery Rate's τ_recovery —
    "back in normal territory"), and the step MSR's own
    ``StabilizationDetector`` confirms a real dwell (what the Drift
    Convergence / Runtime Stability / Lyapunov metrics are evaluated over —
    those are asymptotic/settled-state claims, not border-crossing claims).
    """
    cache = field_to_well_cache(prior)
    theta = np.asarray(origin_state.theta) + offset
    precision = np.asarray(origin_state.precision, dtype=np.float64)
    base_time_s = origin_state.time_s
    detector = StabilizationDetector()
    states: list[MeaningState] = []
    basin_reentry_step: int | None = None

    for step in range(1, MAX_RECOVERY_STEPS + 1):

        def _nvs_gradient(point: np.ndarray = theta) -> np.ndarray:
            result: np.ndarray = PotentialField.gradient(point, cache)
            return result

        gradient_result = quarantine_safe("nvs.PotentialField.gradient", _nvs_gradient)
        if isinstance(gradient_result, HarnessQuarantine):
            return RecoveryTrial(
                basin_reentry_step, None, tuple(states), gradient_result
            )

        previous_theta = theta
        theta = theta - RECOVERY_MOBILITY * gradient_result * RECOVERY_DT
        precision = assimilator.decay(precision, RECOVERY_DT)
        speed = float(np.linalg.norm(theta - previous_theta)) / RECOVERY_DT
        basin_id = prior.basin_of(theta)

        state = MeaningState(
            frame_id=FRAME,
            step_index=step,
            time_s=base_time_s + step * RECOVERY_DT,
            theta=tuple(float(v) for v in theta),
            precision=tuple(tuple(float(v) for v in row) for row in precision),
            speed=speed,
            potential=prior.potential(theta),
            basin_id=basin_id,
            source_observation_id=None,
        )
        states.append(state)
        if basin_reentry_step is None and basin_id is not None:
            basin_reentry_step = step

        stabilized = detector.observe(state)
        if stabilized is not None:
            return RecoveryTrial(basin_reentry_step, stabilized, tuple(states), None)

    return RecoveryTrial(basin_reentry_step, None, tuple(states), None)


def run() -> dict[str, Any]:
    well = GaussianWell.isotropic("c1", WELL_MEAN, width=WELL_WIDTH, depth=WELL_DEPTH)
    prior = FieldPrior(FRAME, DIMENSION, (well,), basin_radius=DEFAULT_BASIN_RADIUS)
    origin_state = _settle(prior)
    origin_theta = np.asarray(origin_state.theta)

    # --- Parity check: NVS-Kernel's independent gradient implementation
    # against MSR's own, on the SAME (converted) field, at the settled point.
    cache = field_to_well_cache(prior)
    nvs_gradient = PotentialField.gradient(origin_theta, cache)
    msr_gradient = prior.gradient(origin_theta)
    parity_error = float(np.linalg.norm(nvs_gradient - msr_gradient))

    assimilator = InformationAssimilator()
    trials = [
        _run_recovery_trial(prior, origin_state, offset, assimilator)
        for offset in PERTURBATIONS
    ]

    quarantine_count = sum(1 for trial in trials if trial.quarantine is not None)
    recovered_count = sum(1 for trial in trials if trial.recovered_within_budget)
    basin_recovery_rate = recovered_count / len(trials)
    yield_count = sum(1 for trial in trials if trial.stabilized is not None)
    stabilized_trajectory_yield_rate = yield_count / len(trials)

    # The Runtime Stability / Drift Convergence / Lyapunov metrics are
    # reported over the first trial that reached a genuine, MSR-confirmed
    # re-stabilization -- not merely "crossed back into the basin," which is
    # a transient the earlier version of this experiment mistakenly reused
    # for these asymptotic checks (see docs/RFC_ALIGNMENT.md).
    exemplar = next((trial for trial in trials if trial.stabilized is not None), None)
    stability_rate: float | None = None
    drift_final: float | None = None
    converged: bool | None = None
    lyapunov_exponent: float | None = None
    is_contracting: bool | None = None
    recovered_trajectory: dict[str, Any] | None = None

    if exemplar is not None and exemplar.stabilized is not None:
        segment = exemplar.stabilized.states
        stability_rate = runtime_stability_rate(segment)

        reference = [origin_state.theta for _ in segment]
        actual = [state.theta for state in segment]
        distances = drift_distance(actual, reference)
        drift_final = distances[-1]
        window = min(DRIFT_WINDOW, len(distances))
        converged = drift_converged(distances, DRIFT_EPSILON, window=window)

        estimate = largest_lyapunov_exponent(segment, prior, mobility=RECOVERY_MOBILITY)
        lyapunov_exponent = estimate.exponent
        is_contracting = estimate.is_contracting

        recovered_trajectory = exemplar.stabilized.as_dict()

    result: dict[str, Any] = {
        "experiment": "EXP-Ubuntu004",
        "field_adapter_parity_error": parity_error,
        "quarantine_count": quarantine_count,
        "quarantine_free": quarantine_count == 0,
        "trials": [
            {
                "offset": offset.tolist(),
                "recovered_within_budget": trial.recovered_within_budget,
                "basin_reentry_step": trial.basin_reentry_step,
                "stabilized": trial.stabilized is not None,
                "steps_recorded": len(trial.states),
            }
            for offset, trial in zip(PERTURBATIONS, trials, strict=True)
        ],
        "basin_recovery_rate": basin_recovery_rate,
        "stabilized_trajectory_yield_rate": stabilized_trajectory_yield_rate,
        "runtime_stability_rate": stability_rate,
        "drift_final_distance": drift_final,
        "drift_converged": converged,
        "lyapunov_exponent": lyapunov_exponent,
        "lyapunov_is_contracting": is_contracting,
        "recovered_trajectory": recovered_trajectory,
    }
    result["pass"] = bool(
        parity_error < PARITY_TOLERANCE
        and quarantine_count == 0
        and basin_recovery_rate == 1.0
        and stabilized_trajectory_yield_rate == 1.0
        and converged is True
        and is_contracting is True
    )
    return result


def main() -> None:
    result = run()
    output = Path(__file__).with_name("results") / "exp_msr_004.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
