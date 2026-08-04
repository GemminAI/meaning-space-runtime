"""EXP-Ubuntu004's read-only bridge from MSR's FieldPrior to NVS-Kernel's WellCache.

This module is experiment-only glue, not MSR library code: it lives here,
not in ``src/msr``, exactly as ``msr.reference`` documents non-normative
stand-ins as experiment/test-only. NVS-Kernel is used strictly as a
read-only decision oracle (its own geometry/field math); this experiment
owns every state transition itself.

**The conversion is structural, not a reinterpretation.** MSR's
``msr.field.GaussianWell`` (``well_id``, ``mean``, ``precision``, ``depth``)
and NVS-Kernel's ``nvs_kernel.field.wells.GaussianWell`` (``label``,
``mean``, ``inverse_covariance``, ``weight``) are the same Gaussian-well
shape under different field names — both are literally
``gᵢ(x) = exp(-½(x-μᵢ)ᵀAᵢ(x-μᵢ))`` with ``Aᵢ`` the precision/inverse
covariance. This experiment's own parity check (in
``exp_msr_004_recovery.py``) verifies the two independent implementations
agree numerically at every point evaluated, not just structurally.

**Error taxonomies do not unify.** NVS-Kernel raises its own typed
``KernelError`` hierarchy in most places, but several of the exact modules
this experiment calls (``field.wells``, ``field.attractor``,
``manifold.euclidean``) raise plain ``ValueError`` instead — read directly
from their source, not assumed. MSR raises ``MSRError``. Neither repo is
modified to unify these; :func:`quarantine_safe` is this experiment's own
boundary, exactly as Meaning Mapper's ``QuarantineReason`` is that repo's
own boundary and not a type shared with anything upstream or downstream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from nvs_kernel.errors import KernelError
from nvs_kernel.field.wells import GaussianWell as NVSGaussianWell
from nvs_kernel.field.wells import WellCache

from msr.errors import MSRError
from msr.field import FieldPrior

T = TypeVar("T")


def field_to_well_cache(prior: FieldPrior) -> WellCache:
    """Convert an MSR ``FieldPrior`` into the NVS-Kernel ``WellCache`` shape.

    A copy of each well's arrays, never a shared reference: the two repos
    must not be able to observe each other's mutations (moot for frozen MSR
    wells today, but the boundary is worth keeping honest regardless).
    """
    return WellCache(
        [
            NVSGaussianWell(
                label=well.well_id,
                mean=well.mean.copy(),
                inverse_covariance=well.precision.copy(),
                weight=well.depth,
            )
            for well in prior.wells
        ]
    )


@dataclass(frozen=True, slots=True)
class HarnessQuarantine:
    """Recorded when a cross-repo call fails. This experiment's own quarantine type."""

    stage: str
    detail: str


def quarantine_safe(stage: str, call: Callable[[], T]) -> T | HarnessQuarantine:
    """Run ``call``; convert any MSRError/KernelError/ValueError into a quarantine
    record.

    Nothing from either repo's own boundary is caught silently: this is the
    harness's Quarantine-Free Continuous Operation guarantee — no uncaught
    failure leaves this function, ever.
    """
    try:
        return call()
    except (MSRError, KernelError, ValueError) as exc:
        return HarnessQuarantine(stage=stage, detail=f"{type(exc).__name__}: {exc}")


__all__ = ["HarnessQuarantine", "field_to_well_cache", "quarantine_safe"]
