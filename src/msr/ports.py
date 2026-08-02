"""Structural ports to the three neighbours. MSR imports none of them.

Each neighbour is a :class:`typing.Protocol`, so NVS-Kernel, CLE and HEKB
satisfy them by shape alone — MSR never imports kernel, CLE or HEKB code, and
therefore can be run, tested and experimented on standalone.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from msr.abi import KernelView, StabilizedTrajectory
from msr.field import FieldPrior


@runtime_checkable
class KernelPort(Protocol):
    """Fast loop: MSR → NVS-Kernel, every step."""

    def observe(self, view: KernelView) -> None: ...


@runtime_checkable
class LiftPort(Protocol):
    """Slow loop, outbound: MSR → CLE, once per stabilization."""

    def lift(self, trajectory: StabilizedTrajectory) -> None: ...


@runtime_checkable
class FieldPriorSource(Protocol):
    """Slow loop, inbound: HEKB → MSR, whenever knowledge changed."""

    def field_prior(self, frame_id: str, dimension: int) -> FieldPrior | None: ...


__all__ = ["FieldPriorSource", "KernelPort", "LiftPort"]
