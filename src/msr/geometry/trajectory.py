"""``TrajectoryInput`` — a hidden-layer / token trajectory with its identity.

MSR compares only trajectories living in the same ``space_id`` (D-69, H6);
the input type therefore makes ``space_id`` mandatory and binds the policy
(how the trajectory was made) and ``model_build_id`` to the points.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from msr.geometry.errors import CrossSpaceComparison, InvalidTrajectory

Source = Literal["T-tok", "T-hid"]
SOURCES: tuple[str, ...] = ("T-tok", "T-hid")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGITS_ONLY = re.compile(r"[0-9]+")


@dataclass(frozen=True, slots=True)
class Policy:
    """Name and digest of the recipe that produced the trajectory."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidTrajectory("policy name must be non-empty")
        if not _SHA256.match(self.sha256):
            raise InvalidTrajectory("policy sha256 must be 64 lowercase hex digits")


@dataclass(frozen=True, slots=True, eq=False)
class TrajectoryInput:
    """T points in D dimensions plus the identity needed to compare them."""

    space_id: str
    source: Source
    points: NDArray[np.float64]
    token_index: tuple[int, ...]
    policy: Policy
    model_build_id: str
    layer: int | str | None = None

    def __post_init__(self) -> None:
        if not self.space_id:
            raise InvalidTrajectory("space_id is required")
        if not self.model_build_id:
            raise InvalidTrajectory("model_build_id is required")
        if self.source not in SOURCES:
            raise InvalidTrajectory(f"source must be one of {SOURCES}")
        if self.source == "T-hid" and self.layer is None:
            raise InvalidTrajectory("layer is required when source is 'T-hid'")
        if self.source == "T-tok" and self.layer is not None:
            raise InvalidTrajectory("layer must be None when source is 'T-tok'")
        if isinstance(self.layer, bool) or not isinstance(
            self.layer, (int, str, type(None))
        ):
            raise InvalidTrajectory("layer must be an int, a str or None")
        if isinstance(self.layer, str):
            # IC-02 allows a string layer (e.g. "final_norm"). A digits-only string
            # is refused so the integer 12 and the string "12" never name the same
            # layer (hidden-trajectory H6).
            if not self.layer:
                raise InvalidTrajectory("layer must not be an empty string")
            if _DIGITS_ONLY.fullmatch(self.layer):
                raise InvalidTrajectory(
                    f"layer {self.layer!r} is digits-only; pass the integer (H6)"
                )
        points = np.array(self.points, dtype=np.float64)  # private copy
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 1:
            raise InvalidTrajectory("points must be a non-empty T x D array")
        if not np.all(np.isfinite(points)):
            raise InvalidTrajectory("points must be finite")
        points.setflags(write=False)
        object.__setattr__(self, "points", points)
        if len(self.token_index) != points.shape[0]:
            raise InvalidTrajectory("token_index must have one entry per point")

    @property
    def length(self) -> int:
        return int(self.points.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.points.shape[1])

    def sha256(self) -> str:
        """Digest of everything that identifies this input (the input ref)."""
        header: dict[str, Any] = {
            "space_id": self.space_id,
            "source": self.source,
            "layer": self.layer,
            "token_index": list(self.token_index),
            "policy": {"name": self.policy.name, "sha256": self.policy.sha256},
            "model_build_id": self.model_build_id,
            "shape": list(self.points.shape),
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        digest.update(np.ascontiguousarray(self.points).tobytes())
        return digest.hexdigest()


def require_same_space(
    left: TrajectoryInput, right: TrajectoryInput, *, allow_cross_space: bool = False
) -> None:
    """Refuse to compare trajectories of different spaces unless told to."""
    if left.space_id != right.space_id and not allow_cross_space:
        raise CrossSpaceComparison(
            f"space_id {left.space_id!r} != {right.space_id!r}; "
            "pass allow_cross_space=True to compare explicitly"
        )


__all__ = ["SOURCES", "Policy", "Source", "TrajectoryInput", "require_same_space"]
