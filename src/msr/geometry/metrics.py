"""Trajectory geometry metrics (EXP-WSP-TRAJ001 §8, MSR-side subset).

Pure functions over a ``T x D`` float64 array. Every metric is returned both as
a whole-session summary and as a prefix series (``series[t]`` = the value
computed from points ``0..t``). Trajectory Completion (M3) is an Outcome and
is deliberately not computed here (D-89).

Definitions fixed here (the spec leaves the exact form open):

* Displacement ``d_t = ||x_t - x_{t-1}||``; the sub-δ run is the longest run of
  consecutive ``d_t < δ``.
* Recurrence (a): among pairs ``i < j`` with ``j - i > w_min``, the fraction
  with ``||x_i - x_j|| < ε``. (b): a *return* happens at ``j`` when some past
  point ``i`` (``j - i > w_min``) has ``||x_i - x_j|| < ε`` while
  ``||x_i - x_{j-1}|| >= ε`` (left the ε-ball, came back); count / length.
* Diversity (a): Shannon entropy (nats) of codebook-cell occupancy;
  (b): participation ratio ``tr(C)^2 / tr(C^2)`` of the covariance (0.0 when
  all points coincide).
* Persistence: points get a cell id (codebook nearest centroid, else online
  leader clustering with radius ε); longest run of one id, and the share of
  points in the most occupied cell.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from msr.geometry.errors import InvalidTrajectory

Array = NDArray[np.float64]
Series = list[float | None]


def _check_points(points: Array) -> None:
    if points.ndim != 2 or points.shape[0] < 1:
        raise InvalidTrajectory("points must be a non-empty T x D array")


def pairwise_distances(points: Array) -> Array:
    """Full Euclidean distance matrix (T x T), float64."""
    _check_points(points)
    diff = points[:, None, :] - points[None, :, :]
    return np.asarray(np.sqrt(np.sum(diff * diff, axis=-1)), dtype=np.float64)


def step_displacements(points: Array) -> Array:
    """``d_t`` for ``t = 1..T-1`` (empty for a single point)."""
    _check_points(points)
    delta = np.diff(points, axis=0)
    return np.asarray(np.sqrt(np.sum(delta * delta, axis=1)), dtype=np.float64)


def assign_cells(
    points: Array, codebook: Array | None, epsilon: float | None
) -> list[int]:
    """Cell id per point: nearest codebook centroid, else leader clustering."""
    if codebook is not None:
        if codebook.ndim != 2 or codebook.shape[1] != points.shape[1]:
            raise InvalidTrajectory("codebook must be K x D with D matching points")
        if codebook.shape[0] < 1:
            raise InvalidTrajectory("codebook must have at least one centroid")
        diff = points[:, None, :] - codebook[None, :, :]
        return [int(i) for i in np.argmin(np.sum(diff * diff, axis=-1), axis=1)]
    if epsilon is None:
        raise InvalidTrajectory("cells need a codebook or epsilon")
    leaders: list[Array] = []
    cells: list[int] = []
    for point in points:
        for index, leader in enumerate(leaders):
            if float(np.linalg.norm(point - leader)) < epsilon:
                cells.append(index)
                break
        else:
            leaders.append(point)
            cells.append(len(leaders) - 1)
    return cells


def _longest_run(flags: Sequence[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def displacement_metrics(
    points: Array, delta: float | None
) -> tuple[dict[str, object], dict[str, Series]]:
    d = step_displacements(points)
    n = len(d)
    summary: dict[str, object] = {
        "count": n,
        "mean": float(np.mean(d)) if n else None,
        "median": float(np.median(d)) if n else None,
        "longest_sub_delta_run": None,
    }
    mean_s: Series = [None]
    median_s: Series = [None]
    run_s: Series = [None]
    run = best = 0
    for t in range(n):
        mean_s.append(float(np.mean(d[: t + 1])))
        median_s.append(float(np.median(d[: t + 1])))
        if delta is not None:
            run = run + 1 if d[t] < delta else 0
            best = max(best, run)
            run_s.append(best)
        else:
            run_s.append(None)
    if delta is not None:
        summary["longest_sub_delta_run"] = _longest_run([bool(x < delta) for x in d])
    series: dict[str, Series] = {
        "displacement_mean": mean_s,
        "displacement_median": median_s,
        "displacement_longest_sub_delta_run": run_s,
    }
    return summary, series


def recurrence_metrics(
    points: Array, epsilon: float | None, w_min: int | None
) -> tuple[dict[str, object], dict[str, Series]]:
    length = points.shape[0]
    empty: Series = [None] * length
    if epsilon is None or w_min is None:
        summary: dict[str, object] = {
            "pair_fraction": None,
            "return_count": None,
            "return_rate": None,
        }
        return summary, {
            "recurrence_pair_fraction": list(empty),
            "recurrence_return_count": list(empty),
            "recurrence_return_rate": list(empty),
        }
    if epsilon <= 0.0 or w_min < 0:
        raise InvalidTrajectory("epsilon must be > 0 and w_min >= 0")
    dist = pairwise_distances(points)
    cum_pairs = cum_hits = cum_returns = 0
    frac_s: Series = []
    count_s: Series = []
    rate_s: Series = []
    for j in range(length):
        eligible = max(0, j - w_min)  # i in [0, eligible): j - i > w_min
        if eligible:
            close = dist[j, :eligible] < epsilon
            cum_pairs += eligible
            cum_hits += int(np.sum(close))
            if j >= 1 and bool(np.any(close & (dist[j - 1, :eligible] >= epsilon))):
                cum_returns += 1
        frac_s.append(cum_hits / cum_pairs if cum_pairs else None)
        count_s.append(cum_returns)
        rate_s.append(cum_returns / (j + 1))
    summary = {
        "pair_fraction": frac_s[-1],
        "return_count": cum_returns,
        "return_rate": rate_s[-1],
    }
    return summary, {
        "recurrence_pair_fraction": frac_s,
        "recurrence_return_count": count_s,
        "recurrence_return_rate": rate_s,
    }


def _entropy(counts: dict[int, int], total: int) -> float:
    return float(-sum((c / total) * np.log(c / total) for c in counts.values()))


def diversity_metrics(
    points: Array, codebook: Array | None
) -> tuple[dict[str, object], dict[str, Series]]:
    length, dim = points.shape
    entropy_s: Series = []
    pr_s: Series = []
    counts: dict[int, int] = {}
    cells = assign_cells(points, codebook, None) if codebook is not None else None
    mean = np.zeros(dim)
    scatter = np.zeros((dim, dim))
    for t in range(length):
        if cells is not None:
            counts[cells[t]] = counts.get(cells[t], 0) + 1
            entropy_s.append(_entropy(counts, t + 1))
        else:
            entropy_s.append(None)
        step = points[t] - mean
        mean = mean + step / (t + 1)
        scatter = scatter + np.outer(step, points[t] - mean)
        if t == 0:
            pr_s.append(None)
            continue
        cov = scatter / (t + 1)
        trace = float(np.trace(cov))
        denom = float(np.sum(cov * cov))
        pr_s.append(trace * trace / denom if denom > 0.0 else 0.0)
    summary: dict[str, object] = {
        "cell_entropy": entropy_s[-1],
        "participation_ratio": pr_s[-1],
    }
    return summary, {
        "diversity_cell_entropy": entropy_s,
        "diversity_participation_ratio": pr_s,
    }


def persistence_metrics(
    points: Array, codebook: Array | None, epsilon: float | None
) -> tuple[dict[str, object], dict[str, Series]]:
    _check_points(points)
    length = points.shape[0]
    if codebook is None and epsilon is None:
        summary: dict[str, object] = {"longest_stay": None, "dominant_cell_share": None}
        return summary, {
            "persistence_longest_stay": [None] * length,
            "persistence_dominant_cell_share": [None] * length,
        }
    cells = assign_cells(points, codebook, epsilon)
    counts: dict[int, int] = {}
    run = best = 0
    stay_s: Series = []
    share_s: Series = []
    for t, cell in enumerate(cells):
        run = run + 1 if t and cells[t - 1] == cell else 1
        best = max(best, run)
        counts[cell] = counts.get(cell, 0) + 1
        stay_s.append(best)
        share_s.append(max(counts.values()) / (t + 1))
    summary = {"longest_stay": best, "dominant_cell_share": share_s[-1]}
    return summary, {
        "persistence_longest_stay": stay_s,
        "persistence_dominant_cell_share": share_s,
    }


def mean_step_distance(left: Array, right: Array) -> float:
    """Mean per-timestep L2 over the common prefix (the mlba-core d_manifold form)."""
    n = min(left.shape[0], right.shape[0])
    if n < 1 or left.shape[1] != right.shape[1]:
        raise InvalidTrajectory("need non-empty trajectories of equal dimension")
    diff = left[:n] - right[:n]
    return float(np.mean(np.sqrt(np.sum(diff * diff, axis=1))))


__all__ = [
    "assign_cells",
    "displacement_metrics",
    "diversity_metrics",
    "mean_step_distance",
    "pairwise_distances",
    "persistence_metrics",
    "recurrence_metrics",
    "step_displacements",
]
