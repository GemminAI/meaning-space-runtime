from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import msr.geometry as g
from msr.geometry.analyze import VERDICT_REASON
from msr.geometry.calibration import Calibration, CalibrationArtifact
from msr.geometry.errors import (
    CalibrationError,
    CrossSpaceComparison,
    InvalidTrajectory,
)
from msr.geometry.metrics import (
    assign_cells,
    displacement_metrics,
    mean_step_distance,
    pairwise_distances,
    persistence_metrics,
    recurrence_metrics,
)

SHA = "a" * 64
POLICY = g.Policy("last-token", SHA)


def traj(
    points: object,
    *,
    space: str = "S1",
    source: str = "T-hid",
    layer: int | None = 12,
    build: str = "build-1",
) -> g.TrajectoryInput:
    arr = np.asarray(points, dtype=np.float64)
    return g.TrajectoryInput(
        space_id=space,
        source=source,  # type: ignore[arg-type]
        layer=layer if source == "T-hid" else None,
        points=arr,
        token_index=tuple(range(arr.shape[0])),
        policy=POLICY,
        model_build_id=build,
    )


def walk(n: int = 200, d: int = 4, seed: int = 0) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(size=(n, d)), axis=0)


def calib(**kw: Any) -> Calibration:
    base: dict[str, Any] = {
        "model_build_id": "build-1",
        "space_id": "S1",
        "noise_floor_lower": 2.0,
        "noise_floor_upper": 5.0,
        "background_mean": 3.0,
        "epsilon": 0.5,
        "delta": 0.1,
        "w_min": 1,
    }
    base.update(kw)
    return Calibration(**base)


def ref(i: int) -> g.ArtifactRef:
    return g.ArtifactRef(f"art:{i:064x}", f"cnt:{i + 1000:064x}")


def ctx(seq: int = 1, **kw: Any) -> g.EmitContext:
    return g.EmitContext(
        "ses:" + "1" * 64,
        "obs:" + "2" * 64,
        seq,
        producer_build="test-build",
        produced_at="2026-11-06T00:00:00Z",
        **kw,
    )


def fake_artifact(c: Calibration) -> CalibrationArtifact:
    """A calibration artifact around given float values (ids are placeholders)."""
    return CalibrationArtifact(
        {"artifact_id": ref(900).artifact_id, "content_id": ref(900).content_id}, c
    )


# ------------------------------------------------------------------ input


def test_input_validation() -> None:
    with pytest.raises(InvalidTrajectory):
        g.Policy("", SHA)
    with pytest.raises(InvalidTrajectory):
        g.Policy("p", "xyz")
    pts = np.zeros((3, 2))
    ok = {"points": pts, "token_index": (0, 1, 2), "policy": POLICY}
    for bad in (
        {"space_id": "", "source": "T-tok", "model_build_id": "m"},
        {"space_id": "s", "source": "T-tok", "model_build_id": ""},
        {"space_id": "s", "source": "bogus", "model_build_id": "m"},
        {"space_id": "s", "source": "T-hid", "model_build_id": "m"},
        {"space_id": "s", "source": "T-tok", "model_build_id": "m", "layer": 3},
        {
            "space_id": "s",
            "source": "T-tok",
            "model_build_id": "m",
            "points": np.zeros(3),
        },
        {
            "space_id": "s",
            "source": "T-tok",
            "model_build_id": "m",
            "points": np.zeros((0, 2)),
        },
        {
            "space_id": "s",
            "source": "T-tok",
            "model_build_id": "m",
            "points": np.full((3, 2), np.nan),
        },
        {
            "space_id": "s",
            "source": "T-tok",
            "model_build_id": "m",
            "token_index": (0,),
        },
    ):
        with pytest.raises(InvalidTrajectory):
            g.TrajectoryInput(**{**ok, **bad})  # type: ignore[arg-type]


def test_input_is_copied_readonly_and_hash_stable() -> None:
    src = np.arange(6.0).reshape(3, 2)
    t = traj(src)
    src[0, 0] = 99
    assert t.points[0, 0] == 0.0
    assert not t.points.flags.writeable
    assert (t.length, t.dimension) == (3, 2)
    assert t.sha256() == traj(np.arange(6.0).reshape(3, 2)).sha256()
    assert t.sha256() != traj(np.arange(6.0).reshape(3, 2), layer=13).sha256()


# ------------------------------------------------------- positive controls


def test_repeated_point_raises_recurrence_and_persistence_zero_displacement() -> None:
    rep = traj(np.ones((50, 3)))
    rw = traj(walk(50, 3))
    a = g.analyze_trajectory(rep, epsilon=0.5, delta=0.1, w_min=1)
    b = g.analyze_trajectory(rw, epsilon=0.5, delta=0.1, w_min=1)
    assert a["metrics"]["displacement"]["mean"] == 0.0
    assert a["metrics"]["recurrence"]["pair_fraction"] == 1.0
    assert b["metrics"]["recurrence"]["pair_fraction"] < 0.5
    assert a["metrics"]["persistence"]["longest_stay"] == 50
    assert a["metrics"]["persistence"]["dominant_cell_share"] == 1.0
    assert b["metrics"]["persistence"]["longest_stay"] < 50
    assert a["metrics"]["displacement"]["longest_sub_delta_run"] == 49


def test_random_walk_has_higher_diversity_than_repeats() -> None:
    codebook = np.random.default_rng(1).normal(size=(8, 3)) * 5
    rep = g.analyze_trajectory(traj(np.ones((200, 3))), codebook=codebook)
    rw = g.analyze_trajectory(traj(walk(200, 3)), codebook=codebook)
    assert rep["metrics"]["diversity"]["cell_entropy"] == 0.0
    assert rep["metrics"]["diversity"]["participation_ratio"] == 0.0
    assert rw["metrics"]["diversity"]["cell_entropy"] > 0.5
    assert rw["metrics"]["diversity"]["participation_ratio"] > 1.0


def _cycle(cycles: int) -> np.ndarray:
    a = np.zeros((3, 2))
    b = np.full((3, 2), 10.0)
    return np.vstack([a, b] * cycles)


def test_periodic_return_count_is_proportional_to_cycles() -> None:
    counts = []
    for cycles in (2, 4, 8):
        res = g.analyze_trajectory(
            traj(_cycle(cycles)), epsilon=1.0, delta=0.1, w_min=1
        )
        counts.append(res["metrics"]["recurrence"]["return_count"])
    assert counts == [2, 6, 14]  # 2 * (cycles - 1): one return per region per repeat
    assert (counts[2] - counts[1]) == 2 * (counts[1] - counts[0])


def test_series_are_prefix_values() -> None:
    res = g.analyze_trajectory(traj(walk(30, 3)), epsilon=3.0, delta=0.5, w_min=2)
    for key, values in res["series"].items():
        assert len(values) == 30, key
    prefix = g.analyze_trajectory(
        traj(walk(30, 3)[:12]), epsilon=3.0, delta=0.5, w_min=2
    )
    for key in (
        "recurrence_pair_fraction",
        "displacement_mean",
        "persistence_longest_stay",
        "diversity_participation_ratio",
        "recurrence_return_count",
    ):
        assert res["series"][key][11] == prefix["series"][key][-1], key
    assert res["series"]["recurrence_pair_fraction"][0] is None


def test_metrics_edge_cases() -> None:
    one = traj(np.zeros((1, 2)))
    res = g.analyze_trajectory(one, epsilon=1.0, delta=1.0, w_min=0)
    assert res["metrics"]["displacement"]["mean"] is None
    assert res["metrics"]["displacement"]["median"] is None
    assert res["metrics"]["recurrence"]["pair_fraction"] is None
    # threshold-free values are returned without thresholds; thresholded are None
    bare = g.analyze_trajectory(traj(walk(10, 2)))
    assert bare["metrics"]["displacement"]["mean"] is not None
    assert bare["metrics"]["displacement"]["longest_sub_delta_run"] is None
    assert bare["metrics"]["recurrence"]["return_count"] is None
    assert bare["metrics"]["persistence"]["longest_stay"] is None
    assert bare["metrics"]["diversity"]["cell_entropy"] is None
    assert bare["series"]["displacement_longest_sub_delta_run"][3] is None
    with pytest.raises(InvalidTrajectory):
        recurrence_metrics(np.zeros((3, 2)), 0.0, 1)
    with pytest.raises(InvalidTrajectory):
        recurrence_metrics(np.zeros((3, 2)), 1.0, -1)
    with pytest.raises(InvalidTrajectory):
        assign_cells(np.zeros((3, 2)), None, None)
    with pytest.raises(InvalidTrajectory):
        assign_cells(np.zeros((3, 2)), np.zeros((2, 5)), None)
    with pytest.raises(InvalidTrajectory):
        assign_cells(np.zeros((3, 2)), np.zeros((0, 2)), None)
    with pytest.raises(InvalidTrajectory):
        persistence_metrics(np.zeros((0, 2)), None, 1.0)
    with pytest.raises(InvalidTrajectory):
        mean_step_distance(np.zeros((2, 2)), np.zeros((2, 3)))
    summary, _ = displacement_metrics(np.zeros((1, 2)), 1.0)
    assert summary["longest_sub_delta_run"] == 0


def test_sub_delta_run_resets() -> None:
    pts = np.array([[0.0], [0.0], [5.0], [5.0], [5.0], [5.0], [9.0]])
    s, series = displacement_metrics(pts, 0.5)
    assert s["longest_sub_delta_run"] == 3
    assert series["displacement_longest_sub_delta_run"][-1] == 3


# ------------------------------------------------------------ calibration

NULLS = [traj(walk(40, 3, seed=s) * 0.1) for s in range(5)]


def test_uncalibrated_without_calibration() -> None:
    res = g.analyze_trajectory(traj(walk(20)))
    assert res["verdict"] == {"state": None, "reason": VERDICT_REASON}
    assert res["calibration"] is None
    assert res["thresholds"]["source"] == {
        "epsilon": None,
        "delta": None,
        "w_min": None,
    }
    assert res["metrics"]["displacement"]["mean"] is not None


# SPEC-MSR-001 v2 section 3.2, floor 2.0 / ceiling 5.0 -- a fixed table that
# does not depend on any neighbour repository being present.
VERDICT_TABLE = [
    (0.0, "STAGNANT"),
    (1.999999, "STAGNANT"),  # below the lower bound
    (2.0, "NOISE_FLOOR_INTERNAL"),  # exactly the lower bound
    (3.5, "NOISE_FLOOR_INTERNAL"),
    (5.0, "NOISE_FLOOR_INTERNAL"),  # exactly the upper bound
    (5.000001, "STRUCTURAL_SHIFT_DETECTED"),  # above the upper bound
    (100.0, "STRUCTURAL_SHIFT_DETECTED"),
]


@pytest.mark.parametrize(("d", "expected"), VERDICT_TABLE)
def test_verdict_boundaries(d: float, expected: str) -> None:
    assert g.evaluate_shift(d, calib()) == expected


@pytest.mark.parametrize("d", [0.0, 3.5, 100.0])
def test_verdict_without_calibration(d: float) -> None:
    assert g.evaluate_shift(d, None) == "UNCALIBRATED"


def test_verdict_matches_sensos_core() -> None:
    path = Path.home() / "GemminAI/sensos-core/sensos_core/manifold_shift.py"
    if not path.exists():
        pytest.skip("sensos-core not present")
    spec = importlib.util.spec_from_file_location("_ms_ref", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["_ms_ref"] = mod  # dataclass needs the module registered
    spec.loader.exec_module(mod)
    ref = mod.Calibration(
        model_build_id="b",
        measured_at="t",
        noise_floor_lower=2.0,
        noise_floor_upper=5.0,
        background_mean=3.0,
        sample_config={},
    )
    for d, _ in VERDICT_TABLE:
        assert g.evaluate_shift(d, calib()) == mod.evaluate_manifold_shift(d, ref)
    assert g.evaluate_shift(1.0, None) == mod.evaluate_manifold_shift(1.0, None)


def test_build_write_load_roundtrip_and_analyze(tmp_path: Path) -> None:
    members = [g.NullMember(t, ref(i)) for i, t in enumerate(NULLS)]
    art = g.build_calibration(members, ctx(2), w_min=1, codebook_ref=ref(77))
    c = art.calibration
    assert c.epsilon is not None and c.epsilon > 0
    assert c.noise_floor_lower <= c.noise_floor_upper
    assert art.payload["sample_config"] == {"trajectories": 5, "points": 40, "dims": 3}
    assert c.codebook_ref == ref(77)
    path = g.write_calibration(tmp_path, art)
    assert path == tmp_path / "calibration" / f"{art.ref.artifact_id}.json"
    loaded = g.load_calibration(tmp_path, art.ref.artifact_id)
    assert loaded.calibration == c and loaded.envelope == art.envelope
    res = g.analyze_trajectory(NULLS[0], calibration=loaded)
    assert res["calibration"] == art.ref.to_dict()
    assert res["thresholds"]["epsilon"] == c.epsilon
    assert res["verdict"]["state"] is None  # analyze never emits a shift state
    assert res["thresholds"]["source"] == {
        "epsilon": "calibration",
        "delta": "calibration",
        "w_min": "calibration",
    }
    big = g.compare_trajectories(
        NULLS[0], traj(NULLS[0].points + 50.0), calibration=loaded
    )
    assert big["verdict"]["state"] == "STRUCTURAL_SHIFT_DETECTED"
    assert big["calibration"] == art.ref.to_dict()
    with pytest.raises(CalibrationError):  # calibration is for another space
        g.compare_trajectories(
            traj(walk(40, 3), space="S2"),
            traj(walk(40, 3), space="S2"),
            calibration=loaded,
        )
    with pytest.raises(CalibrationError):
        g.analyze_trajectory(traj(walk(20, 3), space="S2"), calibration=loaded)
    # explicit thresholds override the calibration
    over = g.analyze_trajectory(
        NULLS[0], calibration=loaded, epsilon=9.0, delta=9.0, w_min=0
    )
    assert over["thresholds"] == {
        "epsilon": 9.0,
        "delta": 9.0,
        "w_min": 0,
        "source": {"epsilon": "explicit", "delta": "explicit", "w_min": "explicit"},
    }


def test_build_calibration_errors() -> None:
    def mem(trajs: list[g.TrajectoryInput]) -> list[g.NullMember]:
        return [g.NullMember(t, ref(i)) for i, t in enumerate(trajs)]

    with pytest.raises(CalibrationError):
        g.build_calibration(mem(NULLS[:1]), ctx())
    with pytest.raises(CalibrationError):
        g.build_calibration(mem([NULLS[0], traj(walk(40, 3), space="other")]), ctx())
    with pytest.raises(CalibrationError):
        g.build_calibration(mem([NULLS[0], traj(walk(40, 3), build="b2")]), ctx())
    with pytest.raises(CalibrationError):
        g.build_calibration(mem([NULLS[0], traj(walk(40, 5))]), ctx())
    with pytest.raises(CalibrationError):
        g.build_calibration(mem(NULLS), ctx(), w_min=-1)
    with pytest.raises(CalibrationError):
        g.build_calibration(
            mem([traj(np.zeros((1, 2))), traj(np.zeros((1, 2)))]), ctx()
        )


# ------------------------------------------------------- space / determinism


def test_cross_space_comparison_refused() -> None:
    a, b = traj(walk(10)), traj(walk(10, seed=3), space="S2")
    with pytest.raises(CrossSpaceComparison):
        g.compare_trajectories(a, b)
    res = g.compare_trajectories(a, b, allow_cross_space=True)
    assert res["verdict"]["state"] == "UNCALIBRATED"
    assert res["calibration"] is None


def test_compare_output_and_alignment() -> None:
    a, b = traj(walk(10)), traj(walk(10, seed=3))
    loaded = fake_artifact(calib())
    same = g.compare_trajectories(a, b, calibration=loaded)
    assert same["distance"]["definition"] == "mean_step_l2"
    assert same["distance"]["value"] == pytest.approx(
        float(np.mean(np.linalg.norm(a.points - b.points, axis=1)))
    )
    assert same["alignment"] == {"mode": "equal_only", "points_used": 10}
    assert same["inputs"]["left"]["sha256"] == a.sha256()
    assert same["inputs"]["right"]["policy"] == {"name": "last-token", "sha256": SHA}
    assert same["inputs"]["right"]["model_build_id"] == "build-1"
    assert same["calibration"] == loaded.ref.to_dict()
    assert g.compare_trajectories(a, a)["distance"]["value"] == 0.0
    short = traj(walk(6, seed=3))
    with pytest.raises(InvalidTrajectory):
        g.compare_trajectories(a, short)
    with pytest.raises(InvalidTrajectory):
        g.compare_trajectories(a, short, alignment="bogus")  # type: ignore[arg-type]
    prefix = g.compare_trajectories(a, short, alignment="common_prefix")
    assert prefix["alignment"] == {"mode": "common_prefix", "points_used": 6}
    assert prefix["distance"]["value"] == pytest.approx(
        float(np.mean(np.linalg.norm(a.points[:6] - short.points, axis=1)))
    )


# ------------------------------------------------------------ block-wise


@pytest.mark.parametrize("w_min", [0, 1, 7])
def test_recurrence_is_block_size_invariant_and_matches_full_matrix(
    w_min: int,
) -> None:
    pts = np.random.default_rng(11).normal(size=(103, 5))
    eps = 3.0
    dist = pairwise_distances(pts)
    pairs = hits = ret = 0
    fracs: list[float | None] = []
    for j in range(len(pts)):
        k = max(0, j - w_min)
        if k:
            close = dist[j, :k] < eps
            pairs += k
            hits += int(close.sum())
            if j and (close & (dist[j - 1, :k] >= eps)).any():
                ret += 1
        fracs.append(hits / pairs if pairs else None)
    base = recurrence_metrics(pts, eps, w_min, block_size=1000)
    assert base[0]["return_count"] == ret
    assert base[1]["recurrence_pair_fraction"] == fracs
    for block in (1, 2, 7, 64, 102, 103, 104):
        assert recurrence_metrics(pts, eps, w_min, block_size=block) == base
    with pytest.raises(InvalidTrajectory):
        recurrence_metrics(pts, eps, w_min, block_size=0)


def test_analyze_passes_block_size() -> None:
    t = traj(walk(40, 3))
    a = g.analyze_trajectory(t, epsilon=3.0, delta=1.0, w_min=1, block_size=5)
    b = g.analyze_trajectory(t, epsilon=3.0, delta=1.0, w_min=1)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_deterministic_and_no_meaning_words() -> None:
    cb = np.random.default_rng(5).normal(size=(6, 3))
    runs = [
        json.dumps(g.analyze_trajectory(NULLS[1], codebook=cb), sort_keys=True)
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    text = runs[0].lower()
    for word in ("loop", "anomaly", "hallucination", '"p"', '"l"', '"u"'):
        assert word not in text


# ---------------------------------------------------------------- boundary

FORBIDDEN = (
    "meaning_mapper",
    "hekb",
    "requests",
    "httpx",
    "urllib",
    "http",
    "socket",
    "aiohttp",
    "sensos",
    "sensos_core",
)


def test_geometry_imports_no_mapper_hekb_or_http() -> None:
    root = Path(g.__file__).parent
    seen = 0
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                seen += 1
                assert name.split(".")[0] not in FORBIDDEN, (path.name, name)
    assert seen > 10
