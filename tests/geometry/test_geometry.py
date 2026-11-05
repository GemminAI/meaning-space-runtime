from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import msr.geometry as g
from msr.geometry.calibration import Calibration
from msr.geometry.errors import (
    CalibrationError,
    CrossSpaceComparison,
    InvalidTrajectory,
)
from msr.geometry.metrics import (
    assign_cells,
    displacement_metrics,
    mean_step_distance,
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
        "measured_at": "2026-11-05",
        "noise_floor_lower": 2.0,
        "noise_floor_upper": 5.0,
        "background_mean": 3.0,
        "epsilon": 0.5,
        "delta": 0.1,
        "w_min": 1,
    }
    base.update(kw)
    return Calibration(**base)


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


def test_uncalibrated_without_file(tmp_path: Path) -> None:
    res = g.analyze_trajectory(traj(walk(20)), data_dir=tmp_path)
    assert res["verdict"]["state"] == "UNCALIBRATED"
    assert res["calibration"] == {"model_build_id": None, "sha256": None}
    assert res["metrics"]["displacement"]["mean"] is not None


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (1.99, "STAGNANT"),
        (2.0, "NOISE_FLOOR_INTERNAL"),
        (5.0, "NOISE_FLOOR_INTERNAL"),
        (5.01, "STRUCTURAL_SHIFT_DETECTED"),
    ],
)
def test_verdict_boundaries(d: float, expected: str) -> None:
    assert g.evaluate_shift(d, calib()) == expected
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
    for d in (0.0, 1.999, 2.0, 3.5, 5.0, 5.0001, 100.0):
        assert g.evaluate_shift(d, calib()) == mod.evaluate_manifold_shift(d, ref)
    assert g.evaluate_shift(1.0, None) == mod.evaluate_manifold_shift(1.0, None)


def test_build_write_load_roundtrip_and_analyze(tmp_path: Path) -> None:
    c = g.build_calibration(
        NULLS,
        w_min=1,
        measured_at="2026-11-05T00:00:00+00:00",
        codebook_ref={"id": "cb", "sha256": None},
    )
    assert c.epsilon is not None and c.epsilon > 0
    assert c.noise_floor_lower <= c.noise_floor_upper
    assert c.provenance["method_version"]
    assert len(c.provenance["input_sha256"]) == 5
    assert c.sample_config == {"trajectories": 5, "points": 40, "dims": 3}
    path = g.write_calibration(tmp_path, c)
    assert path == tmp_path / "calibration" / "build-1.json"
    loaded = g.load_calibration(tmp_path, "build-1")
    assert loaded is not None and loaded.calibration == c
    res = g.analyze_trajectory(NULLS[0], data_dir=tmp_path)
    assert res["calibration"]["model_build_id"] == "build-1"
    assert res["calibration"]["sha256"] == loaded.sha256
    assert res["thresholds"]["epsilon"] == c.epsilon
    assert res["verdict"]["state"] in {
        "STAGNANT",
        "NOISE_FLOOR_INTERNAL",
        "STRUCTURAL_SHIFT_DETECTED",
    }
    big = g.analyze_trajectory(traj(walk(40, 3) * 50.0), data_dir=tmp_path)
    assert big["verdict"]["state"] == "STRUCTURAL_SHIFT_DETECTED"
    # explicit thresholds override the file
    over = g.analyze_trajectory(
        NULLS[0], data_dir=tmp_path, epsilon=9.0, delta=9.0, w_min=0
    )
    assert over["thresholds"] == {"epsilon": 9.0, "delta": 9.0, "w_min": 0}
    one = g.analyze_trajectory(traj(np.zeros((1, 3))), data_dir=tmp_path)
    assert one["verdict"]["state"] is None


def test_build_calibration_defaults_and_errors() -> None:
    assert g.build_calibration(NULLS).measured_at
    with pytest.raises(CalibrationError):
        g.build_calibration(NULLS[:1])
    with pytest.raises(CalibrationError):
        g.build_calibration([NULLS[0], traj(walk(40, 3), space="other")])
    with pytest.raises(CalibrationError):
        g.build_calibration([NULLS[0], traj(walk(40, 5))])
    with pytest.raises(CalibrationError):
        g.build_calibration(NULLS, w_min=-1)
    with pytest.raises(CalibrationError):
        g.build_calibration([traj(np.zeros((1, 2))), traj(np.zeros((1, 2)))])


def test_calibration_file_errors(tmp_path: Path) -> None:
    d = tmp_path / "calibration"
    d.mkdir()
    (d / "b1.json").write_text("{}")
    with pytest.raises(CalibrationError):
        g.load_calibration(tmp_path, "b1")
    (d / "b2.json").write_text(json.dumps(calib(model_build_id="other").to_json_dict()))
    with pytest.raises(CalibrationError):
        g.load_calibration(tmp_path, "b2")
    # SPEC section 3.1 file without space_id is rejected (space_id is required)
    legacy = {k: v for k, v in calib().to_json_dict().items() if k != "space_id"}
    (d / "build-1.json").write_text(json.dumps(legacy))
    with pytest.raises(CalibrationError):
        g.analyze_trajectory(traj(walk(10)), data_dir=tmp_path)


def test_calibration_space_and_codebook_binding(tmp_path: Path) -> None:
    cb = np.random.default_rng(2).normal(size=(4, 3))
    c = calib(codebook_ref={"id": "cb", "sha256": g.codebook_sha256(cb)})
    g.write_calibration(tmp_path, c)
    assert g.analyze_trajectory(traj(walk(20, 3)), data_dir=tmp_path, codebook=cb)
    with pytest.raises(CalibrationError):
        g.analyze_trajectory(traj(walk(20, 3)), data_dir=tmp_path, codebook=cb + 1)
    with pytest.raises(CalibrationError):
        g.analyze_trajectory(traj(walk(20, 3), space="S2"), data_dir=tmp_path)


# ------------------------------------------------------- space / determinism


def test_cross_space_comparison_refused() -> None:
    a, b = traj(walk(10)), traj(walk(10, seed=3), space="S2")
    with pytest.raises(CrossSpaceComparison):
        g.compare_trajectories(a, b)
    res = g.compare_trajectories(a, b, allow_cross_space=True)
    assert res["verdict"] == "UNCALIBRATED"
    same = g.compare_trajectories(a, traj(walk(10, seed=3)), calib=calib())
    assert same["distance"] > 0
    assert same["verdict"] in {
        "STAGNANT",
        "NOISE_FLOOR_INTERNAL",
        "STRUCTURAL_SHIFT_DETECTED",
    }
    assert g.compare_trajectories(a, a)["distance"] == 0.0


def test_deterministic_bytes_and_no_meaning_words(tmp_path: Path) -> None:
    g.write_calibration(tmp_path, g.build_calibration(NULLS, measured_at="t"))
    cb = np.random.default_rng(5).normal(size=(6, 3))
    runs = [
        g.to_json_bytes(g.analyze_trajectory(NULLS[1], data_dir=tmp_path, codebook=cb))
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    out = json.loads(runs[0])
    assert out["producer"]["name"] == "meaning-space-runtime"
    text = runs[0].decode().lower()
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
