#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable

from step1_index import (
    RunPaths,
    VisualizationConfig,
    default_cache_dir,
    distance_key_candidates,
    load_config,
    save_pickle,
    write_stage_metadata,
)

try:
    from shapely import LineString, frechet_distance, hausdorff_distance
except ImportError:  # pragma: no cover - optional dependency
    LineString = None
    frechet_distance = None
    hausdorff_distance = None

PAIR_TO_FIELDS = {
    "fp_lrp": ("fp", "lrp"),
    "fp_lrpFMM": ("fp", "lrp_fmm"),
    "fp_fpDijkstra": ("fp", "fp_dijkstra"),
    "fp_fpFMM": ("fp", "fp_fmm"),
    "fp_fpFSM": ("fp", "fp_fsm"),
}
DISTANCE_TYPES = ("end_point", "hausdorff", "frechet")
VARIANCE_METRIC_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "len_fp": ("len_fp", "len_fp_rk45", "len_fp_m"),
    "len_fp_dijkstra": ("len_fp_dijkstra", "len_fp_dijkstra_m"),
    "len_fp_fmm": ("len_fp_fmm", "len_fp_fmm_m"),
    "len_fp_fsm": ("len_fp_fsm",),
    "len_lrp": ("len_lrp", "len_lrp_m"),
    "len_lrp_fmm": ("len_lrp_fmm", "len_lrp_fmm_m"),
    "travel_time_fp": ("travel_time_fp", "travel_time_rk45", "fpt_m"),
    "travel_time_fp_dijkstra": ("travel_time_dijkstra", "fpt_dijkstra_m"),
    "travel_time_fp_fmm": ("travel_time_fmm", "fpt_fmm_m"),
    "travel_time_fp_fsm": ("travel_time_fsm",),
    "resistance_lrp": ("resistance_lrp", "mhr_m"),
    "resistance_lrp_fmm": ("resistance_lrp_fmm", "mhr_fmm_m"),
}


@dataclass(slots=True)
class PathSet:
    fp: np.ndarray | None = None
    lrp: np.ndarray | None = None
    lrp_fmm: np.ndarray | None = None
    fp_dijkstra: np.ndarray | None = None
    fp_fmm: np.ndarray | None = None
    fp_fsm: np.ndarray | None = None


@dataclass(slots=True)
class SummaryStats:
    mean: float
    std: float
    count: int


@dataclass(slots=True)
class VarianceResults:
    field_size: np.ndarray
    data: dict[str, Any]
    stats: dict[str, Any]
    line_integrals: dict[str, Any]


@dataclass(slots=True)
class StdResults:
    data_path: dict[str, Any]
    stats_path: dict[str, Any]
    stats_std: dict[str, np.ndarray]


def _serialize_variance_results(results: VarianceResults) -> dict[str, Any]:
    return {
        "field_size": results.field_size,
        "data": results.data,
        "stats": results.stats,
        "line_integrals": results.line_integrals,
    }


def _serialize_std_results(results: StdResults | None) -> dict[str, Any] | None:
    if results is None:
        return None
    return {
        "data_path": results.data_path,
        "stats_path": results.stats_path,
        "stats_std": results.stats_std,
    }


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def runpaths_from_dict(raw: dict[str, Any]) -> RunPaths:
    return RunPaths(
        size_dir=Path(raw["size_dir"]),
        size_value=float(raw["size_value"]),
        size_str=str(raw["size_str"]),
        mc_files=[Path(p) for p in raw.get("mc_files", [])],
        main_output_files=[Path(p) for p in raw.get("main_output_files", [])],
        aggregate_output_file=Path(raw["aggregate_output_file"]) if raw.get("aggregate_output_file") else None,
        fp_files=[Path(p) for p in raw.get("fp_files", [])],
        lrp_files=[Path(p) for p in raw.get("lrp_files", [])],
        lrp_fmm_files=[Path(p) for p in raw.get("lrp_fmm_files", [])],
        fp_dijkstra_files=[Path(p) for p in raw.get("fp_dijkstra_files", [])],
        fp_fmm_files=[Path(p) for p in raw.get("fp_fmm_files", [])],
        fp_fsm_files=[Path(p) for p in raw.get("fp_fsm_files", [])],
        logk_files=[Path(p) for p in raw.get("logk_files", [])],
    )


def deserialize_discovered(raw: dict[float, list[dict[str, Any]]]) -> dict[float, list[RunPaths]]:
    return {float(k): [runpaths_from_dict(rp) for rp in v] for k, v in raw.items()}


def load_mc_consolidated(npy_path: Path) -> dict[str, Any]:
    return np.load(npy_path, allow_pickle=True).item()


def _consolidated_to_output_data(mc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    paths = mc.get("paths", {})
    for key in ("fp", "fp_dijkstra", "fp_fmm"):
        p = paths.get(key, {})
        out[f"len_{key}"] = p.get("length", np.nan)
        out[f"travel_time_{key}".replace("travel_time_fp_", "travel_time_")] = p.get("travel_time", np.nan)
    out["travel_time_fp"] = paths.get("fp", {}).get("travel_time", np.nan)
    out["travel_time_dijkstra"] = paths.get("fp_dijkstra", {}).get("travel_time", np.nan)
    out["travel_time_fmm"] = paths.get("fp_fmm", {}).get("travel_time", np.nan)
    out["len_fp_fsm"] = paths.get("fp_fsm", {}).get("length", np.nan)
    out["travel_time_fsm"] = paths.get("fp_fsm", {}).get("travel_time", np.nan)
    for key in ("lrp", "lrp_fmm"):
        p = paths.get(key, {})
        out[f"len_{key}"] = p.get("length", np.nan)
        out[f"resistance_{key}"] = p.get("resistance", np.nan)
    endpoint = mc.get("endpoint", {})
    dy = endpoint.get("dy", {})
    out["endpoint_dy_fp_vs_lrp"] = dy.get("fp_vs_lrp", np.nan)
    out["endpoint_dy_fp_vs_lrp_fmm"] = dy.get("fp_vs_lrp_fmm", np.nan)
    out["endpoint_dy_fp_vs_dijkstra"] = dy.get("fp_vs_dijkstra", np.nan)
    out["endpoint_dy_fp_vs_fmm"] = dy.get("fp_vs_fmm", np.nan)
    out["endpoint_dy_fp_vs_fsm"] = dy.get("fp_vs_fsm", np.nan)
    renyi = mc.get("renyi_entropy", {})
    out["renyi_entropy_S1"] = np.asarray(renyi.get("S1", [np.nan]), dtype=float)
    out["renyi_entropy_S2"] = np.asarray(renyi.get("S2", [np.nan]), dtype=float)
    return out


def _consolidated_to_pathset(mc: dict[str, Any]) -> PathSet:
    paths = mc.get("paths", {})
    return PathSet(
        fp=paths.get("fp", {}).get("coords"),
        lrp=paths.get("lrp", {}).get("coords"),
        lrp_fmm=paths.get("lrp_fmm", {}).get("coords"),
        fp_dijkstra=paths.get("fp_dijkstra", {}).get("coords"),
        fp_fmm=paths.get("fp_fmm", {}).get("coords"),
        fp_fsm=paths.get("fp_fsm", {}).get("coords"),
    )


def _consolidated_to_logk(mc: dict[str, Any]) -> np.ndarray | None:
    return mc.get("fields", {}).get("logk")


def load_main_output(npz_path: Path) -> dict[str, Any]:
    try:
        return _load_main_output_impl(npz_path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        return _load_main_output_impl(npz_path, allow_pickle=True)


def _normalize_npz_value(value: np.ndarray) -> Any:
    arr = np.asarray(value)
    if arr.size != 1:
        return arr
    scalar = arr.item()
    return np.nan if scalar is None else scalar


def _load_main_output_impl(npz_path: Path, *, allow_pickle: bool) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=allow_pickle) as data:
        return {k: _normalize_npz_value(data[k]) for k in data.files}


def _aggregate_mc_count(output_data: dict[str, Any]) -> int:
    for value in output_data.values():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.size > 1:
            return int(arr.shape[0])
    return 0


def _slice_aggregate_output_data(output_data: dict[str, Any], mc_idx: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in output_data.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.size > 1 and mc_idx < arr.shape[0]:
            item = arr[mc_idx]
            if np.asarray(item).size == 1:
                scalar = np.asarray(item).item()
                sliced[key] = np.nan if scalar is None else scalar
            else:
                sliced[key] = item
        else:
            sliced[key] = value
    return sliced


def _safe_load(paths: list[Path], idx: int) -> np.ndarray | None:
    if paths and idx < len(paths):
        return np.load(paths[idx])
    return None


def load_paths(run_paths: RunPaths, mc_idx: int) -> PathSet:
    return PathSet(
        fp=_safe_load(run_paths.fp_files, mc_idx),
        lrp=_safe_load(run_paths.lrp_files, mc_idx),
        lrp_fmm=_safe_load(run_paths.lrp_fmm_files, mc_idx),
        fp_dijkstra=_safe_load(run_paths.fp_dijkstra_files, mc_idx),
        fp_fmm=_safe_load(run_paths.fp_fmm_files, mc_idx),
        fp_fsm=_safe_load(run_paths.fp_fsm_files, mc_idx),
    )


def load_logk_field(run_paths: RunPaths, mc_idx: int) -> np.ndarray | None:
    if run_paths.logk_files and mc_idx < len(run_paths.logk_files):
        return np.load(run_paths.logk_files[mc_idx])
    return None


def resample_by_arclength(points: np.ndarray, ds: float = 1.0) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] == 0:
        return points[[0]].copy()
    s_new = np.arange(0, s[-1] + 1e-9, ds)
    px = np.interp(s_new, s, points[:, 0])
    py = np.interp(s_new, s, points[:, 1])
    return np.column_stack([px, py])


def directed_nn_distances(a: np.ndarray, b: np.ndarray, workers: int = -1) -> np.ndarray:
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1, workers=workers)
    return d


def distance_distribution(a: np.ndarray, b: np.ndarray, ds: float = 1.0, workers: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    au = resample_by_arclength(a, ds=ds)
    bu = resample_by_arclength(b, ds=ds)
    d_ab = directed_nn_distances(au, bu, workers=workers)
    d_ba = directed_nn_distances(bu, au, workers=workers)
    return np.concatenate([d_ab, d_ba]), d_ab, d_ba


def path_to_linestring(path_arr: np.ndarray | None) -> LineString | None:
    if LineString is None:
        return None
    if path_arr is None or len(path_arr) < 2:
        return None
    return LineString(path_arr.tolist())


def shapely_pair_distances(
    fp: np.ndarray | None,
    lrp: np.ndarray | None,
    lrp_fmm: np.ndarray | None,
    fp_dijkstra: np.ndarray | None,
    fp_fmm: np.ndarray | None,
    fp_fsm: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    out = {
        "hausdorff": {"fp_lrp": np.nan, "fp_lrpFMM": np.nan, "fp_fpDijkstra": np.nan, "fp_fpFMM": np.nan, "fp_fpFSM": np.nan},
        "frechet": {"fp_lrp": np.nan, "fp_lrpFMM": np.nan, "fp_fpDijkstra": np.nan, "fp_fpFMM": np.nan, "fp_fpFSM": np.nan},
    }
    ls_fp = path_to_linestring(fp)
    if ls_fp is None:
        return out
    if hausdorff_distance is None or frechet_distance is None:
        return out
    pairs = [("fp_lrp", lrp), ("fp_lrpFMM", lrp_fmm), ("fp_fpDijkstra", fp_dijkstra), ("fp_fpFMM", fp_fmm), ("fp_fpFSM", fp_fsm)]
    for name, other in pairs:
        ls_other = path_to_linestring(other)
        if ls_other is None:
            continue
        out["hausdorff"][name] = hausdorff_distance(ls_fp, ls_other)
        out["frechet"][name] = frechet_distance(ls_fp, ls_other)
    return out


def endpoint_distance(a: np.ndarray | None, b: np.ndarray | None, *, axis: int = 1, signed: bool = False) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return float("nan")
    d = float(a[-1, axis] - b[-1, axis])
    return d if signed else abs(d)


def _empty_distance_dict() -> dict[str, dict[str, float]]:
    return {
        "end_point": {k: np.nan for k in PAIR_TO_FIELDS},
        "hausdorff": {k: np.nan for k in PAIR_TO_FIELDS},
        "frechet": {k: np.nan for k in PAIR_TO_FIELDS},
    }


def _compute_endpoint_for_pair(pathset: PathSet, pair: str) -> float:
    a_name, b_name = PAIR_TO_FIELDS[pair]
    a = getattr(pathset, a_name)
    b = getattr(pathset, b_name)
    return endpoint_distance(a, b, axis=1, signed=True)


def _compute_hausdorff_nn_for_pair(pathset: PathSet, pair: str, workers: int) -> float:
    a_name, b_name = PAIR_TO_FIELDS[pair]
    a = getattr(pathset, a_name)
    b = getattr(pathset, b_name)
    if a is None or b is None:
        return float("nan")
    d, _, _ = distance_distribution(a, b, ds=1.0, workers=workers)
    if len(d) == 0:
        return float("nan")
    return float(np.max(d))


def _compute_frechet_custom(a: np.ndarray | None, b: np.ndarray | None) -> float:
    # Discrete Frechet distance via dynamic programming.
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return float("nan")

    na = len(a)
    nb = len(b)
    ca = np.full((na, nb), np.nan, dtype=float)

    for i in range(na):
        for j in range(nb):
            d = float(np.linalg.norm(a[i] - b[j]))
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)

    return float(ca[-1, -1])


def _compute_frechet_custom_for_pair(pathset: PathSet, pair: str) -> float:
    a_name, b_name = PAIR_TO_FIELDS[pair]
    a = getattr(pathset, a_name)
    b = getattr(pathset, b_name)
    return _compute_frechet_custom(a, b)


def _load_main_output_value(output_data: dict[str, Any] | None, candidates: list[str] | tuple[str, ...]) -> tuple[float, str | None]:
    if output_data is None:
        return float("nan"), None
    for key in candidates:
        if key in output_data:
            return float(output_data[key]), key
    return float("nan"), None


def _add_unique_warning(warnings: list[str], warning_seen: set[str], message: str) -> None:
    if message not in warning_seen:
        warning_seen.add(message)
        warnings.append(message)


def resolve_distances(
    cfg: VisualizationConfig,
    output_data: dict[str, Any] | None,
    pathset: PathSet | None,
    *,
    workers: int,
    hausdorff_compute_method: str,
    frechet_compute_method: str,
    pathset_loader: Callable[[], PathSet | None] | None = None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    out = _empty_distance_dict()
    warnings: list[str] = []
    shapely_cache: dict[str, dict[str, float]] | None = None
    pathset_cached = pathset

    def _get_pathset() -> PathSet | None:
        nonlocal pathset_cached
        if pathset_cached is None and pathset_loader is not None:
            pathset_cached = pathset_loader()
        return pathset_cached

    def _ensure_shapely() -> dict[str, dict[str, float]]:
        nonlocal shapely_cache
        if shapely_cache is None:
            ps = _get_pathset()
            if ps is None:
                return {
                    "hausdorff": {k: np.nan for k in PAIR_TO_FIELDS},
                    "frechet": {k: np.nan for k in PAIR_TO_FIELDS},
                }
            shapely_cache = shapely_pair_distances(ps.fp, ps.lrp, ps.lrp_fmm, ps.fp_dijkstra, ps.fp_fmm, ps.fp_fsm)
        return shapely_cache

    def _compute_value(distance_type: str, pair: str) -> float:
        ps = _get_pathset()
        if ps is None:
            return float("nan")
        if distance_type == "end_point":
            return _compute_endpoint_for_pair(ps, pair)
        if distance_type == "hausdorff":
            if hausdorff_compute_method == "custom":
                return _compute_hausdorff_nn_for_pair(ps, pair, workers)
            return float(_ensure_shapely()["hausdorff"][pair])
        if distance_type == "frechet":
            if frechet_compute_method == "custom":
                return _compute_frechet_custom_for_pair(ps, pair)
            return float(_ensure_shapely()["frechet"][pair])
        return float("nan")

    for distance_type in DISTANCE_TYPES:
        if not cfg.distance_enabled.get(distance_type, True):
            continue
        mode = cfg.distance_mode[distance_type]
        for pair in PAIR_TO_FIELDS:
            value = float("nan")
            loaded = False
            if mode == "load_from_main_output":
                candidates = distance_key_candidates(cfg, distance_type, pair)
                value, loaded_key = _load_main_output_value(output_data, candidates)
                if loaded_key is not None:
                    loaded = True
                else:
                    joined = ", ".join(candidates)
                    warnings.append(f"Missing mainOutput keys [{joined}] for {distance_type}:{pair}")

            if (not loaded) and (mode == "compute_from_paths" or cfg.distance_fallback_on_missing == "compute_from_paths"):
                value = _compute_value(distance_type, pair)

            out[distance_type][pair] = value

    return out, warnings

def compute_line_integral(path: np.ndarray | None, field: np.ndarray, grid_spacing: float = 1.0) -> tuple[float, float] | tuple[None, None]:
    if path is None or len(path) == 0:
        return None, None

    x_idx = np.round(path[:, 0] / grid_spacing).astype(int)
    y_idx = np.round(path[:, 1] / grid_spacing).astype(int)
    x_idx = np.clip(x_idx, 0, field.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, field.shape[1] - 1)

    values = field[x_idx, y_idx]
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    ds = np.concatenate(([0.0], segment_lengths))

    int_logk = float(np.sum(values * ds))
    int_logk2 = float(np.sum((values ** 2) * ds))
    return int_logk, int_logk2


def compute_mc_line_integrals(pathset: PathSet, field: np.ndarray, grid_spacing: float = 1.0) -> dict[str, tuple[float, float] | tuple[None, None]]:
    return {
        "fp": compute_line_integral(pathset.fp, field, grid_spacing),
        "lrp": compute_line_integral(pathset.lrp, field, grid_spacing),
        "lrp_fmm": compute_line_integral(pathset.lrp_fmm, field, grid_spacing),
        "fp_dijkstra": compute_line_integral(pathset.fp_dijkstra, field, grid_spacing),
        "fp_fmm": compute_line_integral(pathset.fp_fmm, field, grid_spacing),
        "fp_fsm": compute_line_integral(pathset.fp_fsm, field, grid_spacing),
    }


def summarize_integral_values(values: list[float]) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "values": []}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "values": values}


def mean_std(arr: list[float] | np.ndarray, *, filter_positive: bool = False, ignore_nan: bool = True) -> SummaryStats:
    a = np.asarray(arr, dtype=float)
    if filter_positive:
        a = a[a > 0]
    if ignore_nan:
        a = a[~np.isnan(a)]
    if a.size == 0:
        return SummaryStats(mean=float("nan"), std=float("nan"), count=0)
    return SummaryStats(mean=float(np.mean(a)), std=float(np.std(a)), count=int(a.size))


def mean_std_matrix(arr_list: list[np.ndarray], *, filter_positive: bool = False) -> np.ndarray:
    out = []
    for arr in arr_list:
        s = mean_std(arr, filter_positive=filter_positive)
        out.append((s.mean, s.std))
    return np.asarray(out, dtype=float)


def _init_lengths() -> dict[str, list[np.ndarray]]:
    return {"fp": [], "fp_dijkstra": [], "fp_fmm": [], "fp_fsm": [], "lrp": [], "lrp_fmm": []}


def _init_times() -> dict[str, list[np.ndarray]]:
    return {"fpt": [], "fpt_dijkstra": [], "fpt_fmm": [], "fpt_fsm": [], "mhr": [], "mhr_fmm": []}


def _init_distances() -> dict[str, dict[str, list[np.ndarray]]]:
    return {
        "hausdorff": {"fp_lrp": [], "fp_lrpFMM": [], "fp_fpDijkstra": [], "fp_fpFMM": [], "fp_fpFSM": []},
        "frechet": {"fp_lrp": [], "fp_lrpFMM": [], "fp_fpDijkstra": [], "fp_fpFMM": [], "fp_fpFSM": []},
        "end_point": {"fp_lrp": [], "fp_lrpFMM": [], "fp_fpDijkstra": [], "fp_fpFMM": [], "fp_fpFSM": []},
    }


def _append_stat_row(container: dict[str, list[np.ndarray]], key: str, values: list[float]) -> None:
    container[key].append(np.asarray(values, dtype=float))


def _summary_stats(lengths: dict[str, list[np.ndarray]], times: dict[str, list[np.ndarray]], distances: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, Any]:
    return {
        "lengths": {
            "fp": mean_std_matrix(lengths["fp"]),
            "fp_dijkstra": mean_std_matrix(lengths["fp_dijkstra"]),
            "fp_fmm": mean_std_matrix(lengths["fp_fmm"]),
            "fp_fsm": mean_std_matrix(lengths["fp_fsm"]),
            "lrp": mean_std_matrix(lengths["lrp"]),
            "lrp_fmm": mean_std_matrix(lengths["lrp_fmm"]),
        },
        "times": {
            "fpt": mean_std_matrix(times["fpt"], filter_positive=True),
            "fpt_dijkstra": mean_std_matrix(times["fpt_dijkstra"], filter_positive=True),
            "fpt_fmm": mean_std_matrix(times["fpt_fmm"], filter_positive=True),
            "fpt_fsm": mean_std_matrix(times["fpt_fsm"], filter_positive=True),
            "mhr": mean_std_matrix(times["mhr"], filter_positive=True),
            "mhr_fmm": mean_std_matrix(times["mhr_fmm"], filter_positive=True),
        },
        "distances": {
            "hausdorff": {
                "fp_lrp": mean_std_matrix(distances["hausdorff"]["fp_lrp"]),
                "fp_lrpFMM": mean_std_matrix(distances["hausdorff"]["fp_lrpFMM"], filter_positive=True),
                "fp_fpDijkstra": mean_std_matrix(distances["hausdorff"]["fp_fpDijkstra"], filter_positive=True),
                "fp_fpFMM": mean_std_matrix(distances["hausdorff"]["fp_fpFMM"], filter_positive=True),
                "fp_fpFSM": mean_std_matrix(distances["hausdorff"]["fp_fpFSM"], filter_positive=True),
            },
            "frechet": {
                "fp_lrp": mean_std_matrix(distances["frechet"]["fp_lrp"]),
                "fp_lrpFMM": mean_std_matrix(distances["frechet"]["fp_lrpFMM"], filter_positive=True),
                "fp_fpDijkstra": mean_std_matrix(distances["frechet"]["fp_fpDijkstra"], filter_positive=True),
                "fp_fpFMM": mean_std_matrix(distances["frechet"]["fp_fpFMM"], filter_positive=True),
                "fp_fpFSM": mean_std_matrix(distances["frechet"]["fp_fpFSM"], filter_positive=True),
            },
            "end_point": {
                "fp_lrp": mean_std_matrix(distances["end_point"]["fp_lrp"]),
                "fp_lrpFMM": mean_std_matrix(distances["end_point"]["fp_lrpFMM"]),
                "fp_fpDijkstra": mean_std_matrix(distances["end_point"]["fp_fpDijkstra"]),
                "fp_fpFMM": mean_std_matrix(distances["end_point"]["fp_fpFMM"]),
                "fp_fpFSM": mean_std_matrix(distances["end_point"]["fp_fpFSM"]),
            },
        },
    }


def _compute_line_integrals_for_var(
    cfg: VisualizationConfig,
    run_paths: list[RunPaths],
    *,
    max_mc_per_size: int | None = None,
) -> dict[str, dict[str, float | list[float]]]:
    target = next((rp for rp in run_paths if int(rp.size_value) == int(cfg.field_size_ref)), None)
    if target is None:
        empty = summarize_integral_values([])
        return {
            "int_logk_fp": empty,
            "int_logk2_fp": empty,
            "int_logk_lrp": empty,
            "int_logk2_lrp": empty,
            "int_logk_lrp_fmm": empty,
            "int_logk2_lrp_fmm": empty,
            "int_logk_fp_dijkstra": empty,
            "int_logk2_fp_dijkstra": empty,
            "int_logk_fp_fmm": empty,
            "int_logk2_fp_fmm": empty,
            "int_logk_fp_fsm": empty,
            "int_logk2_fp_fsm": empty,
        }

    values = {
        "fp": {"logk": [], "logk2": []},
        "lrp": {"logk": [], "logk2": []},
        "lrp_fmm": {"logk": [], "logk2": []},
        "fp_dijkstra": {"logk": [], "logk2": []},
        "fp_fmm": {"logk": [], "logk2": []},
        "fp_fsm": {"logk": [], "logk2": []},
    }

    use_consolidated = bool(target.mc_files)
    n_mc = len(target.mc_files) if use_consolidated else len(target.logk_files)
    if max_mc_per_size is not None:
        n_mc = min(n_mc, max_mc_per_size)

    for mc_idx in range(n_mc):
        if use_consolidated:
            mc = load_mc_consolidated(target.mc_files[mc_idx])
            field = _consolidated_to_logk(mc)
            pathset = _consolidated_to_pathset(mc)
        else:
            field = load_logk_field(target, mc_idx)
            pathset = load_paths(target, mc_idx)
        if field is None:
            continue
        ints = compute_mc_line_integrals(pathset, field, grid_spacing=1.0)
        for key, pair in ints.items():
            if pair[0] is None:
                continue
            values[key]["logk"].append(pair[0])
            values[key]["logk2"].append(pair[1])

    return {
        "int_logk_fp": summarize_integral_values(values["fp"]["logk"]),
        "int_logk2_fp": summarize_integral_values(values["fp"]["logk2"]),
        "int_logk_lrp": summarize_integral_values(values["lrp"]["logk"]),
        "int_logk2_lrp": summarize_integral_values(values["lrp"]["logk2"]),
        "int_logk_lrp_fmm": summarize_integral_values(values["lrp_fmm"]["logk"]),
        "int_logk2_lrp_fmm": summarize_integral_values(values["lrp_fmm"]["logk2"]),
        "int_logk_fp_dijkstra": summarize_integral_values(values["fp_dijkstra"]["logk"]),
        "int_logk2_fp_dijkstra": summarize_integral_values(values["fp_dijkstra"]["logk2"]),
        "int_logk_fp_fmm": summarize_integral_values(values["fp_fmm"]["logk"]),
        "int_logk2_fp_fmm": summarize_integral_values(values["fp_fmm"]["logk2"]),
        "int_logk_fp_fsm": summarize_integral_values(values["fp_fsm"]["logk"]),
        "int_logk2_fp_fsm": summarize_integral_values(values["fp_fsm"]["logk2"]),
    }


def run_variance_analysis_from_discovered(
    cfg: VisualizationConfig,
    discovered: dict[float, list[RunPaths]],
    *,
    show_progress: bool = False,
    hausdorff_compute_method: str = "shapely",
    frechet_compute_method: str = "shapely",
    max_var_values: int | None = None,
    max_sizes_per_var: int | None = None,
    max_mc_per_size: int | None = None,
) -> tuple[VarianceResults, list[str]]:
    data: dict[str, dict[str, Any]] = {}
    stats: dict[str, dict[str, Any]] = {}
    line_integrals: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    warning_seen: set[str] = set()

    var_values = cfg.variance_values[:max_var_values] if max_var_values is not None else cfg.variance_values
    var_iter = tqdm(var_values, desc="Variance sweep", unit="var", leave=False) if show_progress else var_values

    for var_ in var_iter:
        run_paths = discovered.get(var_, [])
        if max_sizes_per_var is not None:
            run_paths = run_paths[:max_sizes_per_var]

        field_size_list: list[float] = []
        lengths = _init_lengths()
        times = _init_times()
        distances = _init_distances()

        run_iter = tqdm(run_paths, desc=f"var={var_} sizes", unit="size", leave=False) if show_progress else run_paths
        for rp in run_iter:
            field_size_list.append(rp.size_value)
            use_consolidated = bool(rp.mc_files)
            use_aggregate = (not use_consolidated) and (not rp.main_output_files) and (rp.aggregate_output_file is not None)
            if not rp.mc_files and not rp.main_output_files and not use_aggregate:
                warnings.append(f"No mainOutput files found for var={var_}, size={rp.size_str}")
                continue
            aggregate_output_data = load_main_output(rp.aggregate_output_file) if use_aggregate and rp.aggregate_output_file else None

            len_fp_m: list[float] = []
            len_lrp_m: list[float] = []
            len_lrp_fmm_m: list[float] = []
            len_fp_dijkstra_m: list[float] = []
            len_fp_fmm_m: list[float] = []
            len_fp_fsm_m: list[float] = []
            fpt_m: list[float] = []
            fpt_dijkstra_m: list[float] = []
            fpt_fmm_m: list[float] = []
            fpt_fsm_m: list[float] = []
            mhr_m: list[float] = []
            mhr_fmm_m: list[float] = []
            haus_fp_lrp: list[float] = []
            haus_fp_lrp_fmm: list[float] = []
            haus_fp_d: list[float] = []
            haus_fp_fmm: list[float] = []
            haus_fp_fsm: list[float] = []
            fre_fp_lrp: list[float] = []
            fre_fp_lrp_fmm: list[float] = []
            fre_fp_d: list[float] = []
            fre_fp_fmm: list[float] = []
            fre_fp_fsm: list[float] = []
            end_fp_lrp: list[float] = []
            end_fp_lrp_fmm: list[float] = []
            end_fp_d: list[float] = []
            end_fp_fmm: list[float] = []
            end_fp_fsm: list[float] = []

            if use_consolidated:
                n_mc = len(rp.mc_files)
            elif use_aggregate:
                n_mc = _aggregate_mc_count(aggregate_output_data or {})
            else:
                n_mc = len(rp.main_output_files)
            if max_mc_per_size is not None:
                n_mc = min(n_mc, max_mc_per_size)

            mc_range = tqdm(range(n_mc), desc=f"var={var_} size={rp.size_str} MC", unit="mc", leave=False) if show_progress else range(n_mc)
            for mc_idx in mc_range:
                mc_consolidated = None
                if use_consolidated:
                    mc_consolidated = load_mc_consolidated(rp.mc_files[mc_idx])
                    output_data = _consolidated_to_output_data(mc_consolidated)
                elif use_aggregate:
                    output_data = _slice_aggregate_output_data(aggregate_output_data or {}, mc_idx)
                else:
                    output_data = load_main_output(rp.main_output_files[mc_idx])
                len_fp_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["len_fp"])
                len_fp_dijkstra_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["len_fp_dijkstra"])
                len_fp_fmm_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["len_fp_fmm"])
                len_fp_fsm_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["len_fp_fsm"])
                len_lrp_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["len_lrp"])
                len_lrp_fmm_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["len_lrp_fmm"])
                fpt_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["travel_time_fp"])
                fpt_dijkstra_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["travel_time_fp_dijkstra"])
                fpt_fmm_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["travel_time_fp_fmm"])
                fpt_fsm_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["travel_time_fp_fsm"])
                mhr_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["resistance_lrp"])
                mhr_fmm_val, _ = _load_main_output_value(output_data, VARIANCE_METRIC_KEY_CANDIDATES["resistance_lrp_fmm"])
                metric_values = {
                    "len_fp": len_fp_val,
                    "len_fp_dijkstra": len_fp_dijkstra_val,
                    "len_fp_fmm": len_fp_fmm_val,
                    "len_fp_fsm": len_fp_fsm_val,
                    "len_lrp": len_lrp_val,
                    "len_lrp_fmm": len_lrp_fmm_val,
                    "travel_time_fp": fpt_val,
                    "travel_time_fp_dijkstra": fpt_dijkstra_val,
                    "travel_time_fp_fmm": fpt_fmm_val,
                    "travel_time_fp_fsm": fpt_fsm_val,
                    "resistance_lrp": mhr_val,
                    "resistance_lrp_fmm": mhr_fmm_val,
                }
                for metric_name, metric_value in metric_values.items():
                    if np.isnan(metric_value):
                        joined = ", ".join(VARIANCE_METRIC_KEY_CANDIDATES[metric_name])
                        _add_unique_warning(
                            warnings,
                            warning_seen,
                            f"var={var_}, size={rp.size_str}, mc={mc_idx}: missing mainOutput keys [{joined}] for {metric_name}",
                        )

                len_fp_m.append(len_fp_val)
                len_fp_dijkstra_m.append(len_fp_dijkstra_val)
                len_fp_fmm_m.append(len_fp_fmm_val)
                len_fp_fsm_m.append(len_fp_fsm_val)
                len_lrp_m.append(len_lrp_val)
                len_lrp_fmm_m.append(len_lrp_fmm_val)
                fpt_m.append(fpt_val)
                fpt_dijkstra_m.append(fpt_dijkstra_val)
                fpt_fmm_m.append(fpt_fmm_val)
                fpt_fsm_m.append(fpt_fsm_val)
                mhr_m.append(mhr_val)
                mhr_fmm_m.append(mhr_fmm_val)

                if use_consolidated and mc_consolidated is not None:
                    pathset_loader_fn = lambda mc=mc_consolidated: _consolidated_to_pathset(mc)
                else:
                    pathset_loader_fn = lambda rp=rp, mc_idx=mc_idx: load_paths(rp, mc_idx)

                dist_values, distance_warnings = resolve_distances(
                    cfg,
                    output_data,
                    None,
                    workers=cfg.workers,
                    hausdorff_compute_method=hausdorff_compute_method,
                    frechet_compute_method=frechet_compute_method,
                    pathset_loader=pathset_loader_fn,
                )
                for warning in distance_warnings:
                    _add_unique_warning(warnings, warning_seen, warning)

                haus_fp_lrp.append(dist_values["hausdorff"]["fp_lrp"])
                haus_fp_lrp_fmm.append(dist_values["hausdorff"]["fp_lrpFMM"])
                haus_fp_d.append(dist_values["hausdorff"]["fp_fpDijkstra"])
                haus_fp_fmm.append(dist_values["hausdorff"]["fp_fpFMM"])
                haus_fp_fsm.append(dist_values["hausdorff"]["fp_fpFSM"])
                fre_fp_lrp.append(dist_values["frechet"]["fp_lrp"])
                fre_fp_lrp_fmm.append(dist_values["frechet"]["fp_lrpFMM"])
                fre_fp_d.append(dist_values["frechet"]["fp_fpDijkstra"])
                fre_fp_fmm.append(dist_values["frechet"]["fp_fpFMM"])
                fre_fp_fsm.append(dist_values["frechet"]["fp_fpFSM"])
                end_fp_lrp.append(dist_values["end_point"]["fp_lrp"])
                end_fp_lrp_fmm.append(dist_values["end_point"]["fp_lrpFMM"])
                end_fp_d.append(dist_values["end_point"]["fp_fpDijkstra"])
                end_fp_fmm.append(dist_values["end_point"]["fp_fpFMM"])
                end_fp_fsm.append(dist_values["end_point"]["fp_fpFSM"])

            _append_stat_row(lengths, "fp", len_fp_m)
            _append_stat_row(lengths, "fp_dijkstra", len_fp_dijkstra_m)
            _append_stat_row(lengths, "fp_fmm", len_fp_fmm_m)
            _append_stat_row(lengths, "fp_fsm", len_fp_fsm_m)
            _append_stat_row(lengths, "lrp", len_lrp_m)
            _append_stat_row(lengths, "lrp_fmm", len_lrp_fmm_m)
            _append_stat_row(times, "fpt", fpt_m)
            _append_stat_row(times, "fpt_dijkstra", fpt_dijkstra_m)
            _append_stat_row(times, "fpt_fmm", fpt_fmm_m)
            _append_stat_row(times, "fpt_fsm", fpt_fsm_m)
            _append_stat_row(times, "mhr", mhr_m)
            _append_stat_row(times, "mhr_fmm", mhr_fmm_m)
            distances["hausdorff"]["fp_lrp"].append(np.asarray(haus_fp_lrp, dtype=float))
            distances["hausdorff"]["fp_lrpFMM"].append(np.asarray(haus_fp_lrp_fmm, dtype=float))
            distances["hausdorff"]["fp_fpDijkstra"].append(np.asarray(haus_fp_d, dtype=float))
            distances["hausdorff"]["fp_fpFMM"].append(np.asarray(haus_fp_fmm, dtype=float))
            distances["hausdorff"]["fp_fpFSM"].append(np.asarray(haus_fp_fsm, dtype=float))
            distances["frechet"]["fp_lrp"].append(np.asarray(fre_fp_lrp, dtype=float))
            distances["frechet"]["fp_lrpFMM"].append(np.asarray(fre_fp_lrp_fmm, dtype=float))
            distances["frechet"]["fp_fpDijkstra"].append(np.asarray(fre_fp_d, dtype=float))
            distances["frechet"]["fp_fpFMM"].append(np.asarray(fre_fp_fmm, dtype=float))
            distances["frechet"]["fp_fpFSM"].append(np.asarray(fre_fp_fsm, dtype=float))
            distances["end_point"]["fp_lrp"].append(np.asarray(end_fp_lrp, dtype=float))
            distances["end_point"]["fp_lrpFMM"].append(np.asarray(end_fp_lrp_fmm, dtype=float))
            distances["end_point"]["fp_fpDijkstra"].append(np.asarray(end_fp_d, dtype=float))
            distances["end_point"]["fp_fpFMM"].append(np.asarray(end_fp_fmm, dtype=float))
            distances["end_point"]["fp_fpFSM"].append(np.asarray(end_fp_fsm, dtype=float))

        key = f"var_{var_}"
        data[key] = {
            "field_size": np.asarray(field_size_list, dtype=float),
            "lengths": lengths,
            "times": times,
            "distances": distances,
        }
        stats[key] = _summary_stats(lengths, times, distances)
        line_integrals[key] = _compute_line_integrals_for_var(cfg, run_paths, max_mc_per_size=max_mc_per_size)

    first_key = f"var_{var_values[0]}" if var_values else ""
    field_size = data.get(first_key, {}).get("field_size", np.asarray([], dtype=float))
    return VarianceResults(field_size=field_size, data=data, stats=stats, line_integrals=line_integrals), warnings


def run_std_analysis_from_discovered(
    cfg: VisualizationConfig,
    discovered: dict[float, list[RunPaths]],
    *,
    show_progress: bool = False,
    hausdorff_compute_method: str = "shapely",
    frechet_compute_method: str = "shapely",
    max_std_values: int | None = None,
    max_sizes_per_std: int | None = None,
    max_mc_per_size: int | None = None,
) -> tuple[StdResults, list[str]]:
    data_path: dict[str, dict[str, Any]] = {}
    stats_path: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    warning_seen: set[str] = set()

    std_values = cfg.std_values[:max_std_values] if max_std_values is not None else cfg.std_values
    std_iter = tqdm(std_values, desc="Std sweep", unit="std", leave=False) if show_progress else std_values

    for std_ in std_iter:
        runs = discovered.get(std_, [])
        if max_sizes_per_std is not None:
            runs = runs[:max_sizes_per_std]

        field_size = []
        hausdorffDist_fp_lrp_sm = []
        frechetDist_fp_lrp_sm = []
        endPointDist_fp_lrp_sm = []
        hausdorffDist_fp_lrpFMM_sm = []
        frechetDist_fp_lrpFMM_sm = []
        endPointDist_fp_lrpFMM_sm = []
        hausdorffDist_fp_fpDijkstra_sm = []
        frechetDist_fp_fpDijkstra_sm = []
        endPointDist_fp_fpDijkstra_sm = []
        hausdorffDist_fp_fpFMM_sm = []
        frechetDist_fp_fpFMM_sm = []
        endPointDist_fp_fpFMM_sm = []

        run_iter = tqdm(runs, desc=f"std={std_} sizes", unit="size", leave=False) if show_progress else runs
        for rp in run_iter:
            field_size.append(rp.size_value)
            use_consolidated = bool(rp.mc_files)
            use_aggregate = (not use_consolidated) and (not rp.main_output_files) and (rp.aggregate_output_file is not None)
            if not rp.mc_files and not (rp.fp_files or rp.lrp_files or rp.lrp_fmm_files or rp.fp_dijkstra_files or rp.fp_fmm_files or rp.main_output_files or use_aggregate):
                warnings.append(f"No path/mainOutput files found for std={std_}, size={rp.size_str}")
                continue
            aggregate_output_data = load_main_output(rp.aggregate_output_file) if use_aggregate and rp.aggregate_output_file else None

            hausdorffDist_fp_lrp_m: list[float] = []
            frechetDist_fp_lrp_m: list[float] = []
            endPointDist_fp_lrp_m: list[float] = []
            hausdorffDist_fp_lrpFMM_m: list[float] = []
            frechetDist_fp_lrpFMM_m: list[float] = []
            endPointDist_fp_lrpFMM_m: list[float] = []
            hausdorffDist_fp_fpDijkstra_m: list[float] = []
            frechetDist_fp_fpDijkstra_m: list[float] = []
            endPointDist_fp_fpDijkstra_m: list[float] = []
            hausdorffDist_fp_fpFMM_m: list[float] = []
            frechetDist_fp_fpFMM_m: list[float] = []
            endPointDist_fp_fpFMM_m: list[float] = []

            if use_consolidated:
                n_mc = len(rp.mc_files)
            elif use_aggregate:
                n_mc = _aggregate_mc_count(aggregate_output_data or {})
            else:
                n_mc = max(
                    len(rp.fp_files),
                    len(rp.lrp_files),
                    len(rp.lrp_fmm_files),
                    len(rp.fp_dijkstra_files),
                    len(rp.fp_fmm_files),
                    len(rp.main_output_files),
                )
            if max_mc_per_size is not None:
                n_mc = min(n_mc, max_mc_per_size)

            mc_range = tqdm(range(n_mc), desc=f"std={std_} size={rp.size_str} MC", unit="mc", leave=False) if show_progress else range(n_mc)
            for mc_idx in mc_range:
                if use_consolidated:
                    mc_consolidated = load_mc_consolidated(rp.mc_files[mc_idx])
                    output_data = _consolidated_to_output_data(mc_consolidated)
                    pathset_loader_fn = lambda mc=mc_consolidated: _consolidated_to_pathset(mc)
                elif use_aggregate:
                    output_data = _slice_aggregate_output_data(aggregate_output_data or {}, mc_idx)

                    def pathset_loader_fn(rp: RunPaths = rp, mc_idx: int = mc_idx) -> PathSet:
                        return PathSet(
                            fp=_safe_load(rp.fp_files, mc_idx),
                            lrp=_safe_load(rp.lrp_files, mc_idx),
                            lrp_fmm=_safe_load(rp.lrp_fmm_files, mc_idx),
                            fp_dijkstra=_safe_load(rp.fp_dijkstra_files, mc_idx),
                            fp_fmm=_safe_load(rp.fp_fmm_files, mc_idx),
                        )
                else:
                    output_data = load_main_output(rp.main_output_files[mc_idx]) if mc_idx < len(rp.main_output_files) else None

                    def pathset_loader_fn(rp: RunPaths = rp, mc_idx: int = mc_idx) -> PathSet:
                        return PathSet(
                            fp=_safe_load(rp.fp_files, mc_idx),
                            lrp=_safe_load(rp.lrp_files, mc_idx),
                            lrp_fmm=_safe_load(rp.lrp_fmm_files, mc_idx),
                            fp_dijkstra=_safe_load(rp.fp_dijkstra_files, mc_idx),
                            fp_fmm=_safe_load(rp.fp_fmm_files, mc_idx),
                        )

                dist_values, distance_warnings = resolve_distances(
                    cfg,
                    output_data,
                    None,
                    workers=cfg.workers,
                    hausdorff_compute_method=hausdorff_compute_method,
                    frechet_compute_method=frechet_compute_method,
                    pathset_loader=pathset_loader_fn,
                )
                for warning in distance_warnings:
                    full = f"std={std_}, size={rp.size_str}, {warning}"
                    _add_unique_warning(warnings, warning_seen, full)

                hausdorffDist_fp_lrp_m.append(dist_values["hausdorff"]["fp_lrp"])
                frechetDist_fp_lrp_m.append(dist_values["frechet"]["fp_lrp"])
                endPointDist_fp_lrp_m.append(dist_values["end_point"]["fp_lrp"])
                hausdorffDist_fp_lrpFMM_m.append(dist_values["hausdorff"]["fp_lrpFMM"])
                frechetDist_fp_lrpFMM_m.append(dist_values["frechet"]["fp_lrpFMM"])
                endPointDist_fp_lrpFMM_m.append(dist_values["end_point"]["fp_lrpFMM"])
                hausdorffDist_fp_fpDijkstra_m.append(dist_values["hausdorff"]["fp_fpDijkstra"])
                frechetDist_fp_fpDijkstra_m.append(dist_values["frechet"]["fp_fpDijkstra"])
                endPointDist_fp_fpDijkstra_m.append(dist_values["end_point"]["fp_fpDijkstra"])
                hausdorffDist_fp_fpFMM_m.append(dist_values["hausdorff"]["fp_fpFMM"])
                frechetDist_fp_fpFMM_m.append(dist_values["frechet"]["fp_fpFMM"])
                endPointDist_fp_fpFMM_m.append(dist_values["end_point"]["fp_fpFMM"])

            hausdorffDist_fp_lrp_sm.append(hausdorffDist_fp_lrp_m)
            frechetDist_fp_lrp_sm.append(frechetDist_fp_lrp_m)
            endPointDist_fp_lrp_sm.append(endPointDist_fp_lrp_m)
            hausdorffDist_fp_lrpFMM_sm.append(hausdorffDist_fp_lrpFMM_m)
            frechetDist_fp_lrpFMM_sm.append(frechetDist_fp_lrpFMM_m)
            endPointDist_fp_lrpFMM_sm.append(endPointDist_fp_lrpFMM_m)
            hausdorffDist_fp_fpDijkstra_sm.append(hausdorffDist_fp_fpDijkstra_m)
            frechetDist_fp_fpDijkstra_sm.append(frechetDist_fp_fpDijkstra_m)
            endPointDist_fp_fpDijkstra_sm.append(endPointDist_fp_fpDijkstra_m)
            hausdorffDist_fp_fpFMM_sm.append(hausdorffDist_fp_fpFMM_m)
            frechetDist_fp_fpFMM_sm.append(frechetDist_fp_fpFMM_m)
            endPointDist_fp_fpFMM_sm.append(endPointDist_fp_fpFMM_m)

        key = f"std_{std_}"
        data_path[key] = {
            "field_size": field_size,
            "hausdorffDist_fp_lrp_sm": hausdorffDist_fp_lrp_sm,
            "frechetDist_fp_lrp_sm": frechetDist_fp_lrp_sm,
            "endPointDist_fp_lrp_sm": endPointDist_fp_lrp_sm,
            "hausdorffDist_fp_lrpFMM_sm": hausdorffDist_fp_lrpFMM_sm,
            "frechetDist_fp_lrpFMM_sm": frechetDist_fp_lrpFMM_sm,
            "endPointDist_fp_lrpFMM_sm": endPointDist_fp_lrpFMM_sm,
            "hausdorffDist_fp_fpDijkstra_sm": hausdorffDist_fp_fpDijkstra_sm,
            "frechetDist_fp_fpDijkstra_sm": frechetDist_fp_fpDijkstra_sm,
            "endPointDist_fp_fpDijkstra_sm": endPointDist_fp_fpDijkstra_sm,
            "hausdorffDist_fp_fpFMM_sm": hausdorffDist_fp_fpFMM_sm,
            "frechetDist_fp_fpFMM_sm": frechetDist_fp_fpFMM_sm,
            "endPointDist_fp_fpFMM_sm": endPointDist_fp_fpFMM_sm,
        }

    for key_c, value in data_path.items():
        stats_path[key_c] = {}
        for key_, arr_lists in value.items():
            if key_ == "field_size":
                continue
            means = [mean_std(lst).mean for lst in arr_lists]
            stds = [mean_std(lst).std for lst in arr_lists]
            stats_path[key_c][key_] = [means, stds]

    stats_std = {
        "hausdorffDist": {"fp_lrp": [], "fp_lrpFMM": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
        "frechetDist": {"fp_lrp": [], "fp_lrpFMM": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
        "endPointDist": {"fp_lrp": [], "fp_lrpFMM": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
    }

    for std_key in stats_path:
        stats_std["hausdorffDist"]["fp_lrp"].append(stats_path[std_key]["hausdorffDist_fp_lrp_sm"])
        stats_std["hausdorffDist"]["fp_lrpFMM"].append(stats_path[std_key]["hausdorffDist_fp_lrpFMM_sm"])
        stats_std["hausdorffDist"]["fp_fpDijkstra"].append(stats_path[std_key]["hausdorffDist_fp_fpDijkstra_sm"])
        stats_std["hausdorffDist"]["fp_fpFMM"].append(stats_path[std_key]["hausdorffDist_fp_fpFMM_sm"])
        stats_std["frechetDist"]["fp_lrp"].append(stats_path[std_key]["frechetDist_fp_lrp_sm"])
        stats_std["frechetDist"]["fp_lrpFMM"].append(stats_path[std_key]["frechetDist_fp_lrpFMM_sm"])
        stats_std["frechetDist"]["fp_fpDijkstra"].append(stats_path[std_key]["frechetDist_fp_fpDijkstra_sm"])
        stats_std["frechetDist"]["fp_fpFMM"].append(stats_path[std_key]["frechetDist_fp_fpFMM_sm"])
        stats_std["endPointDist"]["fp_lrp"].append(stats_path[std_key]["endPointDist_fp_lrp_sm"])
        stats_std["endPointDist"]["fp_lrpFMM"].append(stats_path[std_key]["endPointDist_fp_lrpFMM_sm"])
        stats_std["endPointDist"]["fp_fpDijkstra"].append(stats_path[std_key]["endPointDist_fp_fpDijkstra_sm"])
        stats_std["endPointDist"]["fp_fpFMM"].append(stats_path[std_key]["endPointDist_fp_fpFMM_sm"])

    for key1 in stats_std:
        for key2 in stats_std[key1]:
            stats_std[key1][key2] = np.asarray(stats_std[key1][key2], dtype=float)

    return StdResults(data_path=data_path, stats_path=stats_path, stats_std=stats_std), warnings


def run_stage2(
    cfg: VisualizationConfig,
    *,
    cache_dir: Path | None = None,
    skip_std: bool = False,
    hausdorff_compute_method: str = "shapely",
    frechet_compute_method: str = "shapely",
    max_var_values: int | None = None,
    max_std_values: int | None = None,
    max_sizes_per_value: int | None = None,
    max_mc_per_size: int | None = None,
) -> Path:
    cache_dir = cache_dir or default_cache_dir(cfg)

    index = load_pickle(cache_dir / "stage1_index.pkl")
    discovered_var = deserialize_discovered(index["variance"])
    discovered_std = deserialize_discovered(index["std"])

    has_variance = bool(discovered_var)
    has_std = bool(discovered_std)
    if not has_variance and not has_std:
        raise FileNotFoundError(
            f"No variance or std sweep directories were discovered for run{cfg.run_id} under {cfg.data_root}."
        )

    var_results: VarianceResults | None = None
    warnings_var: list[str] = []
    if has_variance:
        var_results, warnings_var = run_variance_analysis_from_discovered(
            cfg,
            discovered_var,
            show_progress=True,
            hausdorff_compute_method=hausdorff_compute_method,
            frechet_compute_method=frechet_compute_method,
            max_var_values=max_var_values,
            max_sizes_per_var=max_sizes_per_value,
            max_mc_per_size=max_mc_per_size,
        )

    std_results: StdResults | None = None
    warnings_std: list[str] = []
    if has_std and not skip_std:
        std_results, warnings_std = run_std_analysis_from_discovered(
            cfg,
            discovered_std,
            show_progress=True,
            hausdorff_compute_method=hausdorff_compute_method,
            frechet_compute_method=frechet_compute_method,
            max_std_values=max_std_values,
            max_sizes_per_std=max_sizes_per_value,
            max_mc_per_size=max_mc_per_size,
        )

    payload = {
        "var_results": _serialize_variance_results(var_results) if var_results is not None else None,
        "std_results": _serialize_std_results(std_results),
        "warnings": {"variance": warnings_var, "std": warnings_std},
    }
    cache_file = cache_dir / "stage2_metrics.pkl"
    save_pickle(cache_file, payload)

    notes = {
        "cache_file": str(cache_file),
        "warnings_variance": len(warnings_var),
        "warnings_std": len(warnings_std),
        "variance_computed": var_results is not None,
        "std_computed": std_results is not None,
        "hausdorff_compute_method": hausdorff_compute_method,
        "frechet_compute_method": frechet_compute_method,
        "max_var_values": max_var_values,
        "max_std_values": max_std_values,
        "max_sizes_per_value": max_sizes_per_value,
        "max_mc_per_size": max_mc_per_size,
    }
    write_stage_metadata(cache_dir / "stage2_metrics.meta.json", stage="stage2_metrics", config=cfg, notes=notes)
    return cache_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: compute metrics and aggregate raw stats")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--skip-std", action="store_true")
    parser.add_argument(
        "--hausdorff-compute-method",
        type=str,
        choices=["shapely", "custom"],
        default="shapely",
        help="Method for Hausdorff distance computation",
    )
    parser.add_argument(
        "--frechet-compute-method",
        type=str,
        choices=["shapely", "custom"],
        default="shapely",
        help="Method for Frechet distance computation",
    )
    parser.add_argument("--max-var-values", type=int, default=None, help="Limit number of variance values processed")
    parser.add_argument("--max-std-values", type=int, default=None, help="Limit number of std values processed")
    parser.add_argument("--max-sizes-per-value", type=int, default=None, help="Limit number of size runs per var/std value")
    parser.add_argument("--max-mc-per-size", type=int, default=None, help="Limit number of realizations (MC) per size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cache_file = run_stage2(
        cfg,
        cache_dir=args.cache_dir,
        skip_std=args.skip_std,
        hausdorff_compute_method=args.hausdorff_compute_method,
        frechet_compute_method=args.frechet_compute_method,
        max_var_values=args.max_var_values,
        max_std_values=args.max_std_values,
        max_sizes_per_value=args.max_sizes_per_value,
        max_mc_per_size=args.max_mc_per_size,
    )
    print(f"Wrote metrics cache: {cache_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
