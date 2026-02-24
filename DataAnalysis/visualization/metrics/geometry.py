from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:
    from shapely import LineString, frechet_distance, hausdorff_distance
except ImportError:  # pragma: no cover - environment dependent
    LineString = None
    frechet_distance = None
    hausdorff_distance = None


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
    fp_dijkstra: np.ndarray | None,
    fp_fmm: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    out = {
        "hausdorff": {"fp_lrp": np.nan, "fp_fpDijkstra": np.nan, "fp_fpFMM": np.nan},
        "frechet": {"fp_lrp": np.nan, "fp_fpDijkstra": np.nan, "fp_fpFMM": np.nan},
    }
    ls_fp = path_to_linestring(fp)
    if ls_fp is None:
        return out
    if hausdorff_distance is None or frechet_distance is None:
        return out
    pairs = [("fp_lrp", lrp), ("fp_fpDijkstra", fp_dijkstra), ("fp_fpFMM", fp_fmm)]
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
