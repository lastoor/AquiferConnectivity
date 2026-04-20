from __future__ import annotations

from typing import Any, Callable

import numpy as np

from visualization.config import VisualizationConfig
from visualization.metrics.geometry import distance_distribution, endpoint_distance, shapely_pair_distances
from visualization.types import PathSet

PAIR_TO_FIELDS = {
    "fp_lrp": ("fp", "lrp"),
    "fp_fpDijkstra": ("fp", "fp_dijkstra"),
    "fp_fpFMM": ("fp", "fp_fmm"),
}
DISTANCE_TYPES = ("end_point", "hausdorff", "frechet")


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


def resolve_distances(
    cfg: VisualizationConfig,
    output_data: dict[str, Any] | None,
    pathset: PathSet | None,
    *,
    workers: int,
    hausdorff_compute_method: str,
    pathset_loader: Callable[[], PathSet | None] | None = None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Resolve configured distances by loading from mainOutput and/or computing from paths.

    hausdorff_compute_method:
      - 'shapely' for shapely hausdorff/frechet
      - 'nn_max' for hausdorff via nearest-neighbor distribution max
    """
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
                return {"hausdorff": {k: np.nan for k in PAIR_TO_FIELDS}, "frechet": {k: np.nan for k in PAIR_TO_FIELDS}}
            shapely_cache = shapely_pair_distances(ps.fp, ps.lrp, ps.fp_dijkstra, ps.fp_fmm)
        return shapely_cache

    def _compute_value(distance_type: str, pair: str) -> float:
        ps = _get_pathset()
        if ps is None:
            return float("nan")
        if distance_type == "end_point":
            return _compute_endpoint_for_pair(ps, pair)
        if distance_type == "hausdorff":
            if hausdorff_compute_method == "nn_max":
                return _compute_hausdorff_nn_for_pair(ps, pair, workers)
            return float(_ensure_shapely()["hausdorff"][pair])
        if distance_type == "frechet":
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
                key = cfg.distance_main_output_keys[distance_type][pair]
                if output_data is not None and key in output_data:
                    value = float(output_data[key])
                    loaded = True
                else:
                    warnings.append(f"Missing mainOutput key '{key}' for {distance_type}:{pair}")

            if (not loaded) and (mode == "compute_from_paths" or cfg.distance_fallback_on_missing == "compute_from_paths"):
                value = _compute_value(distance_type, pair)

            out[distance_type][pair] = value

    return out, warnings
