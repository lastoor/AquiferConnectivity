from __future__ import annotations

import numpy as np
from tqdm import tqdm

from visualization.config import VisualizationConfig
from visualization.io.discovery import discover_std_runs
from visualization.io.loaders import load_main_output
from visualization.metrics.distance_resolver import resolve_distances
from visualization.stats.aggregate import mean_std
from visualization.types import PathSet, StdResults


def _safe_load(paths, idx):
    if paths and idx < len(paths):
        return np.load(paths[idx])
    return None


def run_std_analysis_from_discovered(
    cfg: VisualizationConfig,
    discovered,
    *,
    show_progress: bool = False,
    max_std_values: int | None = None,
    max_sizes_per_std: int | None = None,
    max_mc_per_size: int | None = None,
) -> tuple[StdResults, list[str]]:
    data_path: dict[str, dict] = {}
    stats_path: dict[str, dict] = {}
    warnings: list[str] = []
    warning_seen: set[str] = set()

    std_values = cfg.std_values[:max_std_values] if max_std_values is not None else cfg.std_values
    std_iter = std_values
    if show_progress:
        std_iter = tqdm(std_values, desc="Std sweep", unit="std", leave=False)

    for std_ in std_iter:
        runs = discovered.get(std_, [])
        if max_sizes_per_std is not None:
            runs = runs[:max_sizes_per_std]
        field_size = []

        histDist_fp_lrp_sm = []
        hausdorffDist_fp_lrp_sm = []
        frechetDist_fp_lrp_sm = []
        endPointDist_fp_lrp_sm = []

        histDist_fp_fpDijkstra_sm = []
        hausdorffDist_fp_fpDijkstra_sm = []
        frechetDist_fp_fpDijkstra_sm = []
        endPointDist_fp_fpDijkstra_sm = []

        histDist_fp_fpFMM_sm = []
        hausdorffDist_fp_fpFMM_sm = []
        frechetDist_fp_fpFMM_sm = []
        endPointDist_fp_fpFMM_sm = []

        run_iter = runs
        if show_progress:
            run_iter = tqdm(runs, desc=f"std={std_} sizes", unit="size", leave=False)

        for rp in run_iter:
            field_size.append(rp.size_value)
            if not (rp.fp_files or rp.lrp_files or rp.fp_dijkstra_files or rp.fp_fmm_files or rp.main_output_files):
                warnings.append(f"No path/mainOutput files found for std={std_}, size={rp.size_str}")
                continue

            histDist_fp_lrp_m = []
            hausdorffDist_fp_lrp_m = []
            frechetDist_fp_lrp_m = []
            endPointDist_fp_lrp_m = []

            histDist_fp_fpDijkstra_m = []
            hausdorffDist_fp_fpDijkstra_m = []
            frechetDist_fp_fpDijkstra_m = []
            endPointDist_fp_fpDijkstra_m = []

            histDist_fp_fpFMM_m = []
            hausdorffDist_fp_fpFMM_m = []
            frechetDist_fp_fpFMM_m = []
            endPointDist_fp_fpFMM_m = []

            n_mc = max(
                len(rp.fp_files),
                len(rp.lrp_files),
                len(rp.fp_dijkstra_files),
                len(rp.fp_fmm_files),
                len(rp.main_output_files),
            )
            if max_mc_per_size is not None:
                n_mc = min(n_mc, max_mc_per_size)

            mc_range = range(n_mc)
            if show_progress:
                mc_range = tqdm(mc_range, desc=f"std={std_} size={rp.size_str} MC", unit="mc", leave=False)

            for mc_idx in mc_range:
                output_data = load_main_output(rp.main_output_files[mc_idx]) if mc_idx < len(rp.main_output_files) else None

                def _load_pathset(rp=rp, mc_idx=mc_idx):
                    return PathSet(
                        fp=_safe_load(rp.fp_files, mc_idx),
                        lrp=_safe_load(rp.lrp_files, mc_idx),
                        fp_dijkstra=_safe_load(rp.fp_dijkstra_files, mc_idx),
                        fp_fmm=_safe_load(rp.fp_fmm_files, mc_idx),
                    )

                dist_values, distance_warnings = resolve_distances(
                    cfg,
                    output_data,
                    None,
                    workers=cfg.workers,
                    hausdorff_compute_method="nn_max",
                    pathset_loader=_load_pathset,
                )
                for w in distance_warnings:
                    full = f"std={std_}, size={rp.size_str}, {w}"
                    if full not in warning_seen:
                        warning_seen.add(full)
                        warnings.append(full)

                hausdorffDist_fp_lrp_m.append(dist_values["hausdorff"]["fp_lrp"])
                frechetDist_fp_lrp_m.append(dist_values["frechet"]["fp_lrp"])
                endPointDist_fp_lrp_m.append(dist_values["end_point"]["fp_lrp"])

                hausdorffDist_fp_fpDijkstra_m.append(dist_values["hausdorff"]["fp_fpDijkstra"])
                frechetDist_fp_fpDijkstra_m.append(dist_values["frechet"]["fp_fpDijkstra"])
                endPointDist_fp_fpDijkstra_m.append(dist_values["end_point"]["fp_fpDijkstra"])

                hausdorffDist_fp_fpFMM_m.append(dist_values["hausdorff"]["fp_fpFMM"])
                frechetDist_fp_fpFMM_m.append(dist_values["frechet"]["fp_fpFMM"])
                endPointDist_fp_fpFMM_m.append(dist_values["end_point"]["fp_fpFMM"])

            histDist_fp_lrp_sm.append(histDist_fp_lrp_m)
            hausdorffDist_fp_lrp_sm.append(hausdorffDist_fp_lrp_m)
            frechetDist_fp_lrp_sm.append(frechetDist_fp_lrp_m)
            endPointDist_fp_lrp_sm.append(endPointDist_fp_lrp_m)

            histDist_fp_fpDijkstra_sm.append(histDist_fp_fpDijkstra_m)
            hausdorffDist_fp_fpDijkstra_sm.append(hausdorffDist_fp_fpDijkstra_m)
            frechetDist_fp_fpDijkstra_sm.append(frechetDist_fp_fpDijkstra_m)
            endPointDist_fp_fpDijkstra_sm.append(endPointDist_fp_fpDijkstra_m)

            histDist_fp_fpFMM_sm.append(histDist_fp_fpFMM_m)
            hausdorffDist_fp_fpFMM_sm.append(hausdorffDist_fp_fpFMM_m)
            frechetDist_fp_fpFMM_sm.append(frechetDist_fp_fpFMM_m)
            endPointDist_fp_fpFMM_sm.append(endPointDist_fp_fpFMM_m)

        k = f"std_{std_}"
        data_path[k] = {
            "field_size": field_size,
            "histDist_fp_lrp_sm": histDist_fp_lrp_sm,
            "hausdorffDist_fp_lrp_sm": hausdorffDist_fp_lrp_sm,
            "frechetDist_fp_lrp_sm": frechetDist_fp_lrp_sm,
            "endPointDist_fp_lrp_sm": endPointDist_fp_lrp_sm,
            "histDist_fp_fpDijkstra_sm": histDist_fp_fpDijkstra_sm,
            "hausdorffDist_fp_fpDijkstra_sm": hausdorffDist_fp_fpDijkstra_sm,
            "frechetDist_fp_fpDijkstra_sm": frechetDist_fp_fpDijkstra_sm,
            "endPointDist_fp_fpDijkstra_sm": endPointDist_fp_fpDijkstra_sm,
            "histDist_fp_fpFMM_sm": histDist_fp_fpFMM_sm,
            "hausdorffDist_fp_fpFMM_sm": hausdorffDist_fp_fpFMM_sm,
            "frechetDist_fp_fpFMM_sm": frechetDist_fp_fpFMM_sm,
            "endPointDist_fp_fpFMM_sm": endPointDist_fp_fpFMM_sm,
        }

    for key_c, value in data_path.items():
        stats_path[key_c] = {}
        for key_, arr_lists in value.items():
            if key_.startswith("histDist") or key_ == "field_size":
                continue
            means = [mean_std(lst).mean for lst in arr_lists]
            stds = [mean_std(lst).std for lst in arr_lists]
            stats_path[key_c][key_] = [means, stds]

    stats_std = {
        "hausdorffDist": {"fp_lrp": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
        "frechetDist": {"fp_lrp": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
        "endPointDist": {"fp_lrp": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
    }

    for std_key in stats_path:
        stats_std["hausdorffDist"]["fp_lrp"].append(stats_path[std_key]["hausdorffDist_fp_lrp_sm"])
        stats_std["hausdorffDist"]["fp_fpDijkstra"].append(stats_path[std_key]["hausdorffDist_fp_fpDijkstra_sm"])
        stats_std["hausdorffDist"]["fp_fpFMM"].append(stats_path[std_key]["hausdorffDist_fp_fpFMM_sm"])
        stats_std["frechetDist"]["fp_lrp"].append(stats_path[std_key]["frechetDist_fp_lrp_sm"])
        stats_std["frechetDist"]["fp_fpDijkstra"].append(stats_path[std_key]["frechetDist_fp_fpDijkstra_sm"])
        stats_std["frechetDist"]["fp_fpFMM"].append(stats_path[std_key]["frechetDist_fp_fpFMM_sm"])
        stats_std["endPointDist"]["fp_lrp"].append(stats_path[std_key]["endPointDist_fp_lrp_sm"])
        stats_std["endPointDist"]["fp_fpDijkstra"].append(stats_path[std_key]["endPointDist_fp_fpDijkstra_sm"])
        stats_std["endPointDist"]["fp_fpFMM"].append(stats_path[std_key]["endPointDist_fp_fpFMM_sm"])

    for key1 in stats_std:
        for key2 in stats_std[key1]:
            stats_std[key1][key2] = np.asarray(stats_std[key1][key2], dtype=float)

    return StdResults(data_path=data_path, stats_path=stats_path, stats_std=stats_std), warnings


def run_std_analysis(
    cfg: VisualizationConfig,
    *,
    max_std_values: int | None = None,
    max_sizes_per_std: int | None = None,
    max_mc_per_size: int | None = None,
) -> tuple[StdResults, list[str]]:
    discovered = discover_std_runs(cfg)
    return run_std_analysis_from_discovered(
        cfg,
        discovered,
        max_std_values=max_std_values,
        max_sizes_per_std=max_sizes_per_std,
        max_mc_per_size=max_mc_per_size,
    )
