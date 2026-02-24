from __future__ import annotations

from itertools import zip_longest

import numpy as np

from visualization.config import VisualizationConfig
from visualization.io.discovery import discover_std_runs
from visualization.metrics.geometry import distance_distribution
from visualization.stats.aggregate import mean_std
from visualization.types import StdResults


def run_std_analysis(cfg: VisualizationConfig) -> tuple[StdResults, list[str]]:
    discovered = discover_std_runs(cfg)
    data_path: dict[str, dict] = {}
    stats_path: dict[str, dict] = {}
    warnings: list[str] = []

    for std_ in cfg.std_values:
        runs = discovered.get(std_, [])
        field_size = []

        histDist_fp_lrp_sm = []
        hausdorffDist_fp_lrp_sm = []
        endPointDist_fp_lrp_sm = []

        histDist_fp_fpDijkstra_sm = []
        hausdorffDist_fp_fpDijkstra_sm = []
        endPointDist_fp_fpDijkstra_sm = []

        histDist_fp_fpFMM_sm = []
        hausdorffDist_fp_fpFMM_sm = []
        endPointDist_fp_fpFMM_sm = []

        for rp in runs:
            field_size.append(rp.size_value)
            if not (rp.fp_files or rp.lrp_files or rp.fp_dijkstra_files or rp.fp_fmm_files):
                warnings.append(f"No path files found for std={std_}, size={rp.size_str}")
                continue

            histDist_fp_lrp_m = []
            hausdorffDist_fp_lrp_m = []
            endPointDist_fp_lrp_m = []

            histDist_fp_fpDijkstra_m = []
            hausdorffDist_fp_fpDijkstra_m = []
            endPointDist_fp_fpDijkstra_m = []

            histDist_fp_fpFMM_m = []
            hausdorffDist_fp_fpFMM_m = []
            endPointDist_fp_fpFMM_m = []

            fp = lrp = fp_dijkstra = fp_fmm = None
            for fp_file, lrp_file, fp_d_file, fp_fmm_file in zip_longest(rp.fp_files, rp.lrp_files, rp.fp_dijkstra_files, rp.fp_fmm_files):
                if fp_file is not None:
                    fp = np.load(fp_file)
                if lrp_file is not None:
                    lrp = np.load(lrp_file)
                if fp_d_file is not None:
                    fp_dijkstra = np.load(fp_d_file)
                if fp_fmm_file is not None:
                    fp_fmm = np.load(fp_fmm_file)

                if fp is None or lrp is None or fp_dijkstra is None or fp_fmm is None:
                    continue

                d, _, _ = distance_distribution(fp, lrp, ds=1.0, workers=cfg.workers)
                histDist_fp_lrp_m.append(d)
                hausdorffDist_fp_lrp_m.append(float(np.max(d)))
                endPointDist_fp_lrp_m.append(float(fp[-1, 1] - lrp[-1, 1]))

                d, _, _ = distance_distribution(fp, fp_dijkstra, ds=1.0, workers=cfg.workers)
                histDist_fp_fpDijkstra_m.append(d)
                hausdorffDist_fp_fpDijkstra_m.append(float(np.max(d)))
                endPointDist_fp_fpDijkstra_m.append(float(fp[-1, 1] - fp_dijkstra[-1, 1]))

                d, _, _ = distance_distribution(fp, fp_fmm, ds=1.0, workers=cfg.workers)
                histDist_fp_fpFMM_m.append(d)
                hausdorffDist_fp_fpFMM_m.append(float(np.max(d)))
                endPointDist_fp_fpFMM_m.append(float(fp[-1, 1] - fp_fmm[-1, 1]))

            histDist_fp_lrp_sm.append(histDist_fp_lrp_m)
            hausdorffDist_fp_lrp_sm.append(hausdorffDist_fp_lrp_m)
            endPointDist_fp_lrp_sm.append(endPointDist_fp_lrp_m)

            histDist_fp_fpDijkstra_sm.append(histDist_fp_fpDijkstra_m)
            hausdorffDist_fp_fpDijkstra_sm.append(hausdorffDist_fp_fpDijkstra_m)
            endPointDist_fp_fpDijkstra_sm.append(endPointDist_fp_fpDijkstra_m)

            histDist_fp_fpFMM_sm.append(histDist_fp_fpFMM_m)
            hausdorffDist_fp_fpFMM_sm.append(hausdorffDist_fp_fpFMM_m)
            endPointDist_fp_fpFMM_sm.append(endPointDist_fp_fpFMM_m)

        k = f"std_{std_}"
        data_path[k] = {
            "field_size": field_size,
            "histDist_fp_lrp_sm": histDist_fp_lrp_sm,
            "hausdorffDist_fp_lrp_sm": hausdorffDist_fp_lrp_sm,
            "endPointDist_fp_lrp_sm": endPointDist_fp_lrp_sm,
            "histDist_fp_fpDijkstra_sm": histDist_fp_fpDijkstra_sm,
            "hausdorffDist_fp_fpDijkstra_sm": hausdorffDist_fp_fpDijkstra_sm,
            "endPointDist_fp_fpDijkstra_sm": endPointDist_fp_fpDijkstra_sm,
            "histDist_fp_fpFMM_sm": histDist_fp_fpFMM_sm,
            "hausdorffDist_fp_fpFMM_sm": hausdorffDist_fp_fpFMM_sm,
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
        "hausdorffDist": {
            "fp_lrp": [],
            "fp_fpDijkstra": [],
            "fp_fpFMM": [],
        },
        "endPointDist": {
            "fp_lrp": [],
            "fp_fpDijkstra": [],
            "fp_fpFMM": [],
        },
    }

    for std_key in stats_path:
        stats_std["hausdorffDist"]["fp_lrp"].append(stats_path[std_key]["hausdorffDist_fp_lrp_sm"])
        stats_std["hausdorffDist"]["fp_fpDijkstra"].append(stats_path[std_key]["hausdorffDist_fp_fpDijkstra_sm"])
        stats_std["hausdorffDist"]["fp_fpFMM"].append(stats_path[std_key]["hausdorffDist_fp_fpFMM_sm"])
        stats_std["endPointDist"]["fp_lrp"].append(stats_path[std_key]["endPointDist_fp_lrp_sm"])
        stats_std["endPointDist"]["fp_fpDijkstra"].append(stats_path[std_key]["endPointDist_fp_fpDijkstra_sm"])
        stats_std["endPointDist"]["fp_fpFMM"].append(stats_path[std_key]["endPointDist_fp_fpFMM_sm"])

    for key1 in stats_std:
        for key2 in stats_std[key1]:
            stats_std[key1][key2] = np.asarray(stats_std[key1][key2], dtype=float)

    return StdResults(data_path=data_path, stats_path=stats_path, stats_std=stats_std), warnings
