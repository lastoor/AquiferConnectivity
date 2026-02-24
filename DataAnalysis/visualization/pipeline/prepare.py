from __future__ import annotations

import numpy as np

from visualization.config import VisualizationConfig
from visualization.stats.transforms import compute_i_y, compute_t_ref, compute_taylor_times
from visualization.types import StdResults, VarianceResults


def collect_primary_stats(stats: dict[str, dict], var_values: list[float]) -> dict[str, np.ndarray]:
    stats_lrp = []
    stats_fp = []
    stats_fp_dijkstra = []
    stats_fp_fmm = []
    stats_mhr = []
    stats_fpt = []
    stats_fpt_dijkstra = []
    stats_fpt_fmm = []
    stats_hausdorff_lrp = []
    stats_hausdorff_dijkstra = []
    stats_hausdorff_fmm = []
    stats_endpoint_lrp = []
    stats_endpoint_dijkstra = []
    stats_endpoint_fmm = []

    for var_ in var_values:
        key = f"var_{var_}"
        entry = stats[key]
        stats_lrp.append(entry["lengths"]["lrp"][0, :])
        stats_fp.append(entry["lengths"]["fp"][0, :])
        stats_fp_dijkstra.append(entry["lengths"]["fp_dijkstra"][0, :])
        stats_fp_fmm.append(entry["lengths"]["fp_fmm"][0, :])
        stats_mhr.append(entry["times"]["mhr"][0, :])
        stats_fpt.append(entry["times"]["fpt"][0, :])
        stats_fpt_dijkstra.append(entry["times"]["fpt_dijkstra"][0, :])
        stats_fpt_fmm.append(entry["times"]["fpt_fmm"][0, :])
        stats_hausdorff_lrp.append(entry["distances"]["hausdorff"]["fp_lrp"][0, :])
        stats_hausdorff_dijkstra.append(entry["distances"]["hausdorff"]["fp_fpDijkstra"][0, :])
        stats_hausdorff_fmm.append(entry["distances"]["hausdorff"]["fp_fpFMM"][0, :])
        stats_endpoint_lrp.append(entry["distances"]["end_point"]["fp_lrp"][0, :])
        stats_endpoint_dijkstra.append(entry["distances"]["end_point"]["fp_fpDijkstra"][0, :])
        stats_endpoint_fmm.append(entry["distances"]["end_point"]["fp_fpFMM"][0, :])

    return {
        "stats_lrp": np.asarray(stats_lrp, dtype=float),
        "stats_fp": np.asarray(stats_fp, dtype=float),
        "stats_fp_dijkstra": np.asarray(stats_fp_dijkstra, dtype=float),
        "stats_fp_fmm": np.asarray(stats_fp_fmm, dtype=float),
        "stats_mhr": np.asarray(stats_mhr, dtype=float),
        "stats_fpt": np.asarray(stats_fpt, dtype=float),
        "stats_fpt_dijkstra": np.asarray(stats_fpt_dijkstra, dtype=float),
        "stats_fpt_fmm": np.asarray(stats_fpt_fmm, dtype=float),
        "stats_hausdorff_lrp": np.asarray(stats_hausdorff_lrp, dtype=float),
        "stats_hausdorff_dijkstra": np.asarray(stats_hausdorff_dijkstra, dtype=float),
        "stats_hausdorff_fmm": np.asarray(stats_hausdorff_fmm, dtype=float),
        "stats_endpoint_lrp": np.asarray(stats_endpoint_lrp, dtype=float),
        "stats_endpoint_dijkstra": np.asarray(stats_endpoint_dijkstra, dtype=float),
        "stats_endpoint_fmm": np.asarray(stats_endpoint_fmm, dtype=float),
    }


def collect_integral_stats(line_integrals: dict[str, dict], var_values: list[float]) -> dict[str, np.ndarray]:
    out = {
        "int_fp": [],
        "int2_fp": [],
        "int_lrp": [],
        "int2_lrp": [],
        "int_fp_dijkstra": [],
        "int2_fp_dijkstra": [],
        "int_fp_fmm": [],
        "int2_fp_fmm": [],
    }
    for var_ in var_values:
        li = line_integrals[f"var_{var_}"]
        out["int_fp"].append([li["int_logk_fp"]["mean"], li["int_logk_fp"]["std"]])
        out["int2_fp"].append([li["int_logk2_fp"]["mean"], li["int_logk2_fp"]["std"]])
        out["int_lrp"].append([li["int_logk_lrp"]["mean"], li["int_logk_lrp"]["std"]])
        out["int2_lrp"].append([li["int_logk2_lrp"]["mean"], li["int_logk2_lrp"]["std"]])
        out["int_fp_dijkstra"].append([li["int_logk_fp_dijkstra"]["mean"], li["int_logk_fp_dijkstra"]["std"]])
        out["int2_fp_dijkstra"].append([li["int_logk2_fp_dijkstra"]["mean"], li["int_logk2_fp_dijkstra"]["std"]])
        out["int_fp_fmm"].append([li["int_logk_fp_fmm"]["mean"], li["int_logk_fp_fmm"]["std"]])
        out["int2_fp_fmm"].append([li["int_logk2_fp_fmm"]["mean"], li["int_logk2_fp_fmm"]["std"]])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def prepare_plot_inputs(cfg: VisualizationConfig, var_results: VarianceResults, std_results: StdResults | None) -> dict:
    var = np.asarray(cfg.variance_values, dtype=float)
    s = collect_primary_stats(var_results.stats, cfg.variance_values)
    ints = collect_integral_stats(var_results.line_integrals, cfg.variance_values)

    field_size = cfg.field_size_ref if var_results.field_size.size == 0 else float(var_results.field_size[0])
    i_y = compute_i_y(field_size, cfg.length_scale)
    t_ref = compute_t_ref(cfg.k_g, cfg.dh, field_size, cfg.length_scale)
    taylor = compute_taylor_times(
        s["stats_fp"],
        s["stats_lrp"],
        s["stats_fp_dijkstra"],
        s["stats_fp_fmm"],
        ints["int_fp"],
        ints["int_lrp"],
        ints["int_fp_dijkstra"],
        ints["int_fp_fmm"],
        ints["int2_fp"],
        ints["int2_lrp"],
        ints["int2_fp_dijkstra"],
        ints["int2_fp_fmm"],
        cfg.dh,
    )

    return {
        "var": var,
        "std_x": np.asarray(cfg.std_values, dtype=float) ** 2,
        "stats": s,
        "integrals": ints,
        "taylor": taylor,
        "i_y": i_y,
        "t_ref": t_ref,
        "stats_std": None if std_results is None else std_results.stats_std,
    }
