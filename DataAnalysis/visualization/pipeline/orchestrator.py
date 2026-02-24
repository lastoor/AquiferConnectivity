from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from visualization.config import VisualizationConfig
from visualization.io.writers import save_figure, write_manifest
from visualization.pipeline.run_std import run_std_analysis
from visualization.pipeline.run_variance import run_variance_analysis
from visualization.plots.distances import (
    plot_endpoint_vs_variance,
    plot_hausdorff_vs_variance,
    plot_std_endpoint,
    plot_std_hausdorff,
)
from visualization.plots.style import configure_style
from visualization.plots.times import (
    plot_characteristic_time,
    plot_scatter_tmin_vs_rmin,
    plot_taylor_comparison,
)
from visualization.stats.transforms import compute_i_y, compute_t_ref, compute_taylor_times
from visualization.types import FigureArtifact


def _collect_primary_stats(stats: dict[str, dict], var_values: list[float]) -> dict[str, np.ndarray]:
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


def _collect_integral_stats(line_integrals: dict[str, dict], var_values: list[float]) -> dict[str, np.ndarray]:
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


def run(cfg: VisualizationConfig) -> list[FigureArtifact]:
    configure_style(cfg)
    artifacts: list[FigureArtifact] = []
    all_warnings: list[str] = []

    var_results, warnings = run_variance_analysis(cfg)
    all_warnings.extend(warnings)

    s = _collect_primary_stats(var_results.stats, cfg.variance_values)
    ints = _collect_integral_stats(var_results.line_integrals, cfg.variance_values)

    var = np.asarray(cfg.variance_values, dtype=float)
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

    if "times" in cfg.sections:
        fig, meta = plot_characteristic_time(var, s["stats_fpt"], s["stats_mhr"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], t_ref)
        artifacts.extend(save_figure(fig, section="times", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
        plt.close(fig)

    if "taylor" in cfg.sections:
        fig, meta = plot_taylor_comparison(var, s["stats_mhr"], s["stats_fpt"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], taylor, t_ref)
        artifacts.extend(save_figure(fig, section="taylor", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
        plt.close(fig)

    if "scatter" in cfg.sections:
        fig, meta = plot_scatter_tmin_vs_rmin(s["stats_fpt"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], s["stats_mhr"], t_ref)
        artifacts.extend(save_figure(fig, section="scatter", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
        plt.close(fig)

    if "hausdorff" in cfg.sections:
        fig, meta = plot_hausdorff_vs_variance(var, s["stats_hausdorff_lrp"], s["stats_hausdorff_fmm"], s["stats_hausdorff_dijkstra"], i_y)
        artifacts.extend(save_figure(fig, section="hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
        plt.close(fig)

    if "endpoint" in cfg.sections:
        fig, meta = plot_endpoint_vs_variance(var, s["stats_endpoint_lrp"], s["stats_endpoint_fmm"], s["stats_endpoint_dijkstra"], i_y)
        artifacts.extend(save_figure(fig, section="endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
        plt.close(fig)

    if "std_hausdorff" in cfg.sections or "std_endpoint" in cfg.sections:
        std_results, std_warnings = run_std_analysis(cfg)
        all_warnings.extend(std_warnings)
        x_var = np.asarray(cfg.std_values, dtype=float) ** 2

        if "std_hausdorff" in cfg.sections:
            fig, meta = plot_std_hausdorff(x_var, std_results.stats_std, i_y)
            artifacts.extend(save_figure(fig, section="std_hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
            plt.close(fig)

        if "std_endpoint" in cfg.sections:
            fig, meta = plot_std_endpoint(x_var, std_results.stats_std, i_y)
            artifacts.extend(save_figure(fig, section="std_endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
            plt.close(fig)

    write_manifest(cfg.output_dir, config=asdict(cfg), artifacts=artifacts, warnings=all_warnings)
    return artifacts
