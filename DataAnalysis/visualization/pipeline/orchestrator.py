from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt

from visualization.config import VisualizationConfig
from visualization.io.writers import save_figure, write_manifest
from visualization.pipeline.prepare import prepare_plot_inputs
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
from visualization.types import FigureArtifact


def run(
    cfg: VisualizationConfig,
    *,
    max_var_values: int | None = None,
    max_std_values: int | None = None,
    max_sizes_per_value: int | None = None,
    max_mc_per_size: int | None = None,
) -> list[FigureArtifact]:
    configure_style(cfg)
    artifacts: list[FigureArtifact] = []
    all_warnings: list[str] = []

    var_results, warnings = run_variance_analysis(
        cfg,
        max_var_values=max_var_values,
        max_sizes_per_var=max_sizes_per_value,
        max_mc_per_size=max_mc_per_size,
    )
    all_warnings.extend(warnings)

    std_results = None
    if "std_hausdorff" in cfg.sections or "std_endpoint" in cfg.sections:
        std_results, std_warnings = run_std_analysis(
            cfg,
            max_std_values=max_std_values,
            max_sizes_per_std=max_sizes_per_value,
            max_mc_per_size=max_mc_per_size,
        )
        all_warnings.extend(std_warnings)

    prepared = prepare_plot_inputs(cfg, var_results, std_results)
    var = prepared["var"]
    s = prepared["stats"]
    taylor = prepared["taylor"]
    i_y = prepared["i_y"]
    t_ref = prepared["t_ref"]
    x_var = prepared["std_x"]

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

    if "std_hausdorff" in cfg.sections and prepared["stats_std"] is not None:
        fig, meta = plot_std_hausdorff(x_var, prepared["stats_std"], i_y)
        artifacts.extend(save_figure(fig, section="std_hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
        plt.close(fig)

    if "std_endpoint" in cfg.sections and prepared["stats_std"] is not None:
        fig, meta = plot_std_endpoint(x_var, prepared["stats_std"], i_y)
        artifacts.extend(save_figure(fig, section="std_endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta))
        plt.close(fig)

    write_manifest(cfg.output_dir, config=asdict(cfg), artifacts=artifacts, warnings=all_warnings)
    return artifacts
