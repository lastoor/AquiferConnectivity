#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from visualization.config import load_config
from visualization.io.writers import save_figure, write_manifest
from visualization.pipeline.cache import default_cache_dir, load_pickle
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4: render figures from cached plotting inputs")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument(
        "--unique-filenames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include timestamp/hash in output figure filenames",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cache_dir = args.cache_dir or default_cache_dir(cfg)

    payload = load_pickle(cache_dir / "stage3_plot_inputs.pkl")
    prepared = payload["plot_inputs"]
    warning_dict = payload.get("warnings", {})
    all_warnings = warning_dict.get("variance", []) + warning_dict.get("std", [])

    configure_style(cfg)
    artifacts = []

    var = prepared["var"]
    s = prepared["stats"]
    taylor = prepared["taylor"]
    i_y = prepared["i_y"]
    t_ref = prepared["t_ref"]
    x_var = prepared["std_x"]

    if "times" in cfg.sections:
        fig, meta = plot_characteristic_time(var, s["stats_fpt"], s["stats_mhr"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], t_ref)
        artifacts.extend(save_figure(fig, section="times", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=args.unique_filenames))
        plt.close(fig)

    if "taylor" in cfg.sections:
        fig, meta = plot_taylor_comparison(var, s["stats_mhr"], s["stats_fpt"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], taylor, t_ref)
        artifacts.extend(save_figure(fig, section="taylor", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=args.unique_filenames))
        plt.close(fig)

    if "scatter" in cfg.sections:
        fig, meta = plot_scatter_tmin_vs_rmin(s["stats_fpt"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], s["stats_mhr"], t_ref)
        artifacts.extend(save_figure(fig, section="scatter", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=args.unique_filenames))
        plt.close(fig)

    if "hausdorff" in cfg.sections:
        fig, meta = plot_hausdorff_vs_variance(var, s["stats_hausdorff_lrp"], s["stats_hausdorff_fmm"], s["stats_hausdorff_dijkstra"], i_y)
        artifacts.extend(save_figure(fig, section="hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=args.unique_filenames))
        plt.close(fig)

    if "endpoint" in cfg.sections:
        fig, meta = plot_endpoint_vs_variance(var, s["stats_endpoint_lrp"], s["stats_endpoint_fmm"], s["stats_endpoint_dijkstra"], i_y)
        artifacts.extend(save_figure(fig, section="endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=args.unique_filenames))
        plt.close(fig)

    if "std_hausdorff" in cfg.sections and prepared["stats_std"] is not None:
        fig, meta = plot_std_hausdorff(x_var, prepared["stats_std"], i_y)
        artifacts.extend(save_figure(fig, section="std_hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=args.unique_filenames))
        plt.close(fig)

    if "std_endpoint" in cfg.sections and prepared["stats_std"] is not None:
        fig, meta = plot_std_endpoint(x_var, prepared["stats_std"], i_y)
        artifacts.extend(save_figure(fig, section="std_endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=args.unique_filenames))
        plt.close(fig)

    write_manifest(cfg.output_dir, config=asdict(cfg), artifacts=artifacts, warnings=all_warnings)
    print(f"Generated {len(artifacts)} figure artifact(s) in {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
