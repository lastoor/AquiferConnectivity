#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from visualization.config import load_config
from visualization.pipeline.cache import default_cache_dir, load_pickle, save_pickle, write_stage_metadata
from visualization.pipeline.prepare import prepare_plot_inputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3: prepare plotting inputs from metric caches")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cache_dir = args.cache_dir or default_cache_dir(cfg)

    metrics = load_pickle(cache_dir / "stage2_metrics.pkl")
    var_results = metrics["var_results"]
    std_results = metrics["std_results"]

    plot_inputs = prepare_plot_inputs(cfg, var_results, std_results)
    payload = {
        "plot_inputs": plot_inputs,
        "warnings": metrics.get("warnings", {}),
    }
    save_pickle(cache_dir / "stage3_plot_inputs.pkl", payload)

    notes = {
        "cache_file": str(cache_dir / "stage3_plot_inputs.pkl"),
        "has_std_inputs": plot_inputs["stats_std"] is not None,
    }
    write_stage_metadata(cache_dir / "stage3_plot_inputs.meta.json", stage="stage3_plot_inputs", config=cfg, notes=notes)
    print(f"Wrote plot-input cache: {cache_dir / 'stage3_plot_inputs.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
