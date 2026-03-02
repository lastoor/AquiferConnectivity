#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from visualization.config import load_config
from visualization.pipeline.cache import (
    default_cache_dir,
    deserialize_discovered,
    load_pickle,
    save_pickle,
    write_stage_metadata,
)
from visualization.pipeline.run_std import run_std_analysis_from_discovered
from visualization.pipeline.run_variance import run_variance_analysis_from_discovered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: compute metrics and aggregate raw stats")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--skip-std", action="store_true")
    p.add_argument("--max-var-values", type=int, default=None, help="Limit number of variance values processed")
    p.add_argument("--max-std-values", type=int, default=None, help="Limit number of std values processed")
    p.add_argument("--max-sizes-per-value", type=int, default=None, help="Limit number of size runs per var/std value")
    p.add_argument("--max-mc-per-size", type=int, default=None, help="Limit number of realizations (MC) per size")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    total_steps = 4 if not args.skip_std else 3
    bar = tqdm(total=total_steps, unit="step", desc="Stage2")
    try:
        bar.set_description("Loading stage1 index")
        cfg = load_config(args.config)
        cache_dir = args.cache_dir or default_cache_dir(cfg)

        index = load_pickle(cache_dir / "stage1_index.pkl")
        discovered_var = deserialize_discovered(index["variance"])
        discovered_std = deserialize_discovered(index["std"])
        bar.update(1)

        bar.set_description("Computing variance metrics")
        var_results, warnings_var = run_variance_analysis_from_discovered(
            cfg,
            discovered_var,
            show_progress=True,
            max_var_values=args.max_var_values,
            max_sizes_per_var=args.max_sizes_per_value,
            max_mc_per_size=args.max_mc_per_size,
        )
        bar.update(1)

        std_results = None
        warnings_std: list[str] = []
        if not args.skip_std:
            bar.set_description("Computing std metrics")
            std_results, warnings_std = run_std_analysis_from_discovered(
                cfg,
                discovered_std,
                show_progress=True,
                max_std_values=args.max_std_values,
                max_sizes_per_std=args.max_sizes_per_value,
                max_mc_per_size=args.max_mc_per_size,
            )
            bar.update(1)

        bar.set_description("Writing stage2 cache")
        payload = {
            "var_results": var_results,
            "std_results": std_results,
            "warnings": {"variance": warnings_var, "std": warnings_std},
        }
        save_pickle(cache_dir / "stage2_metrics.pkl", payload)

        notes = {
            "cache_file": str(cache_dir / "stage2_metrics.pkl"),
            "warnings_variance": len(warnings_var),
            "warnings_std": len(warnings_std),
            "std_computed": std_results is not None,
            "max_var_values": args.max_var_values,
            "max_std_values": args.max_std_values,
            "max_sizes_per_value": args.max_sizes_per_value,
            "max_mc_per_size": args.max_mc_per_size,
        }
        write_stage_metadata(cache_dir / "stage2_metrics.meta.json", stage="stage2_metrics", config=cfg, notes=notes)
        bar.update(1)
    finally:
        bar.close()

    print(f"Wrote metrics cache: {cache_dir / 'stage2_metrics.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
