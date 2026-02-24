#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cache_dir = args.cache_dir or default_cache_dir(cfg)

    index = load_pickle(cache_dir / "stage1_index.pkl")
    discovered_var = deserialize_discovered(index["variance"])
    discovered_std = deserialize_discovered(index["std"])

    var_results, warnings_var = run_variance_analysis_from_discovered(cfg, discovered_var)
    std_results = None
    warnings_std: list[str] = []
    if not args.skip_std:
        std_results, warnings_std = run_std_analysis_from_discovered(cfg, discovered_std)

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
    }
    write_stage_metadata(cache_dir / "stage2_metrics.meta.json", stage="stage2_metrics", config=cfg, notes=notes)
    print(f"Wrote metrics cache: {cache_dir / 'stage2_metrics.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
