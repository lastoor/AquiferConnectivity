#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from visualization.config import load_config
from visualization.io.discovery import discover_std_runs, discover_variance_runs
from visualization.pipeline.cache import (
    default_cache_dir,
    save_pickle,
    serialize_discovered,
    write_stage_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: discover and index data files")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cache_dir = args.cache_dir or default_cache_dir(cfg)

    discovered_var = discover_variance_runs(cfg)
    discovered_std = discover_std_runs(cfg)

    payload = {
        "variance": serialize_discovered(discovered_var),
        "std": serialize_discovered(discovered_std),
    }
    save_pickle(cache_dir / "stage1_index.pkl", payload)

    notes = {
        "variance_entries": {str(k): len(v) for k, v in payload["variance"].items()},
        "std_entries": {str(k): len(v) for k, v in payload["std"].items()},
        "cache_file": str(cache_dir / "stage1_index.pkl"),
    }
    write_stage_metadata(cache_dir / "stage1_index.meta.json", stage="stage1_index", config=cfg, notes=notes)
    print(f"Wrote index cache: {cache_dir / 'stage1_index.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
