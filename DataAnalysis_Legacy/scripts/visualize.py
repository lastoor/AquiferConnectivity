#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualization import load_config, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastMarching visualization pipeline")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument("--section", action="append", default=None, help="Optional section(s) override")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan only")
    parser.add_argument("--max-var-values", type=int, default=None, help="Limit number of variance values processed")
    parser.add_argument("--max-std-values", type=int, default=None, help="Limit number of std values processed")
    parser.add_argument("--max-sizes-per-value", type=int, default=None, help="Limit number of size runs per var/std value")
    parser.add_argument("--max-mc-per-size", type=int, default=None, help="Limit number of realizations (MC) per size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.section:
        cfg.sections = args.section

    if args.dry_run:
        print(json.dumps({
            "run_id": cfg.run_id,
            "data_root": str(cfg.data_root),
            "sections": cfg.sections,
            "output_dir": str(cfg.output_dir),
        }, indent=2))
        return 0

    artifacts = run(
        cfg,
        max_var_values=args.max_var_values,
        max_std_values=args.max_std_values,
        max_sizes_per_value=args.max_sizes_per_value,
        max_mc_per_size=args.max_mc_per_size,
    )
    print(f"Generated {len(artifacts)} figure artifact(s) in {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
