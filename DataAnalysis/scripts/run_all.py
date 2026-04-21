#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import step1_index as step1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 4-stage data analysis pipeline")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument("--section", action="append", default=None, help="Optional section(s) override")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override for Stage 4 figures")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan only")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache directory")
    parser.add_argument("--skip-std", action="store_true", help="Skip std sweep in stage2")
    parser.add_argument("--max-var-values", type=int, default=None, help="Limit number of variance values processed")
    parser.add_argument("--max-std-values", type=int, default=None, help="Limit number of std values processed")
    parser.add_argument("--max-sizes-per-value", type=int, default=None, help="Limit number of size runs per var/std value")
    parser.add_argument("--max-mc-per-size", type=int, default=None, help="Limit number of realizations (MC) per size")
    parser.add_argument(
        "--unique-filenames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include timestamp/hash in output figure filenames",
    )
    return parser.parse_args()


def run_pipeline(cfg: step1.VisualizationConfig, args: argparse.Namespace) -> list[Any]:
    import step2_metrics as step2
    import step3_stats as step3
    import step4_plot as step4

    cache_dir = args.cache_dir
    step1.run_stage1(cfg, cache_dir=cache_dir)
    step2.run_stage2(
        cfg,
        cache_dir=cache_dir,
        skip_std=args.skip_std,
        max_var_values=args.max_var_values,
        max_std_values=args.max_std_values,
        max_sizes_per_value=args.max_sizes_per_value,
        max_mc_per_size=args.max_mc_per_size,
    )
    step3.run_stage3(cfg, cache_dir=cache_dir)
    return step4.run_stage4(cfg, cache_dir=cache_dir, unique_filenames=args.unique_filenames)


def main() -> int:
    args = parse_args()
    cfg = step1.load_config(args.config)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir.expanduser().resolve()
    if args.section:
        cfg.sections = args.section

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": cfg.run_id,
                    "data_root": str(cfg.data_root),
                    "sections": cfg.sections,
                    "output_dir": str(cfg.output_dir),
                    "cache_dir": str(args.cache_dir or step1.default_cache_dir(cfg)),
                    "unique_filenames": args.unique_filenames,
                },
                indent=2,
            )
        )
        return 0

    artifacts = run_pipeline(cfg, args)

    print(f"Generated {len(artifacts)} figure artifact(s) in {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
