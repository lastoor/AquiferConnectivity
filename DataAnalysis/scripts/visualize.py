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

    artifacts = run(cfg)
    print(f"Generated {len(artifacts)} figure artifact(s) in {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
