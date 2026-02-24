"""Backward-compatible entrypoint for visualization workflow.

Use `scripts/visualize.py --config <yaml>` for full control.
"""

from __future__ import annotations

from pathlib import Path

from visualization import load_config, run


def main(config_path: str | Path = "configs/visualization.default.yaml") -> int:
    cfg = load_config(config_path)
    artifacts = run(cfg)
    print(f"Generated {len(artifacts)} figure artifact(s) in {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
