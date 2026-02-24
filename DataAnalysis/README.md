# DataAnalysis

This repository contains post-processing and visualization tooling for Fast Marching Method experiment outputs.

## Current Structure
- `visualization/`: modular analysis package (I/O, metrics, stats, plots, pipeline)
- `scripts/visualize.py`: CLI entrypoint for running visualization pipeline
- `configs/visualization.default.yaml`: default run configuration
- `tests/`: unit and smoke tests for refactored modules
- `visualization.py`: backward-compatible wrapper that runs default config
- `agent/guide.md`: required instructions for CLI agents
- `agent/log.md`: append-only agent work log
- `AGENT.md`: Codex entry instruction that points to `agent/guide.md`

## Run
From repository root:

```bash
PYTHONPATH=. python scripts/visualize.py --config configs/visualization.default.yaml --dry-run
PYTHONPATH=. python scripts/visualize.py --config configs/visualization.default.yaml
```

Backward-compatible wrapper:

```bash
PYTHONPATH=. python visualization.py
```

## Config
Main config fields in `configs/visualization.default.yaml`:
- `data_root`, `run_id`, `length_scale`, `dh`, `field_size_ref`, `k_g`
- `variance_values`, `std_values`
- `output_dir`, `figure_dpi`, `workers`, `save_formats`, `sections`
- `font.*`

## Outputs
Generated figures are saved to `output_dir` using versioned names:
- `run<id>_<section>_<timestamp>_<hash>.<ext>`

A manifest is written at:
- `<output_dir>/manifest.json`

## Dependencies
Expected Python packages include:
- `numpy`, `matplotlib`, `scipy`, `PyYAML`
- Optional: `shapely` (if absent, Shapely-based distances become NaN)
- Optional styling: `cmcrameri`

## Agent Workflow
All CLI agents must:
1. Read `agent/guide.md` first.
2. Update `README.md` when workflows/structure/config/outputs/dependencies change.
3. Append a task entry in `agent/log.md`.
