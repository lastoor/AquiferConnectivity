# DataAnalysis

This repository contains post-processing and visualization tooling for Fast Marching Method experiment outputs.

## Current Structure
- `visualization/`: modular analysis package
- `visualization/io/discovery.py`: file discovery and path-format logic
- `visualization/pipeline/`: orchestration, staged preparation, cache helpers
- `scripts/visualize.py`: one-shot pipeline CLI
- `scripts/step1_index.py`: stage 1, discover and save data index cache
- `scripts/step2_metrics.py`: stage 2, compute metrics/stats from index cache
- `scripts/step3_stats.py`: stage 3, prepare plot-ready arrays from metric cache
- `scripts/step4_plot.py`: stage 4, render plots from plot-input cache
- `configs/visualization.default.yaml`: default run configuration
- `visualization.py`: backward-compatible wrapper that runs default config
- `agent/guide.md`: required instructions for CLI agents
- `agent/log.md`: append-only agent work log
- `AGENT.md`: Codex entry instruction that points to `agent/guide.md`

## One-Shot Run
From repository root:

```bash
PYTHONPATH=. python scripts/visualize.py --config configs/visualization.default.yaml --dry-run
PYTHONPATH=. python scripts/visualize.py --config configs/visualization.default.yaml
```

Backward-compatible wrapper:

```bash
PYTHONPATH=. python visualization.py
```

## Staged Run (Cache Between Steps)
This is useful for notebook-style debugging and incremental reruns.

```bash
# Optional: choose cache location
CACHE_DIR=outputs/cache/run3

PYTHONPATH=. python scripts/step1_index.py  --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=. python scripts/step2_metrics.py --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=. python scripts/step3_stats.py   --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=. python scripts/step4_plot.py    --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
```

Stage artifacts:
- `stage1_index.pkl` + `stage1_index.meta.json`
- `stage2_metrics.pkl` + `stage2_metrics.meta.json`
- `stage3_plot_inputs.pkl` + `stage3_plot_inputs.meta.json`

## Path Format and Custom Data Layout
If your folders do not follow the default pattern, edit:
- `visualization/io/discovery.py`

Default expected layout:
- `.../run<run_id>/var_<v>/lengthScale<ls>_size<sz>/...`
- `.../run<run_id>/std_<s>/lengthScale<ls>_size<sz>/...`

Adjust these glob patterns in:
- `discover_variance_runs(...)`
- `discover_std_runs(...)`

## Config
Main fields in `configs/visualization.default.yaml`:
- `data_root`, `run_id`, `length_scale`, `dh`, `field_size_ref`, `k_g`
- `variance_values`, `std_values`
- `output_dir`, `figure_dpi`, `workers`, `save_formats`, `sections`
- `font.*`

## Outputs
Generated figures are saved to `output_dir` with versioned names:
- `run<id>_<section>_<timestamp>_<hash>.<ext>`

Manifest:
- `<output_dir>/manifest.json`

## Dependencies
Expected Python packages:
- `numpy`, `matplotlib`, `scipy`, `PyYAML`
- Optional: `shapely` (if absent, Shapely-based distances become NaN)
- Optional styling: `cmcrameri`

## Agent Workflow
All CLI agents must:
1. Read `agent/guide.md` first.
2. Update `README.md` when workflows/structure/config/outputs/dependencies change.
3. Append a task entry in `agent/log.md`.
