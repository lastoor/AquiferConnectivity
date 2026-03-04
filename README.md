# DataAnalysis_Simple

`DataAnalysis_Simple` is a debug-first rewrite of `DataAnalysis` with a flat workflow:

1. `scripts/step1_index.py`
2. `scripts/step2_metrics.py`
3. `scripts/step3_stats.py`
4. `scripts/step4_plot.py`

All implementation logic is inside the step scripts (no `visualization/` package).

## Repository Layout

- `configs/visualization.default.yaml`: default pipeline config
- `scripts/step1_index.py`: discover files and cache index
- `scripts/step2_metrics.py`: compute metrics and aggregate raw stats
- `scripts/step3_stats.py`: prepare plot-ready arrays
- `scripts/step4_plot.py`: render and save figures + manifest
- `scripts/run_all.py`: one-shot runner for all four stages
- `tests/`: focused tests for each stage and end-to-end smoke run

## Agent Workflow

- Read `AGENT.md` first, then follow `agent/guide.md`.
- Append task entries to `agent/log.md` for each meaningful change.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

From `DataAnalysis_Simple` root:

```bash
PYTHONPATH=./scripts python scripts/run_all.py --config configs/visualization.default.yaml --dry-run
PYTHONPATH=./scripts python scripts/run_all.py --config configs/visualization.default.yaml
```

## Stage-by-Stage Run

```bash
CACHE_DIR=outputs/cache/run3

PYTHONPATH=./scripts python scripts/step1_index.py  --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=./scripts python scripts/step2_metrics.py --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=./scripts python scripts/step3_stats.py   --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=./scripts python scripts/step4_plot.py    --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
```

## Small Test Run

Use Stage 2 limit flags for faster debugging:

```bash
PYTHONPATH=./scripts python scripts/step2_metrics.py \
  --config configs/visualization.default.yaml \
  --cache-dir "$CACHE_DIR" \
  --max-var-values 2 \
  --max-std-values 2 \
  --max-sizes-per-value 2 \
  --max-mc-per-size 10
```

## Output Naming Policy

Stage 4 and `run_all.py` support both policies:

- Stable names (default): overwrite predictable names like `run3_times.png`
- Unique names: add timestamp/hash suffixes

```bash
# default stable names
PYTHONPATH=./scripts python scripts/step4_plot.py --config configs/visualization.default.yaml

# unique names
PYTHONPATH=./scripts python scripts/step4_plot.py --config configs/visualization.default.yaml --unique-filenames
```

## Cache Contracts

Artifacts written under `outputs/cache/run<id>/`:

- `stage1_index.pkl`, `stage1_index.meta.json`
- `stage2_metrics.pkl`, `stage2_metrics.meta.json`
- `stage3_plot_inputs.pkl`, `stage3_plot_inputs.meta.json`

Each stage metadata file includes timestamp, config snapshot, and summary notes.

## Tests

```bash
pytest -q
```

## Notes

- Optional dependency: `shapely` for geometric distance calculations.
- If `shapely` is missing, related metrics are safely returned as `NaN`.
