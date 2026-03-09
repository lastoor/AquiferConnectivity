# DataAnalysis_Simple

`DataAnalysis_Simple` is a debug-first rewrite of `DataAnalysis` with a flat workflow:

1. `scripts/step1_index.py`
2. `scripts/step2_metrics.py`
3. `scripts/step3_stats.py`
4. `scripts/step4_plot.py`

Additional standalone utility:

- `scripts/step5_single_field_paths.py`: solve and plot `FP`, `FP (FMM)`, `FP (Dijkstra)`, and `LRP` for one selected logK realization.
- `scripts/step6_single_field_arrival_hist.py`: generate one GRF logK field, solve flow/particle tracking, and compare empirical arrival-time CDF of all successful particles against analytical `_g_tau`.

All implementation logic is inside the step scripts (no `visualization/` package).

## Repository Layout

- `configs/visualization.default.yaml`: default pipeline config
- `scripts/step1_index.py`: discover files and cache index
- `scripts/step2_metrics.py`: compute metrics and aggregate raw stats
- `scripts/step3_stats.py`: prepare plot-ready arrays
- `scripts/step4_plot.py`: render and save figures + manifest
- `scripts/run_all.py`: one-shot runner for all four stages
- `scripts/step5_single_field_paths.py`: single-realization path solving + overlay figures (`logK` and minimal arrival time)
- `scripts/step6_single_field_arrival_hist.py`: single-realization arrival-time CDF analysis (`status==0` particles) vs analytical `_g_tau`
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
PYTHONPATH=./scripts python scripts/run_all.py --config configs/visualization.default.yaml --output-dir ./outputs/visualization
```

## Stage-by-Stage Run

```bash
CACHE_DIR=outputs/cache/run3
FIGURE_DIR=outputs/visualization/run3

PYTHONPATH=./scripts python scripts/step1_index.py  --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=./scripts python scripts/step2_metrics.py --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR" --max-var-values 3 --max-mc-per-size 5 --hausdorff-compute-method custom --frechet-compute-method custom
PYTHONPATH=./scripts python scripts/step3_stats.py   --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=./scripts python scripts/step4_plot.py    --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR" --output-dir "$FIGURE_DIR" --plot-variances 0.1 --scatter-mode mean,all_points
```

`configs/visualization.default.yaml` no longer specifies `output_dir`.  
Stage 4 output location can be controlled with `--output-dir` in `step4_plot.py` or `run_all.py`.  
Custom YAML files may still include `output_dir`; CLI `--output-dir` takes precedence.

## Output Directory Resolution

The effective Stage 4 output directory is selected in this order:

1. `--output-dir` passed to `scripts/step4_plot.py` or `scripts/run_all.py`
2. `output_dir` field in your custom YAML config (optional)
3. Built-in fallback: `./outputs/visualization`

## Small Test Run

Use Stage 2 limit flags for faster debugging:

```bash
PYTHONPATH=./scripts python scripts/step2_metrics.py \
  --config configs/visualization.default.yaml \
  --cache-dir "$CACHE_DIR" \
  --hausdorff-compute-method custom \
  --frechet-compute-method custom \
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

# override output directory
PYTHONPATH=./scripts python scripts/step4_plot.py --config configs/visualization.default.yaml --output-dir ./outputs/visualization

# choose variance values for separate path-length CDF and scatter(all_points) figures
PYTHONPATH=./scripts python scripts/step4_plot.py --config configs/visualization.default.yaml --plot-variances 0.1,1.0 --scatter-mode all_points
```

## Single-Field Path Overlay

Use `step5_single_field_paths.py` to select one realization from output and regenerate path families:

```bash
PYTHONPATH=./scripts python scripts/step5_single_field_paths.py \
  --config configs/visualization.default.yaml \
  --sweep variance \
  --value 0.1 \
  --field-size 200 \
  --mc-index 0 \
  --output-dir "$FIGURE_DIR"
```

Outputs:

- `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_logk_paths.png`
- `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_arrival_paths.png`
- `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_speed_streamlines.png`

Notes:

- This script re-solves paths from a selected logK field and keeps the minimal-arrival-time and velocity fields in memory for plotting.
- All Step 5 figures omit title text and include source/target lines plus FP/FP(FMM)/FP(Dijkstra)/LRP paths.
- Spatial axes are normalized as `x/I_Y` and `y/I_Y`.
- Arrival-time background is normalized as `T_min/t_ref`, with `t_ref = I_Y / (k_g * dh)`.
- Speed/streamline background uses dimensionless velocity magnitude `|v|/(k_g*dh)` and streamlines from `v/(k_g*dh)`.
- Solver imports require both `Algorithm/src` and the sibling `FlowSimulation` repo.
- Override FlowSimulation path with `FLOWSIM_PATH=/path/to/FlowSimulation` if needed.

## Single-Field Arrival-Time CDF vs Analytical

Use `step6_single_field_arrival_hist.py` to generate one GRF realization and compare
the empirical CDF of particle arrival times (successful exits only, `status==0`) with
analytical `_g_tau` from `step4_plot.py`.

```bash
PYTHONPATH=./scripts python scripts/step6_single_field_arrival_hist.py \
  --config configs/visualization.default.yaml \
  --field-size 200 \
  --std 1.0 \
  --seed 0 \
  --save-npz \
  --save-json \
  --output-dir "$FIGURE_DIR"
```

Outputs:

- `<output_dir>/run<id>_singlefield_std<std>_var<var>_size<size>_seed<seed>_arrival_cdf_vs_analytical.png`
- Optional (when `--save-npz`): `<output_dir>/run<id>_singlefield_std<std>_var<var>_size<size>_seed<seed>_arrival_times.npz`
- Optional (when `--save-json`): `<output_dir>/run<id>_singlefield_std<std>_var<var>_size<size>_seed<seed>_arrival_summary.json`

Notes:

- GRF generation uses exponential covariance (`ExpCov`) with `GaussianRandomFieldCircEmbed`.
- Hydraulic head and velocity are solved using the same solver stack as Step 5.
- Arrival times use one terminal time per successful particle (`status==0`, final trajectory timestamp), not flattened per-step trajectory times.
- Comparison is done in dimensionless time `tU/I_Y` (`U = k_g * dh`, `I_Y = length_scale * field_size`).
- By default, Step 6 saves only the figure; use `--save-npz` and/or `--save-json` to persist data artifacts.
- Script requires the external `Algorithm/src` + `FlowSimulation` runtime stack; set `FLOWSIM_PATH` if needed.

## Figure Sections

Default `sections` now include:

- `times`
- `taylor`
- `scatter`
- `hausdorff`
- `frechet`
- `endpoint`
- `std_hausdorff`
- `std_endpoint`
- `ratio_mean`: plots `mean(t_min/(R_min/dh))` vs variance for FP, FP(FMM), FP(Dijkstra)
- `ratio_var`: plots `var(t_min/(R_min/dh))` vs variance for FP, FP(FMM), FP(Dijkstra)
- `wasserstein_len`: plots normalized Wasserstein-1 distance between path-length distributions (CDF proxy) for FP-LRP, FP-FMM, FP-Dijkstra
  - also writes separate path-length CDF figure(s) `run<id>_cdf_len_var<value>.png` for selected variance values
  - use `--plot-variances` (comma-separated) to choose variance values; default is the first available variance
- `scatter`: supports `--scatter-mode mean`, `--scatter-mode all_points`, or `--scatter-mode mean,all_points`
  - all-points mode writes `run<id>_scatter_var<value>.png` for selected `--plot-variances`

When enabled, the Frechet section is saved as `run<id>_frechet.png` (or corresponding configured format).

## Cache Contracts

Artifacts written under `outputs/cache/run<id>/`:

- `stage1_index.pkl`, `stage1_index.meta.json`
- `stage2_metrics.pkl`, `stage2_metrics.meta.json`
- `stage3_plot_inputs.pkl`, `stage3_plot_inputs.meta.json`

Each stage metadata file includes timestamp, config snapshot, and summary notes.

Compatibility note:
- Stage 3 can read legacy Stage 2 caches produced by script-mode runs (`python scripts/step2_metrics.py`) and current plain-dict Stage 2 caches.

## Tests

```bash
pytest -q
```

## Hausdorff Comparison Utility

Use `tests/compare_distance_by_shapely_with_custom.py` to compare distances from:
- Shapely (`hausdorff_distance`)
- Stage-2 custom Hausdorff (max symmetric nearest-neighbor distance with `ds=1.0`)
- Shapely (`frechet_distance`)
- Stage-2 custom discrete Frechet implementation

For one selected variance sample and one MC label:

```bash
PYTHONPATH=./scripts python tests/compare_distance_by_shapely_with_custom.py \
  --config configs/visualization.default.yaml \
  --variance 0.1 \
  --field-size 200 \
  --mc 1
```

## Notes

- Optional dependency: `shapely` for geometric distance calculations.
- If `shapely` is missing, related metrics are safely returned as `NaN`.
- `times` plot includes an analytical `t_min` overlay controlled by `analytical_percentile` (default `0.01`).
- Analytical solve failures are recorded as warnings in `<output_dir>/manifest.json`.
- `wasserstein_len` is normalized by `I_Y` and computed from Stage 2 path-length sample arrays.
- Separate CDF figures use empirical CDF of normalized path length (`\ell/I_Y`) for FP, LRP, FP(FMM), FP(Dijkstra).
