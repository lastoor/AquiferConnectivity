# DataAnalysis_Simple

`DataAnalysis_Simple` is a debug-first rewrite of `DataAnalysis` with a flat workflow:

1. `scripts/step1_index.py`
2. `scripts/step2_metrics.py`
3. `scripts/step3_stats.py`
4. `scripts/step4_plot.py`

Additional standalone utility:

- `scripts/step5_single_field_paths.py`: solve and plot `FP`, `FP (FMM)`, `FP (Dijkstra)`, `LRP`, and `LRP (FMM)` for one selected logK realization.
- `scripts/step6_single_field_arrival_hist.py`: generate one GRF logK field, solve flow/particle tracking, and compare empirical arrival-time CDF of all successful particles against analytical `_g_tau`.

All implementation logic is inside the step scripts (no `visualization/` package).

## Repository Layout

- `configs/visualization.default.yaml`: default pipeline config
- `scripts/step1_index.py`: discover files and cache index
- `scripts/step2_metrics.py`: compute metrics and aggregate raw stats
- `scripts/step3_stats.py`: prepare plot-ready arrays
- `scripts/step4_plot.py`: render and save figures + manifest
- `scripts/run_all.py`: one-shot runner for all four stages
- `scripts/step5_single_field_paths.py`: single-realization path solving + overlay figures (`logK`, minimal arrival time, and speed/streamlines)
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
PYTHONPATH=./scripts python scripts/step2_metrics.py --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR" --hausdorff-compute-method custom 
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
- All Step 5 figures omit title text and include source/target lines plus FP/FP(FMM)/FP(Dijkstra)/LRP/LRP(FMM) paths.
- Spatial axes are normalized as `x/I_Y` and `y/I_Y`.
- Arrival-time background is normalized as `T_min/t_ref`, with `t_ref = I_Y / (k_g * dh)`.
- Speed/streamline background uses dimensionless velocity magnitude `|v|/(k_g*dh)` and streamlines from `v/(k_g*dh)`.
- Particle tracking now supports `--solver pollock` or `--solver rk45`; defaults mirror the Algorithm runner: `--solver pollock`, `--velocity-location face`, `--no-align-with-modflow-velocity`.
- Pollock requires `--velocity-location face`; the plotted `FP` path remains solver-neutral and reflects the selected particle-tracking backend.
- Solver imports require both `Algorithm/src` and the sibling `FlowSimulation` repo.
- Override FlowSimulation path with `FLOWSIM_PATH=/path/to/FlowSimulation` if needed.

## Single-Field Arrival-Time CDF vs Analytical

Use `step6_single_field_arrival_hist.py` to generate one GRF realization and compare
the empirical CDF of particle arrival times (successful exits only, `status==0`) with
analytical `_g_tau` from `step4_plot.py` and a lognormal CDF approximation.

The Step 6 geometry is now explicit:

- analytical transport length is derived from `target_x - source_x`
- source seeds use the half-open span `[source_y_start, source_y_end)`
- target y bounds are accepted and saved, but are currently metadata only; arrival stopping still uses `target_x`

```bash
PYTHONPATH=./scripts python scripts/step6_single_field_arrival_hist.py \
  --config configs/visualization.default.yaml \
  --field-size 200 \
  --var 5.0 \
  --n-mc 10 \
  --seed 0 \
  --density-particle 20 \
  --source-x 19 \
  --target-x 179 \
  --source-y-start 19 \
  --source-y-end 179 \
  --refine-tau-grid \
  --tau-split 10 \
  --tau-grid-fine 1024 \
  --tau-grid-coarse 256 \
  --y-exp 0.3 \
  --pdf-hist-bins 48 \
  --save-pdf-plot \
  --solver "pollock" \
  --velocity-location "face" \
  --align-with-modflow-velocity False \
  --save-npz \
  --save-json \
  --output-dir "$FIGURE_DIR"
```

Outputs:

- `<output_dir>/run<id>_singlefield_std<std>_var<var>_size<size>_seed<seed>_arrival_cdf_vs_analytical.png`
- Optional (when `--save-pdf-plot`): `<output_dir>/run<id>_singlefield_std<std>_var<var>_size<size>_seed<seed>_arrival_pdf_vs_analytical.png`
- Optional (when `--save-npz`): `<output_dir>/run<id>_singlefield_std<std>_var<var>_size<size>_seed<seed>_arrival_times.npz`
- Optional (when `--save-json`): `<output_dir>/run<id>_singlefield_std<std>_var<var>_size<size>_seed<seed>_arrival_summary.json`
- Reloaded figure from saved NPZ: `<output_dir>/<saved_npz_stem>_reloaded.png`
- Optional reload density figure (when `--save-pdf-plot`): `<output_dir>/<saved_npz_stem>_reloaded_pdf.png`

Reload a saved Step 6 NPZ artifact and regenerate the comparison figure without rerunning GRF/flow/particle tracking:

```bash
PYTHONPATH=./scripts python scripts/step6_single_field_arrival_hist.py \
  --load-npz "$FIGURE_DIR/run3_singlefield_std1_var1_size200_seed0_mc10_arrival_times.npz"
```

Optionally choose a different output directory or DPI for the regenerated figure:

```bash
PYTHONPATH=./scripts python scripts/step6_single_field_arrival_hist.py \
  --load-npz "$FIGURE_DIR/run3_singlefield_std1_var1_size200_seed0_mc10_arrival_times.npz" \
  --refine-tau-grid \
  --tau-split 10 \
  --tau-grid-fine 1024 \
  --tau-grid-coarse 256 \
  --y-exp 0.3 \
  --pdf-hist-bins 48 \
  --save-pdf-plot \
  --output-dir "$FIGURE_DIR/reloaded" \
  --figure-dpi 200
```

Notes:

- GRF generation uses exponential covariance (`ExpCov`) with `GaussianRandomFieldCircEmbed`.
- Hydraulic head and velocity are solved using the same solver stack as Step 5.
- Arrival times use one terminal time per successful particle (`status==0`, final trajectory timestamp), not flattened per-step trajectory times.
- Comparison is done in dimensionless time `tU/I_Y` (`U = k_g * dh`, `I_Y = length_scale * field_size`).
- Step 6 uses the same particle-tracking CLI as Step 5: `--solver {pollock,rk45}`, `--velocity-location {face,cell}`, and `--align-with-modflow-velocity/--no-align-with-modflow-velocity`, with Pollock as the default.
- `--source-x` and `--target-x` control the horizontal transport distance used both by particle tracking and by the analytical/lognormal overlays.
- `--source-y-start` and `--source-y-end` define a half-open source span `[start, end)` used consistently for particle-count resolution and seed placement.
- `--target-y-start` and `--target-y-end` are currently persisted as metadata only; Step 6 arrival tracking still stops on `target_x`.
- The lognormal overlay uses dimensionless mean `L / I_Y` and variance `X_11(L / U) / I_Y^2`, converted to the standard lognormal parameters of `ln(tU/I_Y)`.
- By default, Step 6 saves only the figure; use `--save-npz` and/or `--save-json` to persist data artifacts.
- Use `--save-pdf-plot` to save an extra probability-density figure. It uses a normalized empirical histogram from `tau_samples`, MC histogram mean/band from `tau_samples_by_mc`, and analytical/lognormal density overlays recomputed from saved model metadata.
- Use `--pdf-hist-bins` to control the number of histogram bins for the probability-density figure. Effective histogram bin edges are saved in the NPZ so reload mode can reproduce the same PDF estimate.
- Use `--refine-tau-grid/--no-refine-tau-grid` to enable or disable the early-`tau` refinement.
- When refinement is enabled, the analytical/lognormal comparison grid is nonuniform: dense below `--tau-split` (quadratically clustered toward `tau_lo`) and coarser above it, controlled by `--tau-grid-fine` and `--tau-grid-coarse`.
- When refinement is disabled, Step 6 uses a uniform `tau_grid` across the full range with `tau_grid_fine + tau_grid_coarse - 1` total points.
- `--y-exp` controls the optional y-axis function scale. `--y-exp 1.0` keeps a linear y-axis; values such as `0.3` or `0.5` expand detail near CDF `0`.
- `--load-npz` switches Step 6 into visualization-only mode, so `--config`, `--std`, and `--var` are not required in that path.
- In reload mode, the same tau-grid options recompute analytical/lognormal/MC curves from saved NPZ metadata when available.
- In reload mode, the PDF figure is recalculated from saved samples and saved PDF histogram metadata when available.
- Older NPZ artifacts still reload; if they do not contain enough metadata and no sibling `_arrival_summary.json` is available, Step 6 falls back to the saved curves and prints a notice.
- Script requires the external `Algorithm/src` + `FlowSimulation` runtime stack; set `FLOWSIM_PATH` if needed.

## Figure Sections

Default `sections` now include:

- `times`
- `times_box`
- `taylor`
- `scatter`
- `hausdorff`
- `hausdorff_box`
- `frechet`
- `endpoint`
- `std_hausdorff`
- `std_endpoint`
- `ratio_mean`: plots `mean(t_min/(R_min/dh))` vs variance for FP, FP(FMM), FP(Dijkstra)
- `ratio_var`: plots `var(t_min/(R_min/dh))` vs variance for FP, FP(FMM), FP(Dijkstra)
- `wasserstein_len`: plots normalized Wasserstein-1 distance between path-length distributions (CDF proxy) for FP-LRP, FP-FMM, FP-Dijkstra
  - also writes separate path-length CDF figure(s) `run<id>_cdf_len_var<value>.png` for selected variance values
  - use `--plot-variances` (comma-separated) to choose variance values; default is the first available variance
- `times`: keeps the summary/error-bar characteristic-time figure, including the analytical `t_min` overlay
- `times_box`: writes a separate grouped box-plot figure of raw FP/LRP/LRP(FMM)/FP(FMM)/FP(Dijkstra) time samples across variances
- `hausdorff`: keeps the summary/error-bar Hausdorff figure
- `hausdorff_box`: writes a separate grouped box-plot figure of raw Hausdorff-distance samples across variances
- `endpoint`: writes a grouped box-plot figure of raw endpoint-distance samples across variances
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
- `times_box`, `hausdorff_box`, and `endpoint` are true sample-based box plots derived from Stage 3 cached raw samples.
- Analytical solve failures are recorded as warnings in `<output_dir>/manifest.json`.
- `wasserstein_len` is normalized by `I_Y` and computed from Stage 2 path-length sample arrays.
- Separate CDF figures use empirical CDF of normalized path length (`\ell/I_Y`) for FP, LRP, FP(FMM), FP(Dijkstra).
