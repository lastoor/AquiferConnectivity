# DataAnalysis_Simple

`DataAnalysis_Simple` is a debug-first rewrite of `DataAnalysis` with a flat workflow:

1. `scripts/step1_index.py`
2. `scripts/step2_metrics.py`
3. `scripts/step3_stats.py`
4. `scripts/step4_plot.py`

Additional standalone utilities:

- `scripts/step5_single_field_paths.py`: solve and plot particle-tracking `FP`, `FP (FMM)`, `LRP (FMM)`, and optional linearized `FP^i` paths for one selected logK realization.
- `scripts/step6_single_field_arrival_hist.py`: generate one GRF logK field, solve flow/particle tracking, and compare empirical arrival-time CDF of all successful particles against analytical `_g_tau`.
- `scripts/visualize_single_logk_paths.py`: load a consolidated per-MC `.npy` file and visualize the logK field with all saved paths overlaid.

All implementation logic is inside the step scripts (no `visualization/` package).

The comparison pipeline recognizes optional solver outputs from `Algorithm`, including `fp_fm_asr`, `fp_fm_asr_iso`, `fp_fm_asr_conic`, and `fp_pypar2`. When present in consolidated per-MC files or flat main-output fields, `fp_fm_asr_conic` is included in endpoint, Hausdorff, Frechet, travel-time, computation-time, path-length CDF, Wasserstein-length, and entropy/Hausdorff summaries.

## Repository Layout

- `configs/visualization.default.yaml`: default pipeline config
- `scripts/step1_index.py`: discover files and cache index
- `scripts/step2_metrics.py`: compute metrics and aggregate raw stats
- `scripts/step3_stats.py`: prepare plot-ready arrays
- `scripts/step4_plot.py`: render and save figures + manifest
- `scripts/run_all.py`: one-shot runner for all four stages
- `scripts/step5_single_field_paths.py`: single-realization path solving + overlay figures (`logK`, minimal arrival time, and speed/streamlines)
- `scripts/step6_single_field_arrival_hist.py`: single-realization arrival-time CDF analysis (`status==0` particles) vs analytical `_g_tau`
- `scripts/visualize_single_logk_paths.py`: quick visualization of a consolidated per-MC `.npy` file (logK + paths overlay + summary table)
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
CACHE_DIR=/project2/fbarros_324/binhaoli/task8_rMin_tMin/FastMarchingMethod/run16p1_pypar2-al0p1-at0p01_asr-ratio0p05-factor10/cache
FIGURE_DIR=outputs/run16p1_pypar2-al0p1-at0p01_asr-ratio0p05-factor10

PYTHONPATH=./scripts python scripts/step1_index.py  --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=./scripts python scripts/step2_metrics.py --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR" 
PYTHONPATH=./scripts python scripts/step3_stats.py   --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR"
PYTHONPATH=./scripts python scripts/step4_plot.py    --config configs/visualization.default.yaml --cache-dir "$CACHE_DIR" --output-dir "$FIGURE_DIR" --plot-variances 0.1,1,5 --scatter-mode mean,all_points --hausdorff-cdf --entropy-hausdorff
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

# ensemble-averaged Hausdorff distance CDF
PYTHONPATH=./scripts python scripts/step4_plot.py --config configs/visualization.default.yaml --plot-variances 0.1,1.0 --hausdorff-cdf
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
  --linearized-fp \
  --linearized-scheme linearized \
  --linearized-outer-max 20 \
  --linearized-damping 0.8 \
  --output-dir "$FIGURE_DIR"
```

Outputs:

- `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_logk_paths.png`
- `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_arrival_paths.png`
- `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_speed_streamlines.png`
- `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_curvature_balance.png`
- Optional (when `--linearized-fp`): `<output_dir>/run<id>_var<value>_size<size>_mc<idx>_linearized_fp_progression.png`

Notes:

- This script re-solves paths from a selected logK field and keeps the minimal-arrival-time and velocity fields in memory for plotting.
- Step 5 now uses FMM on slowness fields directly:
  unperturbed `LRP (FMM)` uses `1 / (K * |dh|)`, while `FP (FMM)` uses `1 / (K * |grad h|)`.
- Step 5 source/target geometry is now shared across particle tracking, `LRP (FMM)`, `FP (FMM)`, and linearized `FP^i`.
  Source seeds use the half-open span `[source_y_start, source_y_end)`, and the default target is the full right boundary `[target_y_start, target_y_end) = [0, field_size)`.
- All Step 5 figures omit title text and include source/target lines plus particle-tracking `FP`, `FP (FMM)`, and `LRP (FMM)`. Dijkstra-based `FP`/`LRP` overlays are no longer shown in the main figures.
- `--linearized-fp` enables the iterative linearized-Eikonal sequence `FP^i`, starting from `LRP (FMM)` and progressively approaching `FP (FMM)` under the full slowness field.
- `--linearized-scheme linearized` uses the current iterative linearized solver from `scripts/fp_and_lrp/linear_eikonal_solver.py`.
- `--linearized-scheme full_eikonal` instead solves the full Eikonal equation by FMM on intermediate slowness fields between the unperturbed and true slowness; `--linearized-outer-max` is the number of intermediate FMM steps.
- The linearized progression figure overlays all `FP^i` paths together, with `LRP (FMM)`, `FP (FMM)`, and particle-tracking `FP` retained as references.
- For `--linearized-scheme linearized`, the solver is controlled by `--linearized-outer-max`, `--linearized-inner-max-sweeps`, `--linearized-inner-tol`, `--linearized-eikonal-tol`, `--linearized-damping`, `--linearized-max-deltat`, `--linearized-monotone-residual/--no-linearized-monotone-residual`, and `--linearized-verbose`.
- `--linearized-damping` is the initial update factor `alpha` for the `linearized` scheme; with `--linearized-monotone-residual`, the solver may still reduce `alpha` internally by halving it during residual backtracking.
- For `--linearized-scheme full_eikonal`, the linearized-only controls `--linearized-inner-max-sweeps`, `--linearized-inner-tol`, `--linearized-eikonal-tol`, `--linearized-damping`, `--linearized-max-deltat`, and `--linearized-monotone-residual` are ignored.
- Use `--no-check-path-geometry` to disable all Step 5 path-geometry validation when exploratory linearized `FP^i` paths fail the current geometry checks.
- Override shared Step 5 geometry with `--source-x`, `--target-x`, `--source-y-start`, `--source-y-end`, `--target-y-start`, and `--target-y-end`.
- Curvature diagnostics now support explicit spline geometry modes with `--curvature-spline-mode {interpolation,approximation}` plus `--curvature-spline-smooth` and `--curvature-spline-degree`.
- Use `--curvature-method savgol` (default) for a Savitzky-Golay-based curvature estimate: arc-length-uniform resampling + `dθ/ds` via 1D SG filter (controls: `--curvature-savgol-window`, `--curvature-savgol-poly`). Use `--curvature-method spline` for the original spline-geometry approach.
- Normal-derivative diagnostics can be smoothed independently with `--curvature-field-smooth-method {none,gaussian}` and `--curvature-field-smooth-sigma`.
- Four `--normal-deriv-method` options are available:
  - `direct` (default): cubic `RegularGridInterpolator` field sampling at `p ± δ·n̂`, finite difference, then 1D SG post-smooth (linear fallback for scipy < 1.9).
  - `savgol2d`: separable 2D Savitzky-Golay gradient on the full field grid, then dot with path normal (window controlled by `--curvature-savgol-window`/`--curvature-savgol-poly`).
  - `spline2d`: `RectBivariateSpline` fitted to the field with analytic ∂/∂n at path points — no finite-difference offset needed (smoothing controlled by `--normal-deriv-spline2d-smooth`; `s=0` is interpolating, `s>0` smoothing).
  - `mls_frenet`: Moving Least Squares quadratic polynomial `f(s,n) ≈ c₁s²+c₂n²+c₃sn+c₄s+c₅n+c₆` fitted in the Frenet-Serret frame at each path point — the `c₅` coefficient gives ∂f/∂n directly. Patch half-widths controlled by `--normal-deriv-mls-ns` (tangent direction) and `--normal-deriv-mls-nn` (normal direction).
- Step 5 evaluates curvature, normals, and normal-derivative terms on the same resampled curve. The curvature-balance figure uses `K_S=P grad(log|grad h|)`, `K_E°=-P grad(log K)`, and `K_E=-P grad(log(K|grad h|))`. Its first three panels compare velocity-direction curvature against `K_S` on the particle-tracked streamline and travel-time-Hessian curvatures against `K_E°` and `K_E` on their respective Eikonal paths. The fourth panel contains only `-K_E` evaluated along `gamma_E°*` and `kappa_S(gamma_S*)-kappa_E°(gamma_E°*)`; `kappa_S` is interpolated from its own normalized-arclength grid onto the `gamma_E°*` grid before subtraction. The four panels share one figure-level y-axis label, `Normalized curvature`. Coordinate-differentiated curve curvature remains available in diagnostics but is not used as the left-hand side in these panels.
- The local linearized-FP backtracking now ranks neighbors by travel-time drop per unit step length, reducing diagonal bias in the discrete descent path.
- Spatial axes are normalized as `x/I_Y` and `y/I_Y`.
- Arrival-time background is normalized as `T_min/t_ref`, with `t_ref = I_Y / (k_g * dh)`.
- Speed/streamline background uses dimensionless velocity magnitude `|v|/(k_g*dh)` and streamlines from `v/(k_g*dh)`.
- Particle tracking now supports `--solver pollock` or `--solver rk45`; defaults mirror the Algorithm runner: `--solver pollock`, `--velocity-location face`, `--no-align-with-modflow-velocity`.
- Pollock requires `--velocity-location face`; the plotted particle-tracking `FP` remains solver-neutral and reflects the selected particle-tracking backend.
- `--fsm-fp` enables the Fast Sweeping Method (FSM) path `γ*_S` (green, dash-dot-dot). Step 5 now imports the shared implementation from `Algorithm/src/fmm_core/fsm.py` instead of carrying a local duplicate solver. FSM solves the advection equation `v·∇T = 1` directly using cell-centred Darcy velocity `vx = -K·∂h/∂x`, `vy = -K·∂h/∂y` via upwind finite differences + 4-direction Gauss-Seidel sweeping, then backtracks the path from the target by RK45 against `-v(x)`. This is physically distinct from the Eikonal-FMM paths, which use a scalar slowness field `1/(K·|∇h|)`. FSM is off by default (`--no-fsm-fp`).
- FSM convergence controls:
  - `--fsm-max-sweeps`: hard sweep cap (default `100`)
  - `--fsm-tol`: update-based stopping tolerance on per-sweep `max_change` (default `1e-6`)
  - `--fsm-mse-tol`: optional mean-squared discrete PDE residual tolerance for `v·∇T - 1`
  - `--fsm-convergence-mode`: `update`, `mse`, `update_or_mse`, or `update_and_mse`
  - `--fsm-mse-check-every`: only evaluate the residual MSE every N eligible sweeps (default `5`)
  - `--fsm-mse-activation-factor`: gate residual checks until `max_change` is near `--fsm-tol` (default `10.0`)
  - `--fsm-consecutive-passes`: require the convergence rule to pass multiple consecutive eligible checks (default `2`)
  - `--fsm-dt`: RK45 output step target for FSM backtracking
  - `--fsm-max-steps`: maximum accepted backtracking points
- `submit_step5_gpu.sh` now passes the FSM convergence options through to `step5_single_field_paths.py`, so batch runs can use the same update/MSE stopping controls as interactive runs.
- The current `submit_step5_gpu.sh` defaults are slightly stricter than the interactive CLI defaults: it sets `fsm_tol="1e-8"` while leaving residual-MSE checking optional unless `fsm_mse_tol` is set explicitly.
- `submit_step5_gpu.sh` is now a dual-mode launcher:
  - on macOS it runs `step5_single_field_paths.py` locally, prefers `--backend gpu`, and falls back to `--backend cpu` if Taichi GPU init fails
  - on Linux with `sbatch` available it keeps the existing Slurm submission flow
  - on Linux without `sbatch` it runs locally instead of failing on missing Slurm
- Local launcher overrides:
  - `STEP5_PYTHON=/path/to/python` chooses the interpreter
  - `STEP5_PYTHONPATH=/path/to/DataAnalysis/scripts` overrides the Step 5 `PYTHONPATH`
  - `STEP5_LOCAL_BACKEND=gpu|cpu` overrides the local backend preference
- On Apple Silicon, `submit_step5_mac_m4.sh` uses the existing `taichi` Conda environment by default and uses `NORMAL_DERIV_METHOD=direct` so curvature right-hand sides are evaluated from the `h` and `K` fields. This setting does not select the left-hand-side curvature estimator: `S` uses velocity-direction derivatives, while `E°` and `E` use their travel-time Hessians. Override settings through environment variables; no additional environment is required.
  - `FLOWSIM_PATH=/path/to/FlowSimulation` overrides the sibling checkout path
- When local execution resolves to CPU, the launcher disables snapshot capture automatically because Step 5 snapshots currently require `--solver pollock --backend gpu`.
- Step 5 FMM backtracking still defaults to `--fmm-backtracking discrete`.
- Solver imports require both `Algorithm/src` and the sibling `FlowSimulation` repo.
- The linearized Eikonal iteration additionally loads `scripts/fp_and_lrp/linear_eikonal_solver.py`.
- Override FlowSimulation path with `FLOWSIM_PATH=/path/to/FlowSimulation` if needed.

## Quick Single-MC Visualization

Use `visualize_single_logk_paths.py` to load a consolidated per-MC `.npy` file and plot the logK field with all saved paths overlaid. No solver re-execution is needed — it reads paths directly from the saved data.

Direct mode (provide the `.npy` file):

```bash
PYTHONPATH=./scripts python scripts/visualize_single_logk_paths.py \
  --npy-file /path/to/LengthScale0.05_Size200/size200_mc5.npy \
  --output-dir "$FIGURE_DIR"
```

Discovery mode (find the file via config):

```bash
PYTHONPATH=./scripts python scripts/visualize_single_logk_paths.py \
  --config configs/visualization.default.yaml \
  --sweep variance \
  --value 5.0 \
  --field-size 200 \
  --mc-index 5 \
  --output-dir "$FIGURE_DIR"
```

Options:

- `--paths fp lrp lrp_fmm`: select a subset of paths to show
- `--format pdf`: save as PDF instead of PNG
- `--dpi 300`: increase resolution
- `--cmap cmc.vik`: colormap for logK field (default: `cmc.vik`)
- `--no-legend`: hide legend

Outputs:

- `<output_dir>/logk_paths_mc<idx>.png` (or `.pdf`)
- Summary table printed to stdout: path lengths, travel times, Hausdorff stats, Shannon entropy

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
FIGURE_DIR=/project2/fbarros_324/binhaoli/FMM_FastestPath/outputs/visualization/cdf_travel_time/cdf_fp_lrp/var_1.0_size500_seed0_mc100
PYTHONPATH=./scripts python scripts/step6_single_field_arrival_hist.py \
  --load-npz "$FIGURE_DIR/runrun8_pollock_gpu_1e6_singlefield_std1_var1_size500_seed0_mc100_arrival_times.npz"
```

Or use the Step 6 launcher in local reload mode (no Slurm/GPU solve):

```bash
./submit_step6_gpu.sh reload \
  "$FIGURE_DIR/runrun8_pollock_gpu_1e6_singlefield_std1_var1_size500_seed0_mc100_arrival_times.npz"
```

Optionally choose a different output directory or DPI for the regenerated figure:

```bash
./submit_step6_gpu.sh reload \
  "$FIGURE_DIR/runrun8_pollock_gpu_1e6_singlefield_std1_var1_size500_seed0_mc100_arrival_times.npz" \
  "$FIGURE_DIR/reloaded"

# Equivalent direct Python invocation:
PYTHONPATH=./scripts python scripts/step6_single_field_arrival_hist.py \
  --load-npz "$FIGURE_DIR/runrun8_pollock_gpu_1e6_singlefield_std1_var1_size500_seed0_mc100_arrival_times.npz" \
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

`submit_step6_gpu.sh` reload-mode overrides:
- `STEP6_LOAD_NPZ=/path/to/file.npz` forces reload mode without positional args
- `STEP6_RELOAD_OUTPUT_DIR=/path/to/out` sets the reload output directory
- `STEP6_PYTHON=/path/to/python` chooses the interpreter
- `STEP6_PYTHONPATH=/path/to/DataAnalysis/scripts` overrides `PYTHONPATH`

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
- `scatter`
- `hausdorff`
- `hausdorff_box`
- `frechet`
- `endpoint`
- `ratio_mean`: plots `mean(t_min/(R_min/dh))` vs variance for FP, FP(FMM), FP(Dijkstra), and FSM when available
- `ratio_var`: plots `var(t_min/(R_min/dh))` vs variance for FP, FP(FMM), FP(Dijkstra), and FSM when available
- `wasserstein_len`: plots normalized Wasserstein-1 distance between path-length distributions (CDF proxy) for FP-LRP, FP-FMM, FP-Dijkstra, and FP-FSM
  - also writes separate path-length CDF figure(s) `run<id>_cdf_len_var<value>.png` for selected variance values
  - use `--plot-variances` (comma-separated) to choose variance values; default is the first available variance
- `times`: keeps the summary/error-bar characteristic-time figure, including the analytical `t_min` overlay
- `times_box`: writes a separate grouped box-plot figure of raw FP/LRP/LRP(FMM)/FP(FMM)/FP(Dijkstra)/FSM time samples across variances when FSM samples are available
- `hausdorff`: keeps the summary/error-bar Hausdorff figure and includes FSM comparisons where available
- `hausdorff_box`: writes a separate grouped box-plot figure of raw Hausdorff-distance samples across variances
- `endpoint`: writes a grouped box-plot figure of raw endpoint-distance samples across variances, including FSM when available
- `scatter`: supports `--scatter-mode mean`, `--scatter-mode all_points`, or `--scatter-mode mean,all_points`; both scatter modes include FSM when Stage 3 provides FSM samples
  - all-points mode writes `run<id>_scatter_var<value>.png` for selected `--plot-variances`

When enabled, the Frechet section is saved as `run<id>_frechet.png` (or corresponding configured format).

### Hausdorff CDF Ensemble Plot

Use `--hausdorff-cdf` to generate ensemble-averaged empirical CDF plots of Hausdorff nearest-neighbor distances. For each selected `--plot-variances` value:

- Stage 3 loads every MC consolidated `.npy` file, extracts the full `['hausdorff'][pair_key]['distances']` array (symmetric nearest-neighbor distances between path pairs), and computes per-MC empirical CDFs evaluated on a common 500-point grid.
- Stage 4 plots the mean CDF curve (solid line) with a shaded +/- 1 std region for each of the 4 distance pairs (FP vs LRP, FP vs LRP(FMM), FP vs FP(Dijkstra), FP vs FP(FMM)).
- Output: `run<id>_hausdorff_cdf_var<value>.png`

### Entropy vs Hausdorff Per-MC Plot

Use `--entropy-hausdorff` to generate per-MC scatter/line plots comparing Shannon entropy of the velocity field with Hausdorff distance summary statistics. For each selected `--plot-variances` value:

- Stage 3 loads every MC consolidated `.npy` file and extracts per-MC Shannon entropy (vmag and angle) and Hausdorff `max`, `mean`, `median` for all 4 path pairs.
- Stage 4 renders 5 vertically stacked panels sharing the x-axis (MC realization index):
  - Panel 1: Shannon entropy H(|v|) and H(θ)
  - Panel 2: Hausdorff max/mean/median for FP vs FP(FMM)
  - Panel 3: Hausdorff max/mean/median for FP vs FP(Dijkstra)
  - Panel 4: Hausdorff max/mean/median for FP vs LRP(FMM)
  - Panel 5: Hausdorff max/mean/median for FP vs LRP
- Output: `run<id>_entropy_hausdorff_var<value>.png`

```bash
PYTHONPATH=./scripts python scripts/step4_plot.py \
  --config configs/visualization.default.yaml \
  --cache-dir "$CACHE_DIR" \
  --plot-variances 0.1,1.0,5.0 \
  --entropy-hausdorff
```

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
- `--hausdorff-cdf` generates ensemble-averaged Hausdorff nearest-neighbor distance CDFs (mean curve + std band) for selected `--plot-variances`; requires Stage 3 to have access to `stage1_index.pkl` and raw MC consolidated files.
