# FMM_FastestPath/Algorithm

Utilities and scripts for fastest-path and Monte Carlo runs in this repository.

## Environment

Use:

```bash
./scripts/run_in_fmm_env.sh <command>
```

Example:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py --help
```

## Local Modules

FMM-specific code lives in `./src/fmm_core`:

- `fmm.py`: travel-time solvers (`custom`, `scikit`) and path backtracking (`discrete`, `continuous`).
- `fsm.py`: velocity-driven FSM travel-time solve/path extraction for `v·∇T = 1`, including residual-field and residual-MSE helpers for convergence checks.
- `fm_asr.py`: 2D Mirebeau-style FM-ASR solver for smooth Finsler norms, with paper-style exterior discrete boundary handling for escape-time solves.
- `fm_asr_conic.py`: mixed Randers/conic FM-ASR solver for arbitrary-wind Zermelo metrics. Subcritical cells use the Randers equivalent; critical/strong-wind cells use one-sided conic stencils. It also provides PyPar2-style dispersion tensor helpers.
- `metrics.py`: path comparison metrics (`distance_distribution`).

## Dependency on FlowSimulation

This project depends on a sibling project at `../../FlowSimulation`.

The wrapper script `./scripts/run_in_fmm_env.sh` injects both `./src` and `../../FlowSimulation` into `PYTHONPATH`.

## Output Schema

`scripts/run_fmm.py` and `scripts/run_fmm_gpu_worker.py` write schema v16 outputs:

- Per MC run: `mainOutput_size{field_size}_mc{mc_index}.npz`
- Per size aggregate: `mainOutput_size{field_size}.npz`

Core fields include status, lengths (`len_*`), travel times (`travel_time_*`), resistance, Hausdorff-like distances, endpoint deltas, FMM diagnostics, particle-tracking solver metadata, exact particle-count metadata, and seed metadata. The saved fastest-path outputs are solver-neutral (`len_fp`, `travel_time_fp`, `fp_size...npy`) regardless of whether the backend is Pollock or RK45.

## Fastest-Path Runner Options

Use the CPU-oriented runner for CPU particle tracking:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \
  --path_workdir ./output/run_pollock \
  --num_mc 10 \
  --field_size 64 \
  --covariance exp \
  --length_scale 0.1 \
  --std 1.0 \
  --solver pollock \
  --backend cpu \
  --velocity-location face \
  --no-align-with-modflow-velocity
```

Use the GPU-worker runner for Pollock GPU:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm_gpu_worker.py \
  --path_workdir ./output/run_pollock_gpu \
  --num_mc 10 \
  --field_size 64 \
  --covariance exp \
  --length_scale 0.1 \
  --std 1.0 \
  --solver pollock \
  --backend gpu \
  --num_particles 1000000 \
  --cpu_workers 6 \
  --velocity-location face \
  --no-align-with-modflow-velocity
```

Options:

- `--covariance`: `exp` or `fbm`
- `--solver`: `pollock` or `rk45`
- `--backend`: `cpu` for `run_fmm.py`, `gpu` for `run_fmm_gpu_worker.py`
- `--num_particles`: exact particle-count override for fastest-path tracking
- `--velocity-location`: `face` or `cell`
- `--align-with-modflow-velocity` / `--no-align-with-modflow-velocity`
- `--fm_asr` / `--no-fm_asr`: enable or disable FM-ASR comparison paths
- `--fm_asr_perp_ratio`: sideways anisotropic speed ratio for FM-ASR, default `0.5`
- `--fm_asr_back_factor`: backward speed reduction factor for FM-ASR, default `2.0`
- `--fm_asr_min_speed`: velocity magnitude floor for FM-ASR; defaults to `--fmm_min_speed`
- `--fm_asr_velocity_multiplier`: dimensionless multiplier applied to the velocity field before FM-ASR Randers construction, default `1.0`
- `--fm_asr_max_steps`: FM-ASR backtracking step cap, default `200000`
- `--fm_asr_conic` / `--no-fm_asr_conic`: enable or disable the mixed Randers/conic FM-ASR comparison path (`fp_fm_asr_conic`)
- `--fm_asr_conic_self_velocity`: self-motivated metric for conic/mixed FM-ASR, `dispersion` or `unit`, default `dispersion`
- `--fm_asr_conic_velocity_multiplier`: multiplier applied to the flow velocity used as Zermelo wind, default `1.0`
- `--fm_asr_conic_lambda_threshold`: regime split; cells with `lambda = 1 - g(W,W)` above this value use Randers, otherwise conic, default `0.0`
- `--fm_asr_conic_dispersion_c`: global calibration factor for the dispersion self-velocity, default `1.0`
- `--fm_asr_conic_dispersion_tau`: global calibration timescale for the dispersion self-velocity, default `1.0`
- `--fm_asr_conic_dispersion_eigenvalue_floor`: floor applied to local dispersion tensor eigenvalues before inversion, default `1e-12`
- `--fm_asr_conic_stencil_radius`: conic stencil search radius for integer directions, default `8`
- `--fm_asr_conic_edge_samples`: edge-search sample count for conic updates, default `64`
- `--fm_asr_conic_max_steps`: mixed/conic FM-ASR backtracking step cap, default `200000`

Defaults:

- `--covariance exp`
- `--solver pollock`
- `scripts/run_fmm.py --backend cpu`
- `scripts/run_fmm_gpu_worker.py --backend gpu`
- `--velocity-location face`
- `--no-align-with-modflow-velocity`
- `--fm_asr`
- `--fm_asr_conic`

Notes:

- `--covariance exp` requires `--length_scale` and exactly one of `--std` / `--var`.
- `--covariance fbm` requires `--fbm_alpha` and `--fbm_c`.
- Output folders are covariance-specific: `Covexp_LengthScale{L}_Size{N}` or `Covfbm_Alpha{A}_C{C}_Size{N}`.
- Pollock uses the full staggered face velocity layout from `simulation.hydraulic.compute_velocity(..., output_location="face", face_output="full")`.
- `--solver pollock` requires `--velocity-location face`.
- If `--num_particles` is provided, it overrides the older density-based seeding logic.
- The saved fastest-path outputs are solver-neutral: `len_fp`, `travel_time_fp`, and `fp_size...npy`.
- `scripts/run_fmm.py` rejects `--backend gpu` and redirects GPU usage to `scripts/run_fmm_gpu_worker.py`.
- `scripts/run_fmm_gpu_worker.py` currently supports `--solver pollock` only.
- The current persistent GPU worker is intended to run one field size and one exact `--num_particles` value per job. Mixed tracker shapes in one worker process are rejected explicitly.
- The GPU worker uses a single dedicated GPU process, compact `seed_spec` payloads, shared-memory transfer for `vx` / `vy` / face arrays, and exact winner replay in arrival mode.
- `scripts/run_fmm_gpu_worker.py` is compatible with newer FlowSimulation Pollock GPU trackers that return extra snapshot arrays from `tracker.run()`; the worker uses the first two outputs (`travel_times`, `status`) and ignores additional values in the summary path.
- FM-ASR comparison paths are non-fatal diagnostics. If an FM-ASR solve fails, the MC trial can still succeed and the FM-ASR fields are saved as `NaN` with an error string in the per-MC consolidated output.
- `fp_fm_asr_conic` uses the same full face Darcy velocity as wind after face-to-cell conversion. With `--fm_asr_conic_self_velocity dispersion`, the local dispersion tensor `D(x)` is built from the PyPar2 parameters (`Dm`, `alpha_l`, `alpha_t`, and optional velocity floor), but the deterministic self-velocity scale is calibrated by global `C` and `tau`:
  `v_i(x) = C * sqrt(lambda_i(D(x)) / tau)`.
  Equivalently, the metric tensor eigenvalues are `tau / (C^2 * lambda_i(D(x)))`. PyPar2 `dt` is not used in this deterministic scale.

Example FBM run:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \
  --path_workdir ./output/run_fbm \
  --num_mc 10 \
  --field_size 64 \
  --covariance fbm \
  --fbm_alpha 0.8 \
  --fbm_c 1.0 \
  --var 1.0 \
  --solver pollock \
  --backend cpu \
  --velocity-location face \
  --no-align-with-modflow-velocity
```

## FMM Solver Options

`scripts/run_fmm.py` supports selecting both the travel-time solver and the backtracking method for all FMM-based paths (`LRP(FMM)` and `FP(FMM)`):

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \
  --path_workdir ./output/run_scikit_continuous \
  --num_mc 10 \
  --field_size 64 \
  --covariance exp \
  --length_scale 0.1 \
  --std 1.0 \
  --fmm_solver scikit \
  --fmm_backtracking continuous
```

Options:

- `--fmm_solver`: `scikit` or `custom`
- `--fmm_backtracking`: `continuous` or `discrete`

Defaults:

- `--fmm_solver scikit`
- `--fmm_backtracking continuous`

## FM-ASR Finsler Solver

`FMASR2D` supports smooth Finsler/Randers norms, including metrics built from a cell-centered velocity field. A source-line travel-time solve can be run directly from a `(ny, nx, 2)` velocity array:

```python
import numpy as np
from fmm_core import FMASR2D

ny, nx = 101, 101
W = np.zeros((ny, nx, 2), dtype=float)
W[..., 0] = 1.0  # rightward local velocity

source = np.zeros((ny, nx), dtype=bool)
source[:, 0] = True

solver = FMASR2D.from_velocity_field(
    W,
    h=1.0,
    perp_ratio=0.5,
    back_factor=2.0,
    direction="from_source",
)
T = solver.solve(source)
```

Use `isotropic_only=True` to reduce the local indicatrix to a circle whose radius is the velocity magnitude. Use `direction="to_source"` for the backward/escape orientation.

## Mixed/Conic FM-ASR Solver

`MixedFMASR2D` handles the arbitrary-wind Zermelo case with a per-cell regime split. It accepts a cell-centered wind field and an optional SPD ground metric field:

```python
import numpy as np
from fmm_core import MixedFMASR2D, extract_mixed_fm_asr_path

ny, nx = 101, 101
W = np.zeros((ny, nx, 2), dtype=float)
W[..., 0] = 2.0

source = np.zeros((ny, nx), dtype=bool)
source[:, 0] = True
target = np.zeros((ny, nx), dtype=bool)
target[:, -1] = True

solver = MixedFMASR2D.from_velocity_field(
    W,
    h=1.0,
    direction="from_source",
    lambda_threshold=0.0,
    stencil_radius=8,
    edge_samples=64,
)
T = solver.solve(source)
path, travel_time, chosen_target = extract_mixed_fm_asr_path(solver, T, source, target)
```

In the Monte Carlo runners, `fp_fm_asr_conic` uses `MixedFMASR2D` with flow velocity as wind. When `self_velocity=dispersion`, local anisotropy and ellipse orientation come from the PyPar2 dispersion tensor `D(x)`, while `--fm_asr_conic_dispersion_c` and `--fm_asr_conic_dispersion_tau` set one global speed scale for the whole run.

## FMM Velocity Options

FMM speed preparation is configurable:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \
  --path_workdir ./output/run1 \
  --num_mc 10 \
  --field_size 64 \
  --covariance exp \
  --length_scale 0.1 \
  --std 1.0 \
  --fmm_velocity_layout face \
  --fmm_face_output full \
  --fmm_face_mapping to_cell_avg \
  --fmm_zero_policy obstacle \
  --fmm_min_speed 1e-12
```

Options:

- `--fmm_velocity_layout`: `face` or `cell`
- `--fmm_face_output`: `full` or `trimmed` (used for `face`, default `full`)
- `--fmm_face_mapping`: `to_cell_avg` or `direct` (used for `face`)
- `--fmm_zero_policy`: `obstacle`, `clip`, or `error`
- `--fmm_min_speed`: speed floor used by `clip`/`obstacle`

Face contracts:

- `full`: staggered physical faces, `vx (ny, nx+1)`, `vy (ny+1, nx)`
- `trimmed`: per-cell right/top faces, `vx (ny, nx)`, `vy (ny, nx)`

Notes:

- `full` is the default for FMM face layout.
- `full` with `--fmm_face_mapping direct` is not supported.
- `to_cell_avg` on `full` uses symmetric adjacent-face averaging of magnitudes before forming the scalar speed field.
- `trimmed` preserves the legacy equal-shape `QXX/QYY`-style contract used by the earlier FMM speed-prep path.

Additional schema fields now include:

- `fmm_velocity_layout`, `fmm_face_output`, `fmm_face_mapping`, `fmm_solver`, `fmm_backtracking`, `fmm_zero_policy`, `fmm_min_speed`
- `fmm_zero_count`, `fmm_nontraversable_count`
- `pt_solver`, `pt_backend`, `pt_num_particles`, `pt_velocity_location`, `pt_align_with_modflow_velocity`
- `len_fp_fm_asr`, `len_fp_fm_asr_iso`, `travel_time_fm_asr`, `travel_time_fm_asr_iso`
- `len_fp_fm_asr_conic`, `travel_time_fm_asr_conic`
- `endpoint_dy_fp_vs_fm_asr`, `endpoint_dy_fp_vs_fm_asr_iso`, `endpoint_dy_fp_vs_fm_asr_conic`
- `x_end_fp_fm_asr`, `y_end_fp_fm_asr`, `x_end_fp_fm_asr_iso`, `y_end_fp_fm_asr_iso`, `x_end_fp_fm_asr_conic`, `y_end_fp_fm_asr_conic`
- `fm_asr_enabled`, `fm_asr_perp_ratio`, `fm_asr_back_factor`, `fm_asr_min_speed`, `fm_asr_velocity_multiplier`, `fm_asr_max_steps`
- `fm_asr_conic_enabled`, `fm_asr_conic_self_velocity`, `fm_asr_conic_velocity_multiplier`, `fm_asr_conic_lambda_threshold`
- `fm_asr_conic_dispersion_c`, `fm_asr_conic_dispersion_tau`, `fm_asr_conic_dispersion_eigenvalue_floor`
- `fm_asr_conic_stencil_radius`, `fm_asr_conic_edge_samples`, `fm_asr_conic_max_steps`
- `comp_time_fp_fm_asr`, `comp_time_fp_fm_asr_iso`, `comp_time_fp_fm_asr_conic`

## Random Seed Control

`scripts/run_fmm.py` supports deterministic seeding for Monte Carlo runs.

Options:

- `--seed`: base seed (if omitted, behavior remains non-deterministic).
- `--seed_mode`: `strict_mc` (recommended), `global_stream`, or `best_effort`.
- `--variance_seed_policy`: `same`, `offset`, or `explicit`.
- `--variance_seed_offset`: offset multiplier used with `offset`.
- `--variance_seed`: explicit run seed used with `explicit`.
- `--variance_rank`: variance index used with `offset`.

Recommended deterministic mode:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \
  --path_workdir ./output/run_seeded \
  --num_mc 50 \
  --field_size 200 \
  --covariance exp \
  --length_scale 0.05 \
  --var 1.0 \
  --seed 12345 \
  --seed_mode strict_mc \
  --variance_seed_policy same
```

Variance sweep behavior:

- Same seeds for all variances:
  - `--seed 12345 --variance_seed_policy same`
- Different deterministic seeds by variance index:
  - `--seed 12345 --variance_seed_policy offset --variance_rank <i>`
- Fully explicit seed per variance run:
  - `--seed 12345 --variance_seed_policy explicit --variance_seed <seed_i>`

## GPU Submission

For GPU Pollock fastest-path sweeps with an exact particle count, use:

```bash
./submit_run4_var_sweep_gpu.sh
```

That script:

- requests `1` GPU per Slurm job
- uses `scripts/run_fmm_gpu_worker.py`
- uses `--solver pollock --backend gpu`
- uses `--num_particles 1000000`
- uses `--cpu_workers 6`

Operational notes:

- Host RAM in the script is set by `--cpus-per-task=10` and `--mem-per-cpu=8GB` in Slurm, which requests about `80GB` total system memory for the job.
- The larger risk for this workflow is GPU memory and runtime, not host RAM.
- Run one pilot job first and inspect `sacct ... MaxRSS`, `sstat`, `nvidia-smi`, and the generated `gpu_profile_*.log` before launching the full sweep.
