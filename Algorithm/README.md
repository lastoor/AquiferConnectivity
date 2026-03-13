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
- `metrics.py`: path comparison metrics (`distance_distribution`).

## Dependency on FlowSimulation

This project depends on a sibling project at `../../FlowSimulation`.

The wrapper script `./scripts/run_in_fmm_env.sh` injects both `./src` and `../../FlowSimulation` into `PYTHONPATH`.

## Output Schema

`scripts/run_fmm.py` and `scripts/run_fmm_gpu_worker.py` write schema v8 `.npz` files:

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

- `--solver`: `pollock` or `rk45`
- `--backend`: `cpu` for `run_fmm.py`, `gpu` for `run_fmm_gpu_worker.py`
- `--num_particles`: exact particle-count override for fastest-path tracking
- `--velocity-location`: `face` or `cell`
- `--align-with-modflow-velocity` / `--no-align-with-modflow-velocity`

Defaults:

- `--solver pollock`
- `scripts/run_fmm.py --backend cpu`
- `scripts/run_fmm_gpu_worker.py --backend gpu`
- `--velocity-location face`
- `--no-align-with-modflow-velocity`

Notes:

- Pollock uses the full staggered face velocity layout from `simulation.hydraulic.compute_velocity(..., output_location="face", face_output="full")`.
- `--solver pollock` requires `--velocity-location face`.
- If `--num_particles` is provided, it overrides the older density-based seeding logic.
- The saved fastest-path outputs are solver-neutral: `len_fp`, `travel_time_fp`, and `fp_size...npy`.
- `scripts/run_fmm.py` rejects `--backend gpu` and redirects GPU usage to `scripts/run_fmm_gpu_worker.py`.
- `scripts/run_fmm_gpu_worker.py` currently supports `--solver pollock` only.
- The current persistent GPU worker is intended to run one field size and one exact `--num_particles` value per job. Mixed tracker shapes in one worker process are rejected explicitly.
- The GPU worker uses a single dedicated GPU process, compact `seed_spec` payloads, shared-memory transfer for `vx` / `vy` / face arrays, and exact winner replay in arrival mode.

## FMM Solver Options

`scripts/run_fmm.py` supports selecting both the travel-time solver and the backtracking method for all FMM-based paths (`LRP(FMM)` and `FP(FMM)`):

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \
  --path_workdir ./output/run_scikit_continuous \
  --num_mc 10 \
  --field_size 64 \
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

## FMM Velocity Options

FMM speed preparation is configurable:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \
  --path_workdir ./output/run1 \
  --num_mc 10 \
  --field_size 64 \
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
