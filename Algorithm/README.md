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

- `fmm.py`: travel-time solve (`fast_marching_2d_regions`) and path extraction.
- `metrics.py`: path comparison metrics (`distance_distribution`).

## Dependency on FlowSimulation

This project depends on a sibling project at `../../FlowSimulation`.

The wrapper script `./scripts/run_in_fmm_env.sh` injects both `./src` and `../../FlowSimulation` into `PYTHONPATH`.

## Output Schema

`scripts/run_fmm.py` writes schema v3 `.npz` files:

- Per MC run: `mainOutput_size{field_size}_mc{mc_index}.npz`
- Per size aggregate: `mainOutput_size{field_size}.npz`

Core fields include status, lengths (`len_*`), travel times (`travel_time_*`), resistance, Hausdorff-like distances, endpoint deltas, FMM diagnostics, and seed metadata.

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
  --fmm_face_mapping to_cell_avg \
  --fmm_zero_policy obstacle \
  --fmm_min_speed 1e-12
```

Options:

- `--fmm_velocity_layout`: `face` or `cell`
- `--fmm_face_mapping`: `to_cell_avg` or `direct` (used for `face`)
- `--fmm_zero_policy`: `obstacle`, `clip`, or `error`
- `--fmm_min_speed`: speed floor used by `clip`/`obstacle`

Additional schema fields now include:

- `fmm_velocity_layout`, `fmm_face_mapping`, `fmm_zero_policy`, `fmm_min_speed`
- `fmm_zero_count`, `fmm_nontraversable_count`

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
