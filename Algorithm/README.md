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

`scripts/run_fmm.py` writes schema v2 `.npz` files:

- Per MC run: `mainOutput_size{field_size}_mc{mc_index}.npz`
- Per size aggregate: `mainOutput_size{field_size}.npz`

Core fields include status, lengths (`len_*`), travel times (`travel_time_*`), resistance, Hausdorff-like distances, and endpoint deltas.

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
