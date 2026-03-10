# Agent Log

## 2026-03-02

- Added `AGENT.md` pointing to `./agent/guide.md`.
- Added `./agent/guide.md` with environment instructions for `fmm_env`.
- Added `./run_in_fmm_env.sh` to run commands with `fmm_env` and `PYTHONPATH` including `../../FlowSimulation`.
- Added `./README.md` and this `./agent/log.md`.

## 2026-03-03

- Refactored `./scripts/run_fmm.py` to use `simulation` package implementations for GRF generation, hydraulic head solving, velocity computation, path finding, and RK45 particle tracking.
- Added local FMM modules in `./src/fmm_core` (`fmm.py`, `metrics.py`) and exported them from `./src/fmm_core/__init__.py`.
- Added FMM speed-preparation module `./src/fmm_core/speed.py` with support for face/cell velocity layouts, face-to-cell mapping, and zero-speed policies (`obstacle`, `clip`, `error`).
- Extended FMM solver/path extraction to support `traversable_mask`.
- Added CLI options to `./scripts/run_fmm.py`: `--fmm_velocity_layout`, `--fmm_face_mapping`, `--fmm_zero_policy`, `--fmm_min_speed`.
- Extended schema v2 outputs with FMM diagnostics fields: layout/mapping/policy metadata and zero/nontraversable counts.
- Added tests for FMM core and speed preparation in `./tests`.

## 2026-03-10

- Updated `./scripts/run_fmm.py` output schema from v2 to v3.
- Added reproducible random seed controls:
  - `--seed`
  - `--seed_mode` (`strict_mc`, `global_stream`, `best_effort`)
  - `--variance_seed_policy` (`same`, `offset`, `explicit`)
  - `--variance_seed_offset`, `--variance_seed`, `--variance_rank`
- Implemented deterministic per-MC/per-trial seeding in `strict_mc` mode for multiprocessing runs.
- Added seed metadata to per-MC and aggregate outputs:
  - `seed_base`, `seed_effective_run`, `seed_mode`
  - `variance_seed_policy`, `variance_rank`, `variance_seed_override`
  - `seed_mc`, `seed_trial`
