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

## 2026-03-11

- Extended `./src/fmm_core/fmm.py` with symmetric solver/backtracking APIs:
  - `solve_travel_time_custom`, `solve_travel_time_scikit`
  - `backtrack_path_discrete`, `backtrack_path_continuous`
- Kept legacy FMM function names as compatibility wrappers and re-exported the new APIs from `./src/fmm_core/__init__.py`.
- Refactored `./scripts/run_fmm.py` to use shared FMM dispatch instead of a local hardcoded FMM solve/backtracking path.
- Added CLI options to `./scripts/run_fmm.py`:
  - `--fmm_solver` (`scikit`, `custom`), default `scikit`
  - `--fmm_backtracking` (`continuous`, `discrete`), default `continuous`
- Applied the selected solver/backtracking mode to both `LRP(FMM)` and `FP(FMM)`.
- Extended per-MC and aggregate outputs with FMM configuration metadata:
  - `fmm_solver`, `fmm_backtracking`
- Added and updated tests for the renamed FMM APIs, solver/backtracking combinations, and `run_fmm.py` CLI coverage.
- Updated `./README.md` with the new solver/backtracking options and corrected the documented output schema version to v4.
- Updated `./agent/guide.md` to require schema-safe `.npz` outputs and explicitly forbid `NaN`/`None`/object sentinels in saved metadata fields.

## 2026-03-12

- Refactored `./scripts/run_fmm.py` fastest-path tracking to support both `Pollock` and `RK45` backends via:
  - `--solver` (`pollock`, `rk45`)
  - `--velocity-location` (`face`, `cell`)
  - `--align-with-modflow-velocity` / `--no-align-with-modflow-velocity`
- Switched the saved fastest-path outputs to solver-neutral field names:
  - `len_fp`, `travel_time_fp`
  - added metadata `pt_solver`, `pt_velocity_location`, `pt_align_with_modflow_velocity`
- Updated `./scripts/run_fmm.py` output schema from v5 to v6.
- Extended `./src/fmm_core/speed.py` and `./scripts/run_fmm.py` to support FMM face-centered inputs in two contracts:
  - `--fmm_face_output full` (default): staggered physical faces
  - `--fmm_face_output trimmed`: per-cell right/top faces
- Implemented symmetric adjacent-face averaging for `full` face inputs under `to_cell_avg`.
- Added validation to reject `--fmm_face_output full --fmm_face_mapping direct`.
- Updated `./tests/test_speed_prep.py`, `./tests/test_run_fmm_integration.py`, and `./tests/run_fmm_compare.py` for the new Pollock and FMM face-output behavior.
- Updated `./README.md` to document the new solver options, FMM face-output mode, and schema v6 fields.
