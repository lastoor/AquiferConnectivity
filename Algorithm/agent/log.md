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

## 2026-03-13

- Updated `./scripts/run_fmm.py` from schema v6 to v8:
  - switched fastest-path extraction to `simulation.particle_tracking.particle_tracking_summary(...)`
  - added `pt_backend` metadata
  - added exact particle-count metadata `pt_num_particles`
  - added `--num_particles` exact-count override for fastest-path tracking
- Split CPU and GPU entrypoints:
  - `./scripts/run_fmm.py` is the CPU-oriented runner and rejects `--backend gpu`
  - added `./scripts/run_fmm_gpu_worker.py` for `--backend gpu`
- Added a dedicated GPU Slurm submission script:
  - `./submit_run4_var_sweep_gpu.sh`
  - requests one GPU and uses `--num_particles 1000000`
- Implemented a single dedicated GPU worker in `./scripts/run_fmm_gpu_worker.py` with:
  - bounded task queue
  - compact `seed_spec` payloads instead of serialized seed arrays
  - shared-memory transfer for `vx`, `vy`, `x_faces`, and `y_faces`
  - exact winner replay in Pollock GPU arrival mode
- Attempted true fused multi-task GPU batching, then stepped back after exposing a Taichi lifecycle issue (`FieldsBuilder finalized`) in FlowSimulation’s Pollock GPU tracker.
- Reworked the GPU-worker execution model to the current stable form:
  - one persistent GPU worker process
  - one GPU task at a time
  - one persistent Pollock GPU tracker reused across repeated tasks with the same grid and exact particle count
  - explicit rejection when later tasks require a different tracker shape/capacity in the same worker process
- Verified the new CPU/GPU runner split and GPU-worker tests in `taichi_env`:
  - CLI guard checks for CPU-vs-GPU entrypoints
  - `tests/test_run_fmm_gpu_worker.py`
  - relevant `tests/test_run_fmm_integration.py` and `tests/test_import_contract.py` slices

## 2026-03-27

- Consolidated per-MC output in `./scripts/run_fmm.py`: all data (fields, paths, endpoints, Hausdorff, Shannon entropy, seeds, config) saved as a single nested dict in `size{N}_mc{i}.npy` with `allow_pickle=True`. Removed old separate saves and `main_output/`, `path/`, `logk_field/` subdirectories.
- Changed output folder naming from `lengthScale{L}_size{N}` to `LengthScale{L}_Size{N}`.
- Added Hausdorff detail stats per path pair: full symmetric minimal distance array (`distances`) plus `max`, `mean`, `std`, `median`.
- Added Shannon entropy metrics for velocity magnitude and angle distribution (`shannon_entropy_vmag`, `shannon_entropy_angle`) as new per-MC scalars in `MC_FIELDS`.
- Removed old Hausdorff scalar fields (`hausdorff_fp_vs_lrp`, etc.) from `MC_FIELDS`.
- Extended `_face_to_cell_speed_full` in `./src/fmm_core/speed.py` with a `return_vector` keyword option to return signed cell-centered `(vx_cell, vy_cell)` components instead of the scalar magnitude, needed for angle-based entropy computation on staggered face velocity fields.
- Updated `./scripts/run_fmm_gpu_worker.py` aggregation save path to match new folder structure.
- Updated `DataAnalysis_Simple` scripts (`step1_index.py`, `step2_metrics.py`) to discover and load data from the new consolidated `.npy` files and `LengthScale{L}_Size{N}` directories, with backward compatibility for older structures.

## 2026-04-13

- Added shared FSM utilities in `./src/fmm_core/fsm.py` and re-exported them from `./src/fmm_core/__init__.py`:
  - `solve_fsm_travel_time`
  - `compute_fsm_path_velocity`
  - `compute_fsm_residual_field`
  - `compute_fsm_residual_mse`
- Extended the shared FSM solver/path API with residual-MSE convergence controls:
  - `mse_tol`
  - `convergence_mode`
  - `mse_check_every`
  - `mse_activation_factor`
  - `consecutive_passes`
- Implemented efficient convergence gating so residual-MSE evaluation is skipped unless it is required by the selected convergence mode and update tolerance state.
- Fixed `./scripts/run_fmm_gpu_worker.py` summary extraction for newer FlowSimulation Pollock GPU tracker return signatures by accepting extra outputs from `tracker.run()` and consuming only `(travel_times, status)`.
- Validation:
  - `python -m py_compile ./src/fmm_core/__init__.py ./src/fmm_core/fsm.py`
  - `python -m py_compile ./scripts/run_fmm_gpu_worker.py`
- Limitations/follow-up:
  - `pytest` is not installed in the current environment, so the FSM unit tests were not executed here.

## 2026-04-14

- Renamed legacy FSM references from `solve_fsm_travel_time_old` to `solve_fsm_travel_time_legacy` and re-exported the legacy solver from `./src/fmm_core/__init__.py`.
- Split FSM update controls in `./src/fmm_core/fsm.py` into independent `order` and `neighbors` parameters:
  - supported combinations are `order=1, neighbors=4`, `order=1, neighbors=8`, and `order=2, neighbors=8`
  - `order=2, neighbors=4` now raises `ValueError`
- Updated FSM tests in `./tests/test_fmm_core.py` to cover:
  - legacy-equivalent first-order 4-neighbor behavior
  - first-order 8-neighbor acceptance
  - rejection of unsupported second-order 4-neighbor mode
- Added covariance-aware CLI wiring to `./scripts/run_fmm.py` and `./scripts/run_fmm_gpu_worker.py`:
  - `--covariance {exp,fbm}`
  - `--fbm_alpha`, `--fbm_c` for FBM runs
  - covariance-specific validation for `--length_scale` versus FBM parameters
- Centralized GRF configuration/building in `./scripts/run_fmm.py` via shared helpers so CPU and GPU runners use the same covariance selection path.
- Changed per-size output folder naming from always `LengthScale{L}_Size{N}` to covariance-specific tags:
  - `Covexp_LengthScale{L}_Size{N}`
  - `Covfbm_Alpha{A}_C{C}_Size{N}`
- Validation:
  - `python -m py_compile ./scripts/run_fmm.py ./scripts/run_fmm_gpu_worker.py`
  - `python -m py_compile ./src/fmm_core/__init__.py ./src/fmm_core/fsm.py ./tests/test_fmm_core.py`
- Limitations/follow-up:
  - Live CLI smoke checks still depend on the external `../../FlowSimulation` import path being available in the runtime environment.
  - `pytest` is still not installed in the current environment, so unit tests were not executed here.

## 2026-05-04

- Reworked `./src/fmm_core/fm_asr.py` toward a faithful smooth-norm Mirebeau FM-ASR implementation:
  - `build_f_acute_mesh` now returns unique cyclic stencil vertices instead of a duplicated closing vertex
  - added stencil validation for elementary F-acute edges
  - added paper-style exterior discrete boundary nodes for escape-time solves
  - preserved in-domain source masks as an explicit compatibility mode
- Exported FM-ASR symbols from `./src/fmm_core/__init__.py` and documented the module in `./README.md`.
- Added focused FM-ASR tests for stencil construction, edge updates, exterior-boundary escape time, and asymmetric source-mask behavior.
- Validation:
  - `PYTHONPYCACHEPREFIX=/private/tmp/fmm_pycache python3 -m py_compile src/fmm_core/fm_asr.py tests/test_fm_asr.py src/fmm_core/__init__.py`
  - `PYTHONPYCACHEPREFIX=/private/tmp/fmm_pycache python3 -m py_compile scripts/run_fmm.py src/fmm_core/*.py`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q tests/test_fm_asr.py` (7 passed)
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q` (11 passed, 5 pre-existing FSM warnings)
- Limitations/follow-up:
  - The preferred wrapper `./scripts/run_in_fmm_env.sh` currently targets missing Conda env `taichi_env`; available envs are `pytorch`, `taichi`, `base`, and `general`, so validation used `conda run -n taichi` directly.

## 2026-05-04

- Integrated velocity-derived Randers/Finsler fields with the FM-ASR solver:
  - added `FMASR2D.from_velocity_field(...)` for `(ny, nx, 2)` cell-centered velocity arrays
  - defaulted the factory to `direction="from_source"` by reversing the local Randers norm for source-outward travel times
  - kept `direction="to_source"` for the existing backward/escape orientation
  - generalized `build_randers_from_W` and validation helpers to rectangular fields
- Made FM-ASR plotting helpers import `matplotlib` and optional `cmasher` lazily so solver imports do not require plotting dependencies.
- Exported Randers helper functions from `./src/fmm_core/__init__.py` and documented source-line velocity-field usage in `./README.md`.
- Added FM-ASR tests for isotropic velocity reduction, source-line scaling, directionality, zero-velocity floors, rectangular fields, and invalid inputs.
- Validation:
  - `PYTHONPYCACHEPREFIX=/private/tmp/fmm_pycache python3 -m py_compile scripts/run_fmm.py src/fmm_core/*.py tests/test_fm_asr.py`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q tests/test_fm_asr.py` (13 passed)
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q` (17 passed, 5 pre-existing FSM warnings)

## 2026-05-04

- Added standalone comparison script `./tests/compare_fm_asr_velocity_paths.py`:
  - generates one logK/K field from the existing GRF framework
  - solves hydraulic head and full face/cell velocity fields
  - runs GPU Pollock particle tracking through `particle_tracking_summary`
  - computes scalar-speed FMM, FM-ASR acentric Randers, and FM-ASR reduced-isotropic paths
  - saves `results.npz`, `summary.json`, `comparison.png`, and optional `indicatrix_field.png`
- Added `./tests/run_compare_fm_asr_velocity_paths.sh` wrapper exposing every Python CLI option through environment-variable defaults while forwarding trailing CLI overrides.
- Validation:
  - `PYTHONPYCACHEPREFIX=/private/tmp/fmm_pycache python3 -m py_compile tests/compare_fm_asr_velocity_paths.py src/fmm_core/*.py`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi python tests/compare_fm_asr_velocity_paths.py --help`
  - `tests/run_compare_fm_asr_velocity_paths.sh --help`
  - `bash -n tests/run_compare_fm_asr_velocity_paths.sh`
- Limitations/follow-up:
  - A smoke run reached Taichi/Metal GPU initialization, then failed on local GPU/cache setup (`ticache.lock` and RHI device map assertion). This is an environment/runtime GPU failure rather than a script syntax/import issue.

## 2026-05-04

- Added FM-ASR fastest-path comparisons to the Monte Carlo runner pipeline:
  - `scripts/run_fmm.py` now computes non-fatal `fp_fm_asr` and `fp_fm_asr_iso` paths from full face Darcy velocity converted to cell-centered vectors
  - saved FM-ASR lengths, travel times, endpoints, endpoint deltas, Hausdorff details, path dictionaries, computation times, and config values
  - bumped `MC_SCHEMA_VERSION` to 12
  - added FM-ASR CLI/config knobs to both `scripts/run_fmm.py` and `scripts/run_fmm_gpu_worker.py`
  - added an `available_cpu_count()` fallback for macOS where `os.sched_getaffinity` is unavailable
- Validation:
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi python -m py_compile scripts/run_fmm.py scripts/run_fmm_gpu_worker.py src/fmm_core/fm_asr.py`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi python scripts/run_fmm_gpu_worker.py --help`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q tests/test_fm_asr.py` (15 passed)
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi python scripts/run_fmm.py --path_workdir /private/tmp/fm_asr_cpu_smoke --num_mc 1 --field_size 32 --covariance exp --length_scale 0.1 --std 1.0 --seed 123 --backend cpu --solver pollock --fm_asr_max_steps 50000`
  - inspected `/private/tmp/fm_asr_cpu_smoke/.../size32.npz` and `size32_mc0.npy` for new FM-ASR keys and finite values
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q` (19 passed, 5 pre-existing FSM warnings)
- Note:
  - a direct GPU-worker smoke could not start on this Mac because `scripts/run_fmm_gpu_worker.py` requires a visible CUDA-style GPU token; Taichi/Metal is not detected by that entrypoint's GPU discovery.

## 2026-05-04

- Updated comparison and indicatrix visualizations:
  - `comparison.png` now shows a shared path legend and relative L2 labels on the two travel-time difference panels
  - `indicatrix_field.png` now marks both grid-point origins and geometric centers of the Randers indicatrices
  - added `randers_indicatrix_center(M, b)` for the geometric center calculation
- Validation:
  - `conda run -n taichi python -m py_compile tests/compare_fm_asr_velocity_paths.py src/fmm_core/fm_asr.py`
  - `bash -n tests/run_compare_fm_asr_velocity_paths.sh`
  - `tests/run_compare_fm_asr_velocity_paths.sh` outside the sandbox, using Taichi/Metal GPU, regenerated `comparison.png` and `indicatrix_field.png`

## 2026-05-04

- Added travel-time L2 error reporting to `./tests/compare_fm_asr_velocity_paths.py`:
  - computes `||T_asr - T_fmm||_2 / ||T_fmm||_2`
  - computes `||T_asr_iso - T_fmm||_2 / ||T_fmm||_2`
  - writes absolute L2, relative L2, reference L2 norm, and finite comparison count to `summary.json`
  - stores scalar L2 and relative L2 values in `results.npz`
- Validation:
  - `conda run -n taichi python -m py_compile tests/compare_fm_asr_velocity_paths.py`
  - `bash -n tests/run_compare_fm_asr_velocity_paths.sh`
  - `tests/run_compare_fm_asr_velocity_paths.sh --no-indicatrix` outside the sandbox, using Taichi/Metal GPU

## 2026-05-04

- Updated `./tests/compare_fm_asr_velocity_paths.py` comparison plotting:
  - replaced the top-middle scalar-speed subplot with `FM-ASR ellipse T - FMM T`
  - replaced the bottom-right path-overlay subplot with `FM-ASR isotropic T - FMM T`
  - uses a zero-centered diverging color scale for signed travel-time differences
- Validation:
  - `conda run -n taichi python -m py_compile tests/compare_fm_asr_velocity_paths.py`
  - `bash -n tests/run_compare_fm_asr_velocity_paths.sh`
  - `tests/run_compare_fm_asr_velocity_paths.sh --no-indicatrix` outside the sandbox, using Taichi/Metal GPU, regenerated `output/fm_asr_demo/comparison.png`

## 2026-05-04

- Fixed the FM-ASR comparison demo source/target setup:
  - `./tests/compare_fm_asr_velocity_paths.py` now uses inset source/target vertical lines matching the `scripts/run_fmm.py` convention instead of physical domain boundaries
  - GPU Pollock seeds start on the inset source column and `right_x` is the inset target column, avoiding false "fastest" paths that exit through the top/bottom domain boundary
  - added `--domain-margin` and `DOMAIN_MARGIN` override support
- Made FM-ASR backtracking more robust:
  - `trace_minimal_path` now accepts `source_mask`
  - if a Hopf-Lax edge step does not decrease interpolated distance, it falls back to the lowest-distance stencil vertex instead of stopping halfway
- Validation:
  - `conda run -n taichi python -m py_compile src/fmm_core/fm_asr.py tests/compare_fm_asr_velocity_paths.py`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q tests/test_fm_asr.py` (15 passed)
  - `tests/run_compare_fm_asr_velocity_paths.sh --help`
  - `tests/run_compare_fm_asr_velocity_paths.sh --no-indicatrix` outside the sandbox, using Taichi/Metal GPU, regenerated `output/fm_asr_demo/comparison.png`

## 2026-05-04

- Integrated FM-ASR Hopf-Lax backtracking into `./src/fmm_core/fm_asr.py`:
  - moved draft backtracking helpers into the existing module structure
  - added `AsymmetricEuclideanNorm.edge_argmin`
  - added `trace_minimal_path` and `extract_fm_asr_path`
  - updated the comparison script to use the core extractor instead of local 8-neighbor descent
- Exported the new FM-ASR path extraction helpers from `./src/fmm_core/__init__.py`.
- Added tests for exact edge argmin and source-to-target FM-ASR path extraction.
- Validation:
  - `PYTHONPYCACHEPREFIX=/private/tmp/fmm_pycache python3 -m py_compile src/fmm_core/fm_asr.py src/fmm_core/__init__.py tests/test_fm_asr.py tests/compare_fm_asr_velocity_paths.py`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q tests/test_fm_asr.py` (15 passed)
  - `tests/run_compare_fm_asr_velocity_paths.sh --help`
  - `PYTHONPATH=../../FlowSimulation:./src conda run -n taichi pytest -q` (19 passed, 5 pre-existing FSM warnings)
