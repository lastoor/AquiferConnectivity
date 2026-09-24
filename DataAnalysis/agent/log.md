# Agent Work Log

## 2026-05-22
- Agent/tool: Codex (GPT-5)
- Summary: Added `fp_fm_asr_conic` to the comparison/plotting pipeline so the mixed Randers/conic FM-ASR path is indexed, summarized, and plotted alongside existing FM-ASR, FM-ASR-iso, PyPar2, FSM, FMM, Dijkstra, and LRP solvers.
- Files changed:
  - `scripts/step1_index.py`
  - `scripts/step2_metrics.py`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step1_index.py DataAnalysis/scripts/step2_metrics.py DataAnalysis/scripts/step3_stats.py DataAnalysis/scripts/step4_plot.py`
  - `git diff --check`
- Limitations/follow-up:
  - Full Stage 2-4 data processing and figure generation were not executed in this local environment.

## 2026-03-10
- Agent/tool: Codex (GPT-5)
- Summary: Updated Stage 4 plotting so `endpoint` renders as a grouped sample-based box plot, added separate `times_box` and `hausdorff_box` grouped box-plot sections while keeping the existing `times` and `hausdorff` summary plots, and refreshed README documentation to match the current path/plot outputs.
- Files changed:
  - `scripts/step1_index.py`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step3_stats.py scripts/step4_plot.py`
  - `python -m py_compile scripts/step1_index.py scripts/step3_stats.py scripts/step4_plot.py`
- Limitations/follow-up:
  - Full Stage 4 runtime was not executed here, so figure generation was validated by compile checks and code-path review rather than a complete plotting run.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Created `DataAnalysis_Simple` as a standalone simplified repository with a flat 4-stage pipeline and no `visualization/` package boundary.
- Files changed:
  - `README.md`
  - `.gitignore`
  - `requirements.txt`
  - `configs/visualization.default.yaml`
  - `scripts/step1_index.py`
  - `scripts/step2_metrics.py`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `scripts/run_all.py`
  - `tests/conftest.py`
  - `tests/test_step1_index.py`
  - `tests/test_step2_metrics.py`
  - `tests/test_step3_stats.py`
  - `tests/test_step4_plot.py`
  - `tests/test_pipeline_smoke.py`
  - `outputs/.gitkeep`
- Validation:
  - `python -m compileall scripts tests`
  - `PYTHONPATH=./scripts python scripts/run_all.py --config configs/visualization.default.yaml --dry-run`
  - End-to-end synthetic-data smoke run via `scripts/run_all.py` completed and generated figure outputs
- Limitations/follow-up:
  - `pytest` is not installed in the current shell environment, so test files were added but not executed here.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added agent workflow files to `DataAnalysis_Simple` mirroring `DataAnalysis` (`AGENT.md`, `agent/guide.md`, `agent/log.md`) and linked this workflow in README.
- Files changed:
  - `AGENT.md`
  - `agent/guide.md`
  - `agent/log.md`
  - `README.md`
- Validation:
  - Manual file review
- Limitations/follow-up:
  - Keep appending new entries below this section for each meaningful task.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added analytical `t_min` comparison overlay to `times` plot in `step4_plot.py`, with configurable `analytical_percentile` and analytical solver warnings included in manifest output.
- Files changed:
  - `scripts/step1_index.py`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `configs/visualization.default.yaml`
  - `tests/test_analytical_tmin.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m compileall scripts tests`
  - `PYTHONPATH=./scripts python scripts/run_all.py --config configs/visualization.default.yaml --dry-run`
- Limitations/follow-up:
  - `pytest` is not installed in the current shell environment, so tests were added but not executed here.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Fixed Stage 3 unpickle failure (`__main__.VarianceResults`) by adding legacy-compat unpickling and by serializing Stage 2 caches as plain dictionaries to prevent recurrence.
- Files changed:
  - `scripts/step2_metrics.py`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m compileall scripts tests`
  - Reproduced workflow using script-mode Stage 2 and confirmed Stage 3 no longer raises `AttributeError: module '__main__' has no attribute 'VarianceResults'`
- Limitations/follow-up:
  - `pytest` is not installed in the current shell environment.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added three new variance-sweep figures (`ratio_mean`, `ratio_var`, `wasserstein_len`), including Stage 3 Wasserstein length-distance preparation and Stage 4 rendering/warning integration.
- Files changed:
  - `scripts/step1_index.py`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `configs/visualization.default.yaml`
  - `tests/conftest.py`
  - `tests/test_step3_stats.py`
  - `tests/test_step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m compileall scripts tests`
  - Synthetic end-to-end run (temporary dataset) calling `step1`/`step2`/`step3`/`step4` verified artifacts include `run3_ratio_mean.png`, `run3_ratio_var.png`, `run3_wasserstein_len.png`
  - `PYTHONPATH=./scripts python scripts/run_all.py --config configs/visualization.default.yaml --dry-run`
- Limitations/follow-up:
  - `pytest` is not installed in the current shell environment, so automated test cases were updated but not executed with `pytest`.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added standalone `step5_single_field_paths.py` to select one logK realization from output, solve FP/LRP/FP(FMM)/FP(Dijkstra) with run_fmm-style workflow, and save two overlay figures (logK and minimal-arrival-time backgrounds).
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m compileall DataAnalysis_Simple/scripts/step5_single_field_paths.py DataAnalysis_Simple/tests/test_step5_single_field_paths.py`
  - `PYTHONPATH=./DataAnalysis_Simple/scripts python DataAnalysis_Simple/scripts/step5_single_field_paths.py --help`
  - Synthetic smoke run via Python snippet with mocked `solve_single_field`, confirming two files saved:
    - `run3_var0p1_size200_mc0_logk_paths.png`
    - `run3_var0p1_size200_mc0_arrival_paths.png`
  - Attempted `pytest -q DataAnalysis_Simple/tests/test_step5_single_field_paths.py` (not available: `pytest: command not found`)
- Limitations/follow-up:
  - Full solver execution still depends on external runtime stack (`FlowSimulation`, PETSc, Numba, and Algorithm/src import path).

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Moved visualization output path out of `configs/visualization.default.yaml` and added CLI override support via `--output-dir` in `scripts/step4_plot.py` and `scripts/run_all.py` (with smoke tests covering override behavior).
- Files changed:
  - `configs/visualization.default.yaml`
  - `scripts/step4_plot.py`
  - `scripts/run_all.py`
  - `tests/test_step4_plot.py`
  - `tests/test_pipeline_smoke.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m compileall scripts tests`
  - `PYTHONPATH=./scripts python scripts/run_all.py --config configs/visualization.default.yaml --dry-run --output-dir ./outputs/cli_override_demo`
  - `PYTHONPATH=./scripts python scripts/step4_plot.py --help`
  - Attempted `pytest -q tests/test_pipeline_smoke.py tests/test_step4_plot.py` (not available: `pytest: command not found`)
- Limitations/follow-up:
  - Full test execution requires installing `pytest` in the active environment.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Clarified README documentation for the output-path migration by adding an explicit Stage 4 output directory resolution order (`--output-dir` override, optional config field, fallback default).
- Files changed:
  - `README.md`
  - `agent/log.md`
- Validation:
  - Manual README review
- Limitations/follow-up:
  - No code-path changes; documentation-only update.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Replaced `tests/compare_hausdorff.py` with a standalone CLI utility that compares Hausdorff distances from Shapely and Stage-2 `nn_max` for one selected variance/field-size/MC sample, and documented usage in README.
- Files changed:
  - `tests/compare_hausdorff.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile tests/compare_hausdorff.py`
  - `PYTHONPATH=./scripts python tests/compare_hausdorff.py --help`
  - `PYTHONPATH=./scripts python tests/compare_hausdorff.py --config configs/visualization.default.yaml --variance 0.1 --field-size 200 --mc 1` (expected nonzero in current env: shapely missing)
- Limitations/follow-up:
  - Full comparison run requires `shapely` to be installed in the active environment.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Fixed wiring for `--hausdorff-compute-method` in `step2_metrics.py` by forwarding the selected method into both variance/std analysis paths, adding CLI value validation (`shapely` or `nn_max`), and recording the selected method in Stage 2 metadata notes.
- Files changed:
  - `scripts/step2_metrics.py`
  - `tests/test_step2_metrics.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step2_metrics.py tests/test_step2_metrics.py`
  - Source verification via `grep -n` confirmed forwarding and argparse choices are present
- Limitations/follow-up:
  - `pytest` is not installed in the current shell environment, so updated tests were not executed.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Extended `tests/compare_hausdorff.py` to also compare Frechet distance using a custom discrete Frechet implementation against `shapely.frechet_distance` for the same selected path pairs.
- Files changed:
  - `tests/compare_hausdorff.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile tests/compare_hausdorff.py`
  - Manual source review for new Frechet comparison table and helper functions
- Limitations/follow-up:
  - End-to-end run of the utility depends on dataset scan/runtime and requires `shapely` in the active environment.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added per-method runtime comparison to `tests/compare_distance_by_shapely_with_custom.py`, including per-pair timing columns and total/average timing summaries for Hausdorff (shapely vs nn_max) and Frechet (shapely vs custom).
- Files changed:
  - `tests/compare_distance_by_shapely_with_custom.py`
  - `agent/log.md`
- Validation:
  - `python -m py_compile tests/compare_distance_by_shapely_with_custom.py`
- Limitations/follow-up:
  - End-to-end timing comparison still requires `shapely` and a matching dataset sample.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added Stage-2 Frechet method option (`--frechet-compute-method`), renamed Hausdorff method value `nn_max` to `custom`, and unified custom Frechet implementation in `step2_metrics.py` for reuse by the comparison utility.
- Files changed:
  - `scripts/step2_metrics.py`
  - `tests/compare_distance_by_shapely_with_custom.py`
  - `tests/test_step2_metrics.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `grep -Rni "nn_max|--frechet-compute-method|frechet_compute_method" scripts tests README.md`
  - `python -m py_compile scripts/step2_metrics.py tests/test_step2_metrics.py tests/compare_distance_by_shapely_with_custom.py`
- Limitations/follow-up:
  - Full runtime verification against real data depends on environment and dataset availability; `pytest` is not installed in this shell.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Updated Stage 5 plotting to remove titles, normalize axes (`x/I_Y`, `y/I_Y`), normalize arrival-time field (`T_min/t_ref`), add source/target lines to all figures, and add a third output figure with speed background plus streamlines.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step5_single_field_paths.py tests/test_step5_single_field_paths.py`
  - `pytest -q tests/test_step5_single_field_paths.py` (`3 passed in 8.49s`)
- Limitations/follow-up:
  - Streamline and solver runtime behavior outside tests still depends on external Algorithm/FlowSimulation stack and dataset environment.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added Stage-4 Frechet plotting (main variance figure), wired Stage-3 Frechet summary stats, and enabled Frechet in default plotting sections/config and test fixture defaults.
- Files changed:
  - `scripts/step1_index.py`
  - `configs/visualization.default.yaml`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `tests/conftest.py`
  - `tests/test_step3_stats.py`
  - `tests/test_step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step1_index.py scripts/step3_stats.py scripts/step4_plot.py tests/test_step3_stats.py tests/test_step4_plot.py`
  - `pytest -q tests/test_step3_stats.py tests/test_step4_plot.py` (`3 passed in 4.17s`)
- Limitations/follow-up:
  - Frechet plotting scope is main variance only (`frechet` section); no std-sweep Frechet figure was added.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Updated Step 5 velocity visualization to use dimensionless velocity (`v/(k_g*dh)`) for both speed background and streamlines, and updated the stream colorbar label accordingly.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step5_single_field_paths.py tests/test_step5_single_field_paths.py`
  - `pytest -q tests/test_step5_single_field_paths.py` (`4 passed`)
- Limitations/follow-up:
  - End-to-end solver runtime/physics behavior still depends on external Algorithm/FlowSimulation and dataset availability.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Added separate Stage-4 path-length CDF figure generation (one figure per selected variance), exposed CDF sample arrays from Stage 3, and added `--cdf-variances` CLI option in `step4_plot.py`.
- Files changed:
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `tests/test_step3_stats.py`
  - `tests/test_step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step3_stats.py scripts/step4_plot.py tests/test_step3_stats.py tests/test_step4_plot.py`
  - `pytest -q tests/test_step3_stats.py tests/test_step4_plot.py`
- Limitations/follow-up:
  - CDF figures are generated only when `wasserstein_len` section is enabled and rely on first-size finite length samples from Stage 2 payloads.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Updated Stage-4 scatter behavior to support `mhr/dh` normalization and optional raw all-point scatter per selected variance; unified variance CLI option to `--plot-variances` for both CDF and scatter(all_points), with comma-list `--scatter-mode` supporting `mean`, `all_points`, or both.
- Files changed:
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `tests/conftest.py`
  - `tests/test_step3_stats.py`
  - `tests/test_step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step3_stats.py scripts/step4_plot.py tests/test_step3_stats.py tests/test_step4_plot.py`
  - `pytest -q tests/test_step3_stats.py tests/test_step4_plot.py`
- Limitations/follow-up:
  - Scatter all-points output depends on Stage-3 `scatter_time_samples` availability and currently pairs samples by aligned index after finite filtering.

## 2026-03-04
- Agent/tool: Codex (GPT-5)
- Summary: Changed Stage-4 ratio plots to use `R_min/dh` normalization and compute moments of the sample-wise ratio (`mean(t_min/(R_min/dh))`, `var(t_min/(R_min/dh))`) instead of ratio-of-moments.
- Files changed:
  - `scripts/step4_plot.py`
  - `tests/test_step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step4_plot.py tests/test_step4_plot.py`
  - `pytest -q tests/test_step4_plot.py`
- Limitations/follow-up:
  - Ratio moments are computed from aligned finite `scatter_time_samples`; missing per-variance samples produce `NaN` points with manifest warnings.

## 2026-03-09
- Agent/tool: Codex (GPT-5)
- Summary: Added standalone `step6_single_field_arrival_hist.py` to generate one GRF logK field, solve hydraulic head/velocity, run particle tracking, and compare empirical arrival-time CDF (`status==0` particles) against analytical `_g_tau` from `step4_plot.py`; updated README documentation and added focused unit tests.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m compileall scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
- Limitations/follow-up:
  - `pytest` is not installed in the current shell environment, so new tests were added but not executed with `pytest`.

## 2026-03-09
- Agent/tool: Codex (GPT-5)
- Summary: Fixed Step 6 arrival-time extraction to use terminal time per successful particle trajectory (instead of flattened per-step times), and added optional artifact-save flags (`--save-npz`, `--save-json`) so figure-only output is default while NPZ/JSON saves are opt-in.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `scripts/step4_plot.py`
  - `tests/test_analytical_tmin.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m compileall scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
  - `python -m compileall scripts/step4_plot.py tests/test_analytical_tmin.py`
  - Lightweight runtime sanity check for `_g_tau` and `compute_analytical_tmin_curve` via `PYTHONPATH=./scripts python - <<'PY' ...`
- Limitations/follow-up:
  - `pytest` is not installed in the current shell environment, so updated tests were compile-checked but not executed with `pytest`.

## 2026-03-12
- Agent/tool: Codex (GPT-5)
- Summary: Added Step 6 reload/visualization support so saved arrival-time NPZ artifacts can be loaded and replotted without rerunning simulation, added a round-trip Step 6 test, and updated README usage for the new `--load-npz` workflow.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
  - Attempted `pytest tests/test_step6_single_field_arrival_hist.py` (`pytest: command not found`)
  - Attempted `python -m pytest tests/test_step6_single_field_arrival_hist.py` (`No module named pytest`)
- Limitations/follow-up:
  - Full automated test execution still requires `pytest` to be installed in the active environment.

## 2026-03-12
- Agent/tool: Codex (GPT-5)
- Summary: Updated Step 6 to use a nonuniform tau grid with denser sampling before `tau=10` by default, added CLI controls for tau-grid density and y-axis exponent scaling (`--y-exp`), extended Step 6 tests, and refreshed README documentation.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
- Limitations/follow-up:
  - Full automated test execution still requires `pytest` to be installed in the active environment.

## 2026-03-14
- Agent/tool: Codex (GPT-5)
- Summary: Cleaned up Step 6 geometry consistency so analytical length now follows `target_x - source_x`, source seed placement honors the half-open source span, forward transport is validated, target-y controls are documented as metadata-only, and the submit/doc surfaces were updated to match.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `submit_step6_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step6_single_field_arrival_hist.py`
- Limitations/follow-up:
  - Full runtime verification with custom geometry settings has not been run end-to-end in this environment.

## 2026-03-16
- Agent/tool: Codex (GPT-5)
- Summary: Integrated the new linearized Eikonal solver into Step 5 as an optional `FP^i` workflow from `LRP (FMM)` toward `FP (FMM)`, removed Dijkstra paths from the main Step 5 visual overlays, added a combined linearized-progression figure, updated Step 5 tests, and refreshed README documentation.
- Files changed:
  - `scripts/fp_and_lrp/linear_eikonal_solver.py`
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/fp_and_lrp/linear_eikonal_solver.py scripts/step5_single_field_paths.py tests/test_step5_single_field_paths.py`
- Limitations/follow-up:
  - Full Step 5 runtime verification with the linearized `FP^i` workflow has not been run end-to-end in this environment.

## 2026-03-16
- Agent/tool: Codex (GPT-5)
- Summary: Added `--check-path-geometry/--no-check-path-geometry` to Step 5, wired it through the GPU submit script, and documented the bypass for exploratory linearized `FP^i` runs that fail the current geometry validator.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step5_single_field_paths.py tests/test_step5_single_field_paths.py`
  - `bash -n submit_step5_gpu.sh`
- Limitations/follow-up:
  - Disabling geometry checks allows invalid paths to propagate into plots without additional safeguards.

## 2026-03-12
- Agent/tool: Codex (GPT-5)
- Summary: Added a Step 6 CLI toggle to enable or disable early-`tau` grid refinement (`--refine-tau-grid/--no-refine-tau-grid`), updated tests for refined vs uniform grid construction, and refreshed README usage.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
- Limitations/follow-up:
  - Full automated test execution still requires `pytest` to be installed in the active environment.

## 2026-03-16
- Agent/tool: Codex (GPT-5)
- Summary: Refactored Step 5 to use one shared source/target geometry across particle tracking, `LRP (FMM)`, `FP (FMM)`, and linearized `FP^i`; added Step 5 geometry CLI overrides (`--source-x`, `--target-x`, `--source-y-start`, `--source-y-end`, `--target-y-start`, `--target-y-end`); aligned particle seeding to half-open source-span semantics; and updated the Step 5 submit script, tests, and README accordingly.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step5_single_field_paths.py tests/test_step5_single_field_paths.py`
- Limitations/follow-up:
  - Step 5 particle tracking still stops on `right_x` and does not enforce a target y-band in the underlying tracker; the default shared target semantics therefore use the full right boundary to stay consistent with the available particle-tracking API.

## 2026-03-16
- Agent/tool: Codex (GPT-5)
- Summary: Added Step 5 `--linearized-scheme` support with `linearized` and `full_eikonal` modes, documented the scheme-specific meaning of `--linearized-damping` (`alpha`) and the controls ignored by `full_eikonal`, and updated the Step 5 submit script to forward the scheme selection and print it in job logs.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step5_single_field_paths.py tests/test_step5_single_field_paths.py scripts/fp_and_lrp/linear_eikonal_solver.py`
  - `bash -n submit_step5_gpu.sh`
- Limitations/follow-up:
  - Full end-to-end Step 5 runs with `--linearized-scheme full_eikonal` were not executed in this environment.

## 2026-03-12
- Agent/tool: Codex (GPT-5)
- Summary: Extended Step 6 reload mode so `--load-npz` can recompute analytical/lognormal/MC curves on a new tau grid using the same refinement options as the compute path, persisted the required reload metadata in NPZ/JSON outputs, added backward-compatible fallback via sibling summary JSON or saved curves, and expanded Step 6 tests accordingly.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
- Limitations/follow-up:
  - Full automated test execution still requires `pytest` to be installed in the active environment.

## 2026-03-12
- Agent/tool: Codex (GPT-5)
- Summary: Added an opt-in extra Step 6 probability-density figure (`--save-pdf-plot`) for both compute and reload modes, using a normalized empirical histogram plus density curves derived numerically from the analytical/lognormal/MC CDFs; updated tests, README, and CLI output accordingly.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
- Limitations/follow-up:
  - Full automated test execution still requires `pytest` to be installed in the active environment.

## 2026-03-17
- Agent/tool: Codex (GPT-5)
- Summary: Reworked Step 5 curvature diagnostics to use a shared spline-geometry module with explicit interpolation vs approximation modes, added spline-path overlays and split curvature-balance panels, introduced optional Gaussian field smoothing for normal-derivative diagnostics, made curvature/normals/normal-derivative sampling spline-consistent, expanded the residual panel with multiple LRP balance combinations, updated the GPU submit script to forward the new curvature controls, and fixed two numerical issues: linearized FP discrete backtracking now ranks neighbors by travel-time drop per unit distance and the raw curvature fallback now uses the full parametric denominator.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `scripts/fp_and_lrp/curvature.py`
  - `tests/test_step5_single_field_paths.py`
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step5_single_field_paths.py scripts/fp_and_lrp/curvature.py tests/test_step5_single_field_paths.py`
  - `bash -n submit_step5_gpu.sh`
- Limitations/follow-up:
  - `pytest` is not installed in this environment, so the new Step 5 regression tests were not executed here.
  - Full end-to-end Step 5 figure generation was not rerun after the latest curvature/residual updates.

## 2026-03-12
- Agent/tool: Codex (GPT-5)
- Summary: Reworked the Step 6 probability-density figure to recalculate densities from saved raw samples and model metadata instead of differentiating saved CDF arrays, added saved histogram metadata (`--pdf-hist-bins` + effective bin edges) for reproducible reload-mode PDF regeneration, and updated tests/README accordingly.
- Files changed:
  - `scripts/step6_single_field_arrival_hist.py`
  - `tests/test_step6_single_field_arrival_hist.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step6_single_field_arrival_hist.py tests/test_step6_single_field_arrival_hist.py`
- Limitations/follow-up:
  - Full automated test execution still requires `pytest` to be installed in the active environment.

## 2026-03-27
- Agent/tool: Cursor (Claude)
- Summary: Added ensemble-averaged empirical CDF plot for Hausdorff nearest-neighbor distances. Stage 3 now loads raw MC consolidated `.npy` files, extracts the full `['hausdorff'][pair_key]['distances']` arrays, computes per-MC empirical CDFs on a common 500-point grid, and stores mean/std CDF arrays per pair per variance. Stage 4 renders the mean CDF curve with std shading for all 4 distance pairs, gated by a new `--hausdorff-cdf` CLI flag.
- Files changed:
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step3_stats.py scripts/step4_plot.py`
- Limitations/follow-up:
  - Full Stage 3/4 runtime with the new feature was not executed here; validated by compile check and code-path review.
  - The ensemble CDF uses only the reference field size (`field_size_ref`) MC files; multi-size ensemble is not supported.

## 2026-03-27
- Agent/tool: Cursor (Claude)
- Summary: Added `visualize_single_logk_paths.py` — a standalone script to load a consolidated per-MC `.npy` file and visualize the logK field with all saved paths overlaid. Supports both direct file path (`--npy-file`) and discovery mode (`--config` + sweep/value/field-size/mc-index). Prints a summary table of path lengths, travel times, Hausdorff stats, and Shannon entropy to stdout.
- Files changed:
  - `scripts/visualize_single_logk_paths.py` (new)
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/visualize_single_logk_paths.py`
  - `python scripts/visualize_single_logk_paths.py --help`
- Limitations/follow-up:
  - Discovery mode depends on `step1_index` and `step2_metrics` imports being available on `PYTHONPATH`.

## 2026-03-31
- Agent/tool: Claude Code (claude-sonnet-4-6)
- Summary: Added Savitzky-Golay curvature method and four normal-derivative methods (`direct` cubic, `savgol2d`, `spline2d`, `mls_frenet`) to Step 5 curvature/balance diagnostics.
  - `curvature.py`: added `compute_savgol_geometry` (dθ/ds via Savitzky-Golay filter on arc-length-uniform tangent angle); extended `compute_curvature_bundle` to support `method="savgol"` with `savgol_window` and `savgol_poly` params.
  - `step5_single_field_paths.py`: upgraded `"direct"` normal derivative to cubic `RegularGridInterpolator` (linear fallback for older scipy); added `_savgol_gradient_2d` (separable 2D SG), `_spline2d_normal_derivative` (`RectBivariateSpline` analytic ∂/∂n), and `_mls_frenet_normal_derivative` (Moving Least Squares quadratic polynomial in the Frenet-Serret (s,n) frame — design matrix precomputed once, O(N·m) vectorized field sampling via `RegularGridInterpolator`); threaded `normal_deriv_method`, `normal_deriv_spline2d_smooth`, `normal_deriv_mls_ns`, `normal_deriv_mls_nn` through the full call chain (`_build_path_balance_diagnostics` → `compute_balance_diagnostics` → `run_step5` → `parse_args` → `main`).
  - `submit_step5_gpu.sh`: added all new curvature/normal-deriv parameters; default changed to `normal_deriv_method="mls_frenet"`.
  - `tests/test_step5_single_field_paths.py`: added 5 tests — `test_compute_savgol_geometry_on_parabola`, `test_normal_derivative_direct_on_linear_field`, `test_savgol_gradient_2d_on_linear_field`, `test_spline2d_normal_derivative_on_linear_field`, `test_mls_frenet_normal_derivative_on_linear_field`; all 5 pass.
- Files changed:
  - `scripts/fp_and_lrp/curvature.py`
  - `scripts/step5_single_field_paths.py`
  - `submit_step5_gpu.sh`
  - `tests/test_step5_single_field_paths.py`
- Validation:
  - `pytest tests/test_step5_single_field_paths.py::test_compute_savgol_geometry_on_parabola tests/test_step5_single_field_paths.py::test_normal_derivative_direct_on_linear_field tests/test_step5_single_field_paths.py::test_savgol_gradient_2d_on_linear_field tests/test_step5_single_field_paths.py::test_spline2d_normal_derivative_on_linear_field tests/test_step5_single_field_paths.py::test_mls_frenet_normal_derivative_on_linear_field` — 5 passed
- Limitations/follow-up:
  - MLS Frenet is exact for linear fields to ~2e-6 (pseudoinverse rounding); tolerance set to `atol=1e-5` in test.
  - End-to-end Step 5 figure generation with the new methods has not been rerun.

## 2026-03-27
- Agent/tool: Cursor (Claude)
- Summary: Added per-MC entropy vs Hausdorff comparison figure. Stage 3 now collects per-MC Shannon entropy (vmag, angle) and Hausdorff summary stats (max, mean, median) for all 4 path pairs from consolidated `.npy` files. Stage 4 renders 5 vertically stacked panels (entropy, FP(FMM), FP(Dijkstra), LRP(FMM), LRP) sharing the MC-index x-axis, gated by `--entropy-hausdorff` CLI flag and `--plot-variances`.
- Files changed:
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step3_stats.py scripts/step4_plot.py`
- Limitations/follow-up:
  - Full Stage 3/4 runtime with the new feature was not executed here; validated by compile check and code-path review.
  - The entropy+Hausdorff samples use only the reference field size (`field_size_ref`) MC files.

## 2026-04-03
- Agent/tool: Claude Code (claude-sonnet-4-6)
- Summary: Added Fast Sweeping Method (FSM) as a fourth path-finding method in Step 5 (`--fsm-fp`). FSM directly solves the advection equation `v·∇T = 1` (where `v = (vx, vy)` is the cell-centred Darcy velocity) via upwind finite differences + 4-direction Gauss-Seidel sweeping, then traces the fastest path backward by Euler integration `dγ/dτ = -v(γ)` using `RegularGridInterpolator`. This is physically distinct from the existing Eikonal FMM (which uses a scalar slowness field); FSM tracks the full directional velocity field. Key implementation details:
  - Array convention: `(n_y, n_x)` (row=y, col=x), correcting the sample-code's transposed convention.
  - Soft monotonicity check in backtracking: warns on T-increase instead of hard-stopping, to tolerate minor numerical drift.
  - New fields `fp_fsm` and `t_min_fsm` added to `PathSolveResult` (slots=True, nullable).
  - FSM path plotted in green (`#2CA02C`, dash-dot-dot) with label `γ*_S` when present.
  - Off by default (`--no-fsm-fp`); existing behavior unchanged.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -c "import ast; ast.parse(open('scripts/step5_single_field_paths.py').read()); print('OK')"` — syntax OK
  - `bash -n submit_step5_gpu.sh`
- Limitations/follow-up:
  - Pure-Python nested loops in `_solve_fsm_travel_time` are O(n_y × n_x × max_sweeps); for 200×200 with 20 sweeps this is ~800K iterations and may take a few seconds. A `@numba.jit(nopython=True)` decorator would give ~50–100× speedup if needed.
  - End-to-end Step 5 figure generation with `--fsm-fp` has not been rerun in this environment.

## 2026-04-13
- Agent/tool: Codex (GPT-5)
- Summary: Updated the analysis pipeline for the new FSM-enabled run outputs and moved Step 5 FSM solving onto the shared `Algorithm/src/fmm_core/fsm.py` implementation. Step 4 removed the Taylor and `std_*` figures, added FSM to the remaining missing method-comparison plots (scatter mean/all-points and ratio mean/variance), and kept the existing FSM-aware time, Hausdorff, endpoint, Frechet, and Wasserstein/path-length plots aligned with the current run cache. Step 5 now imports the shared FSM solver/path code, supports residual-MSE convergence controls (`--fsm-mse-tol`, `--fsm-convergence-mode`, `--fsm-mse-check-every`, `--fsm-mse-activation-factor`, `--fsm-consecutive-passes`), and `submit_step5_gpu.sh` passes those options through in batch mode.
- Files changed:
  - `configs/visualization.default.yaml`
  - `scripts/step1_index.py`
  - `scripts/step2_metrics.py`
  - `scripts/step3_stats.py`
  - `scripts/step4_plot.py`
  - `scripts/step5_single_field_paths.py`
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile scripts/step4_plot.py scripts/step5_single_field_paths.py`
  - `bash -n submit_step5_gpu.sh`
- Limitations/follow-up:
  - End-to-end Stage 4 and Step 5 figure generation was not rerun here.
  - `pytest` is not installed in the active environment, so test files were not executed in this session.

## 2026-04-13
- Agent/tool: Codex (GPT-5)
- Summary: Tightened the batch Step 5 FSM submit-wrapper default from `fsm_tol=1e-6` to `fsm_tol=1e-8` so Slurm runs use a stricter update-based convergence target by default while keeping the new residual-MSE controls optional.
- Files changed:
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `bash -n submit_step5_gpu.sh`
- Limitations/follow-up:
  - This changes only the batch-wrapper default; the `step5_single_field_paths.py` CLI default remains `1e-6` unless overridden explicitly.

## 2026-04-19
- Agent/tool: Codex (GPT-5)
- Summary: Reworked `submit_step5_gpu.sh` into a dual-mode Step 5 launcher that auto-selects Slurm submission on cluster hosts and direct local execution on macOS or non-Slurm machines. Local runs now derive repo-relative paths, honor `STEP5_PYTHON` / `STEP5_PYTHONPATH` / `STEP5_LOCAL_BACKEND`, prefer GPU on macOS with a Taichi preflight, and fall back to CPU while disabling unsupported snapshots when GPU initialization is unavailable. Also updated the Pollock GPU summary to report the actual Taichi backend (`taichi_arch`) so CPU fallback is visible to callers.
- Files changed:
  - `submit_step5_gpu.sh`
  - `README.md`
  - `agent/log.md`
  - `../FlowSimulation/simulation/particle_tracking/pollock_gpu.py`
  - `../FlowSimulation/tests/test_particle_tracking.py`
- Validation:
  - `bash -n DataAnalysis/submit_step5_gpu.sh`
  - `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text()) for p in ['../FlowSimulation/simulation/particle_tracking/pollock_gpu.py','../FlowSimulation/tests/test_particle_tracking.py']]; print('AST OK')"`
- Limitations/follow-up:
  - End-to-end Step 5 execution was not run here, so the new local launcher path was validated by shell parsing and code-path review rather than a full solve.
  - The local GPU preflight only checks Taichi initialization; the actual run can still report `taichi_arch="cpu"` if the runtime falls back later.

## 2026-04-19
- Agent/tool: Codex (GPT-5)
- Summary: Normalized FSM convergence arguments inside `step5_single_field_paths.py` when `--fsm-solver new` is selected, so Step 5 no longer forwards legacy residual-gating controls (`mse_tol`, `convergence_mode`, `consecutive_passes`, etc.) that the new Algorithm FSM backend explicitly ignores and warns about.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `agent/log.md`
- Validation:
  - `python -c "import ast, pathlib; ast.parse(pathlib.Path('DataAnalysis/scripts/step5_single_field_paths.py').read_text()); print('AST OK')"`
- Limitations/follow-up:
  - This removes the warning noise for `fsm_solver=new` but does not change the numerical behavior of the underlying Algorithm FSM backend.

## 2026-04-20
- Agent/tool: Codex (GPT-5)
- Summary: Updated Step 5 FSM path solving to call `compute_fsm_path(..., use_inlet_cut=True, ...)`, aligning the single-column source-mask setup used by Step 5 with the Algorithm FSM backends. This restores the inlet-cut semantics expected by the older solver paths and avoids the `solver='new'` reachability failure where no target cell received a finite travel time.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `agent/log.md`
- Validation:
  - `python -c "import ast, pathlib; ast.parse(pathlib.Path('DataAnalysis/scripts/step5_single_field_paths.py').read_text()); print('AST OK')"`
- Limitations/follow-up:
  - This required a follow-up API patch in `Algorithm/src/fmm_core/fsm.py` because `compute_fsm_path()` did not yet expose `use_inlet_cut`.
  - The full Step 5 run was not re-executed in this session.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Replaced the Step 5 curvature-balance plot with four requested comparisons: streamline curvature computed explicitly as `(v_hat·grad)v_hat` versus the projected `grad(log|grad h|)` term, `K`-only Eikonal geometric curvature versus projected `grad(log K)`, full Eikonal geometric curvature versus the existing signed projected `grad(log(K|grad h|))` term, and `-K_E` versus `K_S-K_E°`, with all field terms evaluated using the normal of the path shown in that panel. Updated mathematical legends and correlation annotations accordingly.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `tests/test_step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - `python -c "import ast, pathlib; ast.parse(pathlib.Path('DataAnalysis/scripts/step5_single_field_paths.py').read_text()); print('AST OK')"`
  - Rendered the four-panel figure with the Agg backend in the `pde_env` Conda environment.
  - Focused tests for the circular velocity-direction curvature estimator and three Step 5 figure-generation paths: `4 passed`.
  - Full `test_step5_single_field_paths.py`: `32 passed, 3 failed`; all three failures are pre-existing expectations unrelated to this task (legacy path-label spelling and white versus black plot halos).
- Limitations/follow-up:
  - The active base environment lacks pytest and Matplotlib; validation used `/Users/binhaoli/.conda/envs/pde_env/bin/python`.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Recorded the user's environment policy prohibiting creation of Python environments without explicit consent, and removed the user-requested `pde_env` Conda environment. Confirmed the Apple Silicon launcher uses the existing `taichi` Conda environment and changed its normal-derivative default to `direct`, ensuring the requested curvature right-hand sides come from the `h` and `K` fields instead of a travel-time Hessian.
- Files changed:
  - `AGENT.md`
  - `submit_step5_mac_m4.sh`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `bash -n DataAnalysis/submit_step5_mac_m4.sh`
  - Confirmed `pde_env` is absent from `conda env list`.
  - Confirmed the existing `taichi` environment imports NumPy, SciPy, Matplotlib, cmcrameri, and Taichi.
  - Confirmed Taichi 1.7.4 initializes the Apple Metal GPU backend.
- Limitations/follow-up:
  - The launcher retains its computationally heavy defaults (one million particles plus optional FSM, FM-ASR, and PyPar2) unless overridden at invocation time.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Separated left-hand-side curvature estimators from the `NORMAL_DERIV_METHOD` right-hand-side derivative setting. The four-panel comparison now uses `(v_hat·grad)v_hat` for `kappa_S` and the corresponding travel-time Hessians for `kappa_E°` and `kappa_E`; it no longer plots coordinate-differentiated curve curvature as any of these three left-hand sides. Physical-field RHS terms remain independently evaluated from `h` and `K` when `NORMAL_DERIV_METHOD=direct`.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 figure tests in the existing `taichi` environment.
- Limitations/follow-up:
  - Coordinate-derived curvature is retained in `PathBalanceDiagnostics.curvature` for other diagnostics, but is not used by the four-panel comparison.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Corrected the definition of `K_E` in the E and identity panels from `grad(log(K|grad h|))` to `grad(log(K/|grad h|))`. The final panel now compares an independently differentiated `-grad(log(K/|grad h|))` against `grad(log|grad h|)-grad(log K)`, so the two curves express the requested identity and should agree up to numerical error.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 tests in the existing `taichi` environment.
- Limitations/follow-up:
  - The two sides use the same path normal and linear differentiation pipeline, so a material discrepancy now indicates a numerical implementation issue rather than the previous algebraic sign error.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Corrected the preceding interpretation after the user clarified the operator definitions: `K_S=P grad(log|grad h|)`, `K_E°=-P grad(log K)`, and `K_E=-P grad(log(K|grad h|))`. Removed the erroneous ratio field. The E° and E panels now use the required negative signs, while the final panel compares `-K_E=grad(log(K|grad h|))` with `K_S-K_E°=grad(log|grad h|)+grad(log K)`. Constructed the effective log-product as `log K + log|grad h|` so optional field smoothing cannot violate the identity through separately smoothing the product before taking its logarithm.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 tests in the existing `taichi` environment.
- Limitations/follow-up:
  - The previous log entry documents the superseded ratio interpretation; this entry records the corrected definitions.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Replaced the fourth curvature panel's two-curve streamline identity comparison with a single `K_E=-P grad(log(K|grad h|))` curve evaluated along `gamma_E°*`, using the E° path's points, tangent, normal, and normalized arclength.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 figure tests in the existing `taichi` environment.
- Limitations/follow-up:
  - The fourth panel intentionally has no correlation annotation because it contains only one curve.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Updated the fourth curvature panel to plot `-K_E` rather than `K_E` along `gamma_E°*`, and restored the requested comparison with `K_S-K_E°`. Both sides are evaluated using the E° path frame and normalized arclength, and the panel again reports their Pearson correlation.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 figure tests in the existing `taichi` environment.
- Limitations/follow-up:
  - The previous log entry records the superseded single-curve fourth panel.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Changed the fourth-panel comparison term from the field operators `K_S-K_E°` evaluated in one frame to the previously computed curvatures `kappa_S(gamma_S*)-kappa_E°(gamma_E°*)`. Interpolated velocity-derived `kappa_S` from its streamline normalized-arclength grid onto the E° normalized-arclength grid before subtracting the E° travel-time-Hessian curvature. Retained `-K_E` evaluated along `gamma_E°*` as the other curve.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 figure tests in the existing `taichi` environment.
- Limitations/follow-up:
  - Equal normalized arclength is a parameter-space comparison, not a spatial point-to-point correspondence between the two different paths.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Added two requested curves to the fourth curvature panel: `[-K_S](gamma_S*)-[-K_E°](gamma_E°*)` and `[-K_E](gamma_S*)`. The mixed-path operator difference interpolates the streamline term onto the E° normalized-arclength grid; the standalone `-K_E` streamline term is plotted on its native normalized-arclength grid.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 figure tests in the existing `taichi` environment.
- Limitations/follow-up:
  - Equal normalized arclength between `gamma_S*` and `gamma_E°*` is not a spatial point correspondence.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Reduced the fourth curvature panel to exactly two curves: `-K_E` evaluated along `gamma_E°*` and `[K_S](gamma_S*)-[K_E°](gamma_E°*)`. Removed the curvature-difference and additional negative-operator curves, and corrected the retained operator-difference signs. The `K_S` term is interpolated onto the E° normalized-arclength grid.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 figure tests in the existing `taichi` environment.
- Limitations/follow-up:
  - Equal normalized arclength between two different paths is not a spatial point correspondence.

## 2026-09-23
- Agent/tool: Codex (GPT-5)
- Summary: Finalized the fourth panel with exactly two requested curves: `-K_E` evaluated along `gamma_E°*` and the previously computed curvature difference `kappa_S(gamma_S*)-kappa_E°(gamma_E°*)`. Replaced the operator-difference curve with the velocity-derived and travel-time-Hessian curvatures, interpolating `kappa_S` onto the E° normalized-arclength grid.
- Files changed:
  - `scripts/step5_single_field_paths.py`
  - `README.md`
  - `agent/log.md`
- Validation:
  - `python -m py_compile DataAnalysis/scripts/step5_single_field_paths.py`
  - Focused Step 5 figure tests in the existing `taichi` environment.
- Limitations/follow-up:
  - Equal normalized arclength between the two paths is not a spatial point correspondence.
