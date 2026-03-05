# Agent Work Log

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
