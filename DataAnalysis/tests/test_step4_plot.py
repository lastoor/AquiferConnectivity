from __future__ import annotations

import csv
import os
import subprocess
import sys

import numpy as np

import step1_index as step1
import step2_metrics as step2
import step3_stats as step3
import step4_plot as step4


def test_ratio_moments_use_t_over_r_over_dh() -> None:
    var = np.asarray([0.1], dtype=float)
    samples = {
        "var_0.1": {
            "fpt": np.asarray([2.0, 4.0], dtype=float),
            "fpt_fmm": np.asarray([1.0, 2.0], dtype=float),
            "fpt_dijkstra": np.asarray([3.0, 6.0], dtype=float),
            "mhr": np.asarray([1.0, 2.0], dtype=float),
        }
    }
    dh = 0.5
    mean_fp, mean_fmm, mean_dij, var_fp, var_fmm, var_dij = step4._compute_ratio_moments_series(
        var,
        scatter_samples_by_var=samples,
        dh=dh,
        warnings=[],
    )
    ratio_fp = np.asarray([2.0, 4.0]) / (np.asarray([1.0, 2.0]) / dh)
    ratio_fmm = np.asarray([1.0, 2.0]) / (np.asarray([1.0, 2.0]) / dh)
    ratio_dij = np.asarray([3.0, 6.0]) / (np.asarray([1.0, 2.0]) / dh)
    assert np.isclose(mean_fp[0], np.mean(ratio_fp))
    assert np.isclose(mean_fmm[0], np.mean(ratio_fmm))
    assert np.isclose(mean_dij[0], np.mean(ratio_dij))
    assert np.isclose(var_fp[0], np.var(ratio_fp))
    assert np.isclose(var_fmm[0], np.var(ratio_fmm))
    assert np.isclose(var_dij[0], np.var(ratio_dij))


def test_paired_travel_cost_bootstrap_and_adjacent_sd_ci(tmp_path) -> None:
    variances = np.asarray([0.1, 0.5], dtype=float)
    base = np.linspace(1.0, 5.0, 20)
    samples_by_var = {}
    for factor, (metric_key, *_rest) in enumerate(step4.TRAVEL_COST_SERIES, start=1):
        samples_by_var.setdefault(0.1, {})[metric_key] = factor * base
        samples_by_var.setdefault(0.5, {})[metric_key] = 20.0 + 0.5 * factor * base

    warnings: list[str] = []
    summary = step4.bootstrap_travel_cost_uncertainty(
        variances,
        samples_by_var,
        t_ref=1.0,
        dh=1.0,
        bootstrap_resamples=2000,
        seed=17,
        expected_n_mc=base.size,
        warnings=warnings,
    )

    fp = summary["metrics"]["fpt"]
    assert np.isclose(fp["cv"][0], fp["sd"][0] / fp["mean"][0])
    delta = summary["adjacent"]["fpt"][0]
    assert delta["delta_sd_ci_upper"] < 0.0
    assert delta["decrease_distinguishable"] is True
    assert warnings == []

    fig_sd, _ = step4.plot_characteristic_time_std(variances, summary)
    fig_cv, _ = step4.plot_characteristic_time_cv(variances, summary)
    assert len(fig_sd.axes[0].lines) == len(step4.TRAVEL_COST_SERIES)
    assert len(fig_sd.axes[0].collections) == len(step4.TRAVEL_COST_SERIES)
    assert len(fig_cv.axes[0].lines) == len(step4.TRAVEL_COST_SERIES)
    assert len(fig_cv.axes[0].collections) == len(step4.TRAVEL_COST_SERIES)

    csv_path = step4.write_travel_cost_uncertainty_csv(summary, tmp_path / "travel_cost_uncertainty.csv")
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    adjacent_rows = [row for row in rows if row["record_type"] == "adjacent_sd"]
    assert len(adjacent_rows) == len(step4.TRAVEL_COST_SERIES)
    assert all(row["decrease_distinguishable"] == "True" for row in adjacent_rows)

    import matplotlib.pyplot as plt

    plt.close(fig_sd)
    plt.close(fig_cv)


def test_minimal_cost_histogram_uses_first_arrival_series_and_shared_bins() -> None:
    samples = {}
    for idx, (metric_key, *_rest) in enumerate(step4.TRAVEL_COST_SERIES):
        values = np.linspace(1.0 + idx, 2.0 + idx, 20)
        samples[metric_key] = values[::-1] if metric_key == "fpt_fm_asr" else values
    fig, metadata = step4.plot_minimal_cost_histogram(
        samples,
        variance_value=0.5,
        t_ref=2.0,
        dh=0.1,
        bins=12,
    )

    assert metadata["variance"] == 0.5
    assert metadata["bins"] == 12
    assert len(metadata["bin_edges"]) == 13
    assert metadata["series"] == [entry[1] for entry in step4.TRAVEL_COST_SERIES]
    assert np.isclose(metadata["pearson_correlations_vs_pollock"]["FP (PAR²)"]["r"], 1.0, atol=1e-12)
    assert np.isclose(metadata["pearson_correlations_vs_pollock"]["FP (FM-ASR)"]["r"], -1.0)
    assert metadata["pearson_correlations_vs_pollock"]["FP (FM-ASR)"]["n"] == 20
    assert len(fig.axes[0].patches) == len(step4.TRAVEL_COST_SERIES)
    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
        entry[4] for entry in step4.TRAVEL_COST_SERIES
    ]

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_minimal_cost_histogram_cli_defaults_off_and_accepts_variance(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["step4_plot.py", "--config", "unused.yaml"])
    default_args = step4.parse_args()
    assert default_args.minimal_cost_hist is False
    assert default_args.minimal_cost_hist_variance is None

    monkeypatch.setattr(
        sys,
        "argv",
        ["step4_plot.py", "--config", "unused.yaml", "--minimal-cost-hist", "--minimal-cost-hist-variance", "0.5"],
    )
    enabled_args = step4.parse_args()
    assert enabled_args.minimal_cost_hist is True
    assert enabled_args.minimal_cost_hist_variance == ["0.5"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["step4_plot.py", "--config", "unused.yaml", "--minimal-cost-hist", "--minimal-cost-hist-variance", "0.1", "0.5"],
    )
    multiple_args = step4.parse_args()
    assert step4._parse_plot_variances(",".join(multiple_args.minimal_cost_hist_variance)) == [0.1, 0.5]

    monkeypatch.setattr(
        sys,
        "argv",
        ["step4_plot.py", "--config", "unused.yaml", "--minimal-cost-hist", "--minimal-cost-hist-variance", "0.1,0.5"],
    )
    comma_args = step4.parse_args()
    assert step4._parse_plot_variances(",".join(comma_args.minimal_cost_hist_variance)) == [0.1, 0.5]
    assert step4._resolve_requested_plot_variances(np.asarray([0.1, 0.5]), [0.5, 0.1]) == [0.5, 0.1]


def test_step4_optional_minimal_cost_histogram(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step2.run_stage2(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step3.run_stage3(cfg, cache_dir=pipeline_fixture["cache_dir"])

    artifacts = step4.run_stage4(
        cfg,
        cache_dir=pipeline_fixture["cache_dir"],
        minimal_cost_hist=True,
    )
    artifact = next(a for a in artifacts if a.section == "minimal_cost_hist_var0p1")
    assert artifact.path.exists()
    assert artifact.metadata["variance"] == 0.1

    with np.testing.assert_raises_regex(ValueError, "Requested plot variance 0.5 is not available"):
        step4.run_stage4(
            cfg,
            cache_dir=pipeline_fixture["cache_dir"],
            minimal_cost_hist=True,
            minimal_cost_hist_variance=0.5,
        )


def test_step4_generates_stable_and_unique_outputs(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step2.run_stage2(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step3.run_stage3(cfg, cache_dir=pipeline_fixture["cache_dir"])

    artifacts_stable = step4.run_stage4(cfg, cache_dir=pipeline_fixture["cache_dir"], unique_filenames=False)
    assert artifacts_stable
    assert any(a.filename == "run3_times.png" for a in artifacts_stable)
    assert any(a.filename == "run3_frechet.png" for a in artifacts_stable)
    assert any(a.filename == "run3_scatter.png" for a in artifacts_stable)
    assert any(a.filename == "run3_ratio_mean.png" for a in artifacts_stable)
    assert any(a.filename == "run3_ratio_var.png" for a in artifacts_stable)
    assert any(a.filename == "run3_wasserstein_len.png" for a in artifacts_stable)
    assert any(a.filename == "run3_cdf_len_var0p1.png" for a in artifacts_stable)

    artifacts_unique = step4.run_stage4(cfg, cache_dir=pipeline_fixture["cache_dir"], unique_filenames=True)
    assert artifacts_unique
    assert any(a.filename.startswith("run3_times_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_frechet_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_scatter_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_ratio_mean_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_ratio_var_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_wasserstein_len_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_cdf_len_var0p1_") and a.filename.endswith(".png") for a in artifacts_unique)

    manifest_path = pipeline_fixture["output_dir"] / "manifest.json"
    assert manifest_path.exists()


def test_step4_cli_output_dir_override(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step2.run_stage2(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step3.run_stage3(cfg, cache_dir=pipeline_fixture["cache_dir"])

    repo_root = pipeline_fixture["repo_root"]
    cfg_path = pipeline_fixture["cfg_path"]
    cache_dir = pipeline_fixture["cache_dir"]
    override_dir = pipeline_fixture["output_dir"].parent / "step4_cli_override"

    env = os.environ.copy()
    env["PYTHONPATH"] = "./scripts"
    cmd = [
        sys.executable,
        "scripts/step4_plot.py",
        "--config",
        str(cfg_path),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(override_dir),
        "--no-unique-filenames",
        "--plot-variances",
        "0.1",
        "--scatter-mode",
        "mean,all_points",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    assert "Generated" in proc.stdout
    assert (override_dir / "manifest.json").exists()
    assert (override_dir / "run3_cdf_len_var0p1.png").exists()
    assert (override_dir / "run3_scatter.png").exists()
    assert (override_dir / "run3_scatter_var0p1.png").exists()
