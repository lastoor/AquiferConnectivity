from __future__ import annotations

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
