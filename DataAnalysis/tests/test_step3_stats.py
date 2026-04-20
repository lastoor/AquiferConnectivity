from __future__ import annotations

import pickle

import numpy as np

import step1_index as step1
import step2_metrics as step2
import step3_stats as step3


def test_step3_prepares_plot_inputs(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step2.run_stage2(cfg, cache_dir=pipeline_fixture["cache_dir"])

    cache_file = step3.run_stage3(cfg, cache_dir=pipeline_fixture["cache_dir"])
    assert cache_file.exists()

    with cache_file.open("rb") as f:
        payload = pickle.load(f)

    prepared = payload["plot_inputs"]
    assert "var" in prepared
    assert "stats" in prepared
    assert "taylor" in prepared
    assert "i_y" in prepared
    assert "t_ref" in prepared
    assert "wasserstein_len" in prepared
    assert "length_cdf_samples" in prepared
    assert "scatter_time_samples" in prepared
    assert "derived_warnings" in prepared
    assert isinstance(prepared["derived_warnings"], list)
    for key in ["fp_lrp", "fp_fpFMM", "fp_fpDijkstra"]:
        assert key in prepared["wasserstein_len"]
        series = np.asarray(prepared["wasserstein_len"][key], dtype=float)
        assert series.shape == prepared["var"].shape

    stats = prepared["stats"]
    for key in ["stats_frechet_lrp", "stats_frechet_fmm", "stats_frechet_dijkstra"]:
        assert key in stats
        arr = np.asarray(stats[key], dtype=float)
        assert arr.shape[0] == prepared["var"].shape[0]
        assert arr.shape[1] == 2

    cdf_samples = prepared["length_cdf_samples"]
    assert isinstance(cdf_samples, dict)
    assert cdf_samples
    first_key = sorted(cdf_samples.keys())[0]
    for key in ["fp", "lrp", "fp_fmm", "fp_dijkstra"]:
        assert key in cdf_samples[first_key]
        arr = np.asarray(cdf_samples[first_key][key], dtype=float)
        assert arr.size > 0

    scatter_samples = prepared["scatter_time_samples"]
    assert isinstance(scatter_samples, dict)
    assert scatter_samples
    first_scatter_key = sorted(scatter_samples.keys())[0]
    for key in ["fpt", "fpt_fmm", "fpt_dijkstra", "mhr"]:
        assert key in scatter_samples[first_scatter_key]
        arr = np.asarray(scatter_samples[first_scatter_key][key], dtype=float)
        assert arr.size > 0
