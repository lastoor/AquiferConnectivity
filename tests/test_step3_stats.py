from __future__ import annotations

import pickle

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
