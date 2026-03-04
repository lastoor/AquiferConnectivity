from __future__ import annotations

import json
import pickle

import step1_index as step1
import step2_metrics as step2


def test_step2_metrics_respects_limits_and_writes_cache(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])

    cache_file = step2.run_stage2(
        cfg,
        cache_dir=pipeline_fixture["cache_dir"],
        max_var_values=1,
        max_std_values=1,
        max_sizes_per_value=1,
        max_mc_per_size=1,
    )

    assert cache_file.exists()

    with cache_file.open("rb") as f:
        payload = pickle.load(f)
    assert "var_results" in payload
    assert "std_results" in payload
    assert "warnings" in payload

    meta = json.loads((pipeline_fixture["cache_dir"] / "stage2_metrics.meta.json").read_text(encoding="utf-8"))
    assert meta["notes"]["max_var_values"] == 1
    assert meta["notes"]["max_std_values"] == 1
    assert meta["notes"]["max_sizes_per_value"] == 1
    assert meta["notes"]["max_mc_per_size"] == 1
