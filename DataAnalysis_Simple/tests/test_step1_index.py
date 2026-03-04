from __future__ import annotations

import pickle

import step1_index as step1


def test_step1_index_writes_cache(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    cache_file = step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])

    assert cache_file.exists()
    assert (pipeline_fixture["cache_dir"] / "stage1_index.meta.json").exists()

    with cache_file.open("rb") as f:
        payload = pickle.load(f)

    assert "variance" in payload
    assert "std" in payload
    assert 0.1 in payload["variance"]
    assert 1.0 in payload["std"]
