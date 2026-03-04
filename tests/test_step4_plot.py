from __future__ import annotations

import step1_index as step1
import step2_metrics as step2
import step3_stats as step3
import step4_plot as step4


def test_step4_generates_stable_and_unique_outputs(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step2.run_stage2(cfg, cache_dir=pipeline_fixture["cache_dir"])
    step3.run_stage3(cfg, cache_dir=pipeline_fixture["cache_dir"])

    artifacts_stable = step4.run_stage4(cfg, cache_dir=pipeline_fixture["cache_dir"], unique_filenames=False)
    assert artifacts_stable
    assert any(a.filename == "run3_times.png" for a in artifacts_stable)

    artifacts_unique = step4.run_stage4(cfg, cache_dir=pipeline_fixture["cache_dir"], unique_filenames=True)
    assert artifacts_unique
    assert any(a.filename.startswith("run3_times_") and a.filename.endswith(".png") for a in artifacts_unique)

    manifest_path = pipeline_fixture["output_dir"] / "manifest.json"
    assert manifest_path.exists()
