from __future__ import annotations

import os
import subprocess
import sys

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
    assert any(a.filename == "run3_ratio_mean.png" for a in artifacts_stable)
    assert any(a.filename == "run3_ratio_var.png" for a in artifacts_stable)
    assert any(a.filename == "run3_wasserstein_len.png" for a in artifacts_stable)

    artifacts_unique = step4.run_stage4(cfg, cache_dir=pipeline_fixture["cache_dir"], unique_filenames=True)
    assert artifacts_unique
    assert any(a.filename.startswith("run3_times_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_ratio_mean_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_ratio_var_") and a.filename.endswith(".png") for a in artifacts_unique)
    assert any(a.filename.startswith("run3_wasserstein_len_") and a.filename.endswith(".png") for a in artifacts_unique)

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
    ]
    proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    assert "Generated" in proc.stdout
    assert (override_dir / "manifest.json").exists()
