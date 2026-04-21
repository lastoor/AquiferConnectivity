from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace

import run_all
import step1_index as step1


def test_run_all_dry_run_and_full(pipeline_fixture) -> None:
    repo_root = pipeline_fixture["repo_root"]
    cfg_path = pipeline_fixture["cfg_path"]
    cache_dir = pipeline_fixture["cache_dir"]
    output_dir = pipeline_fixture["output_dir"].parent / "override_outputs"

    env = os.environ.copy()
    env["PYTHONPATH"] = "./scripts"

    dry_cmd = [
        sys.executable,
        "scripts/run_all.py",
        "--config",
        str(cfg_path),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    dry_proc = subprocess.run(dry_cmd, cwd=repo_root, check=False, capture_output=True, text=True, env=env)
    assert dry_proc.returncode == 0
    assert '"sections": [' in dry_proc.stdout
    assert str(output_dir.resolve()) in dry_proc.stdout

    full_cmd = [
        sys.executable,
        "scripts/run_all.py",
        "--config",
        str(cfg_path),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--no-unique-filenames",
    ]
    full_proc = subprocess.run(full_cmd, cwd=repo_root, check=False, capture_output=True, text=True, env=env)
    assert full_proc.returncode == 0
    assert "Generated" in full_proc.stdout
    assert (output_dir / "manifest.json").exists()


def test_run_all_calls_stages_in_order(monkeypatch, pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    calls: list[str] = []

    class _Step2:
        @staticmethod
        def run_stage2(*args, **kwargs):
            calls.append("stage2")
            assert args[0] is cfg
            assert kwargs["cache_dir"] == pipeline_fixture["cache_dir"]
            assert kwargs["skip_std"] is True
            assert kwargs["max_var_values"] == 1
            assert kwargs["max_std_values"] == 2
            assert kwargs["max_sizes_per_value"] == 3
            assert kwargs["max_mc_per_size"] == 4

    class _Step3:
        @staticmethod
        def run_stage3(*args, **kwargs):
            calls.append("stage3")
            assert args[0] is cfg
            assert kwargs["cache_dir"] == pipeline_fixture["cache_dir"]

    class _Step4:
        @staticmethod
        def run_stage4(*args, **kwargs):
            calls.append("stage4")
            assert args[0] is cfg
            assert kwargs["cache_dir"] == pipeline_fixture["cache_dir"]
            assert kwargs["unique_filenames"] is True
            return ["artifact"]

    def _run_stage1(*args, **kwargs):
        calls.append("stage1")
        assert args[0] is cfg
        assert kwargs["cache_dir"] == pipeline_fixture["cache_dir"]
        return pipeline_fixture["cache_dir"] / "stage1_index.pkl"

    monkeypatch.setattr(run_all.step1, "run_stage1", _run_stage1)
    monkeypatch.setitem(sys.modules, "step2_metrics", _Step2)
    monkeypatch.setitem(sys.modules, "step3_stats", _Step3)
    monkeypatch.setitem(sys.modules, "step4_plot", _Step4)

    args = Namespace(
        cache_dir=pipeline_fixture["cache_dir"],
        skip_std=True,
        max_var_values=1,
        max_std_values=2,
        max_sizes_per_value=3,
        max_mc_per_size=4,
        unique_filenames=True,
    )

    artifacts = run_all.run_pipeline(cfg, args)

    assert artifacts == ["artifact"]
    assert calls == ["stage1", "stage2", "stage3", "stage4"]
