from __future__ import annotations

import os
import subprocess
import sys


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
