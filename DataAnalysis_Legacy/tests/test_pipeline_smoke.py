from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_dry_run(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "data_root: /tmp/nonexistent",
                "run_id: 3",
                "length_scale: 0.05",
                "dh: 0.1",
                "field_size_ref: 200",
                "variance_values: [0.1]",
                "std_values: [1.0]",
                f"output_dir: {tmp_path / 'out'}",
                "save_formats: [png]",
                "sections: [times]",
            ]
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "scripts" / "visualize.py"), "--config", str(cfg), "--dry-run"]
    proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    assert proc.returncode == 0
    assert '"sections": [' in proc.stdout
