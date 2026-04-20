from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visualization.types import FigureArtifact


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _short_hash(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def save_figure(
    fig,
    *,
    section: str,
    output_dir: Path,
    run_id: int,
    dpi: int,
    save_formats: list[str],
    metadata: dict[str, Any],
    unique_filenames: bool = True,
) -> list[FigureArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp() if unique_filenames else None
    h8 = _short_hash(metadata) if unique_filenames else None
    artifacts: list[FigureArtifact] = []
    for ext in save_formats:
        if unique_filenames:
            filename = f"run{run_id}_{section}_{stamp}_{h8}.{ext}"
        else:
            filename = f"run{run_id}_{section}.{ext}"
        path = output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        artifacts.append(
            FigureArtifact(
                id=f"{section}:{ext}",
                filename=filename,
                path=path,
                section=section,
                metadata=metadata,
            )
        )
    return artifacts


def write_manifest(output_dir: Path, *, config: dict[str, Any], artifacts: list[FigureArtifact], warnings: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "warnings": warnings,
        "figures": [
            {
                "id": a.id,
                "section": a.section,
                "filename": a.filename,
                "path": str(a.path),
                "metadata": a.metadata,
            }
            for a in artifacts
        ],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
