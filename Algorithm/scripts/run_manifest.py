"""Small, shared helpers for reproducibility metadata in simulation folders."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"revision": revision, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "working_tree_dirty": None}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def new_run_manifest(
    *,
    output_dir: Path,
    runner: str,
    command: list[str],
    config: dict[str, Any],
    schema_version: int,
) -> tuple[Path, dict[str, Any]]:
    """Create a unique manifest path and its initial running payload."""
    started_at = datetime.now(timezone.utc)
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"
    repo_root = Path(__file__).resolve().parents[2]
    payload: dict[str, Any] = {
        "manifest_version": 1,
        "run_id": run_id,
        "runner": runner,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "command": command,
        "schema_version": int(schema_version),
        "config": config,
        "provenance": {
            "git": _git_provenance(repo_root),
            "python": platform.python_version(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "scipy", "taichi")
            },
        },
    }
    manifest_path = output_dir / f"run_manifest_{run_id}.json"
    write_run_manifest(manifest_path, payload)
    return manifest_path, payload


def finish_run_manifest(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    status: str,
    started_monotonic: float,
    error: BaseException | None = None,
) -> None:
    payload["status"] = status
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["elapsed_seconds"] = max(0.0, time.monotonic() - started_monotonic)
    if error is not None:
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    write_run_manifest(manifest_path, payload)


def write_run_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace the JSON manifest so interruption cannot truncate it."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, manifest_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
