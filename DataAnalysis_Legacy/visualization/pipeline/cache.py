from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from visualization.config import VisualizationConfig
from visualization.types import RunPaths


def default_cache_dir(cfg: VisualizationConfig) -> Path:
    return Path("outputs") / "cache" / f"run{cfg.run_id}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def write_stage_metadata(path: Path, *, stage: str, config: VisualizationConfig, notes: dict) -> None:
    payload = {
        "stage": stage,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "notes": notes,
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def runpaths_to_dict(rp: RunPaths) -> dict:
    return {
        "size_dir": str(rp.size_dir),
        "size_value": rp.size_value,
        "size_str": rp.size_str,
        "main_output_files": [str(p) for p in rp.main_output_files],
        "fp_files": [str(p) for p in rp.fp_files],
        "lrp_files": [str(p) for p in rp.lrp_files],
        "fp_dijkstra_files": [str(p) for p in rp.fp_dijkstra_files],
        "fp_fmm_files": [str(p) for p in rp.fp_fmm_files],
        "logk_files": [str(p) for p in rp.logk_files],
    }


def runpaths_from_dict(raw: dict) -> RunPaths:
    return RunPaths(
        size_dir=Path(raw["size_dir"]),
        size_value=float(raw["size_value"]),
        size_str=str(raw["size_str"]),
        main_output_files=[Path(p) for p in raw["main_output_files"]],
        fp_files=[Path(p) for p in raw["fp_files"]],
        lrp_files=[Path(p) for p in raw["lrp_files"]],
        fp_dijkstra_files=[Path(p) for p in raw["fp_dijkstra_files"]],
        fp_fmm_files=[Path(p) for p in raw["fp_fmm_files"]],
        logk_files=[Path(p) for p in raw["logk_files"]],
    )


def serialize_discovered(discovered: dict[float, list[RunPaths]]) -> dict[float, list[dict]]:
    return {float(k): [runpaths_to_dict(rp) for rp in v] for k, v in discovered.items()}


def deserialize_discovered(raw: dict[float, list[dict]]) -> dict[float, list[RunPaths]]:
    return {float(k): [runpaths_from_dict(rp) for rp in v] for k, v in raw.items()}
