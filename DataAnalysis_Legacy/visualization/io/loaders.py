from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visualization.types import PathSet, RunPaths


def load_main_output(npz_path: Path) -> dict[str, Any]:
    try:
        return _load_main_output_impl(npz_path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        return _load_main_output_impl(npz_path, allow_pickle=True)


def _normalize_npz_value(value: np.ndarray) -> Any:
    arr = np.asarray(value)
    if arr.size != 1:
        return arr
    scalar = arr.item()
    return np.nan if scalar is None else scalar


def _load_main_output_impl(npz_path: Path, *, allow_pickle: bool) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=allow_pickle) as data:
        return {k: _normalize_npz_value(data[k]) for k in data.files}


def _safe_load(paths: list[Path], idx: int) -> np.ndarray | None:
    if paths and idx < len(paths):
        return np.load(paths[idx])
    return None


def load_paths(run_paths: RunPaths, mc_idx: int) -> PathSet:
    return PathSet(
        fp=_safe_load(run_paths.fp_files, mc_idx),
        lrp=_safe_load(run_paths.lrp_files, mc_idx),
        fp_dijkstra=_safe_load(run_paths.fp_dijkstra_files, mc_idx),
        fp_fmm=_safe_load(run_paths.fp_fmm_files, mc_idx),
    )


def load_logk_field(run_paths: RunPaths, mc_idx: int) -> np.ndarray | None:
    if run_paths.logk_files and mc_idx < len(run_paths.logk_files):
        return np.load(run_paths.logk_files[mc_idx])
    return None
