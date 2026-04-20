from __future__ import annotations

from pathlib import Path

from visualization.config import VisualizationConfig
from visualization.types import RunPaths


def _size_sort_key(size_dir: Path) -> int:
    return int(size_dir.name.split("size")[-1])


def _value_dir_candidates(value: float) -> list[str]:
    candidates = [f"{value}", f"{value:g}"]
    rounded = round(value)
    if abs(value - rounded) < 1e-12:
        candidates.extend([str(int(rounded)), f"{int(rounded)}.0"])
    out: list[str] = []
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


def _resolve_value_dir(run_dir: Path, prefix: str, value: float) -> Path:
    for raw in _value_dir_candidates(value):
        candidate = run_dir / f"{prefix}_{raw}"
        if candidate.exists():
            return candidate
    return run_dir / f"{prefix}_{value:g}"


def _collect_run_paths(size_dir: Path, size_str: str) -> RunPaths:
    main_output_files = sorted((size_dir / "main_output").glob(f"mainOutput_size{size_str}_mc*.npz"))
    path_dir = size_dir / "path"
    return RunPaths(
        size_dir=size_dir,
        size_value=float(size_str),
        size_str=size_str,
        main_output_files=main_output_files,
        fp_files=sorted(path_dir.glob(f"fp_size{size_str}_mc*.npy")),
        lrp_files=sorted(path_dir.glob(f"lrp_size{size_str}_mc*.npy")),
        fp_dijkstra_files=sorted(path_dir.glob(f"fp_dijkstra_size{size_str}_mc*.npy")),
        fp_fmm_files=sorted(path_dir.glob(f"fp_fmm_size{size_str}_mc*.npy")),
        logk_files=sorted((size_dir / "logk_field").glob("logKField_*.npy")),
    )


def discover_variance_runs(cfg: VisualizationConfig) -> dict[float, list[RunPaths]]:
    output: dict[float, list[RunPaths]] = {}
    run_dir = cfg.data_root / f"run{cfg.run_id}"
    for var_ in cfg.variance_values:
        base = _resolve_value_dir(run_dir, "var", var_)
        pattern = f"lengthScale{cfg.length_scale}_size*"
        size_dirs = sorted(base.glob(pattern), key=_size_sort_key)
        runs: list[RunPaths] = []
        for size_dir in size_dirs:
            size_str = size_dir.name.split("size")[-1]
            runs.append(_collect_run_paths(size_dir, size_str))
        output[var_] = runs
    return output


def discover_std_runs(cfg: VisualizationConfig) -> dict[float, list[RunPaths]]:
    output: dict[float, list[RunPaths]] = {}
    run_dir = cfg.data_root / f"run{cfg.run_id}"
    for std_ in cfg.std_values:
        base = _resolve_value_dir(run_dir, "std", std_)
        pattern = f"lengthScale{cfg.length_scale}_size*"
        size_dirs = sorted(base.glob(pattern), key=_size_sort_key)
        runs: list[RunPaths] = []
        for size_dir in size_dirs:
            size_str = size_dir.name.split("size")[-1]
            runs.append(_collect_run_paths(size_dir, size_str))
        output[std_] = runs
    return output
