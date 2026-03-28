#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import cmcrameri.cm as cmc
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

import step1_index as step1
import step2_metrics as step2
import step3_stats as step3
import step4_plot as step4


_MC_PATTERN = re.compile(r"(?:_mc|_)(-?\d+)$")

COLORS = {
    "fp": "#000000",
    "fpFMM": "#000000",
    "fpDijkstra": "#000000",
    "lrp": "#00008B",
    "lrpFMM": "#6A3FB6",
}

LINESTYLES = {
    "fp": "-",
    "fpFMM": "--",
    "fpDijkstra": ":",
    "lrp": "-",
    "lrpFMM": "-.",
}

LINEWIDTHS = 2

@dataclass(slots=True)
class SelectedRun:
    sweep: str
    value_key: float
    run_paths: step1.RunPaths
    logk_file: Path
    logk_position: int
    mc_label: int


@dataclass(slots=True)
class Step5Geometry:
    x: np.ndarray
    y: np.ndarray
    source_x: int
    target_x: int
    source_y_start: int
    source_y_end: int
    target_y_start: int
    target_y_end: int

    @property
    def domain_box(self) -> tuple[int, int, int, int]:
        return (
            int(self.source_x),
            int(self.target_x),
            int(min(self.source_y_start, self.target_y_start)),
            int(max(self.source_y_end, self.target_y_end)),
        )

    @property
    def source_rows(self) -> np.ndarray:
        return np.arange(self.source_y_start, self.source_y_end, dtype=int)

    @property
    def target_rows(self) -> np.ndarray:
        return np.arange(self.target_y_start, self.target_y_end, dtype=int)

    @property
    def source_xy(self) -> np.ndarray:
        rows = self.source_rows.astype(float)
        return np.column_stack([np.full(rows.size, self.source_x, dtype=float), rows])

    @property
    def target_xy(self) -> np.ndarray:
        rows = self.target_rows.astype(float)
        return np.column_stack([np.full(rows.size, self.target_x, dtype=float), rows])


@dataclass(slots=True)
class PathSolveResult:
    fp: np.ndarray
    fp_fmm: np.ndarray
    lrp_fmm: np.ndarray
    t_min: np.ndarray
    linearized_fp_paths: list[np.ndarray] | None
    linearized_fp_indices: list[int] | None
    linearized_fp_final: np.ndarray | None
    linearized_fp_history: list[dict[str, float]] | None
    linearized_fp_failed_indices: list[int] | None
    linearized_fp_converged: bool | None
    h: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    source_xy: np.ndarray
    target_xy: np.ndarray


@dataclass(slots=True)
class PathBalanceDiagnostics:
    label: str
    points: np.ndarray
    spline_points: np.ndarray
    s: np.ndarray
    s_norm: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    curvature: np.ndarray
    curvature_raw: np.ndarray
    curvature_spline: np.ndarray
    curvature_circle: np.ndarray
    curvature_method: str
    curvature_spline_mode: str
    dlogk_dn: np.ndarray
    dloggradh_dn: np.ndarray
    dlogv_dn: np.ndarray
    residual: np.ndarray


@dataclass(slots=True)
class LinearizedFpSeries:
    travel_time_fields: list[np.ndarray]
    paths: list[np.ndarray]
    path_indices: list[int]
    history: list[dict[str, float]]
    failed_path_indices: list[int]
    converged: bool


def _validate_index_span(*, field_size: int, start: int, end: int, label: str) -> None:
    if not (0 <= int(start) < int(end) <= int(field_size)):
        raise ValueError(f"Invalid {label} span [{start}, {end}); expected 0 <= start < end <= {field_size}.")


def _default_geometry(field_size: int) -> Step5Geometry:
    source_x = field_size // 20 - 1
    target_x = field_size - field_size // 20 - 1
    source_y_start = field_size // 20 - 1
    source_y_end = field_size - field_size // 20 - 1
    target_y_start = 0
    target_y_end = field_size
    return Step5Geometry(
        x=np.arange(0, field_size, 1.0),
        y=np.arange(0, field_size, 1.0),
        source_x=int(source_x),
        target_x=int(target_x),
        source_y_start=int(source_y_start),
        source_y_end=int(source_y_end),
        target_y_start=int(target_y_start),
        target_y_end=int(target_y_end),
    )


def _resolve_geometry(
    *,
    field_size: int,
    source_x: int | None = None,
    target_x: int | None = None,
    source_y_start: int | None = None,
    source_y_end: int | None = None,
    target_y_start: int | None = None,
    target_y_end: int | None = None,
) -> Step5Geometry:
    geometry = _default_geometry(field_size)
    if source_x is not None:
        geometry.source_x = int(source_x)
    if target_x is not None:
        geometry.target_x = int(target_x)
    if source_y_start is not None:
        geometry.source_y_start = int(source_y_start)
    if source_y_end is not None:
        geometry.source_y_end = int(source_y_end)
    if target_y_start is not None:
        geometry.target_y_start = int(target_y_start)
    if target_y_end is not None:
        geometry.target_y_end = int(target_y_end)

    if not (0 <= geometry.source_x < field_size):
        raise ValueError(f"Invalid source_x={geometry.source_x}; expected 0 <= source_x < {field_size}.")
    if not (0 <= geometry.target_x < field_size):
        raise ValueError(f"Invalid target_x={geometry.target_x}; expected 0 <= target_x < {field_size}.")
    if geometry.target_x <= geometry.source_x:
        raise ValueError(
            f"Invalid source/target ordering: source_x={geometry.source_x}, target_x={geometry.target_x}; expected target_x > source_x."
        )
    _validate_index_span(
        field_size=field_size,
        start=geometry.source_y_start,
        end=geometry.source_y_end,
        label="source_y",
    )
    _validate_index_span(
        field_size=field_size,
        start=geometry.target_y_start,
        end=geometry.target_y_end,
        label="target_y",
    )
    return geometry


def _sweep_token(sweep: str) -> str:
    return "var" if sweep == "variance" else "std"


def _parse_mc_label(path: Path) -> int | None:
    match = _MC_PATTERN.search(path.stem)
    if match:
        return int(match.group(1))
    return None


def _resolve_value_key(discovered: dict[float, list[step1.RunPaths]], value: float) -> float:
    if value in discovered:
        return float(value)
    for key in discovered:
        if abs(float(key) - float(value)) < 1e-12:
            return float(key)
    available = ", ".join(str(k) for k in sorted(discovered))
    raise ValueError(f"Value {value} not found in discovered runs. Available values: [{available}]")


def _select_run_paths_by_size(runs: list[step1.RunPaths], field_size: int) -> step1.RunPaths:
    for rp in runs:
        if int(round(rp.size_value)) == int(field_size):
            return rp
        if rp.size_str == str(field_size):
            return rp
    available = ", ".join(rp.size_str for rp in runs)
    raise ValueError(f"Field size {field_size} not found. Available sizes: [{available}]")


def _select_file_by_mc(paths: list[Path], mc_index: int, kind: str) -> tuple[Path, int, int]:
    if not paths:
        raise ValueError(f"No {kind} files were found for this run.")

    label_matches: list[tuple[int, Path]] = []
    for i, p in enumerate(paths):
        label = _parse_mc_label(p)
        if label is not None and label == mc_index:
            label_matches.append((i, p))
    if label_matches:
        pos, path = label_matches[0]
        return path, pos, mc_index

    if 0 <= mc_index < len(paths):
        path = paths[mc_index]
        label = _parse_mc_label(path)
        return path, mc_index, mc_index if label is None else label

    labels = sorted({_parse_mc_label(p) for p in paths if _parse_mc_label(p) is not None})
    raise ValueError(
        f"Could not resolve mc_index={mc_index} for {kind}. "
        f"Available positional range=[0,{len(paths)-1}], labels={labels}."
    )


def select_run_input(
    cfg: step1.VisualizationConfig,
    *,
    sweep: str,
    value: float,
    field_size: int,
    mc_index: int,
) -> SelectedRun:
    if sweep == "variance":
        discovered = step1.discover_variance_runs(cfg)
    elif sweep == "std":
        discovered = step1.discover_std_runs(cfg)
    else:
        raise ValueError(f"Unsupported sweep={sweep}; expected 'variance' or 'std'.")

    value_key = _resolve_value_key(discovered, value)
    runs = discovered[value_key]
    if not runs:
        raise ValueError(f"No runs found for sweep={sweep}, value={value_key}.")

    run_paths = _select_run_paths_by_size(runs, field_size)
    logk_file, logk_pos, mc_label = _select_file_by_mc(run_paths.logk_files, mc_index, "logK")
    return SelectedRun(
        sweep=sweep,
        value_key=float(value_key),
        run_paths=run_paths,
        logk_file=logk_file,
        logk_position=logk_pos,
        mc_label=mc_label,
    )


def load_selected_logk(selected: SelectedRun) -> np.ndarray:
    # Reuse Stage-2 loading convention first; fallback to direct file read.
    field = step2.load_logk_field(selected.run_paths, selected.logk_position)
    if field is not None:
        return np.asarray(field, dtype=float)
    return np.asarray(np.load(selected.logk_file), dtype=float)


def generate_synthetic_logk(
    *,
    field_size: int,
    amplitude: float,
    frequency_x: float,
    frequency_y: float,
    offset: float,
) -> np.ndarray:
    if field_size <= 1:
        raise ValueError(f"field_size must be at least 2 for synthetic logK generation; got {field_size}.")
    for name, value in {
        "synthetic_amplitude": amplitude,
        "synthetic_frequency_x": frequency_x,
        "synthetic_frequency_y": frequency_y,
        "synthetic_offset": offset,
    }.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite; got {value}.")

    x_norm = np.arange(field_size, dtype=float) / float(field_size)
    y_norm = np.arange(field_size, dtype=float) / float(field_size)
    xx, yy = np.meshgrid(x_norm, y_norm, indexing="xy")

    return np.asarray(
        offset
        + amplitude * np.exp(- (xx ** 2 + yy ** 2) / field_size) * (
            np.sin(2.0 * np.pi * frequency_x * xx)
            * np.cos(2.0 * np.pi * frequency_y * yy)
        ),
        dtype=float,
    )

    # # Sine curve
    # y_sin = np.sin(xx)/2
    # thickness = 0.05  # adjust for thickness
    # matrix = np.abs(yy - y_sin) < thickness
    # matrix = matrix.astype(float)
    # return offset + amplitude * matrix


def _add_search_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _import_solver_modules() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    algo_src = repo_root / "Algorithm" / "src"
    flowsim_path = Path(os.environ.get("FLOWSIM_PATH", str(repo_root.parent / "FlowSimulation"))).expanduser().resolve()

    if not algo_src.is_dir():
        raise RuntimeError(f"Algorithm src path does not exist: {algo_src}")
    if not flowsim_path.is_dir():
        raise RuntimeError(
            f"FlowSimulation path does not exist: {flowsim_path}. "
            "Set FLOWSIM_PATH to your FlowSimulation repository root."
        )

    os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    _add_search_path(algo_src)
    _add_search_path(flowsim_path)

    try:
        from fmm_core.fmm import extract_path_continuous, extract_path_source_to_target, fast_marching_2d_regions
        from simulation import hydraulic as sim_hyd
        from simulation import particle_tracking as sim_pt
        from simulation import path_finding as sim_path
    except Exception as exc:  # pragma: no cover - environment-specific import path
        raise RuntimeError(
            "Failed to import solver dependencies. "
            "Ensure Algorithm/src and FlowSimulation are available and dependencies are installed."
        ) from exc

    return {
        "extract_path_continuous": extract_path_continuous,
        "extract_path_source_to_target": extract_path_source_to_target,
        "fast_marching_2d_regions": fast_marching_2d_regions,
        "sim_hyd": sim_hyd,
        "sim_pt": sim_pt,
        "sim_path": sim_path,
    }


def _import_linear_eikonal_solver() -> Any:
    solver_path = Path(__file__).resolve().parent / "fp_and_lrp" / "linear_eikonal_solver.py"
    if not solver_path.is_file():
        raise RuntimeError(f"Linear Eikonal solver file does not exist: {solver_path}")

    module_name = "step5_linear_eikonal_solver"
    spec = importlib.util.spec_from_file_location(module_name, solver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {solver_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_curvature_module() -> Any:
    module_path = Path(__file__).resolve().parent / "fp_and_lrp" / "curvature.py"
    if not module_path.is_file():
        raise RuntimeError(f"Curvature module file does not exist: {module_path}")

    module_name = "step5_curvature_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_face_coordinates_from_centers(coord_centers: np.ndarray) -> np.ndarray:
    coord_centers = np.asarray(coord_centers, dtype=float).ravel()
    if coord_centers.size < 2:
        raise ValueError("coord_centers must contain at least 2 values")
    faces = np.empty(coord_centers.size + 1, dtype=float)
    faces[1:-1] = 0.5 * (coord_centers[:-1] + coord_centers[1:])
    faces[0] = coord_centers[0] - 0.5 * (coord_centers[1] - coord_centers[0])
    faces[-1] = coord_centers[-1] + 0.5 * (coord_centers[-1] - coord_centers[-2])
    return faces


def _resolve_particle_count(*, num_particles: int | None, density_particle: int, y0: int, y1: int) -> int:
    if num_particles is not None:
        return max(2, int(num_particles))
    return max(2, int(density_particle * abs(y1 - y0)))


def _build_particle_seed_y(*, y_start: int, y_end: int, num_particles: int) -> np.ndarray:
    return np.linspace(float(y_start), float(y_end), int(num_particles), endpoint=False, dtype=np.float32)


def _validate_particle_tracking_options(*, solver: str, velocity_location: str, backend: str = "cpu") -> None:
    if solver not in {"pollock", "rk45"}:
        raise ValueError(f"Unsupported particle-tracking solver: {solver}")
    if velocity_location not in {"face", "cell"}:
        raise ValueError(f"Unsupported velocity_location={velocity_location}; expected 'face' or 'cell'.")
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported backend={backend}; expected 'cpu' or 'gpu'.")
    if solver == "pollock" and velocity_location != "face":
        raise ValueError("Pollock solver requires velocity_location='face'.")
    if solver != "pollock" and backend == "gpu":
        raise ValueError("GPU backend is currently supported only with solver='pollock'.")


def _require_path_valid(sim_path: Any, path: np.ndarray, domain_box: tuple[int, int, int, int], x0: int, x1: int, name: str) -> None:
    if path.size == 0 or not sim_path.is_path_valid(path, domain_box, len_min=abs(x1 - x0)):
        raise RuntimeError(f"Invalid {name} geometry.")


def _maybe_require_path_valid(
    sim_path: Any,
    path: np.ndarray,
    domain_box: tuple[int, int, int, int],
    x0: int,
    x1: int,
    name: str,
    *,
    check_path_geometry: bool,
) -> None:
    if not check_path_geometry:
        return
    _require_path_valid(sim_path, path, domain_box, x0, x1, name)


def _build_source_target_masks(
    *,
    field_shape: tuple[int, int],
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    x0: int,
    x1: int,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_y, n_x = field_shape
    source_mask = np.zeros((n_y, n_x), dtype=bool)
    target_mask = np.zeros((n_y, n_x), dtype=bool)
    source_mask[source_rows, x0] = True
    target_mask[target_rows, x1] = True
    if traversable_mask is None:
        traversable_mask = np.ones((n_y, n_x), dtype=bool)
    else:
        traversable_mask = np.asarray(traversable_mask, dtype=bool)
        if traversable_mask.shape != (n_y, n_x):
            raise ValueError(f"Traversable-mask shape {traversable_mask.shape} does not match field shape {(n_y, n_x)}.")
    source_mask = source_mask & traversable_mask
    target_mask = target_mask & traversable_mask
    return source_mask, target_mask, traversable_mask


def _extract_fmm_path_from_t_field(
    t_field: np.ndarray,
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    extract_path_source_to_target: Any,
    extract_path_continuous: Any,
    backtracking: str,
    continuous_step_size: float,
    traversable_mask: np.ndarray,
) -> np.ndarray:
    if backtracking == "continuous":
        path_ij, _chosen_target = extract_path_continuous(
            t_field,
            source_mask,
            target_mask,
            step_size=continuous_step_size,
            traversable_mask=traversable_mask,
        )
    elif backtracking in {"discrete", "zigzag"}:
        path_ij, _chosen_target = extract_path_source_to_target(
            t_field,
            source_mask,
            target_mask,
            traversable_mask=traversable_mask,
        )
    else:
        raise ValueError(
            f"Unsupported fmm backtracking method {backtracking!r}; expected 'discrete' or 'continuous'."
        )
    if len(path_ij) < 2:
        raise RuntimeError("FMM path extraction failed.")
    return np.asarray(path_ij, dtype=float)[:, [1, 0]][::-1]


def _extract_linearized_path_from_start_cell(
    t_arr: np.ndarray,
    *,
    start_cell: tuple[int, int],
    source_mask: np.ndarray,
    traversable_mask: np.ndarray,
) -> np.ndarray:
    current = tuple(int(v) for v in start_cell)
    path_ij: list[tuple[int, int]] = [current]
    visited = {current}
    ny, nx = t_arr.shape
    max_steps = max(2, 4 * ny * nx)
    decrease_tol = 1e-12
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for _step in range(max_steps):
        if source_mask[current]:
            path_xy = np.asarray(path_ij, dtype=float)[:, [1, 0]][::-1]
            if len(path_xy) < 2:
                raise RuntimeError("Linearized FP path extraction failed: degenerate source-target path.")
            return path_xy

        curr_i, curr_j = current
        curr_t = float(t_arr[curr_i, curr_j])
        best_next: tuple[int, int] | None = None
        best_t = curr_t
        best_drop_rate = -np.inf
        for di, dj in neighbors:
            ni = curr_i + di
            nj = curr_j + dj
            if not (0 <= ni < ny and 0 <= nj < nx):
                continue
            if not traversable_mask[ni, nj]:
                continue
            next_t = float(t_arr[ni, nj])
            if not np.isfinite(next_t):
                continue
            step_dist = float(np.hypot(di, dj))
            drop = curr_t - next_t
            if drop <= decrease_tol:
                continue
            drop_rate = drop / step_dist
            if drop_rate > best_drop_rate or (abs(drop_rate - best_drop_rate) <= decrease_tol and next_t < best_t):
                best_next = (ni, nj)
                best_t = next_t
                best_drop_rate = drop_rate

        if best_next is None:
            raise RuntimeError(
                f"Linearized FP path extraction failed: no descending neighbor from cell {current} with T={curr_t:.6e}."
            )
        if best_next in visited:
            raise RuntimeError(f"Linearized FP path extraction failed: detected descent loop at cell {best_next}.")

        current = best_next
        visited.add(current)
        path_ij.append(current)

    raise RuntimeError("Linearized FP path extraction failed: exceeded maximum descent steps.")


def _extract_linearized_path_from_t_field(
    t_field: np.ndarray,
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    traversable_mask: np.ndarray,
) -> np.ndarray:
    t_arr = np.asarray(t_field, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    traversable_mask = np.asarray(traversable_mask, dtype=bool)
    if t_arr.shape != source_mask.shape or t_arr.shape != target_mask.shape or t_arr.shape != traversable_mask.shape:
        raise ValueError("Linearized path extraction expects matching field and mask shapes.")

    valid_target_mask = target_mask & traversable_mask & np.isfinite(t_arr)
    if not np.any(valid_target_mask):
        raise RuntimeError("Linearized FP path extraction failed: no valid target cells.")

    target_indices = np.argwhere(valid_target_mask)
    target_values = t_arr[valid_target_mask]
    order = np.argsort(target_values, kind="stable")
    failure_messages: list[str] = []
    for target_idx in order.tolist():
        start_cell = tuple(int(v) for v in target_indices[int(target_idx)])
        try:
            return _extract_linearized_path_from_start_cell(
                t_arr,
                start_cell=start_cell,
                source_mask=source_mask,
                traversable_mask=traversable_mask,
            )
        except RuntimeError as exc:
            failure_messages.append(str(exc))
            continue

    first_reason = failure_messages[0] if failure_messages else "unknown failure"
    raise RuntimeError(
        "Linearized FP path extraction failed: "
        f"checked {len(order)} target candidates and none admitted a descending path. "
        f"First failure: {first_reason}"
    )


def _compute_fmm_path(
    field: np.ndarray,
    *,
    geometry: Step5Geometry,
    grid_spacing: float,
    fast_marching_2d_regions: Any,
    extract_path_source_to_target: Any,
    extract_path_continuous: Any,
    backtracking: str = "discrete",
    continuous_step_size: float = 0.5,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if traversable_mask is None:
        traversable_mask = np.isfinite(field) & (field > 0.0)
    source_mask, target_mask, traversable_mask = _build_source_target_masks(
        field_shape=field.shape,
        source_rows=geometry.source_rows,
        target_rows=geometry.target_rows,
        x0=geometry.source_x,
        x1=geometry.target_x,
        traversable_mask=traversable_mask,
    )
    if not np.any(source_mask) or not np.any(target_mask):
        raise RuntimeError("No traversable source/target cells for FMM.")

    t_field, _state = fast_marching_2d_regions(field, source_mask, h=grid_spacing, traversable_mask=traversable_mask)
    path_xy = _extract_fmm_path_from_t_field(
        t_field,
        source_mask=source_mask,
        target_mask=target_mask,
        extract_path_source_to_target=extract_path_source_to_target,
        extract_path_continuous=extract_path_continuous,
        backtracking=backtracking,
        continuous_step_size=continuous_step_size,
        traversable_mask=traversable_mask,
    )
    return path_xy, t_field


def _run_particle_tracking(
    *,
    sim_hyd: Any,
    sim_pt: Any,
    k_field: np.ndarray,
    h_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    geometry: Step5Geometry,
    solver: str,
    backend: str,
    velocity_location: str,
    align_with_modflow_velocity: bool,
    density_particle: int,
    num_particles: int | None,
    eps_speed: float,
    ds_min: float,
    ds_max: float,
    tol: float,
    y_policy: str,
    max_points: int,
    max_iters: int,
    rk45_min_success_frac: float,
    rk45_max_bad_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location, backend=backend)
    n_particles = _resolve_particle_count(
        num_particles=num_particles,
        density_particle=density_particle,
        y0=geometry.source_y_start,
        y1=geometry.source_y_end,
    )
    seeds_x = np.full(n_particles, geometry.source_x, np.float32)
    seeds_y = _build_particle_seed_y(
        y_start=geometry.source_y_start,
        y_end=geometry.source_y_end,
        num_particles=n_particles,
    )

    summary_params: dict[str, Any] = {
        "backend": backend,
        "method": solver,
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "arrival_fraction": 1.0,
        "save_trajectories": False,
        "return_fastest": True,
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": geometry.target_x,
        "max_points": max_points,
        "max_iters": max_iters,
        "threads_per_block": 256,
        "dtype": np.float64,
    }

    if solver == "pollock":
        vx, vy = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h_field,
            output_location="face",
            align_with_modflow=align_with_modflow_velocity,
            face_output="full",
        )
        summary_params.update(
            {
                "x_faces": build_face_coordinates_from_centers(x).astype(np.float32),
                "y_faces": build_face_coordinates_from_centers(y).astype(np.float32),
                "vx": vx,
                "vy": vy,
                "max_steps": max_iters,
            }
        )
    else:
        vx, vy = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h_field,
            output_location=velocity_location,
            align_with_modflow=align_with_modflow_velocity,
        )
        summary_params.update(
            {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "velocity_location": velocity_location,
                "eps_speed": eps_speed,
                "ds_min": ds_min,
                "ds_max": ds_max,
                "tol": tol,
                "y_policy": y_policy,
            }
        )

    summary = sim_pt.particle_tracking_summary(summary_params)
    status_arr = np.asarray(summary["status"], dtype=np.int32)
    count = Counter(status_arr.tolist())
    frac_success = count.get(0, 0) / max(1, n_particles)
    frac_bad = (count.get(1, 0) + count.get(4, 0)) / max(1, n_particles)
    if frac_success < rk45_min_success_frac or frac_bad > rk45_max_bad_frac:
        raise RuntimeError(
            "Invalid velocity field for tracking. "
            f"status={dict(count)}, success_frac={frac_success:.3f}, bad_frac={frac_bad:.3f}"
        )

    result = summary.get("fastest")
    if result is None:
        raise RuntimeError("Particle tracking summary did not return the fastest trajectory.")
    fp = np.vstack((result["x"], result["y"])).T
    return fp, np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)


def _compute_linearized_fp_series(
    *,
    linear_solver: Any,
    slowness_unperturbed: np.ndarray,
    slowness_true: np.ndarray,
    geometry: Step5Geometry,
    fast_marching_2d_regions: Any,
    extract_path_source_to_target: Any,
    extract_path_continuous: Any,
    scheme: str,
    backtracking: str,
    continuous_step_size: float,
    outer_max: int,
    inner_max_sweeps: int,
    inner_tol: float,
    eikonal_tol: float,
    damping: float,
    max_delta_t: float | None,
    monotone_residual: bool,
    verbose: bool,
) -> LinearizedFpSeries:
    traversable_mask = (
        np.isfinite(slowness_unperturbed)
        & np.isfinite(slowness_true)
        & (slowness_unperturbed > 0.0)
        & (slowness_true > 0.0)
    )
    source_mask, target_mask, traversable_mask = _build_source_target_masks(
        field_shape=slowness_unperturbed.shape,
        source_rows=geometry.source_rows,
        target_rows=geometry.target_rows,
        x0=geometry.source_x,
        x1=geometry.target_x,
        traversable_mask=traversable_mask,
    )
    if not np.any(source_mask) or not np.any(target_mask):
        raise RuntimeError("No traversable source/target cells for linearized FP.")

    t_init, _state = fast_marching_2d_regions(
        1/slowness_unperturbed,
        source_mask,
        h=1.0,
        traversable_mask=traversable_mask,
    )
    if scheme == "linearized":
        _t_final, history, iterates = linear_solver.solve_eikonal_by_relinearization(
            T0_fmm=t_init,
            n_true=slowness_true,
            dx=1.0,
            dy=1.0,
            source_mask=source_mask,
            outer_max=outer_max,
            inner_max_sweeps=inner_max_sweeps,
            inner_tol=inner_tol,
            eikonal_tol=eikonal_tol,
            damping=damping,
            max_deltaT=max_delta_t,
            monotone_residual=monotone_residual,
            verbose=verbose,
        )
        extractor = "descent"
        history_out = [{str(k): float(v) for k, v in entry.items()} for entry in history]
        converged = False
        if history_out:
            converged = bool(history_out[-1].get("residual_inf", np.inf) <= eikonal_tol)
    elif scheme == "full_eikonal":
        if outer_max < 1:
            raise ValueError("full_eikonal scheme requires linearized_outer_max >= 1.")
        iterates = [np.asarray(t_init, dtype=float)]
        history_out = []
        for idx in range(1, outer_max + 1):
            lam = float(idx) / float(outer_max)
            slowness_i = (1.0 - lam) * slowness_unperturbed + lam * slowness_true
            t_i, _state_i = fast_marching_2d_regions(
                1/slowness_i,
                source_mask,
                h=1.0,
                traversable_mask=traversable_mask,
            )
            iterates.append(np.asarray(t_i, dtype=float))
            history_out.append({"iter": float(idx), "lambda": float(lam)})
        extractor = "fmm"
        converged = True
        if verbose:
            ignored = [
                "linearized_inner_max_sweeps",
                "linearized_inner_tol",
                "linearized_eikonal_tol",
                "linearized_damping",
                "linearized_max_deltat",
                "linearized_monotone_residual",
            ]
            print(
                "Linearized FP scheme 'full_eikonal' selected; "
                f"ignoring controls: {', '.join(ignored)}"
            )
    else:
        raise ValueError(f"Unsupported linearized scheme {scheme!r}; expected 'linearized' or 'full_eikonal'.")

    paths: list[np.ndarray] = []
    path_indices: list[int] = []
    failed_path_indices: list[int] = []
    for idx, t_field in enumerate(iterates[1:], start=1):
        try:
            if extractor == "descent":
                path_xy = _extract_linearized_path_from_t_field(
                    np.asarray(t_field, dtype=float),
                    source_mask=source_mask,
                    target_mask=target_mask,
                    traversable_mask=traversable_mask,
                )
            else:
                path_xy = _extract_fmm_path_from_t_field(
                    np.asarray(t_field, dtype=float),
                    source_mask=source_mask,
                    target_mask=target_mask,
                    extract_path_source_to_target=extract_path_source_to_target,
                    extract_path_continuous=extract_path_continuous,
                    backtracking=backtracking,
                    continuous_step_size=continuous_step_size,
                    traversable_mask=traversable_mask,
                )
        except RuntimeError as exc:
            if verbose:
                print(f"Linearized FP path extraction failed at iteration {idx}: {exc}")
            failed_path_indices.append(idx)
            continue
        paths.append(path_xy)
        path_indices.append(idx)
    return LinearizedFpSeries(
        travel_time_fields=[np.asarray(t_field, dtype=float) for t_field in iterates],
        paths=paths,
        path_indices=path_indices,
        history=history_out,
        failed_path_indices=failed_path_indices,
        converged=converged,
    )


def solve_single_field(
    logk_field: np.ndarray,
    *,
    field_size: int,
    dh: float,
    solver: str,
    backend: str = "cpu",
    velocity_location: str,
    align_with_modflow_velocity: bool,
    fmm_backtracking: str,
    fmm_continuous_step_size: float,
    linearized_fp: bool,
    linearized_scheme: str,
    linearized_outer_max: int,
    linearized_inner_max_sweeps: int,
    linearized_inner_tol: float,
    linearized_eikonal_tol: float,
    linearized_damping: float,
    linearized_max_deltat: float | None,
    linearized_monotone_residual: bool,
    linearized_verbose: bool,
    check_path_geometry: bool,
    source_x: int | None = None,
    target_x: int | None = None,
    source_y_start: int | None = None,
    source_y_end: int | None = None,
    target_y_start: int | None = None,
    target_y_end: int | None = None,
    density_particle: int,
    num_particles: int | None = None,
    eps_speed: float,
    ds_min: float,
    ds_max: float,
    tol: float,
    y_policy: str,
    max_points: int,
    max_iters: int,
    rk45_min_success_frac: float,
    rk45_max_bad_frac: float,
) -> PathSolveResult:
    modules = _import_solver_modules()
    linear_solver = _import_linear_eikonal_solver() if linearized_fp else None
    extract_path_continuous = modules["extract_path_continuous"]
    sim_hyd = modules["sim_hyd"]
    sim_pt = modules["sim_pt"]
    sim_path = modules["sim_path"]
    fast_marching_2d_regions = modules["fast_marching_2d_regions"]
    extract_path_source_to_target = modules["extract_path_source_to_target"]

    if logk_field.shape != (field_size, field_size):
        raise ValueError(f"logK shape {logk_field.shape} does not match requested field-size={field_size}.")
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location, backend=backend)

    k_field = np.exp(logk_field)
    geometry = _resolve_geometry(
        field_size=field_size,
        source_x=source_x,
        target_x=target_x,
        source_y_start=source_y_start,
        source_y_end=source_y_end,
        target_y_start=target_y_start,
        target_y_end=target_y_end,
    )
    x = geometry.x
    y = geometry.y
    domain_box = geometry.domain_box

    h, msg, is_success = sim_hyd.solve_head(k_field, x, y, dh=dh, rtol=1e-10, atol=1e-20, max_it=10000)
    if not is_success:
        raise RuntimeError(f"Head solver failed: {msg}")

    grad_h_x, grad_h_y = _compute_centered_gradient(h, x, y)
    grad_h_mag = np.hypot(grad_h_x, grad_h_y)
    dh_mean = abs(float(dh))
    if dh_mean <= 0.0 or not np.isfinite(dh_mean):
        raise ValueError(f"Invalid mean hydraulic gradient dh={dh}; expected positive finite magnitude.")
    slowness_unperturbed = 1.0 / np.maximum(k_field * dh_mean, 1e-12)
    slowness_true = 1.0 / np.maximum(k_field * grad_h_mag, 1e-12)
    lrp_fmm, _r_min_field = _compute_fmm_path(
        1/slowness_unperturbed,
        geometry=geometry,
        grid_spacing=1.0,
        fast_marching_2d_regions=fast_marching_2d_regions,
        extract_path_source_to_target=extract_path_source_to_target,
        extract_path_continuous=extract_path_continuous,
        backtracking=fmm_backtracking,
        continuous_step_size=fmm_continuous_step_size,
        traversable_mask=np.isfinite(slowness_unperturbed) & (slowness_unperturbed > 0.0),
    )
    _maybe_require_path_valid(
        sim_path,
        lrp_fmm,
        domain_box,
        geometry.source_x,
        geometry.target_x,
        "LRP (FMM)",
        check_path_geometry=check_path_geometry,
    )

    fp_fmm, t_min = _compute_fmm_path(
        1/slowness_true,
        geometry=geometry,
        grid_spacing=1.0,
        fast_marching_2d_regions=fast_marching_2d_regions,
        extract_path_source_to_target=extract_path_source_to_target,
        extract_path_continuous=extract_path_continuous,
        backtracking=fmm_backtracking,
        continuous_step_size=fmm_continuous_step_size,
        traversable_mask=np.isfinite(slowness_true) & (slowness_true > 0.0),
    )
    _maybe_require_path_valid(
        sim_path,
        fp_fmm,
        domain_box,
        geometry.source_x,
        geometry.target_x,
        "FP (FMM)",
        check_path_geometry=check_path_geometry,
    )

    linearized_series: LinearizedFpSeries | None = None
    if linearized_fp:
        linearized_series = _compute_linearized_fp_series(
            linear_solver=linear_solver,
            slowness_unperturbed=slowness_unperturbed,
            slowness_true=slowness_true,
            geometry=geometry,
            fast_marching_2d_regions=fast_marching_2d_regions,
            extract_path_source_to_target=extract_path_source_to_target,
            extract_path_continuous=extract_path_continuous,
            scheme=linearized_scheme,
            backtracking=fmm_backtracking,
            continuous_step_size=fmm_continuous_step_size,
            outer_max=linearized_outer_max,
            inner_max_sweeps=linearized_inner_max_sweeps,
            inner_tol=linearized_inner_tol,
            eikonal_tol=linearized_eikonal_tol,
            damping=linearized_damping,
            max_delta_t=linearized_max_deltat,
            monotone_residual=linearized_monotone_residual,
            verbose=linearized_verbose,
        )
        for idx, path in enumerate(linearized_series.paths, start=1):
            _maybe_require_path_valid(
                sim_path,
                path,
                domain_box,
                geometry.source_x,
                geometry.target_x,
                f"Linearized FP^{idx}",
                check_path_geometry=check_path_geometry,
            )

    fp, vx_pt, vy_pt = _run_particle_tracking(
        sim_hyd=sim_hyd,
        sim_pt=sim_pt,
        k_field=k_field,
        h_field=h,
        x=x,
        y=y,
        geometry=geometry,
        solver=solver,
        backend=backend,
        velocity_location=velocity_location,
        align_with_modflow_velocity=align_with_modflow_velocity,
        density_particle=density_particle,
        num_particles=num_particles,
        eps_speed=eps_speed,
        ds_min=ds_min,
        ds_max=ds_max,
        tol=tol,
        y_policy=y_policy,
        max_points=max_points,
        max_iters=max_iters,
        rk45_min_success_frac=rk45_min_success_frac,
        rk45_max_bad_frac=rk45_max_bad_frac,
    )
    fp = fp[fp[:, 0] <= geometry.target_x]
    _maybe_require_path_valid(
        sim_path,
        fp,
        domain_box,
        geometry.source_x,
        geometry.target_x,
        "FP",
        check_path_geometry=check_path_geometry,
    )

    return PathSolveResult(
        fp=fp,
        fp_fmm=fp_fmm,
        lrp_fmm=lrp_fmm,
        t_min=t_min,
        linearized_fp_paths=None if linearized_series is None else linearized_series.paths,
        linearized_fp_indices=None if linearized_series is None else linearized_series.path_indices,
        linearized_fp_final=None if linearized_series is None or not linearized_series.paths else linearized_series.paths[-1],
        linearized_fp_history=None if linearized_series is None else linearized_series.history,
        linearized_fp_failed_indices=None if linearized_series is None else linearized_series.failed_path_indices,
        linearized_fp_converged=None if linearized_series is None else linearized_series.converged,
        h=np.asarray(h, dtype=float),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        vx=vx_pt,
        vy=vy_pt,
        source_xy=geometry.source_xy,
        target_xy=geometry.target_xy,
    )


# def _normalized_extent(x: np.ndarray, y: np.ndarray, i_y: float) -> tuple[float, float, float, float]:
#     return (x[0] / i_y, (x[-1] + 1.0) / i_y, y[0] / i_y, (y[-1] + 1.0) / i_y)

def _normalized_extent(x: np.ndarray, y: np.ndarray, i_y: float) -> tuple[float, float, float, float]:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return (
        (x[0] - dx/2) / i_y,
        (x[-1] + dx/2) / i_y,
        (y[0] - dy/2) / i_y,
        (y[-1] + dy/2) / i_y,
    )

def _plot_common_overlays(
    ax: Any,
    result: PathSolveResult,
    i_y: float,
    *,
    fp_diag: PathBalanceDiagnostics | None = None,
    lrp_diag: PathBalanceDiagnostics | None = None,
) -> None:
    lw = LINEWIDTHS
    fp_spline_label = None if fp_diag is None else (
        "FP (FMM, spline interp)" if fp_diag.curvature_spline_mode == "interpolation" else "FP (FMM, spline approx)"
    )
    lrp_spline_label = None if lrp_diag is None else (
        "LRP (FMM, spline interp)" if lrp_diag.curvature_spline_mode == "interpolation" else "LRP (FMM, spline approx)"
    )
    ax.plot(result.source_xy[:, 0] / i_y, result.source_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle='-', label="Source", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.target_xy[:, 0] / i_y, result.target_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle='-', label="Target", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.fp[:, 0] / i_y, result.fp[:, 1] / i_y, color=COLORS["fp"], linestyle=LINESTYLES["fp"], linewidth=lw, label="FP (Particle)", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.fp_fmm[:, 0] / i_y, result.fp_fmm[:, 1] / i_y, color=COLORS["fpFMM"], linestyle=LINESTYLES["fpFMM"], linewidth=lw, label="FP (FMM)", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.lrp_fmm[:, 0] / i_y, result.lrp_fmm[:, 1] / i_y, color=COLORS["lrpFMM"], linestyle=LINESTYLES["lrpFMM"], linewidth=lw, label="LRP (FMM)", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    if fp_diag is not None:
        ax.plot(
            fp_diag.spline_points[:, 0] / i_y,
            fp_diag.spline_points[:, 1] / i_y,
            color=to_rgba(COLORS["fpFMM"], alpha=0.75),
            linestyle="-",
            linewidth=1.0,
            label=fp_spline_label,
        )
    if lrp_diag is not None:
        ax.plot(
            lrp_diag.spline_points[:, 0] / i_y,
            lrp_diag.spline_points[:, 1] / i_y,
            color=to_rgba(COLORS["lrpFMM"], alpha=0.75),
            linestyle="-",
            linewidth=1.0,
            label=lrp_spline_label,
        )
    if result.linearized_fp_final is not None:
        ax.plot(
            result.linearized_fp_final[:, 0] / i_y,
            result.linearized_fp_final[:, 1] / i_y,
            color="#C84C09",
            linestyle="-",
            linewidth=lw,
            label="FP(last, linearized)",
            path_effects=[pe.Stroke(linewidth=1.5 * lw, foreground="white"), pe.Normal()],
        )
    ax.set_title("")
    ax.set_xlabel(r"$x/I_Y$")
    ax.set_ylabel(r"$y/I_Y$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)


def _coerce_velocity_for_streamplot(result: PathSolveResult) -> tuple[np.ndarray, np.ndarray]:
    ny = len(result.y)
    nx = len(result.x)
    vx = np.asarray(result.vx, dtype=float)
    vy = np.asarray(result.vy, dtype=float)

    if vx.shape == (ny, nx) and vy.shape == (ny, nx):
        return vx, vy
    if vx.shape == (ny, nx + 1) and vy.shape == (ny + 1, nx):
        vx_c = 0.5 * (vx[:, :-1] + vx[:, 1:])
        vy_c = 0.5 * (vy[:-1, :] + vy[1:, :])
        return vx_c, vy_c

    m = min(vx.shape[0], vy.shape[0], ny)
    n = min(vx.shape[1], vy.shape[1], nx)
    out_vx = np.zeros((ny, nx), dtype=float)
    out_vy = np.zeros((ny, nx), dtype=float)
    out_vx[:m, :n] = vx[:m, :n]
    out_vy[:m, :n] = vy[:m, :n]
    return out_vx, out_vy


def _dimensionless_velocity(vx: np.ndarray, vy: np.ndarray, q_ref: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(q_ref) or q_ref <= 0.0:
        raise ValueError(f"Invalid reference velocity q_ref={q_ref}; expected a positive finite value.")
    return np.asarray(vx, dtype=float) / q_ref, np.asarray(vy, dtype=float) / q_ref


def _safe_log(field: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    return np.log(np.maximum(arr, floor))


def _smooth_scalar_field(field: np.ndarray, *, method: str, sigma: float) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if method == "none" or sigma <= 0.0:
        return arr
    if method != "gaussian":
        raise ValueError(f"Unsupported curvature field smoothing method {method!r}; expected 'none' or 'gaussian'.")
    return np.asarray(gaussian_filter(arr, sigma=float(sigma), mode="nearest"), dtype=float)


def _compute_centered_gradient(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(np.asarray(field, dtype=float), np.asarray(y, dtype=float), np.asarray(x, dtype=float), edge_order=2)
    return np.asarray(grad_x, dtype=float), np.asarray(grad_y, dtype=float)


def _bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    xg = np.asarray(x, dtype=float)
    yg = np.asarray(y, dtype=float)
    pts = np.asarray(points, dtype=float)
    if values.shape != (yg.size, xg.size):
        raise ValueError(f"Field shape {values.shape} does not match grid ({yg.size}, {xg.size}).")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    px = np.clip(pts[:, 0], xg[0], xg[-1])
    py = np.clip(pts[:, 1], yg[0], yg[-1])

    ix_hi = np.searchsorted(xg, px, side="right")
    iy_hi = np.searchsorted(yg, py, side="right")
    ix_hi = np.clip(ix_hi, 1, xg.size - 1)
    iy_hi = np.clip(iy_hi, 1, yg.size - 1)
    ix_lo = ix_hi - 1
    iy_lo = iy_hi - 1

    x0 = xg[ix_lo]
    x1 = xg[ix_hi]
    y0 = yg[iy_lo]
    y1 = yg[iy_hi]
    tx = np.where(x1 > x0, (px - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 > y0, (py - y0) / (y1 - y0), 0.0)

    f00 = values[iy_lo, ix_lo]
    f10 = values[iy_lo, ix_hi]
    f01 = values[iy_hi, ix_lo]
    f11 = values[iy_hi, ix_hi]
    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def _build_path_balance_diagnostics(
    path: np.ndarray,
    *,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    grad_logk_x: np.ndarray,
    grad_logk_y: np.ndarray,
    grad_loggradh_x: np.ndarray,
    grad_loggradh_y: np.ndarray,
    grad_logv_x: np.ndarray,
    grad_logv_y: np.ndarray,
    curvature_method: str,
    spline_mode: str,
    spline_smooth: float,
    spline_degree: int,
    circle_window: int,
    ds: float = 1.0,
) -> PathBalanceDiagnostics:
    pts = step2.resample_by_arclength(np.asarray(path, dtype=float), ds=ds)
    if pts.shape[0] < 3:
        raise ValueError(f"{label} path must contain at least 3 resampled points to compute curvature.")

    curvature_module = _import_curvature_module()
    geometry = curvature_module.compute_curvature_bundle(
        pts,
        method=curvature_method,
        spline_mode=spline_mode,
        spline_smooth=spline_smooth,
        spline_degree=spline_degree,
        circle_window=circle_window,
    )

    s = np.asarray(geometry.s, dtype=float)
    spline_points = np.asarray(geometry.spline_points, dtype=float)
    tangent = np.asarray(geometry.tangent, dtype=float)
    normal = np.asarray(geometry.normal, dtype=float)
    curvature = np.asarray(geometry.curvature, dtype=float)
    curvature_raw = np.asarray(geometry.curvature_raw, dtype=float)
    curvature_circle = np.asarray(geometry.curvature_circle, dtype=float)
    length = float(s[-1])
    if length <= 0.0:
        raise ValueError(f"{label} path has zero arclength.")

    grad_logk = np.column_stack([
        _bilinear_sample(grad_logk_x, x, y, spline_points),
        _bilinear_sample(grad_logk_y, x, y, spline_points),
    ])
    grad_loggradh = np.column_stack([
        _bilinear_sample(grad_loggradh_x, x, y, spline_points),
        _bilinear_sample(grad_loggradh_y, x, y, spline_points),
    ])
    grad_logv = np.column_stack([
        _bilinear_sample(grad_logv_x, x, y, spline_points),
        _bilinear_sample(grad_logv_y, x, y, spline_points),
    ])

    dlogk_dn = np.einsum("ij,ij->i", grad_logk, normal)
    dloggradh_dn = np.einsum("ij,ij->i", grad_loggradh, normal)
    dlogv_dn = np.einsum("ij,ij->i", grad_logv, normal)
    residual_term = dlogv_dn if label == "FP (FMM)" else dlogk_dn

    return PathBalanceDiagnostics(
        label=label,
        points=spline_points,
        spline_points=spline_points,
        s=s,
        s_norm=s / length,
        tangent=tangent,
        normal=normal,
        curvature=curvature,
        curvature_raw=curvature_raw,
        curvature_spline=curvature,
        curvature_circle=curvature_circle,
        curvature_method=str(geometry.curvature_label),
        curvature_spline_mode=str(geometry.spline_mode),
        dlogk_dn=dlogk_dn,
        dloggradh_dn=dloggradh_dn,
        dlogv_dn=dlogv_dn,
        residual=curvature + residual_term,
    )


def compute_balance_diagnostics(
    *,
    logk_field: np.ndarray,
    result: PathSolveResult,
    curvature_method: str = "spline",
    curvature_spline_mode: str = "interpolation",
    curvature_spline_smooth: float = 0.0,
    curvature_spline_degree: int = 3,
    curvature_field_smooth_method: str = "none",
    curvature_field_smooth_sigma: float = 0.0,
    curvature_circle_window: int = 7,
    ds: float = 1.0,
) -> tuple[PathBalanceDiagnostics, PathBalanceDiagnostics]:
    h = np.asarray(result.h, dtype=float)
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    logk = np.asarray(logk_field, dtype=float)
    logk_smoothed = _smooth_scalar_field(
        logk,
        method=curvature_field_smooth_method,
        sigma=curvature_field_smooth_sigma,
    )
    k_field = np.exp(logk)

    grad_h_x, grad_h_y = _compute_centered_gradient(h, x, y)
    grad_h_mag = np.hypot(grad_h_x, grad_h_y)
    v_mag = k_field * grad_h_mag

    grad_h_mag_smoothed = _smooth_scalar_field(
        grad_h_mag,
        method=curvature_field_smooth_method,
        sigma=curvature_field_smooth_sigma,
    )
    v_mag_smoothed = _smooth_scalar_field(
        v_mag,
        method=curvature_field_smooth_method,
        sigma=curvature_field_smooth_sigma,
    )

    log_grad_h = _safe_log(grad_h_mag_smoothed)
    log_v = _safe_log(v_mag_smoothed)

    grad_logk_x, grad_logk_y = _compute_centered_gradient(logk_smoothed, x, y)
    grad_loggradh_x, grad_loggradh_y = _compute_centered_gradient(log_grad_h, x, y)
    grad_logv_x, grad_logv_y = _compute_centered_gradient(log_v, x, y)

    fp_diag = _build_path_balance_diagnostics(
        result.fp_fmm,
        label="FP (FMM)",
        x=x,
        y=y,
        grad_logk_x=grad_logk_x,
        grad_logk_y=grad_logk_y,
        grad_loggradh_x=grad_loggradh_x,
        grad_loggradh_y=grad_loggradh_y,
        grad_logv_x=grad_logv_x,
        grad_logv_y=grad_logv_y,
        curvature_method=curvature_method,
        spline_mode=curvature_spline_mode,
        spline_smooth=curvature_spline_smooth,
        spline_degree=curvature_spline_degree,
        circle_window=curvature_circle_window,
        ds=ds,
    )
    lrp_diag = _build_path_balance_diagnostics(
        result.lrp_fmm,
        label="LRP (FMM)",
        x=x,
        y=y,
        grad_logk_x=grad_logk_x,
        grad_logk_y=grad_logk_y,
        grad_loggradh_x=grad_loggradh_x,
        grad_loggradh_y=grad_loggradh_y,
        grad_logv_x=grad_logv_x,
        grad_logv_y=grad_logv_y,
        curvature_method=curvature_method,
        spline_mode=curvature_spline_mode,
        spline_smooth=curvature_spline_smooth,
        spline_degree=curvature_spline_degree,
        circle_window=curvature_circle_window,
        ds=ds,
    )
    return fp_diag, lrp_diag


def plot_paths_over_field(
    field: np.ndarray,
    *,
    cmap: str,
    colorbar_label: str,
    result: PathSolveResult,
    i_y: float,
    fp_diag: PathBalanceDiagnostics | None = None,
    lrp_diag: PathBalanceDiagnostics | None = None,
):
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(field, origin="lower", cmap=cmap, interpolation="nearest", extent=_normalized_extent(result.x, result.y, i_y))
    _plot_common_overlays(ax, result, i_y, fp_diag=fp_diag, lrp_diag=lrp_diag)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig


def plot_paths_over_speed_with_streamlines(
    speed_field: np.ndarray,
    vx_field: np.ndarray,
    vy_field: np.ndarray,
    *,
    result: PathSolveResult,
    i_y: float,
    fp_diag: PathBalanceDiagnostics | None = None,
    lrp_diag: PathBalanceDiagnostics | None = None,
    colorbar_label: str = r"$|v|/q_{\mathrm{ref}}$",
):
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(
        speed_field,
        origin="lower",
        cmap=cmc.lajolla,
        interpolation="nearest",
        extent=_normalized_extent(result.x, result.y, i_y),
    )
    x_norm = result.x / i_y
    y_norm = result.y / i_y
    ax.streamplot(
        x_norm,
        y_norm,
        vx_field,
        vy_field,
        color=to_rgba("black", alpha=0.4),
        linewidth=0.8,
        arrowsize=0.6,
    )
    _plot_common_overlays(ax, result, i_y, fp_diag=fp_diag, lrp_diag=lrp_diag)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig


def plot_linearized_fp_progression(
    field: np.ndarray,
    *,
    cmap: str,
    colorbar_label: str,
    result: PathSolveResult,
    i_y: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(field, origin="lower", cmap=cmap, interpolation="nearest", extent=_normalized_extent(result.x, result.y, i_y))
    lw = LINEWIDTHS
    ax.plot(result.source_xy[:, 0] / i_y, result.source_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle="-", label="Source", path_effects=[pe.Stroke(linewidth=1.5 * lw, foreground="white"), pe.Normal()])
    ax.plot(result.target_xy[:, 0] / i_y, result.target_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle="-", label="Target", path_effects=[pe.Stroke(linewidth=1.5 * lw, foreground="white"), pe.Normal()])
    ax.plot(result.lrp_fmm[:, 0] / i_y, result.lrp_fmm[:, 1] / i_y, color=COLORS["lrpFMM"], linestyle=LINESTYLES["lrpFMM"], linewidth=lw, label="LRP (FMM)", path_effects=[pe.Stroke(linewidth=1.5 * lw, foreground="white"), pe.Normal()])
    ax.plot(result.fp_fmm[:, 0] / i_y, result.fp_fmm[:, 1] / i_y, color=COLORS["fpFMM"], linestyle=LINESTYLES["fpFMM"], linewidth=lw, label="FP (FMM)", path_effects=[pe.Stroke(linewidth=1.5 * lw, foreground="white"), pe.Normal()])
    ax.plot(result.fp[:, 0] / i_y, result.fp[:, 1] / i_y, color=COLORS["fp"], linestyle=LINESTYLES["fp"], linewidth=lw, label="FP (Particle)", path_effects=[pe.Stroke(linewidth=1.5 * lw, foreground="white"), pe.Normal()])
    if result.linearized_fp_paths:
        progression_cmap = plt.get_cmap("cividis")
        denom = max(1, len(result.linearized_fp_paths) - 1)
        path_indices = result.linearized_fp_indices or list(range(1, len(result.linearized_fp_paths) + 1))
        for order_idx, (iter_idx, path) in enumerate(zip(path_indices, result.linearized_fp_paths), start=1):
            color = progression_cmap((order_idx - 1) / denom)
            # label = f"FP^{iter_idx}" if order_idx in {1, len(result.linearized_fp_paths)} else None
            label = f"FP^{iter_idx}" if order_idx in range(1, len(result.linearized_fp_paths)+1, len(result.linearized_fp_paths)//10) else None
            ax.plot(path[:, 0] / i_y, path[:, 1] / i_y, color=color, linestyle="-", linewidth=1.8, alpha=0.95, label=label, zorder=5)
    else:
        ax.text(
            0.02,
            0.98,
            r"No linearized FP$^i$ paths extracted",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
        )
    ax.set_xlabel(r"$x/I_Y$")
    ax.set_ylabel(r"$y/I_Y$")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    ax.legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    return fig


def plot_curvature_balance(
    fp_diag: PathBalanceDiagnostics,
    lrp_diag: PathBalanceDiagnostics,
) -> Any:
    fig, axes = plt.subplots(5, 1, figsize=(6, 12), sharex=True)
    lw = 1#LINEWIDTHS
    alpha = 0.5

    axes[0].plot(
        fp_diag.s_norm,
        fp_diag.curvature,
        color=COLORS["fpFMM"],
        linewidth=lw,
        label=rf"$\kappa$ FP (FMM) [{fp_diag.curvature_method}]",
    )
    axes[0].plot(
        fp_diag.s_norm,
        -fp_diag.dlogv_dn,
        color=COLORS["fpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$-\partial_N \ln |v|$ on FP (FMM)",
    )
    axes[0].axhline(0.0, color="0.7", linewidth=1.0)
    axes[0].set_ylabel("FP Spline")
    axes[0].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[1].plot(
        fp_diag.s_norm,
        fp_diag.curvature_circle,
        color=COLORS["fpFMM"],
        linewidth=lw,
        label=r"$\kappa_{\mathrm{circle}}$ FP (FMM)",
    )
    axes[1].plot(
        fp_diag.s_norm,
        -fp_diag.dlogv_dn,
        color=COLORS["fpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$-\partial_N \ln |v|$ on FP (FMM)",
    )
    axes[1].axhline(0.0, color="0.7", linewidth=1.0)
    axes[1].set_ylabel("FP Circle")
    axes[1].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[2].plot(
        lrp_diag.s_norm,
        lrp_diag.curvature,
        color=COLORS["lrpFMM"],
        linewidth=lw,
        label=rf"$\kappa$ LRP (FMM) [{lrp_diag.curvature_method}]",
    )
    axes[2].plot(
        lrp_diag.s_norm,
        -lrp_diag.dlogk_dn,
        color=COLORS["lrpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$-\partial_N \ln K$ on LRP (FMM)",
    )
    axes[2].axhline(0.0, color="0.7", linewidth=1.0)
    axes[2].set_ylabel("LRP Spline")
    axes[2].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[3].plot(
        lrp_diag.s_norm,
        lrp_diag.curvature_circle,
        color=COLORS["lrpFMM"],
        linewidth=lw,
        label=r"$\kappa_{\mathrm{circle}}$ LRP (FMM)",
    )
    axes[3].plot(
        lrp_diag.s_norm,
        -lrp_diag.dlogk_dn,
        color=COLORS["lrpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$-\partial_N \ln K$ on LRP (FMM)",
    )
    axes[3].axhline(0.0, color="0.7", linewidth=1.0)
    axes[3].set_ylabel("LRP Circle")
    axes[3].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[4].plot(
        fp_diag.s_norm,
        fp_diag.residual,
        color=COLORS["fpFMM"],
        linewidth=lw,
        label=r"$R_{\mathrm{FP}}=\kappa + \partial_N \ln |v|$",
    )
    axes[4].plot(
        lrp_diag.s_norm,
        lrp_diag.residual,
        color=COLORS["lrpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$R_{\mathrm{LRP}}=\kappa + \partial_N \ln K$",
    )
    # axes[4].plot(
    #     lrp_diag.s_norm,
    #     lrp_diag.curvature + lrp_diag.dlogv_dn,
    #     color=COLORS["lrpFMM"],
    #     linestyle="--",
    #     linewidth=lw,
    #     label=r"$R_{\mathrm{LRP},v}=\kappa + \partial_N \ln |v|$",
    # )
    axes[4].plot(
        lrp_diag.s_norm,
        lrp_diag.curvature + lrp_diag.dlogk_dn + lrp_diag.dloggradh_dn,
        color=COLORS["lrpFMM"],
        linestyle=":",
        linewidth=lw,
        label=r"$R_{\mathrm{LRP},K+\nabla h}=\kappa + \partial_N \ln K + \partial_N \ln |\nabla h|$",
    )
    axes[4].axhline(0.0, color="0.7", linewidth=1.0)
    axes[4].set_xlabel(r"Normalized arclength $s/L$")
    axes[4].set_ylabel("Residual")
    axes[4].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, color="0.9", linewidth=0.8)

    fig.tight_layout()
    return fig


def _value_slug(value: float) -> str:
    return str(f"{float(value):g}").replace("-", "m").replace(".", "p")


def run_step5(
    cfg: step1.VisualizationConfig,
    *,
    logk_source: str,
    sweep: str | None,
    value: float | None,
    field_size: int,
    mc_index: int | None,
    output_dir: Path | None,
    synthetic_amplitude: float,
    synthetic_frequency_x: float,
    synthetic_frequency_y: float,
    synthetic_offset: float,
    dh: float | None,
    solver: str,
    backend: str = "cpu",
    velocity_location: str,
    align_with_modflow_velocity: bool,
    fmm_backtracking: str,
    fmm_continuous_step_size: float,
    linearized_fp: bool,
    linearized_scheme: str,
    linearized_outer_max: int,
    linearized_inner_max_sweeps: int,
    linearized_inner_tol: float,
    linearized_eikonal_tol: float,
    linearized_damping: float,
    linearized_max_deltat: float | None,
    linearized_monotone_residual: bool,
    linearized_verbose: bool,
    check_path_geometry: bool,
    source_x: int | None = None,
    target_x: int | None = None,
    source_y_start: int | None = None,
    source_y_end: int | None = None,
    target_y_start: int | None = None,
    target_y_end: int | None = None,
    curvature_method: str,
    curvature_spline_mode: str,
    curvature_spline_smooth: float,
    curvature_spline_degree: int,
    curvature_field_smooth_method: str,
    curvature_field_smooth_sigma: float,
    curvature_circle_window: int,
    density_particle: int,
    num_particles: int | None = None,
    eps_speed: float,
    ds_min: float,
    ds_max: float,
    tol: float,
    y_policy: str,
    max_points: int,
    max_iters: int,
    rk45_min_success_frac: float,
    rk45_max_bad_frac: float,
) -> tuple[Path, Path, Path, Path, Path | None]:
    selected: SelectedRun | None = None
    if logk_source == "dataset":
        if sweep is None or value is None or mc_index is None:
            raise ValueError("Dataset logK source requires sweep, value, and mc_index.")
        selected = select_run_input(cfg, sweep=sweep, value=value, field_size=field_size, mc_index=mc_index)
        logk_field = load_selected_logk(selected)
        stem = (
            f"run{cfg.run_id}_{_sweep_token(sweep)}{_value_slug(selected.value_key)}_"
            f"size{field_size}_mc{selected.mc_label}"
        )
    elif logk_source == "synthetic":
        logk_field = generate_synthetic_logk(
            field_size=field_size,
            amplitude=synthetic_amplitude,
            frequency_x=synthetic_frequency_x,
            frequency_y=synthetic_frequency_y,
            offset=synthetic_offset,
        )
        stem = (
            f"synthetic_size{field_size}_"
            f"amp{_value_slug(synthetic_amplitude)}_"
            f"fx{_value_slug(synthetic_frequency_x)}_"
            f"fy{_value_slug(synthetic_frequency_y)}_"
            f"offset{_value_slug(synthetic_offset)}"
        )
    else:
        raise ValueError(f"Unsupported logk_source={logk_source!r}; expected 'dataset' or 'synthetic'.")

    dh_eff = cfg.dh if dh is None else dh
    result = solve_single_field(
        logk_field,
        field_size=field_size,
        dh=dh_eff,
        solver=solver,
        backend=backend,
        velocity_location=velocity_location,
        align_with_modflow_velocity=align_with_modflow_velocity,
        fmm_backtracking=fmm_backtracking,
        fmm_continuous_step_size=fmm_continuous_step_size,
        linearized_fp=linearized_fp,
        linearized_scheme=linearized_scheme,
        linearized_outer_max=linearized_outer_max,
        linearized_inner_max_sweeps=linearized_inner_max_sweeps,
        linearized_inner_tol=linearized_inner_tol,
        linearized_eikonal_tol=linearized_eikonal_tol,
        linearized_damping=linearized_damping,
        linearized_max_deltat=linearized_max_deltat,
        linearized_monotone_residual=linearized_monotone_residual,
        linearized_verbose=linearized_verbose,
        check_path_geometry=check_path_geometry,
        source_x=source_x,
        target_x=target_x,
        source_y_start=source_y_start,
        source_y_end=source_y_end,
        target_y_start=target_y_start,
        target_y_end=target_y_end,
        density_particle=density_particle,
        num_particles=num_particles,
        eps_speed=eps_speed,
        ds_min=ds_min,
        ds_max=ds_max,
        tol=tol,
        y_policy=y_policy,
        max_points=max_points,
        max_iters=max_iters,
        rk45_min_success_frac=rk45_min_success_frac,
        rk45_max_bad_frac=rk45_max_bad_frac,
    )
    if linearized_fp:
        successful_indices = result.linearized_fp_indices or []
        failed_indices = result.linearized_fp_failed_indices or []
        print(f"Linearized FP^i extraction: {len(successful_indices)} succeeded, {len(failed_indices)} failed")
        if successful_indices:
            print(f"Successful FP^i iterations: {successful_indices}")
        if failed_indices:
            print(f"Failed FP^i iterations: {failed_indices}")

    step4.configure_style(cfg)
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    i_y = cfg.length_scale * float(field_size)
    t_ref = step3.compute_t_ref(cfg.k_g, dh_eff, float(field_size), cfg.length_scale)
    q_ref = cfg.k_g * dh_eff
    if not np.isfinite(q_ref) or q_ref <= 0.0:
        raise ValueError(f"Invalid reference velocity q_ref={q_ref}; check k_g and dh.")

    logk_path = out_dir / f"{stem}_logk_paths.png"
    tmin_path = out_dir / f"{stem}_arrival_paths.png"
    stream_path = out_dir / f"{stem}_speed_streamlines.png"
    balance_path = out_dir / f"{stem}_curvature_balance.png"
    progression_path = out_dir / f"{stem}_linearized_fp_progression.png"

    fp_diag, lrp_diag = compute_balance_diagnostics(
        logk_field=logk_field,
        result=result,
        curvature_method=curvature_method,
        curvature_spline_mode=curvature_spline_mode,
        curvature_spline_smooth=curvature_spline_smooth,
        curvature_spline_degree=curvature_spline_degree,
        curvature_field_smooth_method=curvature_field_smooth_method,
        curvature_field_smooth_sigma=curvature_field_smooth_sigma,
        curvature_circle_window=curvature_circle_window,
        ds=1.0,
    )

    fig_logk = plot_paths_over_field(
        logk_field,
        cmap=cmc.lajolla,
        colorbar_label=r"$\ln K$",
        result=result,
        i_y=i_y,
        fp_diag=fp_diag,
        lrp_diag=lrp_diag,
    )
    fig_logk.savefig(logk_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_logk)

    tmin_plot = np.where(np.isfinite(result.t_min), result.t_min / t_ref, np.nan)
    fig_tmin = plot_paths_over_field(
        tmin_plot,
        cmap=cmc.lajolla,
        colorbar_label=r"$T_{\min} U/I_Y$",
        result=result,
        i_y=i_y,
        fp_diag=fp_diag,
        lrp_diag=lrp_diag,
    )
    fig_tmin.savefig(tmin_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_tmin)

    vx_plot, vy_plot = _coerce_velocity_for_streamplot(result)
    vx_star, vy_star = _dimensionless_velocity(vx_plot, vy_plot, q_ref)
    speed_field = np.hypot(vx_star, vy_star)
    fig_stream = plot_paths_over_speed_with_streamlines(
        speed_field,
        vx_star,
        vy_star,
        result=result,
        i_y=i_y,
        fp_diag=fp_diag,
        lrp_diag=lrp_diag,
    )
    fig_stream.savefig(stream_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_stream)

    fig_balance = plot_curvature_balance(fp_diag, lrp_diag)
    fig_balance.savefig(balance_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_balance)
    progression_path_out: Path | None = None
    if linearized_fp:
        fig_progression = plot_linearized_fp_progression(
            tmin_plot,
            cmap=cmc.lajolla,
            colorbar_label=r"$T_{\min} U/I_Y$",
            result=result,
            i_y=i_y,
        )
        fig_progression.savefig(progression_path, dpi=cfg.figure_dpi, bbox_inches="tight")
        plt.close(fig_progression)
        progression_path_out = progression_path
    return logk_path, tmin_path, stream_path, balance_path, progression_path_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5: solve and visualize particle-tracking FP, FP(FMM), LRP(FMM), and optional linearized FP^i for one logK field"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--logk-source", choices=["dataset", "synthetic"], default="dataset")
    parser.add_argument("--sweep", choices=["variance", "std"])
    parser.add_argument("--value", type=float)
    parser.add_argument("--field-size", required=True, type=int)
    parser.add_argument("--mc-index", type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--synthetic-amplitude", type=float, default=1.0)
    parser.add_argument("--synthetic-frequency-x", type=float, default=1.0)
    parser.add_argument("--synthetic-frequency-y", type=float, default=1.0)
    parser.add_argument("--synthetic-offset", type=float, default=0.0)

    parser.add_argument("--dh", type=float, default=None)
    parser.add_argument("--solver", choices=["pollock", "rk45"], default="pollock")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--velocity-location", choices=["face", "cell"], default="face")
    parser.add_argument(
        "--align-with-modflow-velocity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to align face velocities with MODFLOW convention for particle tracking.",
    )
    parser.add_argument("--fmm-backtracking", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--fmm-continuous-step-size", type=float, default=0.5)
    parser.add_argument(
        "--linearized-fp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable iterative linearized-Eikonal FP^i computation from LRP (FMM) toward FP (FMM).",
    )
    parser.add_argument(
        "--linearized-scheme",
        choices=["linearized", "full_eikonal"],
        default="linearized",
        help="Choose the FP^i iteration scheme: current linearized solver or full-Eikonal homotopy solved by FMM each step.",
    )
    parser.add_argument("--linearized-outer-max", type=int, default=20)
    parser.add_argument("--linearized-inner-max-sweeps", type=int, default=200)
    parser.add_argument("--linearized-inner-tol", type=float, default=1e-8)
    parser.add_argument("--linearized-eikonal-tol", type=float, default=1e-6)
    parser.add_argument("--linearized-damping", type=float, default=0.8)
    parser.add_argument("--linearized-max-deltat", type=float, default=None)
    parser.add_argument(
        "--linearized-monotone-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require monotone nonlinear-residual improvement during linearized-Eikonal updates.",
    )
    parser.add_argument(
        "--linearized-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print iterative linearized-Eikonal diagnostics.",
    )
    parser.add_argument(
        "--check-path-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enforce Step 5 path-geometry validation for FP/LRP/FP^i paths.",
    )
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--target-x", type=int, default=None)
    parser.add_argument("--source-y-start", type=int, default=None)
    parser.add_argument("--source-y-end", type=int, default=None)
    parser.add_argument("--target-y-start", type=int, default=None)
    parser.add_argument("--target-y-end", type=int, default=None)
    parser.add_argument("--curvature-method", choices=["spline"], default="spline")
    parser.add_argument("--curvature-spline-mode", choices=["interpolation", "approximation"], default="interpolation")
    parser.add_argument("--curvature-spline-smooth", type=float, default=0.0)
    parser.add_argument("--curvature-spline-degree", type=int, default=3)
    parser.add_argument("--curvature-field-smooth-method", choices=["none", "gaussian"], default="none")
    parser.add_argument("--curvature-field-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--curvature-circle-window", type=int, default=11)

    parser.add_argument("--density-particle", type=int, default=20)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--eps-speed", type=float, default=1e-30)
    parser.add_argument("--ds-min", type=float, default=1e-20)
    parser.add_argument("--ds-max", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--y-policy", choices=["stop", "reflect", "clip"], default="stop")
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--max-iters", type=int, default=1000000)
    parser.add_argument("--rk45-min-success-frac", type=float, default=0.5)
    parser.add_argument("--rk45-max-bad-frac", type=float, default=0.1)
    args = parser.parse_args()
    if args.logk_source == "dataset":
        missing = [
            name
            for name in ("sweep", "value", "mc_index")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "Dataset logK source requires "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
                + "."
            )
    else:
        generate_synthetic_logk(
            field_size=args.field_size,
            amplitude=args.synthetic_amplitude,
            frequency_x=args.synthetic_frequency_x,
            frequency_y=args.synthetic_frequency_y,
            offset=args.synthetic_offset,
        )
    _validate_particle_tracking_options(solver=args.solver, velocity_location=args.velocity_location, backend=args.backend)
    return args


def main() -> int:
    args = parse_args()
    cfg = step1.load_config(args.config)
    logk_path, tmin_path, stream_path, balance_path, progression_path = run_step5(
        cfg,
        logk_source=args.logk_source,
        sweep=args.sweep,
        value=args.value,
        field_size=args.field_size,
        mc_index=args.mc_index,
        output_dir=args.output_dir,
        synthetic_amplitude=args.synthetic_amplitude,
        synthetic_frequency_x=args.synthetic_frequency_x,
        synthetic_frequency_y=args.synthetic_frequency_y,
        synthetic_offset=args.synthetic_offset,
        dh=args.dh,
        solver=args.solver,
        backend=args.backend,
        velocity_location=args.velocity_location,
        align_with_modflow_velocity=args.align_with_modflow_velocity,
        fmm_backtracking=args.fmm_backtracking,
        fmm_continuous_step_size=args.fmm_continuous_step_size,
        linearized_fp=args.linearized_fp,
        linearized_scheme=args.linearized_scheme,
        linearized_outer_max=args.linearized_outer_max,
        linearized_inner_max_sweeps=args.linearized_inner_max_sweeps,
        linearized_inner_tol=args.linearized_inner_tol,
        linearized_eikonal_tol=args.linearized_eikonal_tol,
        linearized_damping=args.linearized_damping,
        linearized_max_deltat=args.linearized_max_deltat,
        linearized_monotone_residual=args.linearized_monotone_residual,
        linearized_verbose=args.linearized_verbose,
        check_path_geometry=args.check_path_geometry,
        source_x=args.source_x,
        target_x=args.target_x,
        source_y_start=args.source_y_start,
        source_y_end=args.source_y_end,
        target_y_start=args.target_y_start,
        target_y_end=args.target_y_end,
        curvature_method=args.curvature_method,
        curvature_spline_mode=args.curvature_spline_mode,
        curvature_spline_smooth=args.curvature_spline_smooth,
        curvature_spline_degree=args.curvature_spline_degree,
        curvature_field_smooth_method=args.curvature_field_smooth_method,
        curvature_field_smooth_sigma=args.curvature_field_smooth_sigma,
        curvature_circle_window=args.curvature_circle_window,
        density_particle=args.density_particle,
        num_particles=args.num_particles,
        eps_speed=args.eps_speed,
        ds_min=args.ds_min,
        ds_max=args.ds_max,
        tol=args.tol,
        y_policy=args.y_policy,
        max_points=args.max_points,
        max_iters=args.max_iters,
        rk45_min_success_frac=args.rk45_min_success_frac,
        rk45_max_bad_frac=args.rk45_max_bad_frac,
    )
    print(f"Saved figure: {logk_path}")
    print(f"Saved figure: {tmin_path}")
    print(f"Saved figure: {stream_path}")
    print(f"Saved figure: {balance_path}")
    if progression_path is not None:
        print(f"Saved figure: {progression_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
