#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from scipy.signal import savgol_filter

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
class PathSolveResult:
    fp: np.ndarray
    fp_fmm: np.ndarray
    fp_dijkstra: np.ndarray
    lrp: np.ndarray
    lrp_fmm: np.ndarray
    t_min: np.ndarray
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
    s: np.ndarray
    s_norm: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    curvature: np.ndarray
    curvature_raw: np.ndarray
    curvature_savgol: np.ndarray
    curvature_circle: np.ndarray
    curvature_method: str
    dlogk_dn: np.ndarray
    dloggradh_dn: np.ndarray
    dlogv_dn: np.ndarray
    residual: np.ndarray


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
        from fmm_core.speed import prepare_fmm_speed
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
        "prepare_fmm_speed": prepare_fmm_speed,
        "sim_hyd": sim_hyd,
        "sim_pt": sim_pt,
        "sim_path": sim_path,
    }


def _build_domain(field_size: int) -> dict[str, Any]:
    x0 = field_size // 20 - 1
    x1 = field_size - field_size // 20 - 1
    y0 = field_size // 20 - 1
    y1 = field_size - field_size // 20 - 1
    return {
        "x": np.arange(0, field_size, 1.0),
        "y": np.arange(0, field_size, 1.0),
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "domain_box": (x0, x1, y0, y1),
        "source": [(yy, x0) for yy in range(y0, y1)],
        "target": [(yy, x1) for yy in range(0, field_size - 1)],
    }


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


def _compute_fmm_path(
    field: np.ndarray,
    *,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    x0: int,
    x1: int,
    grid_spacing: float,
    fast_marching_2d_regions: Any,
    extract_path_source_to_target: Any,
    extract_path_continuous: Any,
    backtracking: str = "discrete",
    continuous_step_size: float = 0.5,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_y, n_x = field.shape
    source_mask = np.zeros((n_y, n_x), dtype=bool)
    target_mask = np.zeros((n_y, n_x), dtype=bool)
    source_mask[source_rows, x0] = True
    target_mask[target_rows, x1] = True
    if traversable_mask is None:
        traversable_mask = np.isfinite(field) & (field > 0.0)
    source_mask = source_mask & traversable_mask
    target_mask = target_mask & traversable_mask
    if not np.any(source_mask) or not np.any(target_mask):
        raise RuntimeError("No traversable source/target cells for FMM.")

    t_field, _state = fast_marching_2d_regions(field, source_mask, h=grid_spacing, traversable_mask=traversable_mask)
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
    path_xy = np.asarray(path_ij, dtype=float)[:, [1, 0]][::-1]
    return path_xy, t_field


def _run_particle_tracking(
    *,
    sim_hyd: Any,
    sim_pt: Any,
    k_field: np.ndarray,
    h_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
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
        y0=y0,
        y1=y1,
    )
    seeds_x = np.full(n_particles, x0, np.float32)
    seeds_y = np.linspace(y0, y1, n_particles, endpoint=True).astype(np.float32)

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
        "right_x": x1,
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


def solve_single_field(
    logk_field: np.ndarray,
    *,
    field_size: int,
    dh: float,
    solver: str,
    backend: str = "cpu",
    velocity_location: str,
    align_with_modflow_velocity: bool,
    fmm_velocity_layout: str,
    fmm_face_mapping: str,
    fmm_zero_policy: str,
    fmm_min_speed: float,
    fmm_backtracking: str,
    fmm_continuous_step_size: float,
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
    extract_path_continuous = modules["extract_path_continuous"]
    sim_hyd = modules["sim_hyd"]
    sim_pt = modules["sim_pt"]
    sim_path = modules["sim_path"]
    prepare_fmm_speed = modules["prepare_fmm_speed"]
    fast_marching_2d_regions = modules["fast_marching_2d_regions"]
    extract_path_source_to_target = modules["extract_path_source_to_target"]

    if logk_field.shape != (field_size, field_size):
        raise ValueError(f"logK shape {logk_field.shape} does not match requested field-size={field_size}.")
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location, backend=backend)

    k_field = np.exp(logk_field)
    domain = _build_domain(field_size)
    x = domain["x"]
    y = domain["y"]
    x0 = int(domain["x0"])
    x1 = int(domain["x1"])
    y0 = int(domain["y0"])
    y1 = int(domain["y1"])
    source = domain["source"]
    target = domain["target"]
    domain_box = domain["domain_box"]
    source_rows = np.arange(y0, y1, dtype=int)
    target_rows = np.arange(0, field_size - 1, dtype=int)

    lrp, _total_r, _s_node, _e_node, _rmap = sim_path.dijkstra_2d_least_resistance(k_field, source, target, average="harmonic")
    _require_path_valid(sim_path, lrp, domain_box, x0, x1, "LRP")
    lrp_fmm, _r_min_field = _compute_fmm_path(
        k_field,
        source_rows=source_rows,
        target_rows=target_rows,
        x0=x0,
        x1=x1,
        grid_spacing=1.0,
        fast_marching_2d_regions=fast_marching_2d_regions,
        extract_path_source_to_target=extract_path_source_to_target,
        extract_path_continuous=extract_path_continuous,
        backtracking=fmm_backtracking,
        continuous_step_size=fmm_continuous_step_size,
    )
    _require_path_valid(sim_path, lrp_fmm, domain_box, x0, x1, "LRP (FMM)")

    h, msg, is_success = sim_hyd.solve_head(k_field, x, y, dh=dh, rtol=1e-10, atol=1e-20, max_it=10000)
    if not is_success:
        raise RuntimeError(f"Head solver failed: {msg}")

    if fmm_velocity_layout == "face":
        vx_fmm, vy_fmm = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h,
            output_location="face",
            face_output="full",
        )
    else:
        vx_fmm, vy_fmm = sim_hyd.compute_velocity(k_field, x, y, h, output_location="cell")
    speed_fmm, traversable_mask, _speed_diag = prepare_fmm_speed(
        vx=vx_fmm,
        vy=vy_fmm,
        layout=fmm_velocity_layout,
        face_mapping=fmm_face_mapping,
        zero_policy=fmm_zero_policy,
        min_speed=fmm_min_speed,
    )

    fp_dijkstra, _total_t_dij, _sd, _ed, _rmap_v = sim_path.dijkstra_2d_least_resistance(speed_fmm, source, target, average="arithmetic")
    _require_path_valid(sim_path, fp_dijkstra, domain_box, x0, x1, "FP (Dijkstra)")

    fp_fmm, t_min = _compute_fmm_path(
        speed_fmm,
        source_rows=source_rows,
        target_rows=target_rows,
        x0=x0,
        x1=x1,
        grid_spacing=1.0,
        fast_marching_2d_regions=fast_marching_2d_regions,
        extract_path_source_to_target=extract_path_source_to_target,
        extract_path_continuous=extract_path_continuous,
        backtracking=fmm_backtracking,
        continuous_step_size=fmm_continuous_step_size,
        traversable_mask=traversable_mask,
    )
    _require_path_valid(sim_path, fp_fmm, domain_box, x0, x1, "FP (FMM)")

    fp, vx_pt, vy_pt = _run_particle_tracking(
        sim_hyd=sim_hyd,
        sim_pt=sim_pt,
        k_field=k_field,
        h_field=h,
        x=x,
        y=y,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
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
    fp = fp[fp[:, 0] <= x1]
    _require_path_valid(sim_path, fp, domain_box, x0, x1, "FP")

    source_xy = np.column_stack([np.full(y1 - y0, x0, dtype=float), np.arange(y0, y1, dtype=float)])
    target_xy = np.column_stack([np.full(field_size - 1, x1, dtype=float), np.arange(0, field_size - 1, dtype=float)])

    return PathSolveResult(
        fp=fp,
        fp_fmm=fp_fmm,
        fp_dijkstra=fp_dijkstra,
        lrp=lrp,
        lrp_fmm=lrp_fmm,
        t_min=t_min,
        h=np.asarray(h, dtype=float),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        vx=vx_pt,
        vy=vy_pt,
        source_xy=source_xy,
        target_xy=target_xy,
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

def _plot_common_overlays(ax: Any, result: PathSolveResult, i_y: float) -> None:
    lw = LINEWIDTHS
    ax.plot(result.source_xy[:, 0] / i_y, result.source_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle='-', label="Source", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.target_xy[:, 0] / i_y, result.target_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle='-', label="Target", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.fp[:, 0] / i_y, result.fp[:, 1] / i_y, color=COLORS["fp"], linestyle=LINESTYLES["fp"], linewidth=lw, label="FP", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.fp_fmm[:, 0] / i_y, result.fp_fmm[:, 1] / i_y, color=COLORS["fpFMM"], linestyle=LINESTYLES["fpFMM"], linewidth=lw, label="FP (FMM)", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(
        result.fp_dijkstra[:, 0] / i_y,
        result.fp_dijkstra[:, 1] / i_y,
        color=COLORS["fpDijkstra"],
        linestyle=LINESTYLES["fpDijkstra"],
        linewidth=lw,
        label="FP (Dijkstra)",
        path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()],
    )
    ax.plot(result.lrp[:, 0] / i_y, result.lrp[:, 1] / i_y, color=COLORS["lrp"], linestyle=LINESTYLES["lrp"], linewidth=lw, label="LRP", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.lrp_fmm[:, 0] / i_y, result.lrp_fmm[:, 1] / i_y, color=COLORS["lrpFMM"], linestyle=LINESTYLES["lrpFMM"], linewidth=lw, label="LRP (FMM)", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
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


def _normalize_tangent(tangent: np.ndarray) -> np.ndarray:
    tan = np.asarray(tangent, dtype=float)
    norms = np.linalg.norm(tan, axis=1)
    return tan / np.maximum(norms[:, None], 1e-12)


def _coerce_odd_window(window: int, n_points: int, minimum: int) -> int:
    max_size = n_points if n_points % 2 == 1 else n_points - 1
    if max_size < 1:
        return 0
    size = max(int(window), 1)
    if size % 2 == 0:
        size += 1
    size = min(size, max_size)
    if size % 2 == 0:
        size -= 1
    if size < 1:
        return 0
    return size


def _compute_raw_geometry(points: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.gradient(points[:, 0], s, edge_order=2)
    ys = np.gradient(points[:, 1], s, edge_order=2)
    xss = np.gradient(xs, s, edge_order=2)
    yss = np.gradient(ys, s, edge_order=2)
    tangent = _normalize_tangent(np.column_stack([xs, ys]))
    curvature = tangent[:, 0] * yss - tangent[:, 1] * xss
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal, curvature


def _compute_savgol_geometry(
    points: np.ndarray,
    *,
    ds: float,
    window: int,
    poly: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = points.shape[0]
    win = _coerce_odd_window(window, n_points, minimum=poly + 2)
    if win <= poly or win < 3 or n_points <= poly:
        return _compute_raw_geometry(points, np.arange(n_points, dtype=float) * ds)

    x = np.asarray(points[:, 0], dtype=float)
    y = np.asarray(points[:, 1], dtype=float)
    dx = savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=ds, mode="interp")
    dy = savgol_filter(y, window_length=win, polyorder=poly, deriv=1, delta=ds, mode="interp")
    ddx = savgol_filter(x, window_length=win, polyorder=poly, deriv=2, delta=ds, mode="interp")
    ddy = savgol_filter(y, window_length=win, polyorder=poly, deriv=2, delta=ds, mode="interp")
    tangent = _normalize_tangent(np.column_stack([dx, dy]))
    curvature = (dx * ddy - dy * ddx) / np.maximum(dx * dx + dy * dy, 1e-12) ** 1.5
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal, np.asarray(curvature, dtype=float)


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones(pts.shape[0], dtype=float)])
    b = x * x + y * y
    coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, d = coeffs
    radius_sq = cx * cx + cy * cy + d
    if not np.isfinite(radius_sq) or radius_sq <= 0.0:
        raise ValueError("Degenerate circle fit")
    return float(cx), float(cy), float(np.sqrt(radius_sq))


def _compute_circle_curvature(points: np.ndarray, window: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n_points = pts.shape[0]
    win = _coerce_odd_window(window, n_points, minimum=3)
    if win < 3:
        return np.full(n_points, np.nan, dtype=float)

    half = win // 2
    curvature = np.full(n_points, np.nan, dtype=float)
    for idx in range(half, n_points - half):
        local_pts = pts[idx - half : idx + half + 1]
        try:
            _cx, _cy, radius = _fit_circle(local_pts)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(radius) or radius <= 1e-12:
            continue
        prev_vec = pts[idx] - pts[idx - 1]
        next_vec = pts[idx + 1] - pts[idx]
        turn = prev_vec[0] * next_vec[1] - prev_vec[1] * next_vec[0]
        sign = 1.0 if turn >= 0.0 else -1.0
        curvature[idx] = sign / radius
    return curvature


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
    savgol_window: int,
    savgol_poly: int,
    circle_window: int,
    ds: float = 1.0,
) -> PathBalanceDiagnostics:
    pts = step2.resample_by_arclength(np.asarray(path, dtype=float), ds=ds)
    if pts.shape[0] < 3:
        raise ValueError(f"{label} path must contain at least 3 resampled points to compute curvature.")

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    length = float(s[-1])
    if length <= 0.0:
        raise ValueError(f"{label} path has zero arclength.")

    tangent_raw, _normal_raw, curvature_raw = _compute_raw_geometry(pts, s)
    tangent, normal, curvature_savgol = _compute_savgol_geometry(
        pts,
        ds=ds,
        window=savgol_window,
        poly=savgol_poly,
    )
    curvature_circle = _compute_circle_curvature(pts, window=circle_window)
    if curvature_method != "savgol":
        raise ValueError(f"Unsupported curvature method {curvature_method!r}; expected 'savgol'.")
    curvature = curvature_savgol
    if not np.all(np.isfinite(curvature)):
        curvature = np.where(np.isfinite(curvature), curvature, curvature_raw)
        tangent = np.where(np.isfinite(curvature_savgol)[:, None], tangent, tangent_raw)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    grad_logk = np.column_stack([
        _bilinear_sample(grad_logk_x, x, y, pts),
        _bilinear_sample(grad_logk_y, x, y, pts),
    ])
    grad_loggradh = np.column_stack([
        _bilinear_sample(grad_loggradh_x, x, y, pts),
        _bilinear_sample(grad_loggradh_y, x, y, pts),
    ])
    grad_logv = np.column_stack([
        _bilinear_sample(grad_logv_x, x, y, pts),
        _bilinear_sample(grad_logv_y, x, y, pts),
    ])

    dlogk_dn = np.einsum("ij,ij->i", grad_logk, normal)
    dloggradh_dn = np.einsum("ij,ij->i", grad_loggradh, normal)
    dlogv_dn = np.einsum("ij,ij->i", grad_logv, normal)
    residual = curvature + dlogv_dn

    return PathBalanceDiagnostics(
        label=label,
        points=pts,
        s=s,
        s_norm=s / length,
        tangent=tangent,
        normal=normal,
        curvature=curvature,
        curvature_raw=curvature_raw,
        curvature_savgol=curvature_savgol,
        curvature_circle=curvature_circle,
        curvature_method=curvature_method,
        dlogk_dn=dlogk_dn,
        dloggradh_dn=dloggradh_dn,
        dlogv_dn=dlogv_dn,
        residual=residual,
    )


def compute_balance_diagnostics(
    *,
    logk_field: np.ndarray,
    result: PathSolveResult,
    curvature_method: str = "savgol",
    curvature_savgol_window: int = 9,
    curvature_savgol_poly: int = 3,
    curvature_circle_window: int = 7,
    ds: float = 1.0,
) -> tuple[PathBalanceDiagnostics, PathBalanceDiagnostics]:
    h = np.asarray(result.h, dtype=float)
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    logk = np.asarray(logk_field, dtype=float)
    k_field = np.exp(logk)

    grad_h_x, grad_h_y = _compute_centered_gradient(h, x, y)
    grad_h_mag = np.hypot(grad_h_x, grad_h_y)
    v_mag = k_field * grad_h_mag

    log_grad_h = _safe_log(grad_h_mag)
    log_v = _safe_log(v_mag)

    grad_logk_x, grad_logk_y = _compute_centered_gradient(logk, x, y)
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
        savgol_window=curvature_savgol_window,
        savgol_poly=curvature_savgol_poly,
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
        savgol_window=curvature_savgol_window,
        savgol_poly=curvature_savgol_poly,
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
):
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(field, origin="lower", cmap=cmap, interpolation="nearest", extent=_normalized_extent(result.x, result.y, i_y))
    _plot_common_overlays(ax, result, i_y)
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
    colorbar_label: str = r"$|v|/q_{\mathrm{ref}}$",
):
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(
        speed_field,
        origin="lower",
        cmap=cmc.lajolla.reversed(),
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
    _plot_common_overlays(ax, result, i_y)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig


def plot_curvature_balance(
    fp_diag: PathBalanceDiagnostics,
    lrp_diag: PathBalanceDiagnostics,
) -> Any:
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    lw = LINEWIDTHS

    axes[0].plot(
        fp_diag.s_norm,
        fp_diag.curvature,
        color=COLORS["fpFMM"],
        linestyle=LINESTYLES["fpFMM"],
        linewidth=lw,
        label=rf"$\kappa$ FP (FMM) [{fp_diag.curvature_method}]",
    )
    axes[0].plot(
        fp_diag.s_norm,
        -fp_diag.dlogv_dn,
        color=COLORS["fp"],
        linestyle=LINESTYLES["fp"],
        linewidth=lw,
        label=r"$-\partial_N \ln |v|$ on FP (FMM)",
    )
    axes[0].plot(
        lrp_diag.s_norm,
        lrp_diag.curvature,
        color=COLORS["lrpFMM"],
        linestyle=LINESTYLES["lrpFMM"],
        linewidth=lw,
        label=rf"$\kappa$ LRP (FMM) [{lrp_diag.curvature_method}]",
    )
    axes[0].plot(
        lrp_diag.s_norm,
        -lrp_diag.dlogk_dn,
        color=COLORS["lrp"],
        linestyle=LINESTYLES["lrp"],
        linewidth=lw,
        label=r"$-\partial_N \ln K$ on LRP (FMM)",
    )
    axes[0].axhline(0.0, color="0.7", linewidth=1.0)
    axes[0].set_ylabel("Balance")
    axes[0].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[1].plot(
        fp_diag.s_norm,
        fp_diag.curvature_circle,
        color=COLORS["fpFMM"],
        linestyle=":",
        linewidth=lw,
        label=r"$\kappa_{\mathrm{circle}}$ FP (FMM)",
    )
    axes[1].plot(
        fp_diag.s_norm,
        -fp_diag.dlogv_dn,
        color=COLORS["fp"],
        linestyle=LINESTYLES["fp"],
        linewidth=lw,
        label=r"$-\partial_N \ln |v|$ on FP (FMM)",
    )
    axes[1].plot(
        lrp_diag.s_norm,
        lrp_diag.curvature_circle,
        color=COLORS["lrpFMM"],
        linestyle=":",
        linewidth=lw,
        label=r"$\kappa_{\mathrm{circle}}$ LRP (FMM)",
    )
    axes[1].plot(
        lrp_diag.s_norm,
        -lrp_diag.dlogk_dn,
        color=COLORS["lrp"],
        linestyle=LINESTYLES["lrp"],
        linewidth=lw,
        label=r"$-\partial_N \ln K$ on LRP (FMM)",
    )
    axes[1].axhline(0.0, color="0.7", linewidth=1.0)
    axes[1].set_ylabel("Circle Fit")
    axes[1].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[2].plot(
        fp_diag.s_norm,
        fp_diag.residual,
        color=COLORS["fp"],
        linestyle=LINESTYLES["fp"],
        linewidth=lw,
        label=r"$R_{\mathrm{FP}}=\kappa+\partial_N \ln |v|$",
    )
    axes[2].plot(
        lrp_diag.s_norm,
        lrp_diag.residual,
        color=COLORS["lrpFMM"],
        linestyle=LINESTYLES["lrpFMM"],
        linewidth=lw,
        label=r"$R_{\mathrm{LRP}}=\kappa+\partial_N \ln |v|$",
    )
    axes[2].axhline(0.0, color="0.7", linewidth=1.0)
    axes[2].set_xlabel(r"Normalized arclength $s/L$")
    axes[2].set_ylabel("Residual")
    axes[2].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

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
    sweep: str,
    value: float,
    field_size: int,
    mc_index: int,
    output_dir: Path | None,
    dh: float | None,
    solver: str,
    backend: str = "cpu",
    velocity_location: str,
    align_with_modflow_velocity: bool,
    fmm_velocity_layout: str,
    fmm_face_mapping: str,
    fmm_zero_policy: str,
    fmm_min_speed: float,
    fmm_backtracking: str,
    fmm_continuous_step_size: float,
    curvature_method: str,
    curvature_savgol_window: int,
    curvature_savgol_poly: int,
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
) -> tuple[Path, Path, Path, Path]:
    selected = select_run_input(cfg, sweep=sweep, value=value, field_size=field_size, mc_index=mc_index)
    logk_field = load_selected_logk(selected)

    dh_eff = cfg.dh if dh is None else dh
    result = solve_single_field(
        logk_field,
        field_size=field_size,
        dh=dh_eff,
        solver=solver,
        backend=backend,
        velocity_location=velocity_location,
        align_with_modflow_velocity=align_with_modflow_velocity,
        fmm_velocity_layout=fmm_velocity_layout,
        fmm_face_mapping=fmm_face_mapping,
        fmm_zero_policy=fmm_zero_policy,
        fmm_min_speed=fmm_min_speed,
        fmm_backtracking=fmm_backtracking,
        fmm_continuous_step_size=fmm_continuous_step_size,
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

    step4.configure_style(cfg)
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"run{cfg.run_id}_{_sweep_token(sweep)}{_value_slug(selected.value_key)}_"
        f"size{field_size}_mc{selected.mc_label}"
    )
    i_y = cfg.length_scale * float(field_size)
    t_ref = step3.compute_t_ref(cfg.k_g, dh_eff, float(field_size), cfg.length_scale)
    q_ref = cfg.k_g * dh_eff
    if not np.isfinite(q_ref) or q_ref <= 0.0:
        raise ValueError(f"Invalid reference velocity q_ref={q_ref}; check k_g and dh.")

    logk_path = out_dir / f"{stem}_logk_paths.png"
    tmin_path = out_dir / f"{stem}_arrival_paths.png"
    stream_path = out_dir / f"{stem}_speed_streamlines.png"
    balance_path = out_dir / f"{stem}_curvature_balance.png"

    fig_logk = plot_paths_over_field(
        logk_field,
        cmap=cmc.lajolla.reversed(),
        colorbar_label=r"$\ln K$",
        result=result,
        i_y=i_y,
    )
    fig_logk.savefig(logk_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_logk)

    tmin_plot = np.where(np.isfinite(result.t_min), result.t_min / t_ref, np.nan)
    fig_tmin = plot_paths_over_field(
        tmin_plot,
        cmap=cmc.lajolla.reversed(),
        colorbar_label=r"$T_{\min} U/I_Y$",
        result=result,
        i_y=i_y,
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
    )
    fig_stream.savefig(stream_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_stream)

    fp_diag, lrp_diag = compute_balance_diagnostics(
        logk_field=logk_field,
        result=result,
        curvature_method=curvature_method,
        curvature_savgol_window=curvature_savgol_window,
        curvature_savgol_poly=curvature_savgol_poly,
        curvature_circle_window=curvature_circle_window,
        ds=1.0,
    )
    fig_balance = plot_curvature_balance(fp_diag, lrp_diag)
    fig_balance.savefig(balance_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_balance)
    return logk_path, tmin_path, stream_path, balance_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5: solve and visualize FP / FP(FMM) / FP(Dijkstra) / LRP / LRP(FMM) for one logK field"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sweep", required=True, choices=["variance", "std"])
    parser.add_argument("--value", required=True, type=float)
    parser.add_argument("--field-size", required=True, type=int)
    parser.add_argument("--mc-index", required=True, type=int)
    parser.add_argument("--output-dir", type=Path, default=None)

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
    parser.add_argument("--fmm-velocity-layout", choices=["face", "cell"], default="face")
    parser.add_argument("--fmm-face-mapping", choices=["to_cell_avg", "direct"], default="to_cell_avg")
    parser.add_argument("--fmm-zero-policy", choices=["obstacle", "clip", "error"], default="obstacle")
    parser.add_argument("--fmm-min-speed", type=float, default=1e-12)
    parser.add_argument("--fmm-backtracking", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--fmm-continuous-step-size", type=float, default=0.5)
    parser.add_argument("--curvature-method", choices=["savgol"], default="savgol")
    parser.add_argument("--curvature-savgol-window", type=int, default=11)
    parser.add_argument("--curvature-savgol-poly", type=int, default=3)
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
    _validate_particle_tracking_options(solver=args.solver, velocity_location=args.velocity_location, backend=args.backend)
    return args


def main() -> int:
    args = parse_args()
    cfg = step1.load_config(args.config)
    logk_path, tmin_path, stream_path, balance_path = run_step5(
        cfg,
        sweep=args.sweep,
        value=args.value,
        field_size=args.field_size,
        mc_index=args.mc_index,
        output_dir=args.output_dir,
        dh=args.dh,
        solver=args.solver,
        backend=args.backend,
        velocity_location=args.velocity_location,
        align_with_modflow_velocity=args.align_with_modflow_velocity,
        fmm_velocity_layout=args.fmm_velocity_layout,
        fmm_face_mapping=args.fmm_face_mapping,
        fmm_zero_policy=args.fmm_zero_policy,
        fmm_min_speed=args.fmm_min_speed,
        fmm_backtracking=args.fmm_backtracking,
        fmm_continuous_step_size=args.fmm_continuous_step_size,
        curvature_method=args.curvature_method,
        curvature_savgol_window=args.curvature_savgol_window,
        curvature_savgol_poly=args.curvature_savgol_poly,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
