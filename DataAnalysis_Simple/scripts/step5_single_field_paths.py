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
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    source_xy: np.ndarray
    target_xy: np.ndarray


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
        from fmm_core.fmm import extract_path_source_to_target, fast_marching_2d_regions
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


def _validate_particle_tracking_options(*, solver: str, velocity_location: str) -> None:
    if solver not in {"pollock", "rk45"}:
        raise ValueError(f"Unsupported particle-tracking solver: {solver}")
    if velocity_location not in {"face", "cell"}:
        raise ValueError(f"Unsupported velocity_location={velocity_location}; expected 'face' or 'cell'.")
    if solver == "pollock" and velocity_location != "face":
        raise ValueError("Pollock solver requires velocity_location='face'.")


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
    path_ij, _chosen_target = extract_path_source_to_target(
        t_field,
        source_mask,
        target_mask,
        traversable_mask=traversable_mask,
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
    velocity_location: str,
    align_with_modflow_velocity: bool,
    density_particle: int,
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
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location)
    n_particles = max(2, int(density_particle * abs(y1 - y0)))
    seeds_x = np.full(n_particles, x0, np.float32)
    seeds_y = np.linspace(y0, y1, n_particles).astype(np.float32)

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
        track_args = {
            "x_faces": build_face_coordinates_from_centers(x).astype(np.float32),
            "y_faces": build_face_coordinates_from_centers(y).astype(np.float32),
            "vx": vx,
            "vy": vy,
            "seeds_x": seeds_x,
            "seeds_y": seeds_y,
            "right_x": x1,
            "max_points": max_points,
            "max_steps": max_iters,
        }
        px, py, pt, length_arr, status_arr = sim_pt.particle_tracking_pollock(track_args)
    else:
        vx, vy = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h_field,
            output_location=velocity_location,
            align_with_modflow=align_with_modflow_velocity,
        )
        track_args = {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "seeds_x": seeds_x,
            "seeds_y": seeds_y,
            "right_x": x1,
            "velocity_location": velocity_location,
            "eps_speed": eps_speed,
            "ds_min": ds_min,
            "ds_max": ds_max,
            "tol": tol,
            "y_policy": y_policy,
            "max_points": max_points,
            "max_iters": max_iters,
            "threads_per_block": 256,
            "dtype": np.float64,
        }
        px, py, pt, length_arr, status_arr = sim_pt.particle_tracking_cpu(track_args)

    count = Counter(status_arr.tolist())
    frac_success = count.get(0, 0) / max(1, n_particles)
    frac_bad = (count.get(1, 0) + count.get(4, 0)) / max(1, n_particles)
    if frac_success < rk45_min_success_frac or frac_bad > rk45_max_bad_frac:
        raise RuntimeError(
            "Invalid velocity field for tracking. "
            f"status={dict(count)}, success_frac={frac_success:.3f}, bad_frac={frac_bad:.3f}"
        )

    result = sim_pt.fastest_trajectory(px, py, pt, length_arr, status_arr)
    fp = np.vstack((result["x"], result["y"])).T
    return fp, np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)


def solve_single_field(
    logk_field: np.ndarray,
    *,
    field_size: int,
    dh: float,
    solver: str,
    velocity_location: str,
    align_with_modflow_velocity: bool,
    fmm_velocity_layout: str,
    fmm_face_mapping: str,
    fmm_zero_policy: str,
    fmm_min_speed: float,
    density_particle: int,
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
    sim_hyd = modules["sim_hyd"]
    sim_pt = modules["sim_pt"]
    sim_path = modules["sim_path"]
    prepare_fmm_speed = modules["prepare_fmm_speed"]
    fast_marching_2d_regions = modules["fast_marching_2d_regions"]
    extract_path_source_to_target = modules["extract_path_source_to_target"]

    if logk_field.shape != (field_size, field_size):
        raise ValueError(f"logK shape {logk_field.shape} does not match requested field-size={field_size}.")
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location)

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
    )
    _require_path_valid(sim_path, lrp_fmm, domain_box, x0, x1, "LRP (FMM)")

    h, msg, is_success = sim_hyd.solve_head(k_field, x, y, dh=dh, rtol=1e-10, atol=1e-20, max_it=10000)
    if not is_success:
        raise RuntimeError(f"Head solver failed: {msg}")

    vx_fmm, vy_fmm = sim_hyd.compute_velocity(k_field, x, y, h, output_location=fmm_velocity_layout)
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
        velocity_location=velocity_location,
        align_with_modflow_velocity=align_with_modflow_velocity,
        density_particle=density_particle,
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
    ax.plot(result.source_xy[:, 0] / i_y, result.source_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle='-', path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.target_xy[:, 0] / i_y, result.target_xy[:, 1] / i_y, color="purple", linewidth=lw, linestyle='-', path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
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
    colorbar_label: str = r"$|\mathbf{v}|/U$",
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
    velocity_location: str,
    align_with_modflow_velocity: bool,
    fmm_velocity_layout: str,
    fmm_face_mapping: str,
    fmm_zero_policy: str,
    fmm_min_speed: float,
    density_particle: int,
    eps_speed: float,
    ds_min: float,
    ds_max: float,
    tol: float,
    y_policy: str,
    max_points: int,
    max_iters: int,
    rk45_min_success_frac: float,
    rk45_max_bad_frac: float,
) -> tuple[Path, Path, Path]:
    selected = select_run_input(cfg, sweep=sweep, value=value, field_size=field_size, mc_index=mc_index)
    logk_field = load_selected_logk(selected)

    dh_eff = cfg.dh if dh is None else dh
    result = solve_single_field(
        logk_field,
        field_size=field_size,
        dh=dh_eff,
        solver=solver,
        velocity_location=velocity_location,
        align_with_modflow_velocity=align_with_modflow_velocity,
        fmm_velocity_layout=fmm_velocity_layout,
        fmm_face_mapping=fmm_face_mapping,
        fmm_zero_policy=fmm_zero_policy,
        fmm_min_speed=fmm_min_speed,
        density_particle=density_particle,
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
    return logk_path, tmin_path, stream_path


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

    parser.add_argument("--density-particle", type=int, default=20)
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
    _validate_particle_tracking_options(solver=args.solver, velocity_location=args.velocity_location)
    return args


def main() -> int:
    args = parse_args()
    cfg = step1.load_config(args.config)
    logk_path, tmin_path, stream_path = run_step5(
        cfg,
        sweep=args.sweep,
        value=args.value,
        field_size=args.field_size,
        mc_index=args.mc_index,
        output_dir=args.output_dir,
        dh=args.dh,
        solver=args.solver,
        velocity_location=args.velocity_location,
        align_with_modflow_velocity=args.align_with_modflow_velocity,
        fmm_velocity_layout=args.fmm_velocity_layout,
        fmm_face_mapping=args.fmm_face_mapping,
        fmm_zero_policy=args.fmm_zero_policy,
        fmm_min_speed=args.fmm_min_speed,
        density_particle=args.density_particle,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
