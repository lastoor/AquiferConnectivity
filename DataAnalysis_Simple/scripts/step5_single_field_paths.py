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

import step1_index as step1
import step2_metrics as step2
import step4_plot as step4


_MC_PATTERN = re.compile(r"(?:_mc|_)(-?\d+)$")


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
    t_min: np.ndarray


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


def _require_path_valid(sim_path: Any, path: np.ndarray, domain_box: tuple[int, int, int, int], x0: int, x1: int, name: str) -> None:
    if path.size == 0 or not sim_path.is_path_valid(path, domain_box, len_min=abs(x1 - x0)):
        raise RuntimeError(f"Invalid {name} geometry.")


def solve_single_field(
    logk_field: np.ndarray,
    *,
    field_size: int,
    dh: float,
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

    lrp, _total_r, _s_node, _e_node, _rmap = sim_path.dijkstra_2d_least_resistance(k_field, source, target, average="harmonic")
    _require_path_valid(sim_path, lrp, domain_box, x0, x1, "LRP")

    h, msg, is_success = sim_hyd.solve_head(k_field, x, y, dh=dh, rtol=1e-10, atol=1e-20, max_it=10000)
    if not is_success:
        raise RuntimeError(f"Head solver failed: {msg}")

    vx, vy = sim_hyd.compute_velocity(k_field, x, y, h, output_location=fmm_velocity_layout)
    speed_fmm, traversable_mask, _speed_diag = prepare_fmm_speed(
        vx=vx,
        vy=vy,
        layout=fmm_velocity_layout,
        face_mapping=fmm_face_mapping,
        zero_policy=fmm_zero_policy,
        min_speed=fmm_min_speed,
    )

    fp_dijkstra, _total_t_dij, _sd, _ed, _rmap_v = sim_path.dijkstra_2d_least_resistance(speed_fmm, source, target, average="arithmetic")
    _require_path_valid(sim_path, fp_dijkstra, domain_box, x0, x1, "FP (Dijkstra)")

    source_mask = np.zeros_like(speed_fmm, dtype=bool)
    target_mask = np.zeros_like(speed_fmm, dtype=bool)
    source_mask[y0:y1, x0] = True
    target_mask[:, x1] = True
    source_mask = source_mask & traversable_mask
    target_mask = target_mask & traversable_mask
    if not np.any(source_mask) or not np.any(target_mask):
        raise RuntimeError("No traversable source/target cells for FMM.")

    t_min, _state = fast_marching_2d_regions(speed_fmm, source_mask, h=1.0, traversable_mask=traversable_mask)
    path_ij, _chosen_target = extract_path_source_to_target(
        t_min,
        source_mask,
        target_mask,
        traversable_mask=traversable_mask,
    )
    if len(path_ij) < 2:
        raise RuntimeError("FMM path extraction failed.")
    fp_fmm = np.asarray(path_ij, dtype=float)[:, [1, 0]][::-1]
    _require_path_valid(sim_path, fp_fmm, domain_box, x0, x1, "FP (FMM)")

    n_particles = max(2, int(density_particle * abs(y1 - y0)))
    seeds_x = np.full(n_particles, x0, np.float32)
    seeds_y = np.linspace(y0, y1, n_particles).astype(np.float32)
    track_args = {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": x1,
        "velocity_location": "face",
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
    fp = fp[fp[:, 0] <= x1]
    _require_path_valid(sim_path, fp, domain_box, x0, x1, "FP")

    return PathSolveResult(fp=fp, fp_fmm=fp_fmm, fp_dijkstra=fp_dijkstra, lrp=lrp, t_min=t_min)


def plot_paths_over_field(
    field: np.ndarray,
    *,
    title: str,
    cmap: str,
    colorbar_label: str,
    result: PathSolveResult,
):
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(field, origin="lower", cmap=cmap, interpolation="nearest")
    ax.plot(result.fp[:, 0], result.fp[:, 1], color=step4.COLORS["fp"], linewidth=1.8, label="FP")
    ax.plot(result.fp_fmm[:, 0], result.fp_fmm[:, 1], color=step4.COLORS["fpFMM"], linewidth=1.6, label="FP (FMM)")
    ax.plot(
        result.fp_dijkstra[:, 0],
        result.fp_dijkstra[:, 1],
        color=step4.COLORS["fpDijkstra"],
        linewidth=1.6,
        label="FP (Dijkstra)",
    )
    ax.plot(result.lrp[:, 0], result.lrp[:, 1], color=step4.COLORS["lrp"], linewidth=1.6, label="LRP")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize="8", loc="upper right")
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
) -> tuple[Path, Path]:
    selected = select_run_input(cfg, sweep=sweep, value=value, field_size=field_size, mc_index=mc_index)
    logk_field = load_selected_logk(selected)

    result = solve_single_field(
        logk_field,
        field_size=field_size,
        dh=cfg.dh if dh is None else dh,
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
    logk_path = out_dir / f"{stem}_logk_paths.png"
    tmin_path = out_dir / f"{stem}_arrival_paths.png"

    fig_logk = plot_paths_over_field(
        logk_field,
        title=f"logK with Paths ({_sweep_token(sweep)}={selected.value_key:g}, size={field_size}, mc={selected.mc_label})",
        cmap="viridis",
        colorbar_label="logK",
        result=result,
    )
    fig_logk.savefig(logk_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_logk)

    tmin_plot = np.where(np.isfinite(result.t_min), result.t_min, np.nan)
    fig_tmin = plot_paths_over_field(
        tmin_plot,
        title=f"Minimal Arrival Time with Paths ({_sweep_token(sweep)}={selected.value_key:g}, size={field_size}, mc={selected.mc_label})",
        cmap="magma",
        colorbar_label=r"$T_{\min}$",
        result=result,
    )
    fig_tmin.savefig(tmin_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_tmin)
    return logk_path, tmin_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5: solve and visualize FP / FP(FMM) / FP(Dijkstra) / LRP for one logK field"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sweep", required=True, choices=["variance", "std"])
    parser.add_argument("--value", required=True, type=float)
    parser.add_argument("--field-size", required=True, type=int)
    parser.add_argument("--mc-index", required=True, type=int)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--dh", type=float, default=None)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = step1.load_config(args.config)
    logk_path, tmin_path = run_step5(
        cfg,
        sweep=args.sweep,
        value=args.value,
        field_size=args.field_size,
        mc_index=args.mc_index,
        output_dir=args.output_dir,
        dh=args.dh,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
