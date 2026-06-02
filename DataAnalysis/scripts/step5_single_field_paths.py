#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter as _savgol_filter

import step1_index as step1
import step2_metrics as step2
import step3_stats as step3
import step4_plot as step4


_MC_PATTERN = re.compile(r"(?:_mc|_)(-?\d+)$")

pathcm = cmc.lapaz.resampled(7)
COLORS = {
    "fp": pathcm(1),
    "lrp": pathcm(5), #"#FC1212",
    "lrpFMM": pathcm(5),
    "fpFMM": pathcm(4), #"#31830B",
    "fpDijkstra": pathcm(4), #"#131EE2",
    "fpFSM": "#2CA02C",
    "fpFmAsr": pathcm(3),
    "fpPypar2": pathcm(2),
}

LINESTYLES = {
    "fp": "-",
    "fpFMM": "-",
    "fpDijkstra": "-",
    "lrp": "-",
    "lrpFMM": "-",
    "fpFSM": "-",
    "fpFmAsr": "-",
    "fpPypar2": "-",
}

LINEWIDTHS = 0.8

def configure_style():
    import glob
    import matplotlib.font_manager as fm

    family: str = "CMU Sans Serif"
    mathtext_fontset: str = "cm"
    unicode_minus: bool = False
    cmu_font_glob: list = ["/home1/binhaoli/.local/share/fonts/cmun*.ttf", "/Users/binhaoli/Library/Fonts/cmun*.ttf"]

    for paths in cmu_font_glob:
        for path in glob.glob(paths):
            fm.fontManager.addfont(path)
            
    plt.rcParams["font.family"] = family
    plt.rcParams["mathtext.fontset"] = mathtext_fontset
    plt.rcParams["axes.unicode_minus"] = unicode_minus

    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 12

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
    fp_arrival_time: float | None
    fp_fmm: np.ndarray
    lrp_fmm: np.ndarray
    t_min: np.ndarray
    t_min_lrp: np.ndarray
    fp_fsm: np.ndarray | None
    t_min_fsm: np.ndarray | None
    fp_fm_asr: np.ndarray | None
    fp_fm_asr_arrival_time: float | None
    t_min_fm_asr: np.ndarray | None
    fm_asr_M: np.ndarray | None
    fm_asr_beta: np.ndarray | None
    fp_pypar2: np.ndarray | None
    fp_pypar2_arrival_time: float | None
    fp_pypar2_error: str | None
    pypar2_replay_meta: dict[str, Any] | None
    pypar2_snapshots: SnapshotData | None
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
    snapshots: SnapshotData | None = None


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
    hamilton_curvature: np.ndarray | None = None
    hamilton_curvature_magnitude: np.ndarray | None = None
    hamilton_curvature_smoothed: np.ndarray | None = None
    hamilton_eikonal_residual: np.ndarray | None = None


@dataclass(slots=True)
class LinearizedFpSeries:
    travel_time_fields: list[np.ndarray]
    paths: list[np.ndarray]
    path_indices: list[int]
    history: list[dict[str, float]]
    failed_path_indices: list[int]
    converged: bool


@dataclass(slots=True)
class SnapshotData:
    times: np.ndarray  # (T,) float64 — resolved snapshot times in physical units
    x: np.ndarray      # (N, T) float64 — x-positions; NaN if particle not active
    y: np.ndarray      # (N, T) float64 — y-positions; NaN if particle not active


def _extract_particle_arrival_time(
    fastest_summary: dict[str, Any],
    summary: dict[str, Any],
    snapshot_data: SnapshotData | None,
) -> float:
    candidate_keys = (
        "arrival_time",
        "travel_time",
        "time",
        "t",
        "t_arrival",
        "arrival",
        "first_arrival_time",
    )
    for container in (fastest_summary, summary):
        for key in candidate_keys:
            value = container.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            if arr.size == 0:
                continue
            scalar = float(arr.reshape(-1)[0])
            if np.isfinite(scalar):
                return scalar

    if snapshot_data is not None:
        times = np.asarray(snapshot_data.times, dtype=float)
        if times.size > 0 and np.all(np.isfinite(times)):
            return float(times[-1])

    available_fastest = sorted(str(k) for k in fastest_summary.keys())
    available_summary = sorted(str(k) for k in summary.keys())
    raise RuntimeError(
        "Particle tracking summary did not expose a usable arrival time. "
        f"fastest keys={available_fastest}, summary keys={available_summary}"
    )


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

    xc1, yc1, a1, b1 = 1/3, 3/4, 1/10, 1/10
    xc2, yc2, a2, b2 = 2/3, 1/4, 1/10, 1/10
    xc3, yc3, a3, b3 = 3/4, 3/4, 1/10, 1/10
    xc4, yc4, a4, b4 = 1/4, 1/4, 1/10, 1/10

    dx1, dy1 = (xx - xc1), (yy - yc1)
    dx2, dy2 = (xx - xc2), (yy - yc2)
    dx3, dy3 = (xx - xc3), (yy - yc3)
    dx4, dy4 = (xx - xc4), (yy - yc4)

    theta1 = 0
    theta2 = 0
    theta3 = np.pi/6
    theta4 = np.pi/6

    ellipse1 = (dx1 * np.cos(theta1) + dy1 * np.sin(theta1)) ** 2 / a1**2 + (dx1 * np.sin(theta1) - dy1 * np.cos(theta1)) ** 2 / b1**2
    ellipse2 = (dx2 * np.cos(theta2) + dy2 * np.sin(theta2)) ** 2 / a2**2 + (dx2 * np.sin(theta2) - dy2 * np.cos(theta2)) ** 2 / b2**2
    ellipse3 = (dx3 * np.cos(theta3) + dy3 * np.sin(theta3)) ** 2 / a3**2 + (dx3 * np.sin(theta3) - dy3 * np.cos(theta3)) ** 2 / b3**2
    ellipse4 = (dx4 * np.cos(theta4) + dy4 * np.sin(theta4)) ** 2 / a4**2 + (dx4 * np.sin(theta4) - dy4 * np.cos(theta4)) ** 2 / b4**2

    # return np.asarray(
    #     offset
    #     + amplitude * (
    #         np.exp(- ellipse1)
    #         + 0.75*np.exp(- ellipse2)
    #         # + 0.1*np.exp(- ellipse3)
    #         # + 0.1*np.exp(- ellipse4)
    #     ),
    #     dtype=float,
    # )

    return np.asarray(
        offset
        + amplitude * (
            np.sin(2.0 * np.pi * frequency_x * xx)
            * np.cos(2.0 * np.pi * frequency_y * yy)
            + np.exp(- ((xx - 3/8) ** 2 + (yy - 3/4) ** 2) / (1/8) ** 2)
            + np.exp(- ((xx - 5/8) ** 2 + (yy - 1/4) ** 2) / (1/8) ** 2)
        ),
        dtype=float,
    )

    # # Sine curve
    # y_sin = np.sin(xx)/2
    # thickness = 0.05  # adjust for thickness
    # matrix = np.abs(yy - y_sin) < thickness
    # matrix = matrix.astype(float)
    # return offset + amplitude * matrix


    # return np.asarray(
    #     offset
    #     + amplitude * (
    #         np.exp(- ( ((xx - 3/8)/(1/6)) ** 2 + ((yy - 3/4)/(1/10)) ** 2) )
    #         + np.exp(- ( ((xx - 5/8)/(1/6)) ** 2 + ((yy - 1/4)/(1/10)) ** 2) )
    #     ),
    #     dtype=float,
    # )


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
        from fmm_core.fm_asr import FMASR2D, build_randers_from_W, extract_fm_asr_path
        from fmm_core.fsm import compute_fsm_path as shared_compute_fsm_path
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
        "FMASR2D": FMASR2D,
        "build_randers_from_W": build_randers_from_W,
        "extract_fm_asr_path": extract_fm_asr_path,
        "compute_fsm_path": shared_compute_fsm_path,
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


def _import_grf_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    flowsim_path = Path(os.environ.get("FLOWSIM_PATH", str(repo_root.parent / "FlowSimulation"))).expanduser().resolve()
    if not flowsim_path.is_dir():
        raise RuntimeError(
            f"FlowSimulation path does not exist: {flowsim_path}. "
            "Set FLOWSIM_PATH to your FlowSimulation repository root."
        )
    _add_search_path(flowsim_path)
    try:
        from simulation import grf_generation as sim_grf
    except Exception as exc:
        raise RuntimeError(
            "Failed to import grf_generation from FlowSimulation. "
            "Ensure FlowSimulation is available and dependencies are installed."
        ) from exc
    return sim_grf


def generate_grf_logk(
    *,
    field_size: int,
    std: float,
    length_scale: float,
    seed: int,
    minpadding: int = 0,
) -> np.ndarray:
    """Generate a logK field as a Gaussian Random Field with exponential covariance.

    Uses the same generation pattern as the Monte Carlo runs:
    ``ExpCov(sigma=std, length_scale=length_scale)`` with circulant embedding.

    Parameters
    ----------
    field_size : int
        Grid resolution (field will be field_size × field_size).
    std : float
        Standard deviation of the GRF (must be > 0).
    length_scale : float
        Correlation length of the exponential covariance (must be > 0).
    seed : int
        Random seed for reproducibility.
    minpadding : int
        Minimum padding for the circulant embedding FFT (default 0).
    """
    if field_size < 2:
        raise ValueError(f"field_size must be at least 2; got {field_size}.")
    if not (np.isfinite(float(std)) and float(std) > 0):
        raise ValueError(f"std must be a positive finite number; got {std}.")
    if not (np.isfinite(float(length_scale)) and float(length_scale) > 0):
        raise ValueError(f"length_scale must be a positive finite number; got {length_scale}.")

    sim_grf = _import_grf_module()
    np.random.seed(int(seed))
    cov = sim_grf.ExpCov(sigma=float(std), length_scale=float(length_scale))
    pts = (np.linspace(0, 1, field_size),) * 2
    mean = np.zeros((field_size, field_size))
    grf_gen = sim_grf.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=int(minpadding))
    return np.asarray(grf_gen.sample(), dtype=float)


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


def _compute_fm_asr_path(
    vx_cell: np.ndarray,
    vy_cell: np.ndarray,
    *,
    geometry: Step5Geometry,
    FMASR2D: Any,
    build_randers_from_W: Any,
    extract_fm_asr_path: Any,
    perp_ratio: float,
    back_factor: float,
    min_speed: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    traversable_mask = (
        np.isfinite(vx_cell)
        & np.isfinite(vy_cell)
        & np.isfinite(np.hypot(vx_cell, vy_cell))
    )
    source_mask, target_mask, _traversable_mask = _build_source_target_masks(
        field_shape=vx_cell.shape,
        source_rows=geometry.source_rows,
        target_rows=geometry.target_rows,
        x0=geometry.source_x,
        x1=geometry.target_x,
        traversable_mask=traversable_mask,
    )
    if not np.any(source_mask) or not np.any(target_mask):
        raise RuntimeError("No traversable source/target cells for FM-ASR.")

    vx_safe = np.where(np.isfinite(vx_cell), vx_cell, 0.0)
    vy_safe = np.where(np.isfinite(vy_cell), vy_cell, 0.0)
    W_cell = np.stack((vx_safe, vy_safe), axis=2)
    _c_iso, M_field, beta_field, _e = build_randers_from_W(
        W_cell,
        perp_ratio=perp_ratio,
        back_factor=back_factor,
        min_speed=min_speed,
        isotropic_only=False,
    )
    solver_beta = -np.asarray(beta_field, dtype=float)
    solver = FMASR2D.from_velocity_field(
        W_cell,
        h=1.0,
        perp_ratio=perp_ratio,
        back_factor=back_factor,
        min_speed=min_speed,
        isotropic_only=False,
        direction="from_source",
    )
    t_field = solver.solve(source_mask)
    path_xy, travel_value, _chosen_target = extract_fm_asr_path(
        solver,
        t_field,
        source_mask,
        target_mask,
        max_steps=max_steps,
    )
    return path_xy, t_field, float(travel_value), np.asarray(M_field, dtype=float), solver_beta


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
    snapshot_mode: str | None = None,
    snapshot_times: Any = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, SnapshotData | None]:
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location, backend=backend)
    if snapshot_mode is not None and not (solver == "pollock" and backend == "gpu"):
        import warnings
        warnings.warn(
            "snapshot_mode is only supported with solver='pollock' and backend='gpu'; "
            "snapshots will be skipped.",
            stacklevel=2,
        )
        snapshot_mode = None
        snapshot_times = None
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
        if snapshot_mode is not None and snapshot_times is not None:
            summary_params["snapshot_mode"] = snapshot_mode
            summary_params["snapshot_times"] = snapshot_times
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

    snap_raw = summary.get("snapshots")
    snapshot_data: SnapshotData | None = None
    if snap_raw is not None:
        snapshot_data = SnapshotData(
            times=np.asarray(snap_raw["snapshot_times"], dtype=float),
            x=np.asarray(snap_raw["x"], dtype=float),
            y=np.asarray(snap_raw["y"], dtype=float),
        )

    fp_arrival_time = _extract_particle_arrival_time(result, summary, snapshot_data)

    return fp, fp_arrival_time, np.asarray(vx, dtype=float), np.asarray(vy, dtype=float), snapshot_data


def _run_pypar2_tracking(
    *,
    sim_pt: Any,
    vx_face: np.ndarray,
    vy_face: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    geometry: Step5Geometry,
    density_particle: int,
    num_particles: int | None,
    pypar2_num_particles: int | None,
    pypar2_dt: float,
    pypar2_steps: int,
    pypar2_dm: float,
    pypar2_alpha_l: float,
    pypar2_alpha_t: float,
    pypar2_velocity_floor: float | None,
    pypar2_taichi_arch: str,
    pypar2_max_points: int,
    pypar2_trajectory_memory_limit: int,
    pypar2_replay_top_k: int,
    pypar2_replay_particles_per_center: int,
    pypar2_replay_radius_cells: float,
    pypar2_replay_tolerance: float,
    pypar2_replay_max_redos: int,
    pypar2_seed: int,
    snapshot_mode: str | None = None,
    snapshot_times: Any = None,
) -> tuple[np.ndarray | None, float | None, str | None, dict[str, Any], SnapshotData | None]:
    n_particles = _resolve_particle_count(
        num_particles=pypar2_num_particles if pypar2_num_particles is not None else num_particles,
        density_particle=density_particle,
        y0=geometry.source_y_start,
        y1=geometry.source_y_end,
    )
    source_rows = geometry.source_rows
    seeds_x = np.full(n_particles, geometry.source_x, dtype=np.float32)
    seeds_y = np.linspace(float(source_rows[0]), float(source_rows[-1]), n_particles, endpoint=True).astype(np.float32)

    params: dict[str, Any] = {
        "method": "pypar2",
        "backend": str(pypar2_taichi_arch),
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "save_trajectories": True,
        "return_fastest": True,
        "x_faces": build_face_coordinates_from_centers(x).astype(np.float32),
        "y_faces": build_face_coordinates_from_centers(y).astype(np.float32),
        "vx": np.asarray(vx_face, dtype=np.float32),
        "vy": np.asarray(vy_face, dtype=np.float32),
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": float(geometry.target_x),
        "dt": float(pypar2_dt),
        "steps": int(pypar2_steps),
        "molecular_diffusion": float(pypar2_dm),
        "alpha_l": float(pypar2_alpha_l),
        "alpha_t": float(pypar2_alpha_t),
        "longitudinal_dispersivity": float(pypar2_alpha_l),
        "transverse_dispersivity": float(pypar2_alpha_t),
        "max_points": int(pypar2_max_points),
        "trajectory_memory_limit": int(pypar2_trajectory_memory_limit),
        "replay_top_k": int(pypar2_replay_top_k),
        "replay_particles_per_center": int(pypar2_replay_particles_per_center),
        "replay_radius_cells": float(pypar2_replay_radius_cells),
        "replay_tolerance": float(pypar2_replay_tolerance),
        "replay_max_redos": int(pypar2_replay_max_redos),
        "seed": int(pypar2_seed),
    }
    if snapshot_mode is not None and snapshot_times is not None:
        params["snapshot_mode"] = snapshot_mode
        params["snapshot_times"] = snapshot_times
    if pypar2_velocity_floor is not None:
        params["velocity_floor"] = float(pypar2_velocity_floor)

    try:
        summary = sim_pt.particle_tracking_summary(params)
    except Exception as exc:
        return None, None, f"PyPar2 dispatcher failed: {exc}", {}, None

    replay_meta = summary.get("replay", {}) or {}
    result = summary.get("fastest")
    snap_raw = summary.get("snapshots")
    snapshot_data: SnapshotData | None = None
    if snap_raw is not None:
        snapshot_data = SnapshotData(
            times=np.asarray(snap_raw["snapshot_times"], dtype=float),
            x=np.asarray(snap_raw["x"], dtype=float),
            y=np.asarray(snap_raw["y"], dtype=float),
        )

    if result is None:
        return None, None, "PyPar2: no particle reached right_x", replay_meta, snapshot_data

    fp = np.vstack((np.asarray(result["x"], dtype=float), np.asarray(result["y"], dtype=float))).T
    travel_time = _extract_particle_arrival_time(result, summary, snapshot_data)
    return fp, float(travel_time), None, replay_meta, snapshot_data


def _compute_linearized_fp_series(
    *,
    linear_solver: Any,
    slowness_unperturbed: np.ndarray,
    slowness_true: np.ndarray,
    geometry: Step5Geometry,
    fast_marching_2d_regions: Any,
    extract_path_source_to_target: Any,
    extract_path_continuous: Any,
    scheme: str = "linearized",
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
    snapshot_mode: str | None = None,
    snapshot_times: Any = None,
    fsm_fp: bool = False,
    fsm_max_sweeps: int = 100,
    fsm_tol: float = 1e-6,
    fsm_mse_tol: float | None = None,
    fsm_convergence_mode: str | None = None,
    fsm_mse_check_every: int = 5,
    fsm_mse_activation_factor: float = 10.0,
    fsm_consecutive_passes: int = 2,
    fsm_solver: str = "fipy",
    fsm_dt: float = 0.5,
    fsm_max_steps: int = 10_000,
    fm_asr_fp: bool = False,
    fm_asr_perp_ratio: float = 0.5,
    fm_asr_back_factor: float = 2.0,
    fm_asr_min_speed: float = 1e-12,
    fm_asr_max_steps: int = 200_000,
    pypar2_fp: bool = False,
    pypar2_dt: float = 0.01,
    pypar2_steps: int = 200_000,
    pypar2_dm: float = 1e-9,
    pypar2_alpha_l: float = 0.1,
    pypar2_alpha_t: float = 0.01,
    pypar2_velocity_floor: float | None = None,
    pypar2_taichi_arch: str = "gpu",
    pypar2_max_points: int = 2_000,
    pypar2_num_particles: int | None = None,
    pypar2_trajectory_memory_limit: int = 5_000_000,
    pypar2_replay_top_k: int = 5,
    pypar2_replay_particles_per_center: int = 40,
    pypar2_replay_radius_cells: float = 0.5,
    pypar2_replay_tolerance: float = 1.05,
    pypar2_replay_max_redos: int = 3,
    pypar2_seed: int = 0,
) -> PathSolveResult:
    modules = _import_solver_modules()
    linear_solver = _import_linear_eikonal_solver() if linearized_fp else None
    extract_path_continuous = modules["extract_path_continuous"]
    sim_hyd = modules["sim_hyd"]
    sim_pt = modules["sim_pt"]
    sim_path = modules["sim_path"]
    fast_marching_2d_regions = modules["fast_marching_2d_regions"]
    extract_path_source_to_target = modules["extract_path_source_to_target"]
    compute_fsm_path = modules["compute_fsm_path"]
    FMASR2D = modules["FMASR2D"]
    build_randers_from_W = modules["build_randers_from_W"]
    extract_fm_asr_path = modules["extract_fm_asr_path"]

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
    vx_cell = -k_field * grad_h_x  # cell-centred Darcy velocity (x-component)
    vy_cell = -k_field * grad_h_y  # cell-centred Darcy velocity (y-component)
    dh_mean = abs(float(dh))
    if dh_mean <= 0.0 or not np.isfinite(dh_mean):
        raise ValueError(f"Invalid mean hydraulic gradient dh={dh}; expected positive finite magnitude.")
    slowness_unperturbed = 1.0 / np.maximum(k_field * dh_mean, 1e-12)
    slowness_true = 1.0 / np.maximum(k_field * grad_h_mag, 1e-12)
    lrp_fmm, t_min_lrp = _compute_fmm_path(
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

    fp_fsm: np.ndarray | None = None
    t_min_fsm: np.ndarray | None = None
    if fsm_fp:
        if fsm_solver in ("new", "fd"):
            # These backends use PDE residual directly; residual-gating controls are unused.
            fsm_effective_mse_tol = None
            fsm_effective_convergence_mode = "update_or_mse"
            fsm_effective_mse_check_every = 5
            fsm_effective_mse_activation_factor = 10.0
            fsm_effective_consecutive_passes = 1
        else:
            fsm_effective_mse_tol = fsm_mse_tol
            if fsm_convergence_mode is None:
                fsm_effective_convergence_mode = "update_and_mse" if fsm_mse_tol is not None else "update"
            else:
                fsm_effective_convergence_mode = fsm_convergence_mode
            fsm_effective_mse_check_every = fsm_mse_check_every
            fsm_effective_mse_activation_factor = fsm_mse_activation_factor
            fsm_effective_consecutive_passes = fsm_consecutive_passes
        vx_face, vy_face = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h,
            output_location="face",
            align_with_modflow=align_with_modflow_velocity,
            face_output="full",
        )
        fsm_source_mask = np.zeros(k_field.shape, dtype=bool)
        fsm_source_mask[geometry.source_rows, geometry.source_x] = True
        fsm_target_mask = np.zeros(k_field.shape, dtype=bool)
        fsm_target_mask[geometry.target_rows, geometry.target_x] = True
        fp_fsm, t_min_fsm, _arrival_time_fsm = compute_fsm_path(
            vx_face,
            vy_face,
            fsm_source_mask,
            fsm_target_mask,
            geometry.x,
            geometry.y,
            max_sweeps=fsm_max_sweeps,
            tol=fsm_tol,
            mse_tol=fsm_effective_mse_tol,
            convergence_mode=fsm_effective_convergence_mode,
            mse_check_every=fsm_effective_mse_check_every,
            mse_activation_factor=fsm_effective_mse_activation_factor,
            consecutive_passes=fsm_effective_consecutive_passes,
            use_inlet_cut=True,
            solver=fsm_solver,
            dt=fsm_dt,
            max_steps=fsm_max_steps,
        )

    fp_fm_asr: np.ndarray | None = None
    fp_fm_asr_arrival_time: float | None = None
    t_min_fm_asr: np.ndarray | None = None
    fm_asr_M: np.ndarray | None = None
    fm_asr_beta: np.ndarray | None = None
    if fm_asr_fp:
        try:
            (
                fp_fm_asr,
                t_min_fm_asr,
                fp_fm_asr_arrival_time,
                fm_asr_M,
                fm_asr_beta,
            ) = _compute_fm_asr_path(
                vx_cell,
                vy_cell,
                geometry=geometry,
                FMASR2D=FMASR2D,
                build_randers_from_W=build_randers_from_W,
                extract_fm_asr_path=extract_fm_asr_path,
                perp_ratio=fm_asr_perp_ratio,
                back_factor=fm_asr_back_factor,
                min_speed=fm_asr_min_speed,
                max_steps=fm_asr_max_steps,
            )
            if check_path_geometry and (
                fp_fm_asr.size == 0
                or not sim_path.is_path_valid(
                    fp_fm_asr,
                    domain_box,
                    len_min=abs(geometry.target_x - geometry.source_x),
                )
            ):
                raise RuntimeError("Invalid FP (FM-ASR) geometry.")
        except Exception as exc:
            print(f"Warning: FM-ASR FP solve failed; continuing without FM-ASR path: {exc}")
            fp_fm_asr = None
            fp_fm_asr_arrival_time = None
            t_min_fm_asr = None
            fm_asr_M = None
            fm_asr_beta = None

    fp_pypar2: np.ndarray | None = None
    fp_pypar2_arrival_time: float | None = None
    fp_pypar2_error: str | None = None
    pypar2_replay_meta: dict[str, Any] | None = None
    pypar2_snapshots: SnapshotData | None = None
    if pypar2_fp:
        try:
            vx_face_pypar2, vy_face_pypar2 = sim_hyd.compute_velocity(
                k_field,
                x,
                y,
                h,
                output_location="face",
                align_with_modflow=align_with_modflow_velocity,
                face_output="full",
            )
            (
                fp_pypar2,
                fp_pypar2_arrival_time,
                fp_pypar2_error,
                pypar2_replay_meta,
                pypar2_snapshots,
            ) = _run_pypar2_tracking(
                sim_pt=sim_pt,
                vx_face=vx_face_pypar2,
                vy_face=vy_face_pypar2,
                x=x,
                y=y,
                geometry=geometry,
                density_particle=density_particle,
                num_particles=num_particles,
                pypar2_num_particles=pypar2_num_particles,
                pypar2_dt=pypar2_dt,
                pypar2_steps=pypar2_steps,
                pypar2_dm=pypar2_dm,
                pypar2_alpha_l=pypar2_alpha_l,
                pypar2_alpha_t=pypar2_alpha_t,
                pypar2_velocity_floor=pypar2_velocity_floor,
                pypar2_taichi_arch=pypar2_taichi_arch,
                pypar2_max_points=pypar2_max_points,
                pypar2_trajectory_memory_limit=pypar2_trajectory_memory_limit,
                pypar2_replay_top_k=pypar2_replay_top_k,
                pypar2_replay_particles_per_center=pypar2_replay_particles_per_center,
                pypar2_replay_radius_cells=pypar2_replay_radius_cells,
                pypar2_replay_tolerance=pypar2_replay_tolerance,
                pypar2_replay_max_redos=pypar2_replay_max_redos,
                pypar2_seed=pypar2_seed,
                snapshot_mode=snapshot_mode,
                snapshot_times=snapshot_times,
            )
            if fp_pypar2 is not None:
                fp_pypar2 = fp_pypar2[fp_pypar2[:, 0] <= geometry.target_x]
                if check_path_geometry and (
                    fp_pypar2.size == 0
                    or not sim_path.is_path_valid(
                        fp_pypar2,
                        domain_box,
                        len_min=abs(geometry.target_x - geometry.source_x),
                    )
                ):
                    fp_pypar2_error = "Invalid FP (PyPar2) geometry."
                    fp_pypar2 = None
                    fp_pypar2_arrival_time = None
            if fp_pypar2_error is not None:
                print(f"Warning: PyPar2 FP solve failed; continuing without PyPar2 path: {fp_pypar2_error}")
            elif fp_pypar2 is not None:
                replay_status = "" if not pypar2_replay_meta else str(pypar2_replay_meta.get("status", ""))
                print(
                    "PyPar2 FP extraction: "
                    f"{len(fp_pypar2)} points, arrival_time={fp_pypar2_arrival_time:.6g}, "
                    f"replay_status={replay_status or 'n/a'}"
                )
        except Exception as exc:
            fp_pypar2_error = str(exc)
            pypar2_replay_meta = {}
            print(f"Warning: PyPar2 FP solve failed; continuing without PyPar2 path: {exc}")
            fp_pypar2 = None
            fp_pypar2_arrival_time = None
            pypar2_snapshots = None

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

    fp, fp_arrival_time, vx_pt, vy_pt, snapshot_data = _run_particle_tracking(
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
        snapshot_mode=snapshot_mode,
        snapshot_times=snapshot_times,
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
        fp_arrival_time=float(fp_arrival_time),
        fp_fmm=fp_fmm,
        lrp_fmm=lrp_fmm,
        t_min=t_min,
        t_min_lrp=t_min_lrp,
        fp_fsm=fp_fsm,
        t_min_fsm=t_min_fsm,
        fp_fm_asr=fp_fm_asr,
        fp_fm_asr_arrival_time=fp_fm_asr_arrival_time,
        t_min_fm_asr=t_min_fm_asr,
        fm_asr_M=fm_asr_M,
        fm_asr_beta=fm_asr_beta,
        fp_pypar2=fp_pypar2,
        fp_pypar2_arrival_time=fp_pypar2_arrival_time,
        fp_pypar2_error=fp_pypar2_error,
        pypar2_replay_meta=pypar2_replay_meta,
        pypar2_snapshots=pypar2_snapshots,
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
        snapshots=snapshot_data,
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
    ax.plot(result.fp[:, 0] / i_y, result.fp[:, 1] / i_y, color=COLORS["fp"], linestyle=LINESTYLES["fp"], linewidth=lw, label=r"$\gamma^{\!\star}_{\mathcal{S}}$", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.fp_fmm[:, 0] / i_y, result.fp_fmm[:, 1] / i_y, color=COLORS["fpFMM"], linestyle=LINESTYLES["fpFMM"], linewidth=lw, label=r"$\gamma^{\!\star}_{\mathcal{E}}$", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    ax.plot(result.lrp_fmm[:, 0] / i_y, result.lrp_fmm[:, 1] / i_y, color=COLORS["lrpFMM"], linestyle=LINESTYLES["lrpFMM"], linewidth=lw, label=r"$\gamma^{\!\star}_{\mathcal{R}}$", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    # if result.fp_fsm is not None and len(result.fp_fsm) >= 2:
    #     ax.plot(result.fp_fsm[:, 0] / i_y, result.fp_fsm[:, 1] / i_y, color=COLORS["fpFSM"], linestyle=LINESTYLES["fpFSM"], linewidth=lw, label=r"$\gamma^{\!\star}_{\mathcal{FSM}}$", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    if result.fp_fm_asr is not None and len(result.fp_fm_asr) >= 2:
        ax.plot(result.fp_fm_asr[:, 0] / i_y, result.fp_fm_asr[:, 1] / i_y, color=COLORS["fpFmAsr"], linestyle=LINESTYLES["fpFmAsr"], linewidth=lw, label=r"$\gamma^{\!\star}_{\mathcal{F}}$", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    if result.fp_pypar2 is not None and len(result.fp_pypar2) >= 2:
        ax.plot(result.fp_pypar2[:, 0] / i_y, result.fp_pypar2[:, 1] / i_y, color=COLORS["fpPypar2"], linestyle=LINESTYLES["fpPypar2"], linewidth=lw, label=r"$\gamma^{\!\star}_{\mathcal{D}}$", path_effects=[pe.Stroke(linewidth=1.5*lw, foreground="white"), pe.Normal()])
    # if fp_diag is not None:
    #     ax.plot(
    #         fp_diag.spline_points[:, 0] / i_y,
    #         fp_diag.spline_points[:, 1] / i_y,
    #         color=to_rgba(COLORS["fpFMM"], alpha=0.75),
    #         linestyle="-",
    #         linewidth=1.0,
    #         label=fp_spline_label,
    #     )
    # if lrp_diag is not None:
    #     ax.plot(
    #         lrp_diag.spline_points[:, 0] / i_y,
    #         lrp_diag.spline_points[:, 1] / i_y,
    #         color=to_rgba(COLORS["lrpFMM"], alpha=0.75),
    #         linestyle="-",
    #         linewidth=1.0,
    #         label=lrp_spline_label,
    #     )
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


def _bilinear_sample_components(field: np.ndarray, x: np.ndarray, y: np.ndarray, points: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.ndim < 3:
        raise ValueError("field must have at least one component dimension")
    flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
    samples = [
        _bilinear_sample(flat[:, :, k], x, y, points)
        for k in range(flat.shape[2])
    ]
    return np.stack(samples, axis=1).reshape((len(points),) + arr.shape[2:])


def _normal_derivative_direct(
    log_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spline_points: np.ndarray,
    normal: np.ndarray,
    *,
    delta: float | None = None,
) -> np.ndarray:
    """Estimate ∂(log_field)/∂n at each path point via centered finite difference.

    Samples the scalar log_field at p ± δ·n̂ using bilinear interpolation, then
    divides by 2δ.  This avoids the compound error of first computing a 2-D
    gradient array with finite differences and then bilinearly sampling that
    gradient — instead the smoother scalar field is sampled directly.
    """
    xg = np.asarray(x, dtype=float)
    yg = np.asarray(y, dtype=float)
    if delta is None:
        dx = xg[1] - xg[0] if xg.size > 1 else 1.0
        dy = yg[1] - yg[0] if yg.size > 1 else 1.0
        delta = 0.5 * max(float(dx), float(dy))

    try:
        interp = RegularGridInterpolator(
            (yg, xg), np.asarray(log_field, dtype=float),
            method="cubic", bounds_error=False, fill_value=np.nan,
        )
    except ValueError:
        # scipy < 1.9 does not support method="cubic" for RegularGridInterpolator
        interp = RegularGridInterpolator(
            (yg, xg), np.asarray(log_field, dtype=float),
            method="linear", bounds_error=False, fill_value=np.nan,
        )

    def _sample(pts: np.ndarray) -> np.ndarray:
        yx = np.column_stack([
            np.clip(pts[:, 1], yg[0], yg[-1]),
            np.clip(pts[:, 0], xg[0], xg[-1]),
        ])
        return interp(yx)

    p_plus = spline_points + delta * normal
    p_minus = spline_points - delta * normal
    return (_sample(p_plus) - _sample(p_minus)) / (2.0 * delta)


def _savgol_gradient_2d(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    poly: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth gradient of a 2D field via separable Savitzky-Golay.

    field shape: (ny, nx), axis=1 is x, axis=0 is y.
    Returns (grad_x, grad_y) arrays of the same shape as field.
    """
    arr = np.asarray(field, dtype=float)
    ny, nx = arr.shape
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0

    min_dim = min(nx, ny)
    win = window if window % 2 == 1 else window + 1
    win = min(win, min_dim if min_dim % 2 == 1 else min_dim - 1)
    min_win = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    win = max(win, min_win)

    # ∂f/∂x: differentiate along axis=1 (x), smooth along axis=0 (y)
    grad_x = _savgol_filter(arr,    win, poly, deriv=1, delta=dx, axis=1)
    grad_x = _savgol_filter(grad_x, win, poly, deriv=0,           axis=0)

    # ∂f/∂y: smooth along axis=1 (x), differentiate along axis=0 (y)
    grad_y = _savgol_filter(arr,    win, poly, deriv=1, delta=dy, axis=0)
    grad_y = _savgol_filter(grad_y, win, poly, deriv=0,           axis=1)

    return np.asarray(grad_x, dtype=float), np.asarray(grad_y, dtype=float)


def _spline2d_normal_derivative(
    log_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spline_points: np.ndarray,
    normal: np.ndarray,
    *,
    smooth: float = 0.0,
) -> np.ndarray:
    """Estimate ∂(log_field)/∂n via 2D bicubic spline analytic derivative.

    Fits RectBivariateSpline to the field (s=smooth controls smoothing),
    then evaluates ∂f/∂x and ∂f/∂y analytically at path points.
    No δ-offset finite difference is needed.
    """
    xg = np.asarray(x, dtype=float)
    yg = np.asarray(y, dtype=float)
    field = np.asarray(log_field, dtype=float)
    pts_y = np.clip(spline_points[:, 1], yg[0], yg[-1])
    pts_x = np.clip(spline_points[:, 0], xg[0], xg[-1])

    spl = RectBivariateSpline(yg, xg, field, kx=3, ky=3, s=smooth)
    # RectBivariateSpline(y, x, z): dx=1 → ∂/∂y, dy=1 → ∂/∂x
    grad_x = spl.ev(pts_y, pts_x, dx=0, dy=1)
    grad_y = spl.ev(pts_y, pts_x, dx=1, dy=0)
    return np.asarray(grad_x * normal[:, 0] + grad_y * normal[:, 1], dtype=float)


def _mls_frenet_normal_derivative(
    log_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spline_points: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    *,
    ns: int = 2,
    nn: int = 2,
) -> np.ndarray:
    """Estimate ∂(log_field)/∂n via 2D Moving Least Squares in the Frenet-Serret frame.

    At each path point, samples a (2ns+1)×(2nn+1) patch of field values on a grid
    aligned with the local tangent t̂ and normal n̂.  Fits the degree-2 polynomial

        f(s, n) ≈ c1·s² + c2·n² + c3·s·n + c4·s + c5·n + c6

    via least squares.  The normal derivative at the centre (s=0, n=0) is c5.

    The design matrix depends only on the patch geometry, so its pseudoinverse is
    precomputed once.  All patch points across all path points are sampled in a
    single vectorised interpolator call.
    """
    xg = np.asarray(x, dtype=float)
    yg = np.asarray(y, dtype=float)
    dx = float(xg[1] - xg[0]) if xg.size > 1 else 1.0
    dy = float(yg[1] - yg[0]) if yg.size > 1 else 1.0
    delta = max(dx, dy)

    # Patch offsets in (s, n) frame — same for every path point
    s_offsets = np.arange(-ns, ns + 1) * delta
    n_offsets = np.arange(-nn, nn + 1) * delta
    SS, NN = np.meshgrid(s_offsets, n_offsets)
    SS_flat = SS.ravel()
    NN_flat = NN.ravel()
    m = len(SS_flat)

    # Design matrix A: degree-2 polynomial in (s, n), 6 coefficients
    A = np.column_stack([
        SS_flat ** 2, NN_flat ** 2, SS_flat * NN_flat,
        SS_flat, NN_flat, np.ones(m),
    ])
    # Precompute extraction weight vector for c5 = ∂f/∂n at (s=0, n=0)
    A_pinv = np.linalg.lstsq(A, np.eye(m), rcond=None)[0]  # shape (6, m)
    w = A_pinv[4, :]  # shape (m,)

    # Patch (x, y) coordinates for all path points — vectorised: shape (N, m)
    patch_x = (spline_points[:, 0:1]
                + SS_flat[None, :] * tangent[:, 0:1]
                + NN_flat[None, :] * normal[:, 0:1])
    patch_y = (spline_points[:, 1:2]
                + SS_flat[None, :] * tangent[:, 1:2]
                + NN_flat[None, :] * normal[:, 1:2])
    patch_x = np.clip(patch_x, xg[0], xg[-1])
    patch_y = np.clip(patch_y, yg[0], yg[-1])

    try:
        interp = RegularGridInterpolator(
            (yg, xg), np.asarray(log_field, dtype=float),
            method="cubic", bounds_error=False, fill_value=np.nan,
        )
    except ValueError:
        interp = RegularGridInterpolator(
            (yg, xg), np.asarray(log_field, dtype=float),
            method="linear", bounds_error=False, fill_value=np.nan,
        )

    yx_query = np.column_stack([patch_y.ravel(), patch_x.ravel()])
    f_vals = interp(yx_query).reshape(len(spline_points), m)  # shape (N, m)

    return np.asarray(f_vals @ w, dtype=float)


def _hessian_T_speed_normal_derivative(
    T: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spline_points: np.ndarray,
    *,
    pre_smooth_sigma: float = 0.0,
    stencil_window: int = 3,
    stencil_poly: int = 2,
) -> np.ndarray:
    """Estimate ∂(ln s)/∂n = -∂(ln c)/∂n along the path from the Hessian of T.

    Uses the identity  κ = T_nt / s  derived from the eikonal equation
    (∇(|∇T|²) = ∇(s²)  ⟹  (D²T)·t̂ = ∇s  ⟹  κ = n̂·(D²T)·t̂ / s),
    where κ = ∂(ln s)/∂n is the normal derivative of log-slowness, equal to
    the negative normal derivative of log-speed.

    All five partial derivatives (Tx, Ty, Txx, Tyy, Txy) are estimated with
    separable Savitzky-Golay filters, generalising the 3×3 Sobel approach:
    stencil_window=3, stencil_poly=2 reproduces the original Sobel result
    exactly; larger windows suppress the noise in the second derivatives.

    Parameters
    ----------
    T                : (ny, nx) travel-time field (inf for unreachable cells).
    x, y             : 1-D physical coordinate arrays (uniform spacing).
    spline_points    : (N, 2) path points as (x, y) at which to sample.
    pre_smooth_sigma : Gaussian sigma in grid cells applied to T before
                       differencing (0 = no smoothing).
    stencil_window   : SG window length (odd integer >= 3).  Larger values
                       give smoother second-derivative estimates at the cost
                       of spatial resolution.
    stencil_poly     : SG polynomial degree (>= 2, enforced internally).

    Returns
    -------
    kappa : (N,) array  = ∂(ln s)/∂n = -∂(ln c)/∂n  at each path point.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.signal import savgol_filter as _sg

    T = np.asarray(T, dtype=float)
    valid = np.isfinite(T)
    h_x = float(x[1] - x[0]) if len(x) > 1 else 1.0
    h_y = float(y[1] - y[0]) if len(y) > 1 else 1.0

    T_work = np.where(valid, T, 0.0)
    if pre_smooth_sigma > 0.0:
        T_num = gaussian_filter(T_work, pre_smooth_sigma)
        T_den = gaussian_filter(np.where(valid, 1.0, 0.0), pre_smooth_sigma)
        T_work = np.where(valid, T_num / np.maximum(T_den, 1e-12), 0.0)

    # Coerce window: odd, fits in the grid, and large enough for 2nd derivatives.
    poly = max(stencil_poly, 2)  # second derivatives need degree >= 2
    win = stencil_window if stencil_window % 2 == 1 else stencil_window + 1
    min_dim = min(T_work.shape)
    win = min(win, min_dim if min_dim % 2 == 1 else min_dim - 1)
    min_win = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    win = max(win, min_win)

    def sg(arr: np.ndarray, deriv: int, delta: float, axis: int) -> np.ndarray:
        return np.asarray(
            _sg(arr, win, poly, deriv=deriv, delta=float(delta), axis=axis, mode='nearest'),
            dtype=float,
        )

    # Each derivative: differentiate along its axis, then smooth perpendicularly.
    # For the cross term: differentiate in x first, then in y.
    Tx  = sg(sg(T_work, 1, h_x, 1), 0, 1.0, 0)
    Ty  = sg(sg(T_work, 1, h_y, 0), 0, 1.0, 1)
    Txx = sg(sg(T_work, 2, h_x, 1), 0, 1.0, 0)
    Tyy = sg(sg(T_work, 2, h_y, 0), 0, 1.0, 1)
    Txy = sg(sg(T_work, 1, h_x, 1), 1, h_y, 0)

    grad2 = Tx**2 + Ty**2

    # Mask only cells where T is not finite or the SG stencil spans the T=0
    # source boundary (T smaller than one stencil half-width of travel time).
    # Do NOT use a relative threshold: inside the FMM domain |∇T| = 1/speed
    # is legitimately small in high-speed regions, which are exactly the cells
    # the FP path passes through.
    t_boundary = float(np.nanpercentile(T[valid], 2.0)) if np.any(valid) else 0.0
    near_source = valid & (T <= max(t_boundary, 0.0))
    reliable = valid & ~near_source & (grad2 > 0.0)

    numerator = Tx * Ty * (Tyy - Txx) + (Tx**2 - Ty**2) * Txy
    kappa_field = np.where(reliable, numerator / np.where(reliable, grad2**1.5, 1.0), np.nan)

    return _bilinear_sample(kappa_field, x, y, spline_points)


def _travel_time_derivative_fields(
    T: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    pre_smooth_sigma: float = 0.0,
    stencil_window: int = 3,
    stencil_poly: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import savgol_filter as _sg

    T = np.asarray(T, dtype=float)
    valid = np.isfinite(T)
    h_x = float(x[1] - x[0]) if len(x) > 1 else 1.0
    h_y = float(y[1] - y[0]) if len(y) > 1 else 1.0

    T_work = np.where(valid, T, 0.0)
    if pre_smooth_sigma > 0.0:
        T_num = gaussian_filter(T_work, pre_smooth_sigma)
        T_den = gaussian_filter(np.where(valid, 1.0, 0.0), pre_smooth_sigma)
        T_work = np.where(valid, T_num / np.maximum(T_den, 1e-12), 0.0)

    poly = max(stencil_poly, 2)
    win = stencil_window if stencil_window % 2 == 1 else stencil_window + 1
    min_dim = min(T_work.shape)
    win = min(win, min_dim if min_dim % 2 == 1 else min_dim - 1)
    min_win = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    win = max(win, min_win)

    def sg(arr: np.ndarray, deriv: int, delta: float, axis: int) -> np.ndarray:
        return np.asarray(
            _sg(arr, win, poly, deriv=deriv, delta=float(delta), axis=axis, mode="nearest"),
            dtype=float,
        )

    Tx = sg(sg(T_work, 1, h_x, 1), 0, 1.0, 0)
    Ty = sg(sg(T_work, 1, h_y, 0), 0, 1.0, 1)
    Txx = sg(sg(T_work, 2, h_x, 1), 0, 1.0, 0)
    Tyy = sg(sg(T_work, 2, h_y, 0), 0, 1.0, 1)
    Txy = sg(sg(T_work, 1, h_x, 1), 1, h_y, 0)
    t_boundary = float(np.nanpercentile(T[valid], 2.0)) if np.any(valid) else 0.0
    near_source = valid & (T <= max(t_boundary, 0.0))
    mask = (
        valid
        & ~near_source
        & np.isfinite(Tx)
        & np.isfinite(Ty)
        & np.isfinite(Txx)
        & np.isfinite(Tyy)
        & np.isfinite(Txy)
        & ((Tx * Tx + Ty * Ty) > 0.0)
    )
    return (
        np.where(mask, Tx, np.nan),
        np.where(mask, Ty, np.nan),
        np.where(mask, Txx, np.nan),
        np.where(mask, Txy, np.nan),
        np.where(mask, Tyy, np.nan),
    )


def _travel_time_gradient_fields(
    T: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    pre_smooth_sigma: float = 0.0,
    stencil_window: int = 3,
    stencil_poly: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import savgol_filter as _sg

    T = np.asarray(T, dtype=float)
    valid = np.isfinite(T)
    h_x = float(x[1] - x[0]) if len(x) > 1 else 1.0
    h_y = float(y[1] - y[0]) if len(y) > 1 else 1.0

    T_work = np.where(valid, T, 0.0)
    if pre_smooth_sigma > 0.0:
        T_num = gaussian_filter(T_work, pre_smooth_sigma)
        T_den = gaussian_filter(np.where(valid, 1.0, 0.0), pre_smooth_sigma)
        T_work = np.where(valid, T_num / np.maximum(T_den, 1e-12), 0.0)

    poly = max(stencil_poly, 1)
    win = stencil_window if stencil_window % 2 == 1 else stencil_window + 1
    min_dim = min(T_work.shape)
    win = min(win, min_dim if min_dim % 2 == 1 else min_dim - 1)
    min_win = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    win = max(win, min_win)

    def sg(arr: np.ndarray, deriv: int, delta: float, axis: int) -> np.ndarray:
        return np.asarray(
            _sg(arr, win, poly, deriv=deriv, delta=float(delta), axis=axis, mode="nearest"),
            dtype=float,
        )

    Tx = sg(sg(T_work, 1, h_x, 1), 0, 1.0, 0)
    Ty = sg(sg(T_work, 1, h_y, 0), 0, 1.0, 1)
    t_boundary = float(np.nanpercentile(T[valid], 2.0)) if np.any(valid) else 0.0
    near_source = valid & (T <= max(t_boundary, 0.0))
    mask = (
        valid
        & ~near_source
        & np.isfinite(Tx)
        & np.isfinite(Ty)
        & ((Tx * Tx + Ty * Ty) > 0.0)
    )
    return np.where(mask, Tx, np.nan), np.where(mask, Ty, np.nan)


def _randers_dual_norm(M: np.ndarray, beta: np.ndarray, p: np.ndarray) -> float:
    M = np.asarray(M, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p = np.asarray(p, dtype=float)
    if M.shape != (2, 2) or beta.shape != (2,) or p.shape != (2,):
        return float("nan")
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(beta)) and np.all(np.isfinite(p))):
        return float("nan")
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return float("nan")
    lam = 1.0 - float(beta @ (Minv @ beta))
    if lam <= 1e-12:
        return float("nan")
    q = float(p @ (Minv @ p))
    r = float(beta @ (Minv @ p))
    disc = lam * q + r * r
    if disc < 0.0:
        return float("nan")
    return (math.sqrt(max(disc, 0.0)) + r) / lam


def _randers_dual_hamiltonian(M: np.ndarray, beta: np.ndarray, p: np.ndarray) -> float:
    F_star = _randers_dual_norm(M, beta, p)
    if not np.isfinite(F_star):
        return float("nan")
    return 0.5 * F_star * F_star


def _sample_fm_asr_metric(
    point: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    M_field: np.ndarray,
    beta_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(point, dtype=float).reshape(1, 2)
    M = _bilinear_sample_components(M_field, x, y, pts)[0]
    beta = _bilinear_sample_components(beta_field, x, y, pts)[0]
    return M, beta


def _bilinear_sample_components_with_gradient(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=float)
    if arr.ndim < 3:
        raise ValueError("field must have at least one component dimension")
    xg = np.asarray(x, dtype=float)
    yg = np.asarray(y, dtype=float)
    px = float(point[0])
    py = float(point[1])
    nx = xg.size
    ny = yg.size
    if nx < 2 or ny < 2:
        nan = np.full(arr.shape[2:], np.nan, dtype=float)
        return nan, nan.copy(), nan.copy()

    i = int(np.searchsorted(xg, px, side="right") - 1)
    j = int(np.searchsorted(yg, py, side="right") - 1)
    i = max(0, min(i, nx - 2))
    j = max(0, min(j, ny - 2))
    dx = float(xg[i + 1] - xg[i])
    dy = float(yg[j + 1] - yg[j])
    if dx == 0.0 or dy == 0.0:
        nan = np.full(arr.shape[2:], np.nan, dtype=float)
        return nan, nan.copy(), nan.copy()

    tx = (px - float(xg[i])) / dx
    ty = (py - float(yg[j])) / dy
    tx = min(1.0, max(0.0, tx))
    ty = min(1.0, max(0.0, ty))

    f00 = arr[j, i]
    f10 = arr[j, i + 1]
    f01 = arr[j + 1, i]
    f11 = arr[j + 1, i + 1]
    value = (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )
    grad_x = ((1.0 - ty) * (f10 - f00) + ty * (f11 - f01)) / dx
    grad_y = ((1.0 - tx) * (f01 - f00) + tx * (f11 - f10)) / dy
    return value, grad_x, grad_y


def _sample_fm_asr_metric_with_gradient(
    point: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    M_field: np.ndarray,
    beta_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M, M_x, M_y = _bilinear_sample_components_with_gradient(M_field, x, y, point)
    beta, beta_x, beta_y = _bilinear_sample_components_with_gradient(beta_field, x, y, point)
    return M, beta, M_x, M_y, beta_x, beta_y


def _fm_asr_hamiltonian_at(
    point: np.ndarray,
    p: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    M_field: np.ndarray,
    beta_field: np.ndarray,
) -> float:
    M, beta = _sample_fm_asr_metric(point, x, y, M_field, beta_field)
    return _randers_dual_hamiltonian(M, beta, p)


def _fm_asr_H_p(
    point: np.ndarray,
    p: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    M_field: np.ndarray,
    beta_field: np.ndarray,
) -> np.ndarray:
    M, beta = _sample_fm_asr_metric(point, x, y, M_field, beta_field)
    return _randers_squared_H_p(M, beta, p)


def _randers_dual_H_p(M: np.ndarray, beta: np.ndarray, p: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p = np.asarray(p, dtype=float)
    if M.shape != (2, 2) or beta.shape != (2,) or p.shape != (2,):
        return np.full(2, np.nan, dtype=float)
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(beta)) and np.all(np.isfinite(p))):
        return np.full(2, np.nan, dtype=float)
    try:
        A = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.full(2, np.nan, dtype=float)
    A_beta = A @ beta
    A_p = A @ p
    lam = 1.0 - float(beta @ A_beta)
    if lam <= 1e-12:
        return np.full(2, np.nan, dtype=float)
    q = float(p @ A_p)
    r = float(beta @ A_p)
    s2 = lam * q + r * r
    if s2 <= 1e-24:
        return np.full(2, np.nan, dtype=float)
    s = math.sqrt(s2)
    return ((lam * A_p + r * A_beta) / s + A_beta) / lam


def _randers_dual_H_pp(M: np.ndarray, beta: np.ndarray, p: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p = np.asarray(p, dtype=float)
    if M.shape != (2, 2) or beta.shape != (2,) or p.shape != (2,):
        return np.full((2, 2), np.nan, dtype=float)
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(beta)) and np.all(np.isfinite(p))):
        return np.full((2, 2), np.nan, dtype=float)
    try:
        A = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.full((2, 2), np.nan, dtype=float)
    A_beta = A @ beta
    A_p = A @ p
    lam = 1.0 - float(beta @ A_beta)
    if lam <= 1e-12:
        return np.full((2, 2), np.nan, dtype=float)
    q = float(p @ A_p)
    r = float(beta @ A_p)
    s2 = lam * q + r * r
    if s2 <= 1e-24:
        return np.full((2, 2), np.nan, dtype=float)
    s = math.sqrt(s2)
    n_vec = lam * A_p + r * A_beta
    B = lam * A + np.outer(A_beta, A_beta)
    return (B / s - np.outer(n_vec, n_vec) / (s * s * s)) / lam


def _randers_squared_H_p(M: np.ndarray, beta: np.ndarray, p: np.ndarray) -> np.ndarray:
    F_star = _randers_dual_norm(M, beta, p)
    F_p = _randers_dual_H_p(M, beta, p)
    if not np.isfinite(F_star) or not np.all(np.isfinite(F_p)):
        return np.full(2, np.nan, dtype=float)
    return F_star * F_p


def _randers_squared_H_pp(M: np.ndarray, beta: np.ndarray, p: np.ndarray) -> np.ndarray:
    F_star = _randers_dual_norm(M, beta, p)
    F_p = _randers_dual_H_p(M, beta, p)
    F_pp = _randers_dual_H_pp(M, beta, p)
    if not (np.isfinite(F_star) and np.all(np.isfinite(F_p)) and np.all(np.isfinite(F_pp))):
        return np.full((2, 2), np.nan, dtype=float)
    return F_star * F_pp + np.outer(F_p, F_p)


def _randers_squared_H_p_spatial_derivative(
    M: np.ndarray,
    beta: np.ndarray,
    p: np.ndarray,
    dM: np.ndarray,
    dbeta: np.ndarray,
) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p = np.asarray(p, dtype=float)
    dM = np.asarray(dM, dtype=float)
    dbeta = np.asarray(dbeta, dtype=float)
    if (
        M.shape != (2, 2)
        or beta.shape != (2,)
        or p.shape != (2,)
        or dM.shape != (2, 2)
        or dbeta.shape != (2,)
    ):
        return np.full(2, np.nan, dtype=float)
    if not (
        np.all(np.isfinite(M))
        and np.all(np.isfinite(beta))
        and np.all(np.isfinite(p))
        and np.all(np.isfinite(dM))
        and np.all(np.isfinite(dbeta))
    ):
        return np.full(2, np.nan, dtype=float)

    try:
        A = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.full(2, np.nan, dtype=float)

    dA = -A @ dM @ A
    c = A @ beta
    dc = dA @ beta + A @ dbeta
    u = A @ p
    du = dA @ p
    lam = 1.0 - float(beta @ c)
    dlam = -float(dbeta @ c + beta @ dc)
    if lam <= 1e-12:
        return np.full(2, np.nan, dtype=float)

    q = float(p @ u)
    dq = float(p @ du)
    r = float(beta @ u)
    dr = float(dbeta @ u + beta @ du)
    s2 = lam * q + r * r
    if s2 <= 1e-24:
        return np.full(2, np.nan, dtype=float)
    s = math.sqrt(s2)
    ds = (dlam * q + lam * dq + 2.0 * r * dr) / (2.0 * s)

    n_vec = lam * u + r * c
    dn = dlam * u + lam * du + dr * c + r * dc
    F_star = (s + r) / lam
    dF = ((ds + dr) * lam - (s + r) * dlam) / (lam * lam)
    F_p = (n_vec / s + c) / lam
    dF_p = ((dn * s - n_vec * ds) / (s * s) + dc) / lam - (n_vec / s + c) * dlam / (lam * lam)
    dH_p = dF * F_p + F_star * dF_p
    if not np.all(np.isfinite(dH_p)):
        return np.full(2, np.nan, dtype=float)
    return dH_p


def _randers_squared_H_spatial_derivative(
    M: np.ndarray,
    beta: np.ndarray,
    p: np.ndarray,
    dM: np.ndarray,
    dbeta: np.ndarray,
) -> float:
    M = np.asarray(M, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p = np.asarray(p, dtype=float)
    dM = np.asarray(dM, dtype=float)
    dbeta = np.asarray(dbeta, dtype=float)
    if (
        M.shape != (2, 2)
        or beta.shape != (2,)
        or p.shape != (2,)
        or dM.shape != (2, 2)
        or dbeta.shape != (2,)
    ):
        return float("nan")
    if not (
        np.all(np.isfinite(M))
        and np.all(np.isfinite(beta))
        and np.all(np.isfinite(p))
        and np.all(np.isfinite(dM))
        and np.all(np.isfinite(dbeta))
    ):
        return float("nan")
    try:
        A = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return float("nan")

    dA = -A @ dM @ A
    c = A @ beta
    dc = dA @ beta + A @ dbeta
    u = A @ p
    du = dA @ p
    lam = 1.0 - float(beta @ c)
    dlam = -float(dbeta @ c + beta @ dc)
    if lam <= 1e-12:
        return float("nan")
    q = float(p @ u)
    dq = float(p @ du)
    r = float(beta @ u)
    dr = float(dbeta @ u + beta @ du)
    s2 = lam * q + r * r
    if s2 <= 1e-24:
        return float("nan")
    s = math.sqrt(s2)
    ds = (dlam * q + lam * dq + 2.0 * r * dr) / (2.0 * s)
    F_star = (s + r) / lam
    dF = ((ds + dr) * lam - (s + r) * dlam) / (lam * lam)
    dH = F_star * dF
    return float(dH) if np.isfinite(dH) else float("nan")


def _point_inside_centered_difference(point: np.ndarray, x: np.ndarray, y: np.ndarray, delta: float) -> bool:
    return (
        float(x[0]) + delta <= float(point[0]) <= float(x[-1]) - delta
        and float(y[0]) + delta <= float(point[1]) <= float(y[-1]) - delta
    )


def _local_frenet_quadratic_T_derivatives(
    T: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spline_points: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    *,
    pre_smooth_sigma: float,
    ns: int = 3,
    nn: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate grad(T) and Hessian(T) from local quadratic fits near a path.

    The fit is performed in each sample's Frenet frame, so the derivative
    stencil follows the path instead of differentiating the full gridded
    viscosity solution globally.
    """
    T = np.asarray(T, dtype=float)
    valid = np.isfinite(T)
    xg = np.asarray(x, dtype=float)
    yg = np.asarray(y, dtype=float)
    pts = np.asarray(spline_points, dtype=float)
    tang = np.asarray(tangent, dtype=float)
    norm = np.asarray(normal, dtype=float)

    dx = float(xg[1] - xg[0]) if xg.size > 1 else 1.0
    dy = float(yg[1] - yg[0]) if yg.size > 1 else 1.0
    delta = max(abs(dx), abs(dy))

    T_work = np.where(valid, T, 0.0)
    if pre_smooth_sigma > 0.0:
        T_num = gaussian_filter(T_work, pre_smooth_sigma)
        T_den = gaussian_filter(np.where(valid, 1.0, 0.0), pre_smooth_sigma)
        T_work = np.where(valid, T_num / np.maximum(T_den, 1e-12), np.nan)
    else:
        T_work = np.where(valid, T, np.nan)

    try:
        interp = RegularGridInterpolator((yg, xg), T_work, method="cubic", bounds_error=False, fill_value=np.nan)
    except ValueError:
        interp = RegularGridInterpolator((yg, xg), T_work, method="linear", bounds_error=False, fill_value=np.nan)

    s_offsets = np.arange(-ns, ns + 1, dtype=float) * delta
    n_offsets = np.arange(-nn, nn + 1, dtype=float) * delta
    SS, NN = np.meshgrid(s_offsets, n_offsets)
    ss = SS.ravel()
    nnv = NN.ravel()
    design = np.column_stack([np.ones_like(ss), ss, nnv, 0.5 * ss * ss, ss * nnv, 0.5 * nnv * nnv])
    weights = np.exp(-0.5 * ((ss / max(ns * delta, delta)) ** 2 + (nnv / max(nn * delta, delta)) ** 2))
    sqrt_w = np.sqrt(weights)

    grad = np.full((len(pts), 2), np.nan, dtype=float)
    hess = np.full((len(pts), 2, 2), np.nan, dtype=float)
    cond = np.full(len(pts), np.nan, dtype=float)
    min_samples = 8

    for idx, point in enumerate(pts):
        if not (np.all(np.isfinite(point)) and np.all(np.isfinite(tang[idx])) and np.all(np.isfinite(norm[idx]))):
            continue
        patch_x = point[0] + ss * tang[idx, 0] + nnv * norm[idx, 0]
        patch_y = point[1] + ss * tang[idx, 1] + nnv * norm[idx, 1]
        vals = interp(np.column_stack([patch_y, patch_x]))
        inside = (
            (patch_x >= xg[0]) & (patch_x <= xg[-1])
            & (patch_y >= yg[0]) & (patch_y <= yg[-1])
            & np.isfinite(vals)
        )
        if int(np.count_nonzero(inside)) < min_samples:
            continue

        A = design[inside] * sqrt_w[inside, None]
        b = vals[inside] * sqrt_w[inside]
        try:
            c, _, _, singular = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if singular.size and singular[-1] > 0.0:
            cond[idx] = float(singular[0] / singular[-1])
        if not np.all(np.isfinite(c)) or (np.isfinite(cond[idx]) and cond[idx] > 1e10):
            continue

        local_grad = np.array([c[1], c[2]], dtype=float)
        local_hess = np.array([[c[3], c[4]], [c[4], c[5]]], dtype=float)
        frame = np.column_stack([tang[idx], norm[idx]])
        grad[idx] = frame @ local_grad
        hess[idx] = frame @ local_hess @ frame.T

    return grad, hess, cond


def _smooth_finite_series(values: np.ndarray, window: int, poly: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = vals.copy()
    finite = np.isfinite(vals)
    if int(np.count_nonzero(finite)) < max(poly + 2, 3):
        return out
    indices = np.arange(vals.size, dtype=float)
    filled = vals.copy()
    filled[~finite] = np.interp(indices[~finite], indices[finite], vals[finite])
    win = window if window % 2 == 1 else window + 1
    max_win = vals.size if vals.size % 2 == 1 else vals.size - 1
    win = min(win, max_win)
    min_win = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    win = min(max(win, min_win), max_win)
    if win < min_win:
        return out
    out[finite] = _savgol_filter(filled, win, poly, mode="interp")[finite]
    return out


def _compute_fm_asr_hamiltonian_curvature(
    *,
    t_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    M_field: np.ndarray,
    beta_field: np.ndarray,
    spline_points: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    pre_smooth_sigma: float,
    stencil_window: int,
    stencil_poly: int,
    smooth_window: int,
    smooth_poly: int,
    t_hessian_method: str = "grid_savgol",
    p_dot_method: str = "canonical",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(spline_points, dtype=float)
    if p_dot_method == "canonical" and t_hessian_method == "grid_savgol":
        Tx, Ty = _travel_time_gradient_fields(
            t_field,
            x,
            y,
            pre_smooth_sigma=pre_smooth_sigma,
            stencil_window=stencil_window,
            stencil_poly=stencil_poly,
        )
        grad_T = np.column_stack([
            _bilinear_sample(Tx, x, y, pts),
            _bilinear_sample(Ty, x, y, pts),
        ])
        H_T = np.full((len(pts), 2, 2), np.nan, dtype=float)
        fit_cond = np.where(np.all(np.isfinite(grad_T), axis=1), 1.0, np.nan)
    elif t_hessian_method == "grid_savgol":
        Tx, Ty, Txx, Txy, Tyy = _travel_time_derivative_fields(
            t_field,
            x,
            y,
            pre_smooth_sigma=pre_smooth_sigma,
            stencil_window=stencil_window,
            stencil_poly=stencil_poly,
        )
        grad_T = np.column_stack([
            _bilinear_sample(Tx, x, y, pts),
            _bilinear_sample(Ty, x, y, pts),
        ])
        H_T = np.stack(
            [
                np.stack([
                    _bilinear_sample(Txx, x, y, pts),
                    _bilinear_sample(Txy, x, y, pts),
                ], axis=1),
                np.stack([
                    _bilinear_sample(Txy, x, y, pts),
                    _bilinear_sample(Tyy, x, y, pts),
                ], axis=1),
            ],
            axis=1,
        )
        fit_cond = np.where(
            np.all(np.isfinite(grad_T), axis=1),
            1.0,
            np.nan,
        )
    elif t_hessian_method == "frenet_quadratic":
        win = stencil_window if stencil_window % 2 == 1 else stencil_window + 1
        half_window = max(2, (win - 1) // 2)
        grad_T, H_T, fit_cond = _local_frenet_quadratic_T_derivatives(
            t_field,
            x,
            y,
            pts,
            tangent,
            normal,
            pre_smooth_sigma=pre_smooth_sigma,
            ns=half_window,
            nn=half_window,
        )
    else:
        raise ValueError(f"Unsupported FM-ASR Hamiltonian T-Hessian method: {t_hessian_method!r}")

    h_grid = min(float(abs(x[1] - x[0])) if len(x) > 1 else 1.0, float(abs(y[1] - y[0])) if len(y) > 1 else 1.0)
    dx_eps = 1.0 * h_grid
    signed = np.full(len(pts), np.nan, dtype=float)
    magnitude = np.full(len(pts), np.nan, dtype=float)
    eikonal_residual = np.full(len(pts), np.nan, dtype=float)
    min_alignment = math.cos(math.radians(60.0))
    endpoint_pad = max(2, min(5, len(pts) // 20))

    for idx, point in enumerate(pts):
        if idx < endpoint_pad or idx >= len(pts) - endpoint_pad:
            continue
        p = grad_T[idx]
        if not (np.all(np.isfinite(point)) and np.all(np.isfinite(p)) and np.isfinite(fit_cond[idx])):
            continue
        if p_dot_method == "hessian_T" and not np.all(np.isfinite(H_T[idx])):
            continue
        if not _point_inside_centered_difference(point, x, y, dx_eps):
            continue

        M, beta, M_x, M_y, beta_x, beta_y = _sample_fm_asr_metric_with_gradient(
            point,
            x,
            y,
            M_field,
            beta_field,
        )
        F_star = _randers_dual_norm(M, beta, p)
        if not np.isfinite(F_star):
            continue
        eikonal_residual[idx] = F_star - 1.0

        v = _randers_squared_H_p(M, beta, p)
        speed = float(np.linalg.norm(v))
        if not np.all(np.isfinite(v)) or speed <= 1e-12:
            continue
        t_hat = v / speed
        alignment = float(tangent[idx] @ t_hat)
        if not np.isfinite(alignment) or alignment <= min_alignment:
            continue

        Hpp = _randers_squared_H_pp(M, beta, p)
        Hxp = np.column_stack(
            [
                _randers_squared_H_p_spatial_derivative(M, beta, p, M_x, beta_x),
                _randers_squared_H_p_spatial_derivative(M, beta, p, M_y, beta_y),
            ]
        )
        Hx = np.array(
            [
                _randers_squared_H_spatial_derivative(M, beta, p, M_x, beta_x),
                _randers_squared_H_spatial_derivative(M, beta, p, M_y, beta_y),
            ],
            dtype=float,
        )

        if not (np.all(np.isfinite(Hpp)) and np.all(np.isfinite(Hxp)) and np.all(np.isfinite(Hx))):
            continue
        if p_dot_method == "canonical":
            p_dot = -Hx
        elif p_dot_method == "hessian_T":
            p_dot = H_T[idx] @ v
        else:
            raise ValueError(f"Unsupported FM-ASR Hamiltonian p-dot method: {p_dot_method!r}")
        accel = Hxp @ v + Hpp @ p_dot
        projector = np.eye(2) - np.outer(t_hat, t_hat)
        k_vec = projector @ accel / (speed * speed)
        if not np.all(np.isfinite(k_vec)):
            continue
        magnitude[idx] = float(np.linalg.norm(k_vec))
        signed[idx] = float(k_vec @ normal[idx])

    smoothed = _smooth_finite_series(signed, smooth_window, max(smooth_poly, 2))
    return signed, magnitude, smoothed, eikonal_residual


def _build_path_balance_diagnostics(
    path: np.ndarray,
    *,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    logk_field: np.ndarray,
    log_grad_h_field: np.ndarray,
    log_v_field: np.ndarray,
    curvature_method: str,
    spline_mode: str,
    spline_smooth: float,
    spline_degree: int,
    circle_window: int,
    savgol_window: int = 11,
    savgol_poly: int = 3,
    normal_deriv_method: str = "direct",
    normal_deriv_spline2d_smooth: float = 0.0,
    normal_deriv_mls_ns: int = 2,
    normal_deriv_mls_nn: int = 2,
    normal_deriv_hessian_smooth: float = 0.0,
    normal_deriv_hessian_window: int = 3,
    normal_deriv_hessian_poly: int = 2,
    t_field: np.ndarray | None = None,
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
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
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

    if normal_deriv_method == "mls_frenet":
        dlogk_dn = _mls_frenet_normal_derivative(logk_field, x, y, spline_points, tangent, normal, ns=normal_deriv_mls_ns, nn=normal_deriv_mls_nn)
        dloggradh_dn = _mls_frenet_normal_derivative(log_grad_h_field, x, y, spline_points, tangent, normal, ns=normal_deriv_mls_ns, nn=normal_deriv_mls_nn)
        dlogv_dn = _mls_frenet_normal_derivative(log_v_field, x, y, spline_points, tangent, normal, ns=normal_deriv_mls_ns, nn=normal_deriv_mls_nn)
    elif normal_deriv_method == "spline2d":
        dlogk_dn = _spline2d_normal_derivative(logk_field, x, y, spline_points, normal, smooth=normal_deriv_spline2d_smooth)
        dloggradh_dn = _spline2d_normal_derivative(log_grad_h_field, x, y, spline_points, normal, smooth=normal_deriv_spline2d_smooth)
        dlogv_dn = _spline2d_normal_derivative(log_v_field, x, y, spline_points, normal, smooth=normal_deriv_spline2d_smooth)
    elif normal_deriv_method == "savgol2d":
        grad_logk_x, grad_logk_y = _savgol_gradient_2d(logk_field, x, y, savgol_window, savgol_poly)
        grad_loggradh_x, grad_loggradh_y = _savgol_gradient_2d(log_grad_h_field, x, y, savgol_window, savgol_poly)
        grad_logv_x, grad_logv_y = _savgol_gradient_2d(log_v_field, x, y, savgol_window, savgol_poly)
        dlogk_dn = np.einsum("ij,ij->i", np.column_stack([
            _bilinear_sample(grad_logk_x, x, y, spline_points),
            _bilinear_sample(grad_logk_y, x, y, spline_points),
        ]), normal)
        dloggradh_dn = np.einsum("ij,ij->i", np.column_stack([
            _bilinear_sample(grad_loggradh_x, x, y, spline_points),
            _bilinear_sample(grad_loggradh_y, x, y, spline_points),
        ]), normal)
        dlogv_dn = np.einsum("ij,ij->i", np.column_stack([
            _bilinear_sample(grad_logv_x, x, y, spline_points),
            _bilinear_sample(grad_logv_y, x, y, spline_points),
        ]), normal)
    elif normal_deriv_method == "hessian_T":
        # κ = T_nt/s  ⟹  ∂(ln s)/∂n = -∂(ln c)/∂n = kappa_T
        # Replace only the "primary" combined term for this path's eikonal field;
        # use the direct method for the individually decomposed components.
        kappa_T = (
            _hessian_T_speed_normal_derivative(
                t_field, x, y, spline_points,
                pre_smooth_sigma=normal_deriv_hessian_smooth,
                stencil_window=normal_deriv_hessian_window,
                stencil_poly=normal_deriv_hessian_poly,
            )
            if t_field is not None
            else np.full(len(spline_points), np.nan)
        )
        dlogk_dn_d    = _normal_derivative_direct(logk_field,       x, y, spline_points, normal)
        dloggradh_dn_d = _normal_derivative_direct(log_grad_h_field, x, y, spline_points, normal)
        dlogv_dn_d    = _normal_derivative_direct(log_v_field,      x, y, spline_points, normal)
        if label == "FP (FMM)":
            # speed = v = k·|∇h|  →  ∂(ln s)/∂n = -dlogv_dn
            dlogk_dn    = dlogk_dn_d
            dloggradh_dn = dloggradh_dn_d
            dlogv_dn    = -kappa_T
        else:
            # LRP: speed = k·dh_mean  →  ∂(ln s)/∂n = -dlogk_dn
            dlogk_dn    = -kappa_T
            dloggradh_dn = dloggradh_dn_d
            dlogv_dn    = dlogv_dn_d
    else:  # "direct"
        dlogk_dn = _normal_derivative_direct(logk_field, x, y, spline_points, normal)
        dloggradh_dn = _normal_derivative_direct(log_grad_h_field, x, y, spline_points, normal)
        dlogv_dn = _normal_derivative_direct(log_v_field, x, y, spline_points, normal)
        n_pts = len(spline_points)
        win = curvature_module._coerce_odd_window(savgol_window, n_pts)
        if win >= savgol_poly + 2:
            ds_path = length / (n_pts - 1) if n_pts > 1 else 1.0
            dlogk_dn = _savgol_filter(dlogk_dn, win, savgol_poly, deriv=0, delta=ds_path)
            dloggradh_dn = _savgol_filter(dloggradh_dn, win, savgol_poly, deriv=0, delta=ds_path)
            dlogv_dn = _savgol_filter(dlogv_dn, win, savgol_poly, deriv=0, delta=ds_path)

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
    curvature_savgol_window: int = 11,
    curvature_savgol_poly: int = 3,
    normal_deriv_method: str = "direct",
    normal_deriv_spline2d_smooth: float = 0.0,
    normal_deriv_mls_ns: int = 2,
    normal_deriv_mls_nn: int = 2,
    normal_deriv_hessian_smooth: float = 0.0,
    normal_deriv_hessian_window: int = 3,
    normal_deriv_hessian_poly: int = 2,
    fm_asr_hamiltonian_t_hessian: str = "grid_savgol",
    fm_asr_hamiltonian_p_dot: str = "canonical",
    ds: float = 1.0,
) -> tuple[PathBalanceDiagnostics, PathBalanceDiagnostics, PathBalanceDiagnostics | None, PathBalanceDiagnostics | None]:
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

    fp_diag = _build_path_balance_diagnostics(
        result.fp_fmm,
        label="FP (FMM)",
        x=x,
        y=y,
        logk_field=logk_smoothed,
        log_grad_h_field=log_grad_h,
        log_v_field=log_v,
        curvature_method=curvature_method,
        spline_mode=curvature_spline_mode,
        spline_smooth=curvature_spline_smooth,
        spline_degree=curvature_spline_degree,
        circle_window=curvature_circle_window,
        savgol_window=curvature_savgol_window,
        savgol_poly=curvature_savgol_poly,
        normal_deriv_method=normal_deriv_method,
        normal_deriv_spline2d_smooth=normal_deriv_spline2d_smooth,
        normal_deriv_mls_ns=normal_deriv_mls_ns,
        normal_deriv_mls_nn=normal_deriv_mls_nn,
        normal_deriv_hessian_smooth=normal_deriv_hessian_smooth,
        normal_deriv_hessian_window=normal_deriv_hessian_window,
        normal_deriv_hessian_poly=normal_deriv_hessian_poly,
        t_field=result.t_min,
        ds=ds,
    )
    lrp_diag = _build_path_balance_diagnostics(
        result.lrp_fmm,
        label="LRP (FMM)",
        x=x,
        y=y,
        logk_field=logk_smoothed,
        log_grad_h_field=log_grad_h,
        log_v_field=log_v,
        curvature_method=curvature_method,
        spline_mode=curvature_spline_mode,
        spline_smooth=curvature_spline_smooth,
        spline_degree=curvature_spline_degree,
        circle_window=curvature_circle_window,
        savgol_window=curvature_savgol_window,
        savgol_poly=curvature_savgol_poly,
        normal_deriv_method=normal_deriv_method,
        normal_deriv_spline2d_smooth=normal_deriv_spline2d_smooth,
        normal_deriv_mls_ns=normal_deriv_mls_ns,
        normal_deriv_mls_nn=normal_deriv_mls_nn,
        normal_deriv_hessian_smooth=normal_deriv_hessian_smooth,
        normal_deriv_hessian_window=normal_deriv_hessian_window,
        normal_deriv_hessian_poly=normal_deriv_hessian_poly,
        t_field=result.t_min_lrp,
        ds=ds,
    )
    fm_asr_diag: PathBalanceDiagnostics | None = None
    if (
        result.fp_fm_asr is not None
        and result.t_min_fm_asr is not None
        and result.fm_asr_M is not None
        and result.fm_asr_beta is not None
        and len(result.fp_fm_asr) >= 3
    ):
        try:
            fm_asr_diag = _build_path_balance_diagnostics(
                result.fp_fm_asr,
                label="FP (FM-ASR)",
                x=x,
                y=y,
                logk_field=logk_smoothed,
                log_grad_h_field=log_grad_h,
                log_v_field=log_v,
                curvature_method=curvature_method,
                spline_mode=curvature_spline_mode,
                spline_smooth=curvature_spline_smooth,
                spline_degree=curvature_spline_degree,
                circle_window=curvature_circle_window,
                savgol_window=curvature_savgol_window,
                savgol_poly=curvature_savgol_poly,
                normal_deriv_method="direct",
                ds=ds,
            )
            signed, magnitude, smoothed, eikonal_residual = _compute_fm_asr_hamiltonian_curvature(
                t_field=result.t_min_fm_asr,
                x=x,
                y=y,
                M_field=result.fm_asr_M,
                beta_field=result.fm_asr_beta,
                spline_points=fm_asr_diag.spline_points,
                tangent=fm_asr_diag.tangent,
                normal=fm_asr_diag.normal,
                pre_smooth_sigma=normal_deriv_hessian_smooth,
                stencil_window=normal_deriv_hessian_window,
                stencil_poly=normal_deriv_hessian_poly,
                smooth_window=curvature_savgol_window,
                smooth_poly=curvature_savgol_poly,
                t_hessian_method=fm_asr_hamiltonian_t_hessian,
                p_dot_method=fm_asr_hamiltonian_p_dot,
            )
            fm_asr_diag.hamilton_curvature = signed
            fm_asr_diag.hamilton_curvature_magnitude = magnitude
            fm_asr_diag.hamilton_curvature_smoothed = smoothed
            fm_asr_diag.hamilton_eikonal_residual = eikonal_residual
        except Exception as exc:
            print(f"Warning: FM-ASR Hamiltonian curvature failed; omitting FM-ASR curvature panel: {exc}")
            fm_asr_diag = None
    streamline_diag: PathBalanceDiagnostics | None = None
    if result.fp is not None and len(result.fp) >= 3:
        try:
            streamline_diag = _build_path_balance_diagnostics(
                result.fp,
                label="FP (Streamline)",
                x=x,
                y=y,
                logk_field=logk_smoothed,
                log_grad_h_field=log_grad_h,
                log_v_field=log_v,
                curvature_method=curvature_method,
                spline_mode=curvature_spline_mode,
                spline_smooth=curvature_spline_smooth,
                spline_degree=curvature_spline_degree,
                circle_window=curvature_circle_window,
                savgol_window=curvature_savgol_window,
                savgol_poly=curvature_savgol_poly,
                normal_deriv_method="direct",
                ds=ds,
            )
        except Exception as exc:
            print(f"Warning: Streamline FP curvature failed; omitting streamline curvature overlay: {exc}")
            streamline_diag = None
    return fp_diag, lrp_diag, fm_asr_diag, streamline_diag


def plot_paths_over_field(
    field: np.ndarray,
    *,
    cmap: str,
    colorbar_label: str,
    result: PathSolveResult,
    i_y: float,
    fp_diag: PathBalanceDiagnostics | None = None,
    lrp_diag: PathBalanceDiagnostics | None = None,
    log_scale: bool = False,
):
    norm = LogNorm() if log_scale else None
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(field, origin="lower", norm=norm, cmap=cmap, interpolation="nearest", extent=_normalized_extent(result.x, result.y, i_y))
    _plot_common_overlays(ax, result, i_y, fp_diag=fp_diag, lrp_diag=lrp_diag)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if isinstance(im.norm, LogNorm):
        cbar.locator = LogLocator(base=10)
        cbar.formatter = LogFormatterMathtext(base=10)
        cbar.update_ticks()
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
    colorbar_label: str = r"$\| \mathbf{v} \|/U$",
):
    fig, ax = plt.subplots(figsize=[5, 4])
    im = ax.imshow(
        speed_field,
        origin="lower",
        cmap=cmc.batlowW_r,
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
        label_step = max(1, len(result.linearized_fp_paths) // 10)
        for order_idx, (iter_idx, path) in enumerate(zip(path_indices, result.linearized_fp_paths), start=1):
            color = progression_cmap((order_idx - 1) / denom)
            # label = f"FP^{iter_idx}" if order_idx in {1, len(result.linearized_fp_paths)} else None
            label = f"FP^{iter_idx}" if order_idx in range(1, len(result.linearized_fp_paths) + 1, label_step) else None
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
    *,
    fm_asr_diag: PathBalanceDiagnostics | None = None,
    streamline_diag: PathBalanceDiagnostics | None = None,
    i_y: float = 1.0,
) -> Any:
    has_fm_asr_path = (
        fm_asr_diag is not None
        and np.any(np.isfinite(fm_asr_diag.curvature))
    )
    has_fm_asr_hamiltonian = (
        fm_asr_diag is not None
        and fm_asr_diag.hamilton_curvature is not None
        and np.any(np.isfinite(fm_asr_diag.hamilton_curvature))
    )
    nrows = 4 if has_fm_asr_path else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(6, 2.0 * nrows), sharex=True)
    lw = 1#LINEWIDTHS
    alpha = 0.5

    def _pearson_r(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return float("nan")
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    def _pearson_r_on_s(sa, a, sb, b):
        sa = np.asarray(sa, dtype=float)
        sb = np.asarray(sb, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask_a = np.isfinite(sa) & np.isfinite(a)
        mask_b = np.isfinite(sb) & np.isfinite(b)
        if mask_a.sum() < 2 or mask_b.sum() < 2:
            return float("nan")
        lo = max(float(np.nanmin(sa[mask_a])), float(np.nanmin(sb[mask_b])))
        hi = min(float(np.nanmax(sa[mask_a])), float(np.nanmax(sb[mask_b])))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float("nan")
        n = int(min(mask_a.sum(), mask_b.sum()))
        if n < 2:
            return float("nan")
        grid = np.linspace(lo, hi, n)
        a_i = np.interp(grid, sa[mask_a], a[mask_a])
        b_i = np.interp(grid, sb[mask_b], b[mask_b])
        return _pearson_r(a_i, b_i)

    r_fp    = _pearson_r(fp_diag.curvature,  -fp_diag.dlogv_dn)
    r_lrp_k = _pearson_r(lrp_diag.curvature, -lrp_diag.dlogk_dn)
    r_lrp_v = _pearson_r(lrp_diag.curvature, -lrp_diag.dlogv_dn)
    r_fm_asr = (
        _pearson_r(fm_asr_diag.curvature, fm_asr_diag.hamilton_curvature_smoothed)
        if has_fm_asr_hamiltonian and fm_asr_diag is not None and fm_asr_diag.hamilton_curvature_smoothed is not None
        else float("nan")
    )
    r_fm_stream = (
        _pearson_r_on_s(
            fm_asr_diag.s_norm,
            fm_asr_diag.curvature,
            streamline_diag.s_norm,
            streamline_diag.curvature,
        )
        if has_fm_asr_path and fm_asr_diag is not None and streamline_diag is not None
        else float("nan")
    )

    axes[0].plot(
        fp_diag.s_norm,
        fp_diag.curvature * i_y,
        color=COLORS["fpFMM"],
        linewidth=lw,
        label=r"$\kappa^*_{\mathcal{E}}$",
    )
    axes[0].plot(
        fp_diag.s_norm,
        -fp_diag.dlogv_dn * i_y,
        color=COLORS["fpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$-\partial_{\mathbf{n}} \ln K -\partial_{\mathbf{n}} \ln \| \nabla{h} \|$ along $\gamma^{\!\star}_{\mathcal{E}}$",
    )
    axes[0].axhline(0.0, color="0.7", linewidth=1.0)
    axes[0].set_ylabel(r"FP $[\times I_Y]$")
    axes[0].legend(fontsize="8", loc="upper left", facecolor="white", edgecolor="black", framealpha=1.0)
    axes[0].text(
        0.97, 0.97,
        rf"$r = {r_fp:.2f}$",
        transform=axes[0].transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1.0),
    )

    # axes[1].plot(
    #     fp_diag.s_norm,
    #     fp_diag.curvature_circle * i_y,
    #     color=COLORS["fpFMM"],
    #     linewidth=lw,
    #     label=r"$\kappa_{\mathrm{circle}}$ FP (FMM)",
    # )
    # axes[1].plot(
    #     fp_diag.s_norm,
    #     -fp_diag.dlogv_dn * i_y,
    #     color=COLORS["fpFMM"],
    #     alpha=alpha,
    #     linewidth=lw,
    #     label=r"$-\partial_N \ln |v|$ on FP (FMM)",
    # )
    # axes[1].axhline(0.0, color="0.7", linewidth=1.0)
    # axes[1].set_ylabel("FP Circle")
    # axes[1].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[1].plot(
        lrp_diag.s_norm,
        lrp_diag.curvature * i_y,
        color=COLORS["lrpFMM"],
        linewidth=lw,
        label=r"$\kappa^*_{\mathcal{R}}$",
    )
    axes[1].plot(
        lrp_diag.s_norm,
        -lrp_diag.dlogk_dn * i_y,
        color=COLORS["lrpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$-\partial_{\mathbf{n}} \ln K$ along $\gamma^{\!\star}_{\mathcal{R}}$",
    )
    axes[1].plot(
        lrp_diag.s_norm,
        (- lrp_diag.dlogk_dn - lrp_diag.dloggradh_dn) * i_y,
        color=COLORS["lrpFMM"],
        linestyle=":",
        linewidth=lw,
        label=r"$- \partial_{\mathbf{n}} \ln K -\partial_{\mathbf{n}} \ln \| \nabla{h} \|$ along $\gamma^{\!\star}_{\mathcal{R}}$",
    )
    axes[1].axhline(0.0, color="0.7", linewidth=1.0)
    axes[1].set_ylabel(r"LRP $[\times I_Y]$")
    axes[1].legend(fontsize="8", loc="upper left", facecolor="white", edgecolor="black", framealpha=1.0)
    axes[1].text(
        0.98, 0.97,
        rf"$r(\kappa,\,-\partial_n\ln K) = {r_lrp_k:.2f}$" + "\n" +
        rf"$r(\kappa,\,-\partial_n\ln v) = {r_lrp_v:.2f}$",
        transform=axes[1].transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1.0),
    )

    # axes[3].plot(
    #     lrp_diag.s_norm,
    #     lrp_diag.curvature_circle * i_y,
    #     color=COLORS["lrpFMM"],
    #     linewidth=lw,
    #     label=r"$\kappa_{\mathrm{circle}}$ LRP (FMM)",
    # )
    # axes[3].plot(
    #     lrp_diag.s_norm,
    #     -lrp_diag.dlogk_dn * i_y,
    #     color=COLORS["lrpFMM"],
    #     alpha=alpha,
    #     linewidth=lw,
    #     label=r"$-\partial_N \ln K$ on LRP (FMM)",
    # )
    # axes[3].axhline(0.0, color="0.7", linewidth=1.0)
    # axes[3].set_ylabel("LRP Circle")
    # axes[3].legend(fontsize="8", loc="best", facecolor="white", edgecolor="black", framealpha=1.0)

    axes[2].plot(
        fp_diag.s_norm,
        fp_diag.residual * i_y,
        color=COLORS["fpFMM"],
        linewidth=lw,
        label=r"$\kappa^*_{\mathcal{E}} + \partial_{\mathbf{n}} \ln K + \partial_{\mathbf{n}} \ln \| \nabla{h} \|$ along $\gamma^{\!\star}_{\mathcal{E}}$",
    )
    axes[2].plot(
        lrp_diag.s_norm,
        lrp_diag.residual * i_y,
        color=COLORS["lrpFMM"],
        alpha=alpha,
        linewidth=lw,
        label=r"$\kappa^*_{\mathcal{R}} + \partial_{\mathbf{n}} \ln K$ along $\gamma^{\!\star}_{\mathcal{R}}$",
    )
    # axes[2].plot(
    #     lrp_diag.s_norm,
    #     (lrp_diag.curvature + lrp_diag.dlogv_dn) * i_y,
    #     color=COLORS["lrpFMM"],
    #     linestyle="--",
    #     linewidth=lw,
    #     label=r"$R_{\mathrm{LRP},v}=\kappa + \partial_N \ln |v|$",
    # )
    axes[2].plot(
        lrp_diag.s_norm,
        (lrp_diag.curvature + lrp_diag.dlogk_dn + lrp_diag.dloggradh_dn) * i_y,
        color=COLORS["lrpFMM"],
        linestyle=":",
        linewidth=lw,
        label=r"$\kappa^*_{\mathcal{R}} + \partial_{\mathbf{n}} \ln K + \partial_{\mathbf{n}} \ln \| \nabla{h} \|$ along $\gamma^{\!\star}_{\mathcal{R}}$",
    )
    axes[2].axhline(0.0, color="0.7", linewidth=1.0)
    axes[2].set_xlabel(r"Normalized arclength $\ell/L$")
    axes[2].set_ylabel(r"Residual $[\times I_Y]$")
    axes[2].legend(fontsize="8", loc="upper left", facecolor="white", edgecolor="black", framealpha=1.0)

    if has_fm_asr_path and fm_asr_diag is not None:
        axes[3].plot(
            fm_asr_diag.s_norm,
            fm_asr_diag.curvature * i_y,
            color=COLORS["fpFmAsr"],
            linewidth=lw,
            label=r"$\kappa^*_{\mathcal{F}}$",
        )
        if streamline_diag is not None:
            axes[3].plot(
                streamline_diag.s_norm,
                streamline_diag.curvature * i_y,
                color=COLORS["fp"],
                alpha=0.85,
                linestyle=LINESTYLES["fp"],
                linewidth=lw,
                label=r"$\kappa^*_{\mathcal{S}}$",
            )
        if has_fm_asr_hamiltonian and fm_asr_diag.hamilton_curvature is not None:
            axes[3].plot(
                fm_asr_diag.s_norm,
                fm_asr_diag.hamilton_curvature * i_y,
                color=COLORS["fpFmAsr"],
                alpha=0.25,
                linestyle=LINESTYLES["fpFmAsr"],
                linewidth=lw,
                label=r"$\kappa_{\mathcal{F},H}$ raw",
            )
        if has_fm_asr_hamiltonian and fm_asr_diag.hamilton_curvature_smoothed is not None:
            axes[3].plot(
                fm_asr_diag.s_norm,
                fm_asr_diag.hamilton_curvature_smoothed * i_y,
                color=COLORS["fpFmAsr"],
                alpha=alpha,
                linestyle=LINESTYLES["fpFmAsr"],
                linewidth=lw,
                label=r"$\kappa_{\mathcal{F},H}$ smoothed",
            )
        axes[3].axhline(0.0, color="0.7", linewidth=1.0)
        axes[3].set_xlabel(r"Normalized arclength $\ell/L$")
        axes[3].set_ylabel(r"FM-ASR $[\times I_Y]$")
        axes[3].legend(fontsize="8", loc="upper left", facecolor="white", edgecolor="black", framealpha=1.0)
        axes[3].text(
            0.97, 0.97,
            rf"$r(\kappa_\mathcal{{F}}^*,\kappa_{{\mathcal{{F}},H}}) = {r_fm_asr:.2f}$" + "\n" +
            rf"$r(\kappa_\mathcal{{F}}^*,\kappa_\mathcal{{S}}^*) = {r_fm_stream:.2f}$",
            transform=axes[3].transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1.0),
        )

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, color="0.9", linewidth=0.8)
    
    ymin = min(ax.get_ylim()[0] for ax in axes.flat)
    ymax = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    return fig




def _plot_snapshot_particles_line(
    ax: Any,
    x_points: np.ndarray,
    y_points: np.ndarray,
    *,
    i_y: float,
    color: Any,
    linewidth: float,
    linestyle: str = "--",
    halo_foreground: str = "white",
    halo_scale: float = 1.5,
    zorder: int = 1,
) -> bool:
    mask = np.isfinite(x_points) & np.isfinite(y_points)
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return False
    x_line = np.asarray(x_points[idx], dtype=float)
    y_line = np.asarray(y_points[idx], dtype=float)
    line, = ax.plot(
        x_line / i_y,
        y_line / i_y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )
    line.set_path_effects([
        pe.Stroke(linewidth=halo_scale * linewidth, foreground=halo_foreground),
        pe.Normal(),
    ])
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")
    return True


def _plot_snapshot_density_contour(
    ax: Any,
    x_points: np.ndarray,
    y_points: np.ndarray,
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    i_y: float,
    color: Any,
    linestyle: Any,
    linewidth: float,
    quantile: float = 0.70,
    smooth_sigma: float = 1.0,
    alpha_max: float = 0.01,
    zorder: int = 1,
) -> bool:
    del linestyle, linewidth
    xg = np.asarray(x_grid, dtype=float)
    yg = np.asarray(y_grid, dtype=float)
    x_edges = build_face_coordinates_from_centers(xg)
    y_edges = build_face_coordinates_from_centers(yg)

    x_arr = np.asarray(x_points, dtype=float)
    y_arr = np.asarray(y_points, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    in_domain = (
        finite
        & (x_arr >= x_edges[0])
        & (x_arr <= x_edges[-1])
        & (y_arr >= y_edges[0])
        & (y_arr <= y_edges[-1])
    )
    if int(np.count_nonzero(in_domain)) < 1:
        return False

    hist, _x_edges, _y_edges = np.histogram2d(
        x_arr[in_domain],
        y_arr[in_domain],
        bins=[x_edges, y_edges],
    )
    density = hist.T.astype(float)
    if smooth_sigma > 0.0:
        density = gaussian_filter(density, sigma=float(smooth_sigma), mode="nearest")
    positive = density[density > 0.0]
    if positive.size == 0:
        return False

    density_max = float(np.nanmax(density))
    if not np.isfinite(density_max) or density_max <= 0.0:
        return False
    floor = float(np.quantile(positive, float(quantile)))
    if not np.isfinite(floor) or floor <= 0.0 or floor >= density_max:
        floor = 0.0
    density_plot = np.clip(
        (density - floor) / max(density_max - floor, np.finfo(float).eps),
        0.0,
        1.0,
    )
    if not np.any(density_plot > 0.0):
        density_plot = np.where(density > 0.0, density / density_max, 0.0)
    if not np.any(density_plot > 0.0):
        return False

    rgba = to_rgba(color)
    plume_alpha = float(alpha_max) * (rgba[3] if rgba[3] < 1.0 else 1.0)
    plume_cmap = LinearSegmentedColormap.from_list(
        "snapshot_plume",
        [(rgba[0], rgba[1], rgba[2], 0.0), (rgba[0], rgba[1], rgba[2], plume_alpha)],
        N=256,
    )
    ax.pcolormesh(
        x_edges / i_y,
        y_edges / i_y,
        density_plot,
        cmap=plume_cmap,
        shading="flat",
        rasterized=True,
        zorder=zorder,
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:]) / i_y
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:]) / i_y
    contour_level = min(0.35, 0.5 * float(np.nanmax(density_plot)))
    if contour_level > 0.0 and np.nanmax(density_plot) > contour_level:
        ax.contour(
            x_centers,
            y_centers,
            density_plot,
            levels=[contour_level],
            colors=[color],
            linewidths=max(1.0, float(LINEWIDTHS)),
            linestyles=["-"],
            zorder=zorder,
        )
    ax.scatter(
        x_arr[in_domain] / i_y,
        y_arr[in_domain] / i_y,
        s=8.0,
        color=color,
        alpha=min(0.65, max(0.25, plume_alpha)),
        linewidths=0,
        rasterized=True,
        zorder=zorder,
    )
    return True


def _plot_isochrone_contour(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    level: float,
    *,
    color: Any,
    linewidth: float,
    linestyle: str | tuple = "-",
    halo_foreground: str = "white",
    halo_scale: float = 1.2,
    zorder: int = 1,
) -> None:
    cs = ax.contour(
        x,
        y,
        field,
        levels=[level],
        linewidths=linewidth,
        colors=[color],
        linestyles=[linestyle],
        zorder=zorder,
    )
    effect = [
        pe.Stroke(linewidth=halo_scale * linewidth, foreground=halo_foreground),
        pe.Normal(),
    ]
    if hasattr(cs, "collections"):
        for collection in cs.collections:
            collection.set_path_effects(effect)
    else:
        cs.set_path_effects(effect)


def plot_isochrones_with_plume(
    logk_field: np.ndarray,
    t_min_dimless: np.ndarray,
    t_min_lrp_dimless: np.ndarray,
    snap_fractions: np.ndarray,
    *,
    result: PathSolveResult,
    i_y: float,
    fp_arrival_time_dimless: float | None = None,
    pypar2_arrival_time_dimless: float | None = None,
    t_min_fsm_dimless: np.ndarray | None = None,
    t_min_fm_asr_dimless: np.ndarray | None = None,
) -> plt.Figure | None:
    """Multi-panel figure: one panel per snapshot time showing isochrone + particle plume.

    Each panel shows:
    - logK field as a semi-transparent background
    - FP (FMM) isochrone contour at fraction f of its own first arrival time
    - LRP isochrone contour at fraction f of its own first arrival time
    - FSM isochrone contour at fraction f of its own first arrival time (when available)
    - FM-ASR isochrone contour at fraction f of its own first arrival time (when available)
    - FP particle snapshot as dashed polyline (raw seed-ordered positions, white halo)
    - FP (FMM), LRP (FMM), FSM, and FM-ASR paths as thin overlays
    - Source line

    Parameters
    ----------
    logk_field : ndarray (ny, nx)
        2-D logK field.
    t_min_dimless : ndarray (ny, nx)
        FP travel-time field divided by t_ref (NaN where unreachable).
    t_min_lrp_dimless : ndarray (ny, nx)
        LRP travel-time field divided by t_ref (NaN where unreachable).
    snap_fractions : ndarray (T,)
        Fractions of first arrival time for each snapshot panel (e.g. 0.25, 0.5, 0.75, 1.0).
        Each isochrone is drawn at ``f * t_fp_arrival``, ``f * t_lrp_arrival``, and
        ``f * t_fsm_arrival`` respectively, so each contour is relative to its own first arrival.
    result : PathSolveResult
    i_y : float
        Characteristic length for axis normalisation.
    t_min_fsm_dimless : ndarray (ny, nx) or None
        FSM travel-time field divided by t_ref (NaN where unreachable). When None,
        no FSM isochrone or path is drawn.
    t_min_fm_asr_dimless : ndarray (ny, nx) or None
        FM-ASR travel-time field divided by t_ref (NaN where unreachable). When None,
        no FM-ASR isochrone or path is drawn.

    Returns None if result.snapshots is None or has no snapshots.
    """
    snaps = result.snapshots
    pypar2_snaps = result.pypar2_snapshots
    if snaps is None and pypar2_snaps is None:
        return None

    n_snaps = len(snaps.times) if snaps is not None else len(pypar2_snaps.times)
    if n_snaps == 0:
        return None

    ncols = min(4, n_snaps)
    nrows = int(np.ceil(n_snaps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    extent = _normalized_extent(result.x, result.y, i_y)
    x_norm = result.x / i_y
    y_norm = result.y / i_y

    # First arrival time at target for FP, LRP, and FSM fields
    ix_target = int(np.argmin(np.abs(result.x - result.target_xy[0, 0])))
    t_fp_pt_arrival = (
        float(fp_arrival_time_dimless)
        if fp_arrival_time_dimless is not None and np.isfinite(fp_arrival_time_dimless)
        else np.nan
    )
    t_fp_arrival = float(np.nanmin(t_min_dimless[:, ix_target]))
    t_lrp_arrival = float(np.nanmin(t_min_lrp_dimless[:, ix_target]))

    t_valid = t_min_dimless[np.isfinite(t_min_dimless)]
    t_min_val = float(np.nanmin(t_valid)) if t_valid.size > 0 else 0.0
    t_max_val = float(np.nanmax(t_valid)) if t_valid.size > 0 else 1.0

    t_lrp_valid = t_min_lrp_dimless[np.isfinite(t_min_lrp_dimless)]
    t_lrp_min = float(np.nanmin(t_lrp_valid)) if t_lrp_valid.size > 0 else 0.0
    t_lrp_max = float(np.nanmax(t_lrp_valid)) if t_lrp_valid.size > 0 else 1.0

    if t_min_fsm_dimless is not None:
        t_fsm_arrival = float(np.nanmin(t_min_fsm_dimless[:, ix_target]))
        t_fsm_valid = t_min_fsm_dimless[np.isfinite(t_min_fsm_dimless)]
        t_fsm_min = float(np.nanmin(t_fsm_valid)) if t_fsm_valid.size > 0 else 0.0
        t_fsm_max = float(np.nanmax(t_fsm_valid)) if t_fsm_valid.size > 0 else 1.0
    else:
        t_fsm_arrival = t_fsm_min = t_fsm_max = 0.0

    if t_min_fm_asr_dimless is not None:
        t_fm_asr_arrival = float(np.nanmin(t_min_fm_asr_dimless[:, ix_target]))
        t_fm_asr_valid = t_min_fm_asr_dimless[np.isfinite(t_min_fm_asr_dimless)]
        t_fm_asr_min = float(np.nanmin(t_fm_asr_valid)) if t_fm_asr_valid.size > 0 else 0.0
        t_fm_asr_max = float(np.nanmax(t_fm_asr_valid)) if t_fm_asr_valid.size > 0 else 1.0
    else:
        t_fm_asr_arrival = t_fm_asr_min = t_fm_asr_max = 0.0

    from matplotlib.lines import Line2D
    
    im_ref = None
    for j in range(n_snaps):
        ax = axes_flat[j]

        # FP path
        line, = ax.plot(
            result.fp[:, 0] / i_y, result.fp[:, 1] / i_y,
            color=COLORS["fp"], linestyle=LINESTYLES["fp"],
            linewidth=LINEWIDTHS, zorder=15
        )
        line.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 1, foreground="black"), pe.Normal()])
        line.set_solid_capstyle("round")
        line.set_dash_capstyle("round")

        # FP (FMM) path
        line, = ax.plot(
            result.fp_fmm[:, 0] / i_y, result.fp_fmm[:, 1] / i_y,
            color=COLORS["fpFMM"], linestyle=LINESTYLES["fpFMM"],
            linewidth=LINEWIDTHS, zorder=14
        )
        line.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 1, foreground="black"), pe.Normal()])
        line.set_solid_capstyle("round")
        line.set_dash_capstyle("round")

        # LRP (FMM) path
        line, = ax.plot(
            result.lrp_fmm[:, 0] / i_y, result.lrp_fmm[:, 1] / i_y,
            color=COLORS["lrpFMM"], linestyle=LINESTYLES["lrpFMM"],
            linewidth=LINEWIDTHS, zorder=13
        )
        line.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 1, foreground="black"), pe.Normal()])
        line.set_solid_capstyle("round")
        line.set_dash_capstyle("round")

        # # FSM path
        # if result.fp_fsm is not None and len(result.fp_fsm) >= 2:
        #     line, = ax.plot(
        #         result.fp_fsm[:, 0] / i_y, result.fp_fsm[:, 1] / i_y,
        #         color=COLORS["fpFSM"], linestyle=LINESTYLES["fpFSM"],
        #         linewidth=LINEWIDTHS,
        #     )
        #     line.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 2, foreground="black"), pe.Normal()])

        if result.fp_fm_asr is not None and len(result.fp_fm_asr) >= 2:
            line, = ax.plot(
                result.fp_fm_asr[:, 0] / i_y, result.fp_fm_asr[:, 1] / i_y,
                color=COLORS["fpFmAsr"], linestyle=LINESTYLES["fpFmAsr"],
                linewidth=LINEWIDTHS, zorder=12
            )
            line.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 1, foreground="black"), pe.Normal()])
            line.set_solid_capstyle("round")
            line.set_dash_capstyle("round")

        if result.fp_pypar2 is not None and len(result.fp_pypar2) >= 2:
            line, = ax.plot(
                result.fp_pypar2[:, 0] / i_y, result.fp_pypar2[:, 1] / i_y,
                color=COLORS["fpPypar2"], linestyle=LINESTYLES["fpPypar2"],
                linewidth=LINEWIDTHS, zorder=11
            )
            line.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 1, foreground="black"), pe.Normal()])
            line.set_solid_capstyle("round")
            line.set_dash_capstyle("round")

        # Source line
        ax.plot(
            result.source_xy[:, 0] / i_y, result.source_xy[:, 1] / i_y,
            color="purple", linewidth=LINEWIDTHS, zorder=10
        )

        # Target line
        ax.plot(
            result.target_xy[:, 0] / i_y, result.target_xy[:, 1] / i_y,
            color="purple", linewidth=LINEWIDTHS, zorder=10
        )

        # Background: logK field
        im_ref = ax.imshow(
            np.exp(logk_field), norm=LogNorm(), origin="lower", cmap=cmc.batlowW_r,
            interpolation="nearest", extent=extent, alpha=0.7,
        )

        f = float(snap_fractions[j])
        t_level_fp = f * t_fp_arrival
        t_level_lrp = f * t_lrp_arrival

        # FP particle snapshot plume (dashed polyline in seed order)
        if snaps is not None and j < snaps.x.shape[1]:
            _plot_snapshot_particles_line(
                ax,
                snaps.x[:, j],
                snaps.y[:, j],
                i_y=i_y,
                color=COLORS["fp"],
                linestyle="-",
                linewidth=LINEWIDTHS,
                zorder=3,
            )

        if pypar2_snaps is not None and j < pypar2_snaps.x.shape[1]:
            plume_drawn = _plot_snapshot_density_contour(
                ax,
                pypar2_snaps.x[:, j],
                pypar2_snaps.y[:, j],
                x_grid=result.x,
                y_grid=result.y,
                i_y=i_y,
                color=COLORS["fpPypar2"],
                linestyle=LINESTYLES["fpPypar2"],
                linewidth=LINEWIDTHS,
                zorder=1,
            )
            if not plume_drawn:
                finite_count = int(np.count_nonzero(np.isfinite(pypar2_snaps.x[:, j]) & np.isfinite(pypar2_snaps.y[:, j])))
                print(f"Warning: skipped PyPar2 snapshot plume {j}; no drawable in-domain density from {finite_count} finite points.")

        if t_min_fm_asr_dimless is not None:
            t_level_fm_asr = f * t_fm_asr_arrival
            if t_fm_asr_min <= t_level_fm_asr <= t_fm_asr_max:
                _plot_isochrone_contour(
                    ax, x_norm, y_norm, t_min_fm_asr_dimless, t_level_fm_asr,
                    color=COLORS["fpFmAsr"],
                    linewidth=LINEWIDTHS,
                    linestyle="-",
                    zorder=4,
                )

        # FP (FMM) isochrone — relative to FP first arrival
        if t_min_val <= t_level_fp <= t_max_val:
            _plot_isochrone_contour(
                ax, x_norm, y_norm, t_min_dimless, t_level_fp,
                color=COLORS["fpFMM"],
                linewidth=LINEWIDTHS,
                linestyle="-",
                zorder=5,
            )

        # LRP isochrone — relative to LRP first arrival
        if t_lrp_min <= t_level_lrp <= t_lrp_max:
            _plot_isochrone_contour(
                ax, x_norm, y_norm, t_min_lrp_dimless, t_level_lrp,
                color=COLORS["lrpFMM"],
                linewidth=LINEWIDTHS,
                linestyle="-",
                zorder=6,
            )

        # # FSM isochrone — relative to FSM first arrival
        # if t_min_fsm_dimless is not None:
        #     t_level_fsm = f * t_fsm_arrival
        #     if t_fsm_min <= t_level_fsm <= t_fsm_max:
        #         _plot_isochrone_contour(
        #             ax, x_norm, y_norm, t_min_fsm_dimless, t_level_fsm,
        #             color=COLORS["fpFSM"],
        #             linewidth=LINEWIDTHS,
        #         )

        # # Legend on first panel only
        # if j == 0:
        #     h_fp = Line2D([0], [0], color=COLORS["fp"], lw=LINEWIDTHS, ls=LINESTYLES["fp"],
        #                   label=r"$\gamma^{\!\star}_\mathcal{S}$")
        #     h_fp.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 2, foreground="black"), pe.Normal()])
        #     h_fp.set_solid_capstyle("round")
        #     h_fp.set_dash_capstyle("round")

        #     h_fp_fmm = Line2D([0], [0], color=COLORS["fpFMM"], lw=LINEWIDTHS, ls=LINESTYLES["fpFMM"],
        #                       label=r"$\gamma^{\!\star}_\mathcal{E}$")
        #     h_fp_fmm.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 2, foreground="black"), pe.Normal()])
        #     h_fp_fmm.set_solid_capstyle("round")
        #     h_fp_fmm.set_dash_capstyle("round")

        #     h_lrp_fmm = Line2D([0], [0], color=COLORS["lrpFMM"], lw=LINEWIDTHS, ls=LINESTYLES["lrpFMM"],
        #                        label=r"$\gamma^{\!\star}_\mathcal{R}$")
        #     h_lrp_fmm.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 2, foreground="black"), pe.Normal()])
        #     h_lrp_fmm.set_solid_capstyle("round")
        #     h_lrp_fmm.set_dash_capstyle("round")

        #     h_fm_asr = Line2D([0], [0], color=COLORS["fpFmAsr"], lw=LINEWIDTHS, ls=LINESTYLES["fpFmAsr"],
        #                       label=r"$\gamma^{\!\star}_\mathcal{F}$")
        #     h_fm_asr.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 2, foreground="black"), pe.Normal()])
        #     h_fm_asr.set_solid_capstyle("round")
        #     h_fm_asr.set_dash_capstyle("round")

        #     h_pypar2 = Line2D([0], [0], color=COLORS["fpPypar2"], lw=LINEWIDTHS, ls=LINESTYLES["fpPypar2"],
        #                       label=r"$\gamma^{\!\star}_\mathcal{P}$")
        #     h_pypar2.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 2, foreground="black"), pe.Normal()])
        #     h_pypar2.set_solid_capstyle("round")
        #     h_pypar2.set_dash_capstyle("round")

        #     handles = [
        #         h_fp, h_fp_fmm, h_lrp_fmm,
        #         Line2D([0], [0], color=COLORS["fp"], lw=LINEWIDTHS, label=r"$\mathcal{T}^*_\mathcal{S}$ isochrone"),
        #         Line2D([0], [0], color=COLORS["fpFMM"], lw=LINEWIDTHS, label=r"$\mathcal{T}^*_\mathcal{E}$ isochrone"),
        #         Line2D([0], [0], color=COLORS["lrpFMM"], lw=LINEWIDTHS, label=r"$\mathcal{T}^*_\mathcal{R}$ isochrone"),
        #     ]
        #     if result.fp_fm_asr is not None and t_min_fm_asr_dimless is not None:
        #         handles += [
        #             h_fm_asr,
        #             Line2D([0], [0], color=COLORS["fpFmAsr"], lw=LINEWIDTHS, ls=LINESTYLES["fpFmAsr"], label=r"$\mathcal{T}^*_\mathcal{F}$ isochrone"),
        #         ]
        #     if result.fp_pypar2 is not None:
        #         handles.append(h_pypar2)
        #     if result.pypar2_snapshots is not None:
        #         handles.append(Line2D([0], [0], color=COLORS["fpPypar2"], lw=LINEWIDTHS, ls=LINESTYLES["fpPypar2"], label=r"PyPar2 snapshot contour"))
        #     # if t_min_fsm_dimless is not None:
        #     #     h_fp_fsm = Line2D([0], [0], color=COLORS["fpFSM"], lw=LINEWIDTHS, ls=LINESTYLES["fpFSM"],
        #     #                       label=r"$\gamma^{\!\star}_\mathcal{S}$")
        #     #     h_fp_fsm.set_path_effects([pe.Stroke(linewidth=LINEWIDTHS + 2, foreground="black"), pe.Normal()])
        #     #     handles += [
        #     #         h_fp_fsm,
        #     #         Line2D([0], [0], color=COLORS["fpFSM"], lw=LINEWIDTHS, label=r"$\mathcal{T}^*_\mathcal{S}$ isochrone"),
        #     #     ]
        #     ax.legend(handles=handles, fontsize=6, loc="upper left", framealpha=0.5)

        if j == n_snaps - 1:
            arrival_lines = [rf"$t_{{\mathcal{{S}}}}^{{\!\star}} U/I_Y = {t_fp_pt_arrival:.3g}$",]
            if pypar2_arrival_time_dimless is not None and np.isfinite(pypar2_arrival_time_dimless):
                arrival_lines.append(rf"$t_{{\mathcal{{D}}}}^{{\!\star}} U/I_Y = {pypar2_arrival_time_dimless:.3g}$")
            if t_min_fm_asr_dimless is not None:
                arrival_lines.append(rf"$t_{{\mathcal{{F}}}}^{{\!\star}} U/I_Y = {t_fm_asr_arrival:.3g}$")
            arrival_lines.append(rf"$t_{{\mathcal{{E}}}}^{{\!\star}} U/I_Y = {t_fp_arrival:.3g}$")
            arrival_lines.append(rf"$t_{{\mathcal{{R}}}}^{{\!\star}} U/I_Y = {t_lrp_arrival:.3g}$")
            # if t_min_fsm_dimless is not None:
            #     arrival_lines.append(rf"$t_{{a,\mathcal{{S}}}}^* = {t_fsm_arrival:.3g}$")
            # ax.text(
            #     0.98,
            #     0.02,
            #     "\n".join(arrival_lines),
            #     transform=ax.transAxes,
            #     ha="right",
            #     va="bottom",
            #     fontsize=7,
            #     bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.2", "alpha": 0.8},
            #     zorder=15
            # )

        ax.set_title(r"$f = {:.2f}$".format(f), fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        if j % ncols == 0:
            ax.set_ylabel(r"$y/I_Y$")
        if j >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$x/I_Y$")

    # Turn off unused axes
    for j in range(n_snaps, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared colorbar for K (log scale)
    if im_ref is not None:
        cbar = fig.colorbar(
            im_ref, ax=axes_flat[:n_snaps].tolist(),
            label=r"$K$", fraction=0.02, pad=0.02,
        )
        cbar.locator = LogLocator(base=10)
        cbar.formatter = LogFormatterMathtext(base=10)
        cbar.update_ticks()
    # fig.tight_layout()
    return fig


def _value_slug(value: float) -> str:
    return str(f"{float(value):g}").replace("-", "m").replace(".", "p")


def run_step5(
    cfg: step1.VisualizationConfig,
    *,
    logk_source: str = "dataset",
    sweep: str | None = None,
    value: float | None = None,
    field_size: int = 200,
    mc_index: int | None = None,
    output_dir: Path | None = None,
    synthetic_amplitude: float = 1.0,
    synthetic_frequency_x: float = 1.0,
    synthetic_frequency_y: float = 1.0,
    synthetic_offset: float = 0.0,
    grf_std: float = 1.0,
    grf_length_scale: float | None = None,
    grf_seed: int = 0,
    grf_minpadding: int = 0,
    dh: float | None = None,
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
    curvature_method: str = "savgol",
    curvature_spline_mode: str = "interpolation",
    curvature_spline_smooth: float = 0.0,
    curvature_spline_degree: int = 3,
    curvature_field_smooth_method: str = "none",
    curvature_field_smooth_sigma: float = 0.0,
    curvature_circle_window: int = 11,
    curvature_savgol_window: int = 11,
    curvature_savgol_poly: int = 3,
    normal_deriv_method: str = "direct",
    normal_deriv_spline2d_smooth: float = 0.0,
    normal_deriv_mls_ns: int = 2,
    normal_deriv_mls_nn: int = 2,
    normal_deriv_hessian_smooth: float = 0.0,
    normal_deriv_hessian_window: int = 3,
    normal_deriv_hessian_poly: int = 2,
    fm_asr_hamiltonian_t_hessian: str = "grid_savgol",
    fm_asr_hamiltonian_p_dot: str = "canonical",
    curvature_resample_ds: float = 0.5,
    density_particle: int = 20,
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
    snapshot_mode: str | None = None,
    snapshot_times: Any = None,
    fsm_fp: bool = False,
    fsm_max_sweeps: int = 100,
    fsm_tol: float = 1e-6,
    fsm_mse_tol: float | None = None,
    fsm_convergence_mode: str | None = None,
    fsm_mse_check_every: int = 5,
    fsm_mse_activation_factor: float = 10.0,
    fsm_consecutive_passes: int = 2,
    fsm_solver: str = "fipy",
    fsm_dt: float = 0.5,
    fsm_max_steps: int = 10_000,
    fm_asr_fp: bool = False,
    fm_asr_perp_ratio: float = 0.5,
    fm_asr_back_factor: float = 2.0,
    fm_asr_min_speed: float = 1e-12,
    fm_asr_max_steps: int = 200_000,
    pypar2_fp: bool = False,
    pypar2_dt: float = 0.01,
    pypar2_steps: int = 200_000,
    pypar2_dm: float = 1e-9,
    pypar2_alpha_l: float = 0.1,
    pypar2_alpha_t: float = 0.01,
    pypar2_velocity_floor: float | None = None,
    pypar2_taichi_arch: str = "gpu",
    pypar2_max_points: int = 2_000,
    pypar2_num_particles: int | None = None,
    pypar2_trajectory_memory_limit: int = 5_000_000,
    pypar2_replay_top_k: int = 5,
    pypar2_replay_particles_per_center: int = 40,
    pypar2_replay_radius_cells: float = 0.5,
    pypar2_replay_tolerance: float = 1.05,
    pypar2_replay_max_redos: int = 3,
    pypar2_seed: int = 0,
) -> tuple[Path, Path, Path, Path, Path | None, Path | None]:
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
    elif logk_source == "grf":
        ls = cfg.length_scale if grf_length_scale is None else grf_length_scale
        logk_field = generate_grf_logk(
            field_size=field_size,
            std=grf_std,
            length_scale=ls,
            seed=grf_seed,
            minpadding=grf_minpadding,
        )
        stem = (
            f"grf_size{field_size}_std{_value_slug(grf_std)}_"
            f"ls{_value_slug(ls)}_seed{grf_seed}"
        )
    else:
        raise ValueError(f"Unsupported logk_source={logk_source!r}; expected 'dataset', 'synthetic', or 'grf'.")

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
        snapshot_mode=snapshot_mode,
        snapshot_times=snapshot_times,
        fsm_fp=fsm_fp,
        fsm_max_sweeps=fsm_max_sweeps,
        fsm_tol=fsm_tol,
        fsm_mse_tol=fsm_mse_tol,
        fsm_convergence_mode=fsm_convergence_mode,
        fsm_mse_check_every=fsm_mse_check_every,
        fsm_mse_activation_factor=fsm_mse_activation_factor,
        fsm_consecutive_passes=fsm_consecutive_passes,
        fsm_solver=fsm_solver,
        fsm_dt=fsm_dt,
        fsm_max_steps=fsm_max_steps,
        fm_asr_fp=fm_asr_fp,
        fm_asr_perp_ratio=fm_asr_perp_ratio,
        fm_asr_back_factor=fm_asr_back_factor,
        fm_asr_min_speed=fm_asr_min_speed,
        fm_asr_max_steps=fm_asr_max_steps,
        pypar2_fp=pypar2_fp,
        pypar2_dt=pypar2_dt,
        pypar2_steps=pypar2_steps,
        pypar2_dm=pypar2_dm,
        pypar2_alpha_l=pypar2_alpha_l,
        pypar2_alpha_t=pypar2_alpha_t,
        pypar2_velocity_floor=pypar2_velocity_floor,
        pypar2_taichi_arch=pypar2_taichi_arch,
        pypar2_max_points=pypar2_max_points,
        pypar2_num_particles=pypar2_num_particles,
        pypar2_trajectory_memory_limit=pypar2_trajectory_memory_limit,
        pypar2_replay_top_k=pypar2_replay_top_k,
        pypar2_replay_particles_per_center=pypar2_replay_particles_per_center,
        pypar2_replay_radius_cells=pypar2_replay_radius_cells,
        pypar2_replay_tolerance=pypar2_replay_tolerance,
        pypar2_replay_max_redos=pypar2_replay_max_redos,
        pypar2_seed=pypar2_seed,
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

    fp_diag, lrp_diag, fm_asr_diag, streamline_diag = compute_balance_diagnostics(
        logk_field=logk_field,
        result=result,
        curvature_method=curvature_method,
        curvature_spline_mode=curvature_spline_mode,
        curvature_spline_smooth=curvature_spline_smooth,
        curvature_spline_degree=curvature_spline_degree,
        curvature_field_smooth_method=curvature_field_smooth_method,
        curvature_field_smooth_sigma=curvature_field_smooth_sigma,
        curvature_circle_window=curvature_circle_window,
        curvature_savgol_window=curvature_savgol_window,
        curvature_savgol_poly=curvature_savgol_poly,
        normal_deriv_method=normal_deriv_method,
        normal_deriv_spline2d_smooth=normal_deriv_spline2d_smooth,
        normal_deriv_mls_ns=normal_deriv_mls_ns,
        normal_deriv_mls_nn=normal_deriv_mls_nn,
        normal_deriv_hessian_smooth=normal_deriv_hessian_smooth,
        normal_deriv_hessian_window=normal_deriv_hessian_window,
        normal_deriv_hessian_poly=normal_deriv_hessian_poly,
        fm_asr_hamiltonian_t_hessian=fm_asr_hamiltonian_t_hessian,
        fm_asr_hamiltonian_p_dot=fm_asr_hamiltonian_p_dot,
        ds=curvature_resample_ds,
    )

    fig_logk = plot_paths_over_field(
        np.exp(logk_field),
        cmap=cmc.batlowW_r,
        colorbar_label=r"$K$",
        result=result,
        i_y=i_y,
        fp_diag=fp_diag,
        lrp_diag=lrp_diag,
        log_scale=True,
    )
    fig_logk.savefig(logk_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_logk)

    tmin_plot = np.where(np.isfinite(result.t_min), result.t_min / t_ref, np.nan)
    fig_tmin = plot_paths_over_field(
        tmin_plot,
        cmap=cmc.batlowW_r,
        colorbar_label=r"$\mathcal{T}^* U/I_Y$",
        result=result,
        i_y=i_y,
        fp_diag=fp_diag,
        lrp_diag=lrp_diag,
        log_scale=True,
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

    fig_balance = plot_curvature_balance(
        fp_diag,
        lrp_diag,
        fm_asr_diag=fm_asr_diag,
        streamline_diag=streamline_diag,
        i_y=i_y,
    )
    fig_balance.savefig(balance_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig_balance)
    progression_path_out: Path | None = None
    if linearized_fp:
        fig_progression = plot_linearized_fp_progression(
            tmin_plot,
            cmap=cmc.batlowW_r,
            colorbar_label=r"$T_{\min} U/I_Y$",
            result=result,
            i_y=i_y,
        )
        fig_progression.savefig(progression_path, dpi=cfg.figure_dpi, bbox_inches="tight")
        plt.close(fig_progression)
        progression_path_out = progression_path

    isochrone_plume_path_out: Path | None = None
    if result.snapshots is not None or result.pypar2_snapshots is not None:
        snap_fracs = np.asarray(snapshot_times, dtype=float)
        tmin_lrp_plot = np.where(np.isfinite(result.t_min_lrp), result.t_min_lrp / t_ref, np.nan)
        fp_arrival_time_dimless = (
            float(result.fp_arrival_time) / float(t_ref)
            if result.fp_arrival_time is not None and np.isfinite(result.fp_arrival_time)
            else None
        )
        tmin_fsm_plot = (
            np.where(np.isfinite(result.t_min_fsm), result.t_min_fsm / t_ref, np.nan)
            if result.t_min_fsm is not None else None
        )
        tmin_fm_asr_plot = (
            np.where(np.isfinite(result.t_min_fm_asr), result.t_min_fm_asr / t_ref, np.nan)
            if result.t_min_fm_asr is not None else None
        )
        isochrone_plume_path = out_dir / f"{stem}_isochrone_plume.png"
        fig_plume = plot_isochrones_with_plume(
            logk_field,
            tmin_plot,
            tmin_lrp_plot,
            snap_fracs,
            result=result,
            i_y=i_y,
            fp_arrival_time_dimless=fp_arrival_time_dimless,
            pypar2_arrival_time_dimless=(
                float(result.fp_pypar2_arrival_time) / float(t_ref)
                if result.fp_pypar2_arrival_time is not None and np.isfinite(result.fp_pypar2_arrival_time)
                else None
            ),
            t_min_fsm_dimless=tmin_fsm_plot,
            t_min_fm_asr_dimless=tmin_fm_asr_plot,
        )
        if fig_plume is not None:
            fig_plume.savefig(isochrone_plume_path, dpi=cfg.figure_dpi, bbox_inches="tight")
            plt.close(fig_plume)
            isochrone_plume_path_out = isochrone_plume_path

    return logk_path, tmin_path, stream_path, balance_path, progression_path_out, isochrone_plume_path_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5: solve and visualize particle-tracking FP, FP(FMM), LRP(FMM), and optional linearized FP^i for one logK field"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--logk-source", choices=["dataset", "synthetic", "grf"], default="dataset")
    parser.add_argument("--sweep", choices=["variance", "std"])
    parser.add_argument("--value", type=float)
    parser.add_argument("--field-size", required=True, type=int)
    parser.add_argument("--mc-index", type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--synthetic-amplitude", type=float, default=1.0)
    parser.add_argument("--synthetic-frequency-x", type=float, default=1.0)
    parser.add_argument("--synthetic-frequency-y", type=float, default=1.0)
    parser.add_argument("--synthetic-offset", type=float, default=0.0)

    parser.add_argument("--grf-std", type=float, default=1.0,
        help="Standard deviation for GRF logK generation.")
    parser.add_argument("--grf-length-scale", type=float, default=None,
        help="Correlation length for GRF logK. Defaults to cfg.length_scale.")
    parser.add_argument("--grf-seed", type=int, default=0,
        help="Random seed for GRF logK generation.")
    parser.add_argument("--grf-minpadding", type=int, default=0,
        help="Minimum padding for circulant embedding in GRF generation.")

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
    parser.add_argument(
        "--fsm-fp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable FSM (upwind FD + fast sweeping) travel-time and Euler-backtracking path.",
    )
    parser.add_argument("--fsm-max-sweeps", type=int, default=100,
        help="Maximum number of 4-direction sweep cycles for the FSM solver.")
    parser.add_argument("--fsm-tol", type=float, default=1e-6,
        help="Convergence tolerance for the FSM travel-time solver.")
    parser.add_argument("--fsm-mse-tol", type=float, default=None,
        help="Optional mean-squared residual tolerance for FSM convergence based on the discrete PDE residual v·∇T - 1.")
    parser.add_argument("--fsm-convergence-mode", choices=["update", "mse", "update_or_mse", "update_and_mse"], default=None,
        help="FSM convergence rule. Defaults to 'update' without --fsm-mse-tol and 'update_and_mse' when --fsm-mse-tol is set.")
    parser.add_argument("--fsm-mse-check-every", type=int, default=5,
        help="Evaluate FSM residual MSE at most every N sweeps when MSE checking is enabled.")
    parser.add_argument("--fsm-mse-activation-factor", type=float, default=10.0,
        help="Only start FSM residual-MSE checks once max_change falls below this factor times --fsm-tol for update-screened modes.")
    parser.add_argument("--fsm-consecutive-passes", type=int, default=2,
        help="Require the FSM convergence rule to pass this many consecutive eligible checks before stopping.")
    parser.add_argument("--fsm-solver", choices=["fipy", "sweep", "legacy", "new", "fd"], default="fipy",
        help="FSM backend. 'fipy' uses the FiPy/PETSc solver with sweep fallback in Algorithm; 'sweep' forces the current sweep backend; 'legacy' uses the old cell-centered sweep solver; 'new' selects the two-phase FSM backend with explicit source-mask support; 'fd' uses upwind finite-difference Jacobi iteration with PDE-residual loss convergence.")
    parser.add_argument("--fsm-dt", type=float, default=0.5,
        help="RK45 max step size (grid cells) for FSM path backtracking.")
    parser.add_argument("--fsm-max-steps", type=int, default=10000,
        help="Maximum RK45 backtracking steps for FSM path extraction.")
    parser.add_argument(
        "--fm-asr-fp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable FM-ASR Finsler fastest-path travel-time solve and backtracking path.",
    )
    parser.add_argument("--fm-asr-perp-ratio", type=float, default=0.5,
        help="FM-ASR sideways speed ratio relative to local velocity magnitude.")
    parser.add_argument("--fm-asr-back-factor", type=float, default=2.0,
        help="FM-ASR backward speed reduction factor.")
    parser.add_argument("--fm-asr-min-speed", type=float, default=1e-12,
        help="FM-ASR velocity magnitude floor.")
    parser.add_argument("--fm-asr-max-steps", type=int, default=200000,
        help="Maximum FM-ASR Hopf-Lax backtracking steps.")
    parser.add_argument(
        "--pypar2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Taichi PAR2 random-walk particle-tracking comparison path.",
    )
    parser.add_argument("--pypar2-dt", "--pypar2_dt", dest="pypar2_dt", type=float, default=0.01,
        help="PyPar2 time step.")
    parser.add_argument("--pypar2-steps", "--pypar2_steps", dest="pypar2_steps", type=int, default=200_000,
        help="PyPar2 max number of time steps.")
    parser.add_argument("--pypar2-dm", "--pypar2_dm", "--pypar2-molecular-diffusion", "--pypar2_molecular_diffusion",
        dest="pypar2_dm", type=float, default=1e-9,
        help="PyPar2 molecular diffusion coefficient.")
    parser.add_argument("--pypar2-alpha-l", "--pypar2_alpha_l", dest="pypar2_alpha_l", type=float, default=0.1,
        help="PyPar2 longitudinal dispersivity.")
    parser.add_argument("--pypar2-alpha-t", "--pypar2_alpha_t", dest="pypar2_alpha_t", type=float, default=0.01,
        help="PyPar2 transverse dispersivity.")
    parser.add_argument("--pypar2-velocity-floor", "--pypar2_velocity_floor", dest="pypar2_velocity_floor", type=float, default=None,
        help="PyPar2 velocity-floor regularization. Defaults to dispatcher auto behavior.")
    parser.add_argument("--pypar2-taichi-arch", "--pypar2_taichi_arch", dest="pypar2_taichi_arch", choices=["gpu", "cpu"], default="gpu",
        help="Taichi backend forwarded to PyPar2.")
    parser.add_argument("--pypar2-max-points", "--pypar2_max_points", dest="pypar2_max_points", type=int, default=2_000,
        help="PyPar2 trajectory max stored points per particle.")
    parser.add_argument("--pypar2-num-particles", "--pypar2_num_particles", dest="pypar2_num_particles", type=int, default=None,
        help="PyPar2 pilot-pass particle count. Defaults to --num-particles or density-derived count.")
    parser.add_argument("--pypar2-trajectory-memory-limit", "--pypar2_trajectory_memory_limit", dest="pypar2_trajectory_memory_limit", type=int, default=5_000_000,
        help="PyPar2 trajectory_memory_limit forwarded to the dispatcher.")
    parser.add_argument("--pypar2-replay-top-k", "--pypar2_replay_top_k", dest="pypar2_replay_top_k", type=int, default=5,
        help="PyPar2 replay top-K fastest pilot particles.")
    parser.add_argument("--pypar2-replay-particles-per-center", "--pypar2_replay_particles_per_center", dest="pypar2_replay_particles_per_center", type=int, default=40,
        help="PyPar2 replay particles per top-K release point.")
    parser.add_argument("--pypar2-replay-radius-cells", "--pypar2_replay_radius_cells", dest="pypar2_replay_radius_cells", type=float, default=0.5,
        help="PyPar2 replay seeding radius in grid cells.")
    parser.add_argument("--pypar2-replay-tolerance", "--pypar2_replay_tolerance", dest="pypar2_replay_tolerance", type=float, default=1.05,
        help="PyPar2 replay acceptance tolerance relative to pilot best time.")
    parser.add_argument("--pypar2-replay-max-redos", "--pypar2_replay_max_redos", dest="pypar2_replay_max_redos", type=int, default=3,
        help="PyPar2 replay maximum additional retries.")
    parser.add_argument("--pypar2-seed", "--pypar2_seed", dest="pypar2_seed", type=int, default=0,
        help="Random seed forwarded to PyPar2.")
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--target-x", type=int, default=None)
    parser.add_argument("--source-y-start", type=int, default=None)
    parser.add_argument("--source-y-end", type=int, default=None)
    parser.add_argument("--target-y-start", type=int, default=None)
    parser.add_argument("--target-y-end", type=int, default=None)
    parser.add_argument("--curvature-method", choices=["spline", "savgol", "circle"], default="spline")
    parser.add_argument("--curvature-spline-mode", choices=["interpolation", "approximation"], default="interpolation")
    parser.add_argument("--curvature-spline-smooth", type=float, default=0.0)
    parser.add_argument("--curvature-spline-degree", type=int, default=3)
    parser.add_argument("--curvature-field-smooth-method", choices=["none", "gaussian"], default="none")
    parser.add_argument("--curvature-field-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--curvature-circle-window", type=int, default=11)
    parser.add_argument("--curvature-savgol-window", type=int, default=11)
    parser.add_argument("--curvature-savgol-poly", type=int, default=3)
    parser.add_argument("--normal-deriv-method", choices=["direct", "savgol2d", "spline2d", "mls_frenet", "hessian_T"], default="direct")
    parser.add_argument("--normal-deriv-spline2d-smooth", type=float, default=0.0)
    parser.add_argument("--normal-deriv-mls-ns", type=int, default=2)
    parser.add_argument("--normal-deriv-mls-nn", type=int, default=2)
    parser.add_argument("--normal-deriv-hessian-smooth", type=float, default=0.0,
                        help="Gaussian pre-smoothing sigma (grid cells) for hessian_T normal-deriv method.")
    parser.add_argument("--normal-deriv-hessian-window", type=int, default=3,
                        help="SG stencil window for hessian_T (odd, >= 3); window=3 reproduces original Sobel.")
    parser.add_argument("--normal-deriv-hessian-poly", type=int, default=2,
                        help="SG polynomial degree for hessian_T (>= 2 enforced internally).")
    parser.add_argument("--fm-asr-hamiltonian-t-hessian", choices=["grid_savgol", "frenet_quadratic"], default="grid_savgol",
                        help="T Hessian estimator for FM-ASR Hamiltonian curvature. Default 'grid_savgol' matches the Euclidean/FMM Hessian estimator; 'frenet_quadratic' uses a local path-aligned quadratic fit.")
    parser.add_argument("--fm-asr-hamiltonian-p-dot", choices=["canonical", "hessian_T"], default="canonical",
                        help="Momentum evolution for FM-ASR Hamiltonian curvature. Default 'canonical' uses p_dot=-H_x from the Randers metric only; 'hessian_T' uses p_dot=(nabla^2 T) H_p.")
    parser.add_argument("--curvature-resample-ds", type=float, default=0.5,
                        help="Arc-length resampling interval (grid cells) applied to FP/LRP paths before curvature computation.")

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
    parser.add_argument(
        "--snapshot-mode",
        choices=["fraction_of_first_arrival", "dt_interval", "physical"],
        default=None,
        help=(
            "Particle-tracking snapshot mode for plume visualization. "
            "Only supported with --backend=gpu --solver=pollock."
        ),
    )
    parser.add_argument(
        "--snapshot-times",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Snapshot time specification. "
            "For 'fraction_of_first_arrival': fractions e.g. 0.1 0.3 0.5 1.0. "
            "For 'dt_interval': single scalar dt. "
            "For 'physical': absolute physical times."
        ),
    )
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
    elif args.logk_source == "synthetic":
        generate_synthetic_logk(
            field_size=args.field_size,
            amplitude=args.synthetic_amplitude,
            frequency_x=args.synthetic_frequency_x,
            frequency_y=args.synthetic_frequency_y,
            offset=args.synthetic_offset,
        )
    elif args.logk_source == "grf":
        if args.grf_std <= 0 or not np.isfinite(args.grf_std):
            parser.error("--grf-std must be a positive finite number.")
        if args.grf_length_scale is not None and (args.grf_length_scale <= 0 or not np.isfinite(args.grf_length_scale)):
            parser.error("--grf-length-scale must be a positive finite number.")
    else:
        parser.error(f"Unsupported --logk-source={args.logk_source!r}; expected 'dataset', 'synthetic', or 'grf'.")
    _validate_particle_tracking_options(solver=args.solver, velocity_location=args.velocity_location, backend=args.backend)
    if args.fm_asr_perp_ratio <= 0.0 or not np.isfinite(args.fm_asr_perp_ratio):
        parser.error("--fm-asr-perp-ratio must be a positive finite number.")
    if args.fm_asr_back_factor < 1.0 or not np.isfinite(args.fm_asr_back_factor):
        parser.error("--fm-asr-back-factor must be finite and >= 1.")
    if args.fm_asr_min_speed <= 0.0 or not np.isfinite(args.fm_asr_min_speed):
        parser.error("--fm-asr-min-speed must be a positive finite number.")
    if args.fm_asr_max_steps < 1:
        parser.error("--fm-asr-max-steps must be a positive integer.")
    if args.pypar2_dt <= 0.0 or not np.isfinite(args.pypar2_dt):
        parser.error("--pypar2-dt must be a positive finite number.")
    if args.pypar2_steps < 1:
        parser.error("--pypar2-steps must be a positive integer.")
    if args.pypar2_dm < 0.0 or not np.isfinite(args.pypar2_dm):
        parser.error("--pypar2-dm must be a non-negative finite number.")
    if args.pypar2_alpha_l < 0.0 or not np.isfinite(args.pypar2_alpha_l):
        parser.error("--pypar2-alpha-l must be a non-negative finite number.")
    if args.pypar2_alpha_t < 0.0 or not np.isfinite(args.pypar2_alpha_t):
        parser.error("--pypar2-alpha-t must be a non-negative finite number.")
    if args.pypar2_velocity_floor is not None and (args.pypar2_velocity_floor <= 0.0 or not np.isfinite(args.pypar2_velocity_floor)):
        parser.error("--pypar2-velocity-floor must be a positive finite number when provided.")
    if args.pypar2_max_points < 2:
        parser.error("--pypar2-max-points must be at least 2.")
    if args.pypar2_num_particles is not None and args.pypar2_num_particles < 2:
        parser.error("--pypar2-num-particles must be at least 2 when provided.")
    if args.pypar2_trajectory_memory_limit < 1:
        parser.error("--pypar2-trajectory-memory-limit must be a positive integer.")
    if args.pypar2_replay_top_k < 1:
        parser.error("--pypar2-replay-top-k must be a positive integer.")
    if args.pypar2_replay_particles_per_center < 1:
        parser.error("--pypar2-replay-particles-per-center must be a positive integer.")
    if args.pypar2_replay_radius_cells < 0.0 or not np.isfinite(args.pypar2_replay_radius_cells):
        parser.error("--pypar2-replay-radius-cells must be a non-negative finite number.")
    if args.pypar2_replay_tolerance <= 0.0 or not np.isfinite(args.pypar2_replay_tolerance):
        parser.error("--pypar2-replay-tolerance must be a positive finite number.")
    if args.pypar2_replay_max_redos < 0:
        parser.error("--pypar2-replay-max-redos must be non-negative.")
    if args.snapshot_mode is not None and args.snapshot_times is None:
        parser.error("--snapshot-times is required when --snapshot-mode is specified.")
    if args.snapshot_times is not None and args.snapshot_mode is None:
        parser.error("--snapshot-mode is required when --snapshot-times is specified.")
    if args.snapshot_mode == "dt_interval" and args.snapshot_times is not None and len(args.snapshot_times) != 1:
        parser.error("--snapshot-times must be a single scalar when --snapshot-mode=dt_interval.")
    return args


def main() -> int:
    configure_style()
    args = parse_args()
    cfg = step1.load_config(args.config)
    (logk_path, tmin_path, stream_path, balance_path,
     progression_path, isochrone_plume_path) = run_step5(
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
        grf_std=args.grf_std,
        grf_length_scale=args.grf_length_scale,
        grf_seed=args.grf_seed,
        grf_minpadding=args.grf_minpadding,
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
        curvature_savgol_window=args.curvature_savgol_window,
        curvature_savgol_poly=args.curvature_savgol_poly,
        normal_deriv_method=args.normal_deriv_method,
        normal_deriv_spline2d_smooth=args.normal_deriv_spline2d_smooth,
        normal_deriv_mls_ns=args.normal_deriv_mls_ns,
        normal_deriv_mls_nn=args.normal_deriv_mls_nn,
        normal_deriv_hessian_smooth=args.normal_deriv_hessian_smooth,
        normal_deriv_hessian_window=args.normal_deriv_hessian_window,
        normal_deriv_hessian_poly=args.normal_deriv_hessian_poly,
        fm_asr_hamiltonian_t_hessian=args.fm_asr_hamiltonian_t_hessian,
        fm_asr_hamiltonian_p_dot=args.fm_asr_hamiltonian_p_dot,
        curvature_resample_ds=args.curvature_resample_ds,
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
        snapshot_mode=args.snapshot_mode,
        snapshot_times=args.snapshot_times,
        fsm_fp=args.fsm_fp,
        fsm_max_sweeps=args.fsm_max_sweeps,
        fsm_tol=args.fsm_tol,
        fsm_mse_tol=args.fsm_mse_tol,
        fsm_convergence_mode=args.fsm_convergence_mode,
        fsm_mse_check_every=args.fsm_mse_check_every,
        fsm_mse_activation_factor=args.fsm_mse_activation_factor,
        fsm_consecutive_passes=args.fsm_consecutive_passes,
        fsm_solver=args.fsm_solver,
        fsm_dt=args.fsm_dt,
        fsm_max_steps=args.fsm_max_steps,
        fm_asr_fp=args.fm_asr_fp,
        fm_asr_perp_ratio=args.fm_asr_perp_ratio,
        fm_asr_back_factor=args.fm_asr_back_factor,
        fm_asr_min_speed=args.fm_asr_min_speed,
        fm_asr_max_steps=args.fm_asr_max_steps,
        pypar2_fp=args.pypar2,
        pypar2_dt=args.pypar2_dt,
        pypar2_steps=args.pypar2_steps,
        pypar2_dm=args.pypar2_dm,
        pypar2_alpha_l=args.pypar2_alpha_l,
        pypar2_alpha_t=args.pypar2_alpha_t,
        pypar2_velocity_floor=args.pypar2_velocity_floor,
        pypar2_taichi_arch=args.pypar2_taichi_arch,
        pypar2_max_points=args.pypar2_max_points,
        pypar2_num_particles=args.pypar2_num_particles,
        pypar2_trajectory_memory_limit=args.pypar2_trajectory_memory_limit,
        pypar2_replay_top_k=args.pypar2_replay_top_k,
        pypar2_replay_particles_per_center=args.pypar2_replay_particles_per_center,
        pypar2_replay_radius_cells=args.pypar2_replay_radius_cells,
        pypar2_replay_tolerance=args.pypar2_replay_tolerance,
        pypar2_replay_max_redos=args.pypar2_replay_max_redos,
        pypar2_seed=args.pypar2_seed,
    )
    print(f"Saved figure: {logk_path}")
    print(f"Saved figure: {tmin_path}")
    print(f"Saved figure: {stream_path}")
    print(f"Saved figure: {balance_path}")
    if progression_path is not None:
        print(f"Saved figure: {progression_path}")
    if isochrone_plume_path is not None:
        print(f"Saved figure: {isochrone_plume_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
