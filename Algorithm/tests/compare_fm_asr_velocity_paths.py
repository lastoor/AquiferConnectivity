#!/usr/bin/env python3
"""Compare GPU particle tracking, scalar FMM, and FM-ASR Finsler paths.

This is a standalone visualization/debug script, not a pytest test. It follows
the GPU Pollock payload framework used by scripts/run_fmm_gpu_worker.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
FLOWSIM_ROOT = REPO_ROOT.parents[1] / "FlowSimulation"

for path in (SRC_ROOT, SCRIPTS_ROOT, FLOWSIM_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import run_fmm as base
from fmm_core.fm_asr import FMASR2D, extract_fm_asr_path, visualize_indicatrix_field
from fmm_core.fmm import compute_fmm_path, solve_travel_time
from fmm_core.speed import _face_to_cell_speed_full, prepare_fmm_speed
from simulation import hydraulic as sim_hyd
from simulation import particle_tracking as sim_pt

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual comparison of fastest paths from GPU PT, scalar FMM, and FM-ASR."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/fm_asr_demo"))
    parser.add_argument("--field-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--covariance", choices=["exp", "fbm"], default="exp")
    parser.add_argument("--length-scale", type=float, default=0.1)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--fbm-alpha", type=float, default=0.8)
    parser.add_argument("--fbm-c", type=float, default=1.0)
    parser.add_argument("--minpadding", type=int, default=0)
    parser.add_argument(
        "--domain-margin",
        type=int,
        default=None,
        help=(
            "Inset for the vertical source/target lines in grid cells. "
            "Default matches scripts/run_fmm.py: field_size//20 - 1."
        ),
    )
    parser.add_argument("--dh", type=float, default=0.01)
    parser.add_argument("--num-particles", type=int, default=20_000)
    parser.add_argument("--perp-ratio", type=float, default=0.5)
    parser.add_argument("--back-factor", type=float, default=2.0)
    parser.add_argument("--min-speed", type=float, default=1e-12)
    parser.add_argument("--velocity-multiplier", type=float, default=1.0)
    parser.add_argument("--fmm-solver", choices=["custom", "scikit"], default="scikit")
    parser.add_argument("--fmm-backtracking", choices=["discrete", "continuous"], default="continuous")
    parser.add_argument("--align-with-modflow-velocity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--max-iters", type=int, default=1_000_000)
    parser.add_argument("--asr-max-steps", type=int, default=200_000)
    parser.add_argument("--indicatrix-stride", type=int, default=8)
    parser.add_argument("--indicatrix-glyph-scale", type=float, default=0.75)
    parser.add_argument("--no-indicatrix", action="store_true")
    return parser.parse_args()


def build_grf_arg(args: argparse.Namespace) -> dict[str, Any]:
    if args.covariance == "exp":
        return {
            "covariance": "exp",
            "length_scale": float(args.length_scale),
            "std": float(args.std),
            "minpadding": int(args.minpadding),
        }
    return {
        "covariance": "fbm",
        "fbm_alpha": float(args.fbm_alpha),
        "fbm_c": float(args.fbm_c),
        "minpadding": int(args.minpadding),
    }


def generate_k_field(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(int(np.uint32(args.seed & 0xFFFFFFFF)))
    grf = base.make_grf_generator(build_grf_arg(args), int(args.field_size))
    logk = np.asarray(grf.sample(), dtype=float)
    if logk.shape != (args.field_size, args.field_size):
        raise RuntimeError(f"Unexpected logK shape {logk.shape}")
    return np.exp(logk), logk


def resolve_domain_margin(args: argparse.Namespace, ny: int, nx: int) -> int:
    if args.domain_margin is None:
        margin = min(ny, nx) // 20 - 1
    else:
        margin = int(args.domain_margin)
    margin = max(0, margin)
    max_margin = max(0, (min(ny, nx) - 3) // 2)
    if margin > max_margin:
        raise ValueError(
            f"--domain-margin={margin} leaves no room for separated source/target lines "
            f"on a {ny}x{nx} grid; maximum is {max_margin}."
        )
    return margin


def make_masks(
    ny: int,
    nx: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    margin = resolve_domain_margin(args, ny, nx)
    source_col = margin
    target_col = nx - margin - 1
    source_row0 = margin
    source_row1 = ny - margin - 1
    if source_col >= target_col:
        raise ValueError(
            f"Source column {source_col} must be left of target column {target_col}."
        )

    source_mask = np.zeros((ny, nx), dtype=bool)
    target_mask = np.zeros((ny, nx), dtype=bool)
    source_rows = np.arange(source_row0, source_row1 + 1, dtype=int)
    target_rows = np.arange(ny, dtype=int)
    source_mask[source_rows, source_col] = True
    target_mask[target_rows, target_col] = True
    return source_mask, target_mask, source_rows, target_rows, source_col, target_col


def run_gpu_pollock(
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_faces: np.ndarray,
    y_faces: np.ndarray,
    vx_face: np.ndarray,
    vy_face: np.ndarray,
    source_rows: np.ndarray,
    source_col: int,
    target_col: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    params: dict[str, Any] = {
        "backend": "gpu",
        "method": "pollock",
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "arrival_fraction": 1.0,
        "save_trajectories": False,
        "return_fastest": True,
        "x_faces": x_faces.astype(np.float32),
        "y_faces": y_faces.astype(np.float32),
        "vx": np.asarray(vx_face),
        "vy": np.asarray(vy_face),
        "seeds_x": np.full(int(args.num_particles), float(x[source_col]), dtype=np.float32),
        "seeds_y": np.linspace(
            float(y[source_rows[0]]),
            float(y[source_rows[-1]]),
            int(args.num_particles),
            endpoint=True,
            dtype=np.float32,
        ),
        "right_x": float(x[target_col]),
        "max_points": int(args.max_points),
        "max_steps": int(args.max_iters),
    }
    summary = sim_pt.particle_tracking_summary(params)
    backend_used = str(summary.get("backend", "gpu"))
    if backend_used != "gpu":
        raise RuntimeError(f"GPU Pollock did not run on GPU; backend_used={backend_used!r}")

    status = np.asarray(summary["status"])
    success_fraction = float(np.count_nonzero(status == 0) / max(1, status.size))
    if success_fraction < 0.5:
        raise RuntimeError(f"GPU Pollock success fraction too low: {success_fraction:.3f}")

    fastest = summary.get("fastest")
    if fastest is None:
        raise RuntimeError("GPU Pollock summary did not include a fastest path")
    path = np.vstack((fastest["x"], fastest["y"])).T.astype(float)
    right_x = float(x[target_col])
    if path.size == 0 or path[-1, 0] < right_x - 1e-4:
        raise RuntimeError(
            "GPU Pollock fastest replay did not reach the target line: "
            f"last_x={path[-1, 0] if path.size else np.nan:.6g}, right_x={right_x:.6g}"
        )
    travel_time = float(fastest["travel_time"])
    meta = {
        "backend": backend_used,
        "success_fraction": success_fraction,
        "status_counts": {
            str(int(code)): int(np.count_nonzero(status == code))
            for code in np.unique(status)
        },
        "travel_time": travel_time,
        "particle_id": int(fastest.get("particle_id", -1)),
        "source_col": int(source_col),
        "target_col": int(target_col),
        "right_x": right_x,
    }
    return path, travel_time, meta


def solve_asr_path(
    W_cell: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    h: float,
    args: argparse.Namespace,
    isotropic_only: bool,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    solver = FMASR2D.from_velocity_field(
        W_cell,
        h=h,
        perp_ratio=float(args.perp_ratio),
        back_factor=float(args.back_factor),
        min_speed=float(args.min_speed),
        velocity_multiplier=float(args.velocity_multiplier),
        isotropic_only=bool(isotropic_only),
        direction="from_source",
    )
    T = solver.solve(source_mask)
    path, travel_time, target = extract_fm_asr_path(
        solver,
        T,
        source_mask,
        target_mask,
        max_steps=int(args.asr_max_steps),
    )
    return T, path, travel_time, target


def save_results_npz(path: Path, results: dict[str, Any]) -> None:
    np.savez(
        path,
        **{
            key: value
            for key, value in results.items()
            if isinstance(value, np.ndarray) or np.isscalar(value)
        },
    )


def json_safe_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def travel_time_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    """Return L2 and relative L2 errors on mutually finite travel-time entries."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(candidate)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "finite_count": 0,
            "l2_error": float("nan"),
            "relative_l2_error": float("nan"),
            "reference_l2_norm": float("nan"),
        }

    diff = candidate[mask] - reference[mask]
    ref = reference[mask]
    l2_error = float(np.linalg.norm(diff.ravel()))
    reference_l2_norm = float(np.linalg.norm(ref.ravel()))
    relative_l2_error = l2_error / reference_l2_norm if reference_l2_norm > 0.0 else float("nan")
    return {
        "finite_count": count,
        "l2_error": l2_error,
        "relative_l2_error": float(relative_l2_error),
        "reference_l2_norm": reference_l2_norm,
    }


def plot_comparison(results: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm

    logk = results["logk"]
    T_fmm = results["T_fmm"]
    T_asr = results["T_asr"]
    T_asr_iso = results["T_asr_iso"]
    diff_asr = T_asr - T_fmm
    diff_asr_iso = T_asr_iso - T_fmm
    paths = {
        "GPU Pollock": (results["path_pt"], "white", 2.4),
        "FMM speed": (results["path_fmm"], "tab:orange", 1.8),
        "FM-ASR ellipse": (results["path_asr"], "tab:red", 1.8),
        "FM-ASR isotropic": (results["path_asr_iso"], "tab:cyan", 1.8),
    }

    def diff_norm(arr: np.ndarray) -> TwoSlopeNorm | None:
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        vmax = float(np.nanpercentile(np.abs(finite), 99.0))
        if vmax <= 0.0:
            vmax = 1.0
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    rel_l2_asr = float(results.get("relative_l2_error_asr_vs_fmm", np.nan))
    rel_l2_asr_iso = float(results.get("relative_l2_error_asr_iso_vs_fmm", np.nan))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], logk, "log K", "viridis", None, None),
        (
            axes[0, 1],
            diff_asr,
            "FM-ASR ellipse T - FMM T",
            "coolwarm",
            diff_norm(diff_asr),
            rel_l2_asr,
        ),
        (axes[0, 2], T_fmm, "FMM travel time", "plasma", None, None),
        (axes[1, 0], T_asr, "FM-ASR acentric ellipse T", "plasma", None, None),
        (axes[1, 1], T_asr_iso, "FM-ASR reduced isotropic T", "plasma", None, None),
        (
            axes[1, 2],
            diff_asr_iso,
            "FM-ASR isotropic T - FMM T",
            "coolwarm",
            diff_norm(diff_asr_iso),
            rel_l2_asr_iso,
        ),
    ]
    for ax, arr, title, cmap, norm, rel_l2 in panels:
        im = ax.imshow(arr, origin="lower", cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_aspect("equal")
        for label, (path, color, linewidth) in paths.items():
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth, label=label)
        if rel_l2 is not None:
            ax.text(
                0.02,
                0.96,
                f"rel L2 = {rel_l2:.4g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.35",
                    "alpha": 0.82,
                },
            )
        ax.set_xlim(0, logk.shape[1] - 1)
        ax.set_ylim(0, logk.shape[0] - 1)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=linewidth, label=label)
        for label, (_path, color, linewidth) in paths.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        fontsize=9,
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    configure_style()

    args = parse_args()
    if args.field_size < 4:
        raise ValueError("--field-size must be at least 4")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    K, logk = generate_k_field(args)
    ny, nx = K.shape
    h_grid = 1.0
    x = np.arange(nx, dtype=float) * h_grid
    y = np.arange(ny, dtype=float) * h_grid
    source_mask, target_mask, source_rows, target_rows, source_col, target_col = make_masks(
        ny,
        nx,
        args,
    )

    head, msg, ok = sim_hyd.solve_head(
        K,
        x,
        y,
        dh=float(args.dh),
        rtol=1e-10,
        atol=1e-20,
        max_it=10000,
    )
    if not ok:
        raise RuntimeError(f"Head solver failed: {msg}")

    x_faces = base.build_face_coordinates_from_centers(x)
    y_faces = base.build_face_coordinates_from_centers(y)
    vx_face, vy_face = sim_hyd.compute_velocity(
        K,
        x,
        y,
        head,
        output_location="face",
        align_with_modflow=bool(args.align_with_modflow_velocity),
        face_output="full",
    )
    vx_cell, vy_cell = _face_to_cell_speed_full(vx_face, vy_face, return_vector=True)
    W_cell = np.stack((vx_cell, vy_cell), axis=2)

    speed, traversable_mask, speed_diag = prepare_fmm_speed(
        vx_face,
        vy_face,
        layout="face",
        face_output="full",
        face_mapping="to_cell_avg",
        zero_policy="clip",
        min_speed=float(args.min_speed),
    )

    path_pt, t_pt, pt_meta = run_gpu_pollock(
        x=x,
        y=y,
        x_faces=x_faces,
        y_faces=y_faces,
        vx_face=vx_face,
        vy_face=vy_face,
        source_rows=source_rows,
        source_col=source_col,
        target_col=target_col,
        args=args,
    )

    path_fmm, t_fmm, target_fmm = compute_fmm_path(
        speed,
        source_mask,
        target_mask,
        h=h_grid,
        traversable_mask=traversable_mask,
        solver=str(args.fmm_solver),
        backtracking=str(args.fmm_backtracking),
    )
    T_fmm, _ = solve_travel_time(
        speed,
        source_mask,
        h=h_grid,
        traversable_mask=traversable_mask,
        method=str(args.fmm_solver),
    )

    T_asr, path_asr, t_asr, target_asr = solve_asr_path(
        W_cell,
        source_mask,
        target_mask,
        h=h_grid,
        args=args,
        isotropic_only=False,
    )
    T_asr_iso, path_asr_iso, t_asr_iso, target_asr_iso = solve_asr_path(
        W_cell,
        source_mask,
        target_mask,
        h=h_grid,
        args=args,
        isotropic_only=True,
    )

    error_metrics = {
        "fm_asr_vs_fmm": travel_time_error_metrics(T_fmm, T_asr),
        "fm_asr_isotropic_vs_fmm": travel_time_error_metrics(T_fmm, T_asr_iso),
    }
    config = json_safe_config(args)
    summary = {
        "particle_tracking": pt_meta,
        "fmm": {"travel_time": float(t_fmm), "target": tuple(map(int, target_fmm))},
        "fm_asr": {"travel_time": float(t_asr), "target": tuple(map(int, target_asr))},
        "fm_asr_isotropic": {
            "travel_time": float(t_asr_iso),
            "target": tuple(map(int, target_asr_iso)),
        },
        "domain": {
            "source_col": int(source_col),
            "target_col": int(target_col),
            "source_row0": int(source_rows[0]),
            "source_row1": int(source_rows[-1]),
            "target_row0": int(target_rows[0]),
            "target_row1": int(target_rows[-1]),
        },
        "travel_time_errors": error_metrics,
        "speed_diag": speed_diag,
        "runtime_seconds": float(time.perf_counter() - t0),
        "config": config,
    }

    results = {
        "K": K,
        "logk": logk,
        "head": head,
        "vx_face": np.asarray(vx_face),
        "vy_face": np.asarray(vy_face),
        "vx_cell": vx_cell,
        "vy_cell": vy_cell,
        "speed": speed,
        "traversable_mask": traversable_mask,
        "T_fmm": T_fmm,
        "T_asr": T_asr,
        "T_asr_iso": T_asr_iso,
        "path_pt": path_pt,
        "path_fmm": path_fmm,
        "path_asr": path_asr,
        "path_asr_iso": path_asr_iso,
        "travel_time_pt": np.array(float(t_pt)),
        "travel_time_fmm": np.array(float(t_fmm)),
        "travel_time_asr": np.array(float(t_asr)),
        "travel_time_asr_iso": np.array(float(t_asr_iso)),
        "l2_error_asr_vs_fmm": np.array(float(error_metrics["fm_asr_vs_fmm"]["l2_error"])),
        "relative_l2_error_asr_vs_fmm": np.array(
            float(error_metrics["fm_asr_vs_fmm"]["relative_l2_error"])
        ),
        "l2_error_asr_iso_vs_fmm": np.array(
            float(error_metrics["fm_asr_isotropic_vs_fmm"]["l2_error"])
        ),
        "relative_l2_error_asr_iso_vs_fmm": np.array(
            float(error_metrics["fm_asr_isotropic_vs_fmm"]["relative_l2_error"])
        ),
    }
    save_results_npz(args.output_dir / "results.npz", results)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_comparison(results, args.output_dir / "comparison.png")
    if not args.no_indicatrix:
        import matplotlib

        matplotlib.use("Agg")
        visualize_indicatrix_field(
            W_cell,
            perp_ratio=float(args.perp_ratio),
            back_factor=float(args.back_factor),
            h=h_grid,
            stride=int(args.indicatrix_stride),
            glyph_scale=float(args.indicatrix_glyph_scale),
            K_field=K,
        )
        import matplotlib.pyplot as plt

        plt.savefig(args.output_dir / "indicatrix_field.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
