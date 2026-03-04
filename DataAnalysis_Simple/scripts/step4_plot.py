#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import brentq
from scipy.special import erfc, expi

from step1_index import VisualizationConfig, default_cache_dir, load_config

COLORS = {
    "fp": "black",
    "lrp": "#5278C4",
    "fpFMM": "#C46552",
    "fpDijkstra": "#B1C452",
}


@dataclass(slots=True)
class FigureArtifact:
    id: str
    filename: str
    path: Path
    section: str
    metadata: dict[str, Any]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
            if "__main__" not in str(exc):
                raise
            import __main__
            import step2_metrics as step2

            setattr(__main__, "VarianceResults", step2.VarianceResults)
            setattr(__main__, "StdResults", step2.StdResults)
            setattr(__main__, "SummaryStats", step2.SummaryStats)
            f.seek(0)
            return pickle.load(f)


def configure_style(cfg: VisualizationConfig) -> None:
    for path in glob.glob(cfg.font.cmu_font_glob):
        fm.fontManager.addfont(path)
    plt.rcParams["font.family"] = cfg.font.family
    plt.rcParams["mathtext.fontset"] = cfg.font.mathtext_fontset
    plt.rcParams["axes.unicode_minus"] = cfg.font.unicode_minus


def compute_effective_length(field_size: float) -> float:
    field_size_int = int(field_size)
    margin = field_size_int // 20
    return float(field_size_int - 2 * margin)


def _g_tau(t: float, *, length: float, velocity: float, i_y: float, sigma_y_sq: float, euler_const: float) -> float:
    tiny = np.finfo(float).tiny
    t_prime = max((t * velocity) / i_y, tiny)
    term_bracket = expi(-t_prime) + (np.exp(-t_prime) * (1.0 + t_prime) - 1.0) / (t_prime**2)
    f_norm = 2.0 * t_prime - 3.0 * np.log(t_prime) + 1.5 - 3.0 * euler_const + 3.0 * term_bracket
    x11 = max(sigma_y_sq * (i_y**2) * f_norm, tiny)
    x1 = velocity * t
    return 0.5 * erfc((length - x1) / np.sqrt(2.0 * x11))


def _find_root_bracket(
    *,
    sigma_y_sq: float,
    length: float,
    velocity: float,
    i_y: float,
    percentile: float,
    euler_const: float,
    lo_factor: float,
    hi_factor: float,
    max_expand_steps: int = 12,
) -> tuple[float, float] | None:
    base_time = length / velocity
    lo = max(base_time * lo_factor, np.finfo(float).tiny)
    hi = base_time * hi_factor
    objective = lambda t: _g_tau(t, length=length, velocity=velocity, i_y=i_y, sigma_y_sq=sigma_y_sq, euler_const=euler_const) - percentile
    f_lo = objective(lo)
    f_hi = objective(hi)
    if np.signbit(f_lo) != np.signbit(f_hi):
        return lo, hi

    for _ in range(max_expand_steps):
        lo = max(lo / 2.0, np.finfo(float).tiny)
        hi = hi * 2.0
        f_lo = objective(lo)
        f_hi = objective(hi)
        if np.signbit(f_lo) != np.signbit(f_hi):
            return lo, hi
    return None


def compute_analytical_tmin_curve(
    var: np.ndarray,
    *,
    length: float,
    velocity: float,
    i_y: float,
    percentile: float,
    lo_factor: float = 0.1,
    hi_factor: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    euler_const = float(getattr(constants, "euler_gamma", np.euler_gamma))
    values = np.asarray(var, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    warnings: list[str] = []
    if length <= 0 or velocity <= 0 or i_y <= 0:
        warnings.append("Analytical tmin skipped because length, velocity, and i_y must be positive.")
        return out, warnings
    if not (0.0 < percentile < 1.0):
        warnings.append("Analytical tmin skipped because percentile must be in (0, 1).")
        return out, warnings

    for idx, sigma_y_sq in enumerate(values):
        if sigma_y_sq <= 0:
            warnings.append(f"Analytical tmin skipped for non-positive variance sigma2={sigma_y_sq}.")
            continue
        bracket = _find_root_bracket(
            sigma_y_sq=float(sigma_y_sq),
            length=length,
            velocity=velocity,
            i_y=i_y,
            percentile=percentile,
            euler_const=euler_const,
            lo_factor=lo_factor,
            hi_factor=hi_factor,
        )
        if bracket is None:
            warnings.append(f"Analytical tmin root bracket failed for sigma2={sigma_y_sq}.")
            continue
        objective = lambda t: _g_tau(
            t,
            length=length,
            velocity=velocity,
            i_y=i_y,
            sigma_y_sq=float(sigma_y_sq),
            euler_const=euler_const,
        ) - percentile
        try:
            t_solution = brentq(objective, bracket[0], bracket[1])
        except ValueError:
            warnings.append(f"Analytical tmin root solve failed for sigma2={sigma_y_sq}.")
            continue
        out[idx] = t_solution * velocity / i_y
    return out, warnings


def plot_characteristic_time(
    var: np.ndarray,
    stats_fpt: np.ndarray,
    stats_mhr: np.ndarray,
    stats_fpt_fmm: np.ndarray,
    stats_fpt_dijkstra: np.ndarray,
    t_ref: float,
    analytical_tmin: np.ndarray | None = None,
    analytical_label: str = "Analytical $t_{\\mathrm{min}}$",
):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_fpt[:, 0] / t_ref, stats_fpt[:, 1] / t_ref, color=COLORS["fp"], linestyle="-", marker="o", markerfacecolor="None", label="FP")
    ax.errorbar(var, stats_mhr[:, 0] / t_ref, stats_mhr[:, 1] / t_ref, color=COLORS["lrp"], linestyle="-", marker="o", markerfacecolor="None", label="LRP")
    ax.errorbar(var, stats_fpt_fmm[:, 0] / t_ref, stats_fpt_fmm[:, 1] / t_ref, color=COLORS["fpFMM"], linestyle="-", marker="o", markerfacecolor="None", label="FP (FMM)")
    ax.errorbar(var, stats_fpt_dijkstra[:, 0] / t_ref, stats_fpt_dijkstra[:, 1] / t_ref, color=COLORS["fpDijkstra"], linestyle="-", marker="o", markerfacecolor="None", label="FP (Dijkstra)")
    if analytical_tmin is not None:
        ax.plot(var, analytical_tmin, color="black", linestyle="--", linewidth=1.8, label=analytical_label)
    ax.set_yscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[ tU / I_Y \right]$")
    ax.legend()
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "characteristic_time"}


def plot_taylor_comparison(var: np.ndarray, stats_mhr: np.ndarray, stats_fpt: np.ndarray, stats_fpt_fmm: np.ndarray, stats_fpt_dijkstra: np.ndarray, taylor: dict[str, np.ndarray], t_ref: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_mhr[:, 0] / t_ref, stats_mhr[:, 1] / t_ref, color=COLORS["lrp"], linestyle="-", marker="o", markerfacecolor="None", label="$R_{\\mathrm{min}}$")
    ax.errorbar(var, stats_fpt[:, 0] / t_ref, stats_fpt[:, 1] / t_ref, color=COLORS["fp"], linestyle="-", marker="o", markerfacecolor="None", label="$t_{\\mathrm{min}}$")
    ax.errorbar(var, stats_fpt_fmm[:, 0] / t_ref, stats_fpt_fmm[:, 1] / t_ref, color=COLORS["fpFMM"], linestyle="-", marker="o", markerfacecolor="None", label="$t_{\\mathrm{min}}$ (FMM)")
    ax.errorbar(var, stats_fpt_dijkstra[:, 0] / t_ref, stats_fpt_dijkstra[:, 1] / t_ref, color=COLORS["fpDijkstra"], linestyle="-", marker="o", markerfacecolor="None", label="$t_{\\mathrm{min}}$ (Dijkstra)")
    ax.plot(var, taylor["lrp"] / t_ref, "--", color=COLORS["lrp"], linewidth=1.5, label="Taylor Expansion (LRP)")
    ax.plot(var, taylor["fp"] / t_ref, "--", color=COLORS["fp"], linewidth=1.5, label="Taylor Expansion (FP)")
    ax.plot(var, taylor["fp_fmm"] / t_ref, "--", color=COLORS["fpFMM"], linewidth=1.5, label="Taylor Expansion (FP, FMM)")
    ax.plot(var, taylor["fp_dijkstra"] / t_ref, "--", color=COLORS["fpDijkstra"], linewidth=1.5, label="Taylor Expansion (FP, Dijkstra)")
    ax.set_yscale("log")
    ax.set_ylim([10 ** (-2), 10 ** 2])
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}[t_{\mathrm{min}}] U / I_Y$")
    ax.legend(fontsize="8", ncol=2)
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "taylor_comparison"}


def plot_scatter_tmin_vs_rmin(stats_fpt: np.ndarray, stats_fpt_fmm: np.ndarray, stats_fpt_dijkstra: np.ndarray, stats_mhr: np.ndarray, t_ref: float):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_fpt_fmm[:, 0] / t_ref, color="orange", label="$t_{\\text{min}}\\text{(FMM)}$", alpha=1)
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_fpt_dijkstra[:, 0] / t_ref, color="blue", label="$t_{\\text{min}}\\text{(Dijkstra)}$", alpha=0.6)
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_mhr[:, 0] / t_ref, color="darkred", marker="^", label="$R_{\\text{min}}$", alpha=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t_{\\text{min}}\\text{(RK45)}$ (s)")
    ax.set_ylabel("$\\tau$ (s)")
    ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Characteristic Time Scale")
    ax.legend()
    fig.tight_layout()
    return fig, {"kind": "scatter_tmin_vs_rmin"}


def _extract_std_series(stats_std: dict[str, np.ndarray], metric: str, pair: str, x_len: int, i_y: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(stats_std.get(metric, {}).get(pair, []), dtype=float)
    if arr.ndim < 3 or arr.shape[0] == 0 or arr.shape[1] < 2 or arr.shape[2] == 0:
        nan = np.full(x_len, np.nan, dtype=float)
        return nan, nan
    y = np.asarray(arr[:, 0, 0], dtype=float) / i_y
    yerr = np.asarray(arr[:, 1, 0], dtype=float) / i_y
    return y, yerr


def plot_hausdorff_vs_variance(var: np.ndarray, stats_hausdorff_lrp: np.ndarray, stats_hausdorff_fmm: np.ndarray, stats_hausdorff_dijkstra: np.ndarray, i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_hausdorff_lrp[:, 0] / i_y, yerr=stats_hausdorff_lrp[:, 1] / i_y, label="$d_H$[FP, LRP]", linestyle="-", color=COLORS["lrp"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_hausdorff_fmm[:, 0] / i_y, yerr=stats_hausdorff_fmm[:, 1] / i_y, label="$d_H$[FP, FP(FMM)]", linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_hausdorff_dijkstra[:, 0] / i_y, yerr=stats_hausdorff_dijkstra[:, 1] / i_y, label="$d_H$[FP, FP(Dijkstra)]", linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None")
    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[ d_H / I_Y \right]$")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "hausdorff_variance"}


def plot_endpoint_vs_variance(var: np.ndarray, stats_end_lrp: np.ndarray, stats_end_fmm: np.ndarray, stats_end_dijkstra: np.ndarray, i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_end_lrp[:, 0] / i_y, yerr=stats_end_lrp[:, 1] / i_y, label="$d_{EP}$[FP, LRP]", linestyle="-", color=COLORS["lrp"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_end_fmm[:, 0] / i_y, yerr=stats_end_fmm[:, 1] / i_y, label="$d_{EP}$[FP, FP(FMM)]", linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_end_dijkstra[:, 0] / i_y, yerr=stats_end_dijkstra[:, 1] / i_y, label="$d_{EP}$[FP, FP(Dijkstra)]", linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None")
    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[ d_{EP} / I_Y \right]$")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "endpoint_variance"}


def plot_std_hausdorff(x_var: np.ndarray, stats_std: dict[str, np.ndarray], i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    y, yerr = _extract_std_series(stats_std, "hausdorffDist", "fp_lrp", len(x_var), i_y)
    ax.errorbar(x_var, y, yerr=yerr, label="$d(\\text{FP}, \\text{LRP})$", linestyle="-", color="blue", marker="o", markerfacecolor="None")
    y, yerr = _extract_std_series(stats_std, "hausdorffDist", "fp_fpFMM", len(x_var), i_y)
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(FMM)}]$", linestyle="-", color="orange", marker="o", markerfacecolor="None")
    y, yerr = _extract_std_series(stats_std, "hausdorffDist", "fp_fpDijkstra", len(x_var), i_y)
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(Dijkstra)}]$", linestyle="-", color="red", marker="o", markerfacecolor="None")
    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel("$\\mathbb{E}\\left[ \\frac{d_{\\text{Hausdorff}}}{I_Y} \\right]$")
    ax.set_title("Variation of Hausdorff Distance with Variance")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "hausdorff_std"}


def plot_std_endpoint(x_var: np.ndarray, stats_std: dict[str, np.ndarray], i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    y, yerr = _extract_std_series(stats_std, "endPointDist", "fp_lrp", len(x_var), i_y)
    ax.errorbar(x_var, y, yerr=yerr, label="$d(\\text{FP}, \\text{LRP})$", linestyle="-", color="blue", marker="o", markerfacecolor="None")
    y, yerr = _extract_std_series(stats_std, "endPointDist", "fp_fpFMM", len(x_var), i_y)
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(FMM)}]$", linestyle="-", color="orange", marker="o", markerfacecolor="None")
    y, yerr = _extract_std_series(stats_std, "endPointDist", "fp_fpDijkstra", len(x_var), i_y)
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(Dijkstra)}]$", linestyle="-", color="red", marker="o", markerfacecolor="None")
    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel("$\\mathbb{E}\\left[ \\frac{d_{\\text{end-point}}}{I_Y} \\right]$")
    ax.set_title("Variation of End-Point Distance with Variance")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "endpoint_std"}


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _short_hash(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def save_figure(
    fig,
    *,
    section: str,
    output_dir: Path,
    run_id: int,
    dpi: int,
    save_formats: list[str],
    metadata: dict[str, Any],
    unique_filenames: bool = False,
) -> list[FigureArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp() if unique_filenames else None
    h8 = _short_hash(metadata) if unique_filenames else None
    artifacts: list[FigureArtifact] = []

    for ext in save_formats:
        if unique_filenames:
            filename = f"run{run_id}_{section}_{stamp}_{h8}.{ext}"
        else:
            filename = f"run{run_id}_{section}.{ext}"
        path = output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        artifacts.append(FigureArtifact(id=f"{section}:{ext}", filename=filename, path=path, section=section, metadata=metadata))

    return artifacts


def write_manifest(output_dir: Path, *, config: dict[str, Any], artifacts: list[FigureArtifact], warnings: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "warnings": warnings,
        "figures": [
            {
                "id": a.id,
                "section": a.section,
                "filename": a.filename,
                "path": str(a.path),
                "metadata": a.metadata,
            }
            for a in artifacts
        ],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def run_stage4(cfg: VisualizationConfig, *, cache_dir: Path | None = None, unique_filenames: bool = False) -> list[FigureArtifact]:
    cache_dir = cache_dir or default_cache_dir(cfg)
    payload = load_pickle(cache_dir / "stage3_plot_inputs.pkl")
    prepared = payload["plot_inputs"]
    warning_dict = payload.get("warnings", {})
    all_warnings = warning_dict.get("variance", []) + warning_dict.get("std", [])

    configure_style(cfg)
    artifacts: list[FigureArtifact] = []

    var = prepared["var"]
    s = prepared["stats"]
    taylor = prepared["taylor"]
    i_y = prepared["i_y"]
    t_ref = prepared["t_ref"]
    x_var = prepared["std_x"]
    field_size = float(prepared.get("field_size", cfg.field_size_ref))

    if "times" in cfg.sections:
        u1 = cfg.k_g * cfg.dh
        length = compute_effective_length(field_size)
        analytical_tmin, analytical_warnings = compute_analytical_tmin_curve(
            var,
            length=length,
            velocity=u1,
            i_y=i_y,
            percentile=cfg.analytical_percentile,
        )
        all_warnings.extend(analytical_warnings)
        fig, meta = plot_characteristic_time(
            var,
            s["stats_fpt"],
            s["stats_mhr"],
            s["stats_fpt_fmm"],
            s["stats_fpt_dijkstra"],
            t_ref,
            analytical_tmin=analytical_tmin,
            analytical_label=f"Analytical $t_{{\\mathrm{{min}}}}$ (p={cfg.analytical_percentile:g})",
        )
        artifacts.extend(save_figure(fig, section="times", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "taylor" in cfg.sections:
        fig, meta = plot_taylor_comparison(var, s["stats_mhr"], s["stats_fpt"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], taylor, t_ref)
        artifacts.extend(save_figure(fig, section="taylor", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "scatter" in cfg.sections:
        fig, meta = plot_scatter_tmin_vs_rmin(s["stats_fpt"], s["stats_fpt_fmm"], s["stats_fpt_dijkstra"], s["stats_mhr"], t_ref)
        artifacts.extend(save_figure(fig, section="scatter", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "hausdorff" in cfg.sections:
        fig, meta = plot_hausdorff_vs_variance(var, s["stats_hausdorff_lrp"], s["stats_hausdorff_fmm"], s["stats_hausdorff_dijkstra"], i_y)
        artifacts.extend(save_figure(fig, section="hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "endpoint" in cfg.sections:
        fig, meta = plot_endpoint_vs_variance(var, s["stats_endpoint_lrp"], s["stats_endpoint_fmm"], s["stats_endpoint_dijkstra"], i_y)
        artifacts.extend(save_figure(fig, section="endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "std_hausdorff" in cfg.sections and prepared["stats_std"] is not None:
        fig, meta = plot_std_hausdorff(x_var, prepared["stats_std"], i_y)
        artifacts.extend(save_figure(fig, section="std_hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "std_endpoint" in cfg.sections and prepared["stats_std"] is not None:
        fig, meta = plot_std_endpoint(x_var, prepared["stats_std"], i_y)
        artifacts.extend(save_figure(fig, section="std_endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    write_manifest(cfg.output_dir, config=asdict(cfg), artifacts=artifacts, warnings=all_warnings)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: render figures from cached plotting inputs")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--unique-filenames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include timestamp/hash in output figure filenames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    artifacts = run_stage4(cfg, cache_dir=args.cache_dir, unique_filenames=args.unique_filenames)
    print(f"Generated {len(artifacts)} figure artifact(s) in {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
