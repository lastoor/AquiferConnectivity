#!/usr/bin/env python3
"""Generate shared-scale indicatrix/path figures for one seeded GRF conductivity field."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import cmcrameri.cm as cmc

import step1_index as step1
import step5_single_field_paths as step5


GLYPH_LINEWIDTH = 0.85
GLYPH_OUTLINE_WIDTH = 1.8
STREAMLINE_ARROW_EDGE_WIDTH = 0.35
SAMPLE_POINT_SIZE = 10.0
SAMPLE_POINT_LINEWIDTH = 0.65
TITLE_FONTSIZE = 15.0
AXIS_LABEL_FONTSIZE = 13.0
LATEX_AVAILABLE = shutil.which("latex") is not None


@dataclass(frozen=True, slots=True)
class IndicatrixScale:
    """One glyph scale shared by every indicatrix representation."""

    value: float
    max_radius: float
    stride: int
    grid_spacing: float


@dataclass(frozen=True, slots=True)
class IndicatrixOutputs:
    speed: Path
    conductivity: Path
    streamline: Path
    finsler: Path
    combined_pdf: Path | None = None


def _sample_indices(shape: tuple[int, int], stride: int) -> tuple[np.ndarray, np.ndarray]:
    if int(stride) < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    ny, nx = shape

    def centered_axis(size: int) -> np.ndarray:
        # Keep the requested spacing while centering the resulting lattice in
        # the available index range instead of anchoring it at index zero.
        count = (size - 1) // int(stride) + 1
        start = int(round(((size - 1) - (count - 1) * int(stride)) / 2.0))
        return start + np.arange(count, dtype=int) * int(stride)

    return centered_axis(ny), centered_axis(nx)


def _validate_field(name: str, field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def compute_shared_glyph_scale(
    speed_field: np.ndarray,
    conductivity_speed_field: np.ndarray,
    vx_field: np.ndarray,
    vy_field: np.ndarray,
    M_field: np.ndarray,
    beta_field: np.ndarray,
    randers_points: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    stride: int,
    grid_spacing: float = 1.0,
    directions: int = 180,
    fill_fraction: float = 0.8,
) -> IndicatrixScale:
    """Compute a single field-unit-to-grid-unit scale for all four glyph types."""
    speed = np.asarray(speed_field, dtype=float)
    shape = speed.shape
    conductivity_speed = _validate_field("conductivity_speed_field", conductivity_speed_field, shape)
    vx = _validate_field("vx_field", vx_field, shape)
    vy = _validate_field("vy_field", vy_field, shape)
    M = np.asarray(M_field, dtype=float)
    beta = np.asarray(beta_field, dtype=float)
    if M.shape != shape + (2, 2):
        raise ValueError(f"M_field shape {M.shape} does not match {shape + (2, 2)}")
    if beta.shape != shape + (2,):
        raise ValueError(f"beta_field shape {beta.shape} does not match {shape + (2,)}")
    if not (np.isfinite(grid_spacing) and grid_spacing > 0.0):
        raise ValueError("grid_spacing must be positive and finite")
    if not (0.0 < fill_fraction <= 1.0):
        raise ValueError("fill_fraction must be in (0, 1]")
    if int(directions) < 24:
        raise ValueError("directions must be at least 24")

    ys, xs = _sample_indices(shape, int(stride))
    max_radius = 0.0
    for j in ys:
        for i in xs:
            for radius in (speed[j, i], conductivity_speed[j, i], np.hypot(vx[j, i], vy[j, i])):
                if np.isfinite(radius) and radius > max_radius:
                    max_radius = float(radius)
            gx, gy = randers_points(M[j, i], beta[j, i], K=int(directions))
            finite_radius = np.hypot(gx, gy)
            if np.any(np.isfinite(finite_radius)):
                max_radius = max(max_radius, float(np.nanmax(finite_radius)))

    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError("Could not compute a positive finite glyph radius")
    scale = float(fill_fraction) * float(stride) * float(grid_spacing) / max_radius
    return IndicatrixScale(
        value=scale,
        max_radius=max_radius,
        stride=int(stride),
        grid_spacing=float(grid_spacing),
    )


def _normalized_extent(result: step5.PathSolveResult, i_y: float) -> tuple[float, float, float, float]:
    return step5._normalized_extent(result.x, result.y, i_y)


def _make_k_norm(K_field: np.ndarray) -> LogNorm:
    positive = K_field[K_field > 0.0]
    if positive.size == 0:
        raise ValueError("K_field must contain positive values")
    return LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))


def _add_background(
    ax: Any,
    fig: Any,
    K_field: np.ndarray,
    result: step5.PathSolveResult,
    i_y: float,
    norm: LogNorm,
    *,
    add_colorbar: bool = True,
) -> Any:
    image = ax.imshow(
        K_field,
        origin="lower",
        cmap=cmc.batlowW_r,
        norm=norm,
        interpolation="nearest",
        extent=_normalized_extent(result, i_y),
    )
    if add_colorbar:
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(r"$K$", fontsize=AXIS_LABEL_FONTSIZE)
    return image


def _add_path(ax: Any, result: step5.PathSolveResult, path: np.ndarray, label: str, color: Any, i_y: float) -> None:
    lw = 1.8
    effect = [pe.Stroke(linewidth=lw + 1.8, foreground="white"), pe.Normal()]
    ax.plot(
        result.source_xy[:, 0] / i_y,
        result.source_xy[:, 1] / i_y,
        color="purple",
        linewidth=1.2,
        label="_nolegend_",
        path_effects=[pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()],
        zorder=6,
    )
    ax.plot(
        result.target_xy[:, 0] / i_y,
        result.target_xy[:, 1] / i_y,
        color="purple",
        linewidth=1.2,
        label="_nolegend_",
        path_effects=[pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()],
        zorder=6,
    )
    path_arr = np.asarray(path, dtype=float)
    if path_arr.ndim != 2 or path_arr.shape[0] < 2 or path_arr.shape[1] != 2:
        raise ValueError(f"{label} must have shape (n, 2) with n >= 2")
    ax.plot(
        path_arr[:, 0] / i_y,
        path_arr[:, 1] / i_y,
        color=color,
        linewidth=lw,
        label=label,
        path_effects=effect,
        zorder=7,
    )


def _add_sample_points(
    ax: Any,
    result: step5.PathSolveResult,
    i_y: float,
    ys: np.ndarray,
    xs: np.ndarray,
) -> None:
    """Mark every sampled glyph center with the same visible point marker."""
    X, Y = np.meshgrid(result.x[xs] / i_y, result.y[ys] / i_y)
    ax.scatter(
        X.ravel(),
        Y.ravel(),
        s=SAMPLE_POINT_SIZE,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=SAMPLE_POINT_LINEWIDTH,
        zorder=5,
    )


def _set_equal_tick_spacing(ax: Any, extent: tuple[float, float, float, float]) -> None:
    """Use one numeric major-tick interval for both normalized axes."""
    x_span = float(extent[1] - extent[0])
    y_span = float(extent[3] - extent[2])
    tick_step = min(x_span, y_span) / 5.0
    if not np.isfinite(tick_step) or tick_step <= 0.0:
        return
    locator = MultipleLocator(tick_step)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))


def _set_title(ax: Any, title: str) -> None:
    if LATEX_AVAILABLE and r"\mathfrak" in title:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
        ax.set_title(title, usetex=True)
    else:
        fallback = (
            title.replace(r"\mathfrak", r"\mathrm")
            .replace(r"\mathsf", r"\mathrm")
            .replace(r"\boldsymbol", r"\mathbf")
        )
        ax.set_title(fallback)


def _finish_axes(
    ax: Any,
    title: str,
    result: step5.PathSolveResult,
    i_y: float,
    *,
    show_legend: bool = True,
    show_ylabel: bool = True,
) -> None:
    _set_title(ax, title)
    ax.set_xlabel(r"$x/I_Y$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$y/I_Y$" if show_ylabel else "", fontsize=AXIS_LABEL_FONTSIZE)
    ax.title.set_fontsize(TITLE_FONTSIZE)
    extent = _normalized_extent(result, i_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    _set_equal_tick_spacing(ax, extent)
    if show_legend:
        ax.legend(fontsize=11, loc="lower left", facecolor="white", edgecolor="black", framealpha=0.9)


def _create_figure(
    title: str,
    K_field: np.ndarray,
    result: step5.PathSolveResult,
    i_y: float,
    norm: LogNorm,
    plot_glyphs: Callable[[Any], None],
    path: np.ndarray,
    path_label: str,
    path_color: Any,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    _add_background(ax, fig, K_field, result, i_y, norm)
    plot_glyphs(ax)
    _add_path(ax, result, path, path_label, path_color, i_y)
    _finish_axes(ax, title, result, i_y)
    fig.tight_layout()
    return fig


def _scalar_glyph_plotter(
    field: np.ndarray,
    result: step5.PathSolveResult,
    i_y: float,
    scale: IndicatrixScale,
    *,
    stride: int,
    point_stride: int,
    directions: int,
    color: Any,
) -> Callable[[Any], None]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(directions), endpoint=True)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ys, xs = _sample_indices(field.shape, stride)
    point_ys, point_xs = _sample_indices(field.shape, point_stride)

    def plot(ax: Any) -> None:
        for j in ys:
            for i in xs:
                radius = float(field[j, i])
                if not np.isfinite(radius) or radius <= 0.0:
                    continue
                x0 = float(result.x[i]) / i_y
                y0 = float(result.y[j]) / i_y
                x = x0 + scale.value * radius * cos_theta / i_y
                y = y0 + scale.value * radius * sin_theta / i_y
                line, = ax.plot(x, y, color=color, linewidth=GLYPH_LINEWIDTH, alpha=0.9, zorder=3)
                line.set_path_effects([pe.Stroke(linewidth=GLYPH_OUTLINE_WIDTH, foreground="black"), pe.Normal()])
        _add_sample_points(ax, result, i_y, point_ys, point_xs)

    return plot


def _streamline_glyph_plotter(
    vx: np.ndarray,
    vy: np.ndarray,
    result: step5.PathSolveResult,
    i_y: float,
    scale: IndicatrixScale,
    *,
    stride: int,
    point_stride: int,
) -> Callable[[Any], None]:
    ys, xs = _sample_indices(vx.shape, stride)
    point_ys, point_xs = _sample_indices(vx.shape, point_stride)
    X, Y = np.meshgrid(result.x[xs] / i_y, result.y[ys] / i_y)
    U = scale.value * vx[np.ix_(ys, xs)] / i_y
    V = scale.value * vy[np.ix_(ys, xs)] / i_y

    def plot(ax: Any) -> None:
        quiver = ax.quiver(
            X,
            Y,
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="white",
            edgecolor="black",
            linewidth=STREAMLINE_ARROW_EDGE_WIDTH,
            width=0.005,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
            zorder=4,
        )
        _add_sample_points(ax, result, i_y, point_ys, point_xs)

    return plot


def _finsler_glyph_plotter(
    M_field: np.ndarray,
    beta_field: np.ndarray,
    randers_points: Callable[..., tuple[np.ndarray, np.ndarray]],
    result: step5.PathSolveResult,
    i_y: float,
    scale: IndicatrixScale,
    *,
    stride: int,
    point_stride: int,
    directions: int,
) -> Callable[[Any], None]:
    ys, xs = _sample_indices(M_field.shape[:2], stride)
    point_ys, point_xs = _sample_indices(M_field.shape[:2], point_stride)

    def plot(ax: Any) -> None:
        for j in ys:
            for i in xs:
                gx, gy = randers_points(M_field[j, i], beta_field[j, i], K=int(directions))
                x = float(result.x[i]) / i_y + scale.value * gx / i_y
                y = float(result.y[j]) / i_y + scale.value * gy / i_y
                line, = ax.plot(x, y, color="white", linewidth=GLYPH_LINEWIDTH, alpha=0.95, zorder=3)
                line.set_path_effects([pe.Stroke(linewidth=GLYPH_OUTLINE_WIDTH, foreground="black"), pe.Normal()])
        _add_sample_points(ax, result, i_y, point_ys, point_xs)

    return plot


def _build_stem(field_size: int, std: float, length_scale: float, seed: int) -> str:
    return (
        f"grf_size{field_size}_std{step5._value_slug(std)}_"
        f"ls{step5._value_slug(length_scale)}_seed{int(seed)}"
    )


def generate_and_plot(
    cfg: step1.VisualizationConfig,
    *,
    field_size: int,
    grf_std: float = 1.0,
    grf_length_scale: float | None = None,
    grf_seed: int = 0,
    grf_minpadding: int = 0,
    dh: float | None = None,
    output_dir: Path | None = None,
    indicatrix_stride: int = 16,
    sample_point_stride: int | None = None,
    indicatrix_directions: int = 180,
    indicatrix_size: float = 1.5,
    fm_asr_perp_ratio: float = 0.2,
    fm_asr_back_factor: float = 5.0,
    fm_asr_min_speed: float = 1e-12,
    merge_pdf: bool = True,
    source_x: int | None = None,
    target_x: int | None = None,
    source_y_start: int | None = None,
    source_y_end: int | None = None,
    target_y_start: int | None = None,
    target_y_end: int | None = None,
) -> IndicatrixOutputs:
    if int(field_size) < 2:
        raise ValueError("field_size must be at least 2")
    if source_x is None and int(field_size) < 20:
        raise ValueError(
            "field_size must be at least 20 when using Step 5's default source geometry; "
            "provide --source-x and compatible source/target spans for smaller fields"
        )
    if not (np.isfinite(grf_std) and grf_std > 0.0):
        raise ValueError("grf_std must be positive and finite")
    length_scale = cfg.length_scale if grf_length_scale is None else float(grf_length_scale)
    if not (np.isfinite(length_scale) and length_scale > 0.0):
        raise ValueError("grf_length_scale must be positive and finite")
    if int(grf_minpadding) < 0:
        raise ValueError("grf_minpadding must be non-negative")
    point_stride = int(indicatrix_stride if sample_point_stride is None else sample_point_stride)
    if point_stride < 1:
        raise ValueError("sample_point_stride must be >= 1")
    if not (np.isfinite(indicatrix_size) and indicatrix_size > 0.0):
        raise ValueError("indicatrix_size must be positive and finite")
    if not (np.isfinite(fm_asr_perp_ratio) and fm_asr_perp_ratio > 0.0):
        raise ValueError("fm_asr_perp_ratio must be positive and finite")
    if not (np.isfinite(fm_asr_back_factor) and fm_asr_back_factor >= 1.0):
        raise ValueError("fm_asr_back_factor must be at least 1 and finite")
    if not (np.isfinite(fm_asr_min_speed) and fm_asr_min_speed > 0.0):
        raise ValueError("fm_asr_min_speed must be positive and finite")

    logk_field = step5.generate_grf_logk(
        field_size=int(field_size),
        std=float(grf_std),
        length_scale=length_scale,
        seed=int(grf_seed),
        minpadding=int(grf_minpadding),
    )
    K_field = np.exp(logk_field)
    dh_eff = cfg.dh if dh is None else float(dh)

    result = step5.solve_single_field(
        logk_field,
        field_size=int(field_size),
        dh=dh_eff,
        solver="pollock",
        backend="cpu",
        velocity_location="face",
        align_with_modflow_velocity=False,
        fmm_backtracking="continuous",
        fmm_continuous_step_size=0.5,
        linearized_fp=False,
        linearized_scheme="linearized",
        linearized_outer_max=20,
        linearized_inner_max_sweeps=200,
        linearized_inner_tol=1e-8,
        linearized_eikonal_tol=1e-6,
        linearized_damping=0.8,
        linearized_max_deltat=None,
        linearized_monotone_residual=True,
        linearized_verbose=False,
        check_path_geometry=True,
        source_x=source_x,
        target_x=target_x,
        source_y_start=source_y_start,
        source_y_end=source_y_end,
        target_y_start=target_y_start,
        target_y_end=target_y_end,
        density_particle=20,
        num_particles=None,
        eps_speed=1e-30,
        ds_min=1e-20,
        ds_max=0.5,
        tol=1e-4,
        y_policy="stop",
        max_points=100_000,
        max_iters=1_000_000,
        rk45_min_success_frac=0.5,
        rk45_max_bad_frac=0.1,
        fsm_fp=False,
        fm_asr_fp=True,
        fm_asr_perp_ratio=float(fm_asr_perp_ratio),
        fm_asr_back_factor=float(fm_asr_back_factor),
        fm_asr_min_speed=float(fm_asr_min_speed),
        fm_asr_max_steps=200_000,
        pypar2_fp=False,
    )
    if result.fp_fm_asr is None:
        raise RuntimeError("FM-ASR path was not produced")

    modules = step5._import_solver_modules()
    build_randers_from_W = modules["build_randers_from_W"]
    randers_indicatrix_points = modules["randers_indicatrix_points"]

    vx = np.asarray(result.vx_cell, dtype=float)
    vy = np.asarray(result.vy_cell, dtype=float)
    W = np.stack((vx, vy), axis=-1)
    _c_iso, M_field, beta_field, _directions = build_randers_from_W(
        W,
        perp_ratio=float(fm_asr_perp_ratio),
        back_factor=float(fm_asr_back_factor),
        min_speed=float(fm_asr_min_speed),
        isotropic_only=False,
    )
    speed_field = np.hypot(vx, vy)
    conductivity_speed_field = K_field * abs(float(dh_eff))
    scale = compute_shared_glyph_scale(
        speed_field,
        conductivity_speed_field,
        vx,
        vy,
        M_field,
        beta_field,
        randers_indicatrix_points,
        stride=int(indicatrix_stride),
        directions=int(indicatrix_directions),
    )
    scale = IndicatrixScale(
        value=scale.value * float(indicatrix_size),
        max_radius=scale.max_radius,
        stride=scale.stride,
        grid_spacing=scale.grid_spacing,
    )
    K_norm = _make_k_norm(K_field)

    i_y = cfg.length_scale * float(field_size)
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _build_stem(int(field_size), float(grf_std), length_scale, int(grf_seed))
    output_paths = IndicatrixOutputs(
        speed=out_dir / f"{stem}_indicatrix_speed.png",
        conductivity=out_dir / f"{stem}_indicatrix_conductivity.png",
        streamline=out_dir / f"{stem}_indicatrix_streamline.png",
        finsler=out_dir / f"{stem}_indicatrix_finsler.png",
        combined_pdf=(out_dir / f"{stem}_indicatrix_combined.pdf") if merge_pdf else None,
    )

    figures = (
        (
            output_paths.streamline,
            r"$C_{\mathfrak d \mid \mathsf S}(\mathbf x, \boldsymbol{\tau})$",
            _streamline_glyph_plotter(
                vx,
                vy,
                result,
                i_y,
                scale,
                stride=indicatrix_stride,
                point_stride=point_stride,
            ),
            result.fp,
            r"$\gamma^\star_{\mathfrak{d}\mid\mathsf{S}}$",
            step5.COLORS["fp"],
        ),
        (
            output_paths.finsler,
            r"$C_{\mathfrak d \mid \mathsf F}(\mathbf x, \boldsymbol{\tau})$",
            _finsler_glyph_plotter(
                M_field,
                beta_field,
                randers_indicatrix_points,
                result,
                i_y,
                scale,
                stride=indicatrix_stride,
                point_stride=point_stride,
                directions=indicatrix_directions,
            ),
            result.fp_fm_asr,
            r"$\gamma^\star_{\mathfrak{d}\mid\mathsf{F}}$",
            step5.COLORS["fpFmAsr"],
        ),
        (
            output_paths.speed,
            r"$C_{\mathfrak d \mid \mathsf I}(\mathbf x, \boldsymbol{\tau})$",
            _scalar_glyph_plotter(
                speed_field,
                result,
                i_y,
                scale,
                stride=indicatrix_stride,
                point_stride=point_stride,
                directions=indicatrix_directions,
                color="white",
            ),
            result.fp_fmm,
            r"$\gamma^\star_{\mathfrak{d}\mid\mathsf{I}}$",
            step5.COLORS["fpFMM"],
        ),
        (
            output_paths.conductivity,
            r"$C_{\mathfrak s \mid \mathsf I}(\mathbf x, \boldsymbol{\tau})$",
            _scalar_glyph_plotter(
                conductivity_speed_field,
                result,
                i_y,
                scale,
                stride=indicatrix_stride,
                point_stride=point_stride,
                directions=indicatrix_directions,
                color="white",
            ),
            result.lrp_fmm,
            r"$\gamma^\star_{\mathfrak{s}\mid\mathsf{I}}$",
            step5.COLORS["lrpFMM"],
        ),
    )
    for path, title, glyph_plotter, path_data, path_label, path_color in figures:
        fig = _create_figure(
            title,
            K_field,
            result,
            i_y,
            K_norm,
            glyph_plotter,
            path_data,
            path_label,
            path_color,
        )
        fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {path}")
    if merge_pdf and output_paths.combined_pdf is not None:
        combined_fig, combined_axes = plt.subplots(
            1,
            4,
            figsize=(20, 5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.asarray(combined_axes).reshape(-1)
        images = []
        for panel_index, (ax, (_path, title, glyph_plotter, path_data, path_label, path_color)) in enumerate(zip(axes, figures)):
            images.append(_add_background(ax, combined_fig, K_field, result, i_y, K_norm, add_colorbar=False))
            glyph_plotter(ax)
            _add_path(ax, result, path_data, path_label, path_color, i_y)
            _finish_axes(ax, title, result, i_y, show_ylabel=panel_index == 0)
        colorbar = combined_fig.colorbar(images[-1], ax=axes.tolist(), fraction=0.015, pad=0.02)
        colorbar.set_label(r"$K$", fontsize=AXIS_LABEL_FONTSIZE)
        combined_fig.savefig(output_paths.combined_pdf, dpi=cfg.figure_dpi, bbox_inches="tight")
        plt.close(combined_fig)
        print(f"Saved figure: {output_paths.combined_pdf}")
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four shared-scale indicatrix/path figures from a seeded log-Gaussian K field."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--field-size", required=True, type=int)
    parser.add_argument("--grf-std", type=float, default=1.0)
    parser.add_argument("--grf-length-scale", type=float, default=None)
    parser.add_argument("--grf-seed", type=int, default=0)
    parser.add_argument("--grf-minpadding", type=int, default=0)
    parser.add_argument("--dh", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--indicatrix-stride", type=int, default=24)
    parser.add_argument(
        "--sample-point-stride",
        type=int,
        default=None,
        help="Grid stride for sample-point markers; defaults to --indicatrix-stride.",
    )
    parser.add_argument("--indicatrix-directions", type=int, default=180)
    parser.add_argument(
        "--indicatrix-size",
        type=float,
        default=1.5,
        help="Multiplicative size of all indicatrix glyphs (default: 1.5).",
    )
    parser.add_argument(
        "--fm-asr-perp-ratio",
        "--finsler-perp-ratio",
        dest="fm_asr_perp_ratio",
        type=float,
        default=0.2,
        help="Finsler/Randers perpendicular metric ratio (default: 0.2).",
    )
    parser.add_argument(
        "--fm-asr-back-factor",
        "--finsler-back-factor",
        dest="fm_asr_back_factor",
        type=float,
        default=5.0,
        help="Finsler/Randers backward-direction penalty (default: 5.0).",
    )
    parser.add_argument(
        "--fm-asr-min-speed",
        "--finsler-min-speed",
        dest="fm_asr_min_speed",
        type=float,
        default=1e-12,
        help="Speed floor used when constructing the Finsler/Randers field (default: 1e-12).",
    )
    parser.add_argument(
        "--merge-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a combined 1x4 PDF with one shared K colorbar (default: enabled).",
    )
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--target-x", type=int, default=None)
    parser.add_argument("--source-y-start", type=int, default=None)
    parser.add_argument("--source-y-end", type=int, default=None)
    parser.add_argument("--target-y-start", type=int, default=None)
    parser.add_argument("--target-y-end", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grf_std <= 0.0 or not np.isfinite(args.grf_std):
        raise SystemExit("--grf-std must be positive and finite")
    if args.grf_length_scale is not None and (args.grf_length_scale <= 0.0 or not np.isfinite(args.grf_length_scale)):
        raise SystemExit("--grf-length-scale must be positive and finite")
    if args.grf_minpadding < 0:
        raise SystemExit("--grf-minpadding must be non-negative")
    if args.indicatrix_stride < 1:
        raise SystemExit("--indicatrix-stride must be >= 1")
    if args.sample_point_stride is not None and args.sample_point_stride < 1:
        raise SystemExit("--sample-point-stride must be >= 1")
    if args.indicatrix_directions < 24:
        raise SystemExit("--indicatrix-directions must be >= 24")
    if args.indicatrix_size <= 0.0 or not np.isfinite(args.indicatrix_size):
        raise SystemExit("--indicatrix-size must be positive and finite")
    if args.fm_asr_perp_ratio <= 0.0 or not np.isfinite(args.fm_asr_perp_ratio):
        raise SystemExit("--fm-asr-perp-ratio must be positive and finite")
    if args.fm_asr_back_factor < 1.0 or not np.isfinite(args.fm_asr_back_factor):
        raise SystemExit("--fm-asr-back-factor must be at least 1 and finite")
    if args.fm_asr_min_speed <= 0.0 or not np.isfinite(args.fm_asr_min_speed):
        raise SystemExit("--fm-asr-min-speed must be positive and finite")

    cfg = step1.load_config(args.config)
    step5.configure_style()
    generate_and_plot(
        cfg,
        field_size=args.field_size,
        grf_std=args.grf_std,
        grf_length_scale=args.grf_length_scale,
        grf_seed=args.grf_seed,
        grf_minpadding=args.grf_minpadding,
        dh=args.dh,
        output_dir=args.output_dir,
        indicatrix_stride=args.indicatrix_stride,
        sample_point_stride=args.sample_point_stride,
        indicatrix_directions=args.indicatrix_directions,
        indicatrix_size=args.indicatrix_size,
        fm_asr_perp_ratio=args.fm_asr_perp_ratio,
        fm_asr_back_factor=args.fm_asr_back_factor,
        fm_asr_min_speed=args.fm_asr_min_speed,
        merge_pdf=args.merge_pdf,
        source_x=args.source_x,
        target_x=args.target_x,
        source_y_start=args.source_y_start,
        source_y_end=args.source_y_end,
        target_y_start=args.target_y_start,
        target_y_end=args.target_y_end,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
