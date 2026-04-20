from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visualization.plots.style import COLORS


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
    return fig, {"kind": "endpoint_std"}
